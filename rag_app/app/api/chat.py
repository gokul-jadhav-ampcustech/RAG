"""
Chat API — Document-First RAG + HITL + Automatic Long-Term Memory

Endpoints:
  POST /conversations                       -> create conversation
  GET  /conversations                       -> list user's conversations
  GET  /conversations/{id}/messages         -> get conversation messages
  DELETE /conversations/{id}               -> delete conversation
  POST /chat                               -> send message
  POST /chat/hitl-action                   -> approve/reject web search
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path, Header
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.database import get_db
from app.db.models import Conversation, Message, HITLPending
from app.schemas.chat import ChatRequest, ChatResponse, ToolResult
from app.schemas.conversation import (
    ConversationHistoryResponse,
    ConversationListResponse,
    ConversationSummary,
    MessageSchema,
)
from app.services import rag
from app.services import mcp_client, mcp_router
from app.services.conversation import (
    create_conversation,
    get_conversation_history,
    get_user_conversations,
    delete_conversation,
    save_user_message,
    save_assistant_message,
    update_conversation_title,
)
from app.services.embeddings import EmbeddingError, embed_text
from app.services.llm import LLMError, _call_groq
from app.services.tools import ToolExecutor
from app.services import web_search as web_search_svc
from app.memory.pipeline import run_memory_pipeline
from app.memory.retriever import retrieve_relevant, format_for_context
from app.api.auth import get_current_user_from_token
from app.db.memory_models import User

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(tags=["chat"])


def _get_optional_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> Optional[User]:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        from app.services import auth
        token = authorization.split(" ")[1]
        payload = auth.verify_token(token)
        return auth.get_user_by_id(db, payload.get("user_id"))
    except Exception:
        return None


# ── Conversation endpoints ────────────────────────────────────────────────────

@router.post("/conversations")
def create_new_conversation(
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    conversation = create_conversation(db, user_id=current_user.id)
    return {"conversation_id": conversation.id, "title": conversation.title}


@router.get("/conversations", response_model=ConversationListResponse)
def list_conversations(
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    conversations = get_user_conversations(db, current_user.id)
    summaries = []
    for conv in conversations:
        msg_count = db.query(Message).filter(Message.conversation_id == conv.id).count()
        summaries.append(ConversationSummary(
            id=conv.id,
            title=conv.title or f"Conversation #{conv.id}",
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            message_count=msg_count,
        ))
    return ConversationListResponse(conversations=summaries, total=len(summaries))


@router.get("/conversations/{conversation_id}/messages", response_model=ConversationHistoryResponse)
def get_conversation_messages(
    conversation_id: int = Path(..., gt=0),
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    conversation = get_conversation_history(db, conversation_id, user_id=current_user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
    messages = [MessageSchema.from_orm(msg) for msg in conversation.messages]
    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        title=conversation.title,
        messages=messages,
        total_messages=len(messages),
    )


@router.delete("/conversations/{conversation_id}")
def remove_conversation(
    conversation_id: int = Path(..., gt=0),
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    success = delete_conversation(db, conversation_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation deleted"}


# ── Main chat endpoint ────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[User] = Depends(_get_optional_user),
    db: Session = Depends(get_db)
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    user_id = current_user.id if current_user else None
    logger.info("[CHAT] User message received (user_id=%s)", user_id)

    # ── Conversation setup ────────────────────────────────────
    if request.conversation_id:
        conversation = get_conversation_history(db, request.conversation_id, user_id=user_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conversation_id = conversation.id
    else:
        conversation = create_conversation(db, user_id=user_id)
        conversation_id = conversation.id

    user_msg = save_user_message(db, conversation_id, request.question)

    if not conversation.title:
        title = request.question[:60].strip() + ("…" if len(request.question) > 60 else "")
        update_conversation_title(db, conversation_id, title, user_id=user_id)

    # ── Tool detection (existing local tools) ────────────────
    try:
        tool_name, tool_params = ToolExecutor.detect_tool(request.question)
        if tool_name:
            tool_result_data = ToolExecutor.execute(tool_name, tool_params)
            if tool_result_data.get("status") == "success":
                if tool_name == "calculator":
                    answer = f"The result of {tool_result_data.get('expression')} is **{tool_result_data.get('result')}**."
                elif tool_name == "weather":
                    answer = (f"Weather in {tool_result_data.get('city')}: "
                              f"{tool_result_data.get('temperature')}°F, "
                              f"Humidity {tool_result_data.get('humidity')}%, "
                              f"Wind {tool_result_data.get('wind_speed')} mph")
                elif tool_name == "youtube":
                    answer = f"Here are YouTube videos about '{tool_result_data.get('query')}':"
                else:
                    answer = str(tool_result_data)
            else:
                answer = f"Tool error: {tool_result_data.get('message', 'Unknown error')}"
            save_assistant_message(db, conversation_id, answer)
            return ChatResponse(
                status="tool_result",
                answer=answer,
                tool_result=ToolResult(tool=tool_name, status=tool_result_data.get("status"), data=tool_result_data),
                conversation_id=conversation_id,
            )
    except Exception as e:
        logger.warning("[CHAT] Tool detection error: %s", e)

    # ── MCP tool routing ──────────────────────────────────────
    mcp_route = "rag"           # default: skip MCP
    mcp_tool_name: str | None = None
    mcp_tool_result: str | None = None

    if settings.mcp_enabled:
        try:
            mcp_tools = mcp_client.get_tools()
            if mcp_tools:
                intent = mcp_router.classify_intent(request.question, mcp_tools)
                mcp_route = intent["route"]
                logger.info(
                    "[MCP] route=%s tool=%s reason=%s",
                    mcp_route, intent.get("tool"), intent.get("reason"),
                )

                if mcp_route in ("mcp", "rag+mcp") and intent.get("tool"):
                    success, mcp_tool_result = mcp_router.execute_mcp_tool(
                        intent["tool"], intent.get("arguments") or {}
                    )
                    if success:
                        mcp_tool_name = intent["tool"]
                    else:
                        logger.warning("[MCP] Tool failed — degrading to rag")
                        mcp_route = "rag"

                # Pure MCP: answer entirely from tool result, skip RAG
                if mcp_route == "mcp" and mcp_tool_result:
                    answer = mcp_router.format_mcp_only_answer(
                        request.question, mcp_tool_name, mcp_tool_result
                    )
                    save_assistant_message(db, conversation_id, answer)
                    logger.info("[ANSWER] Response generated (type=mcp_tool)")
                    return ChatResponse(
                        status="answered",
                        answer=answer,
                        requires_human_approval=False,
                        conversation_id=conversation_id,
                        response_type="mcp_tool",
                        retrieval_stats={"mcp_tool": mcp_tool_name, "route": mcp_route},
                    )
        except Exception as exc:
            logger.warning("[MCP] Routing error (non-fatal): %s", exc)
            mcp_route = "rag"

    # ── Retrieve relevant long-term memories ──────────────────
    memory_context = ""
    if user_id:
        try:
            query_emb = embed_text(request.question)
            memories = retrieve_relevant(db, user_id, query_emb)
            memory_context = format_for_context(memories)
            if memory_context:
                logger.info("[MEMORY] Relevant memories retrieved")
        except Exception as e:
            logger.warning("[MEMORY] Memory retrieval skipped: %s", e)

    # ── Document-first RAG ────────────────────────────────────
    logger.info("[DOCUMENT_SEARCH] Search started")
    try:
        result = rag.answer_question(db, request.question, memory_context=memory_context)
    except EmbeddingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("[CHAT] Unexpected error: %s", exc)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from exc

    # ── HITL: pause and ask user before any web search ────────
    if result.status == "awaiting_human_approval":
        logger.info("[HITL] Waiting for user approval")
        expires = datetime.utcnow() + timedelta(minutes=settings.hitl_pending_ttl_minutes)
        pending = HITLPending(
            user_id=user_id or 0,
            conversation_id=conversation_id,
            pending_query=request.question,
            memory_context=memory_context,
            status="pending",
            expires_at=expires,
        )
        db.add(pending)
        db.commit()
        save_assistant_message(db, conversation_id, result.answer)
        return ChatResponse(
            status="awaiting_human_approval",
            answer=result.answer,
            requires_human_approval=True,
            conversation_id=conversation_id,
            response_type="hitl_pending",
            retrieval_stats={"pending_id": pending.id},
        )

    logger.info("[DOCUMENT_SEARCH] Search completed")
    if result.sources:
        logger.info("[DOCUMENT_SEARCH] Relevant context found (%d sources)", len(result.sources))

    # ── RAG + MCP synthesis ────────────────────────────────────
    final_answer = result.answer
    final_response_type = result.response_type
    if mcp_route == "rag+mcp" and mcp_tool_result and mcp_tool_name:
        logger.info("[MCP] Synthesizing RAG + MCP answer")
        final_answer = mcp_router.synthesize_rag_and_mcp(
            request.question, result.answer, mcp_tool_name, mcp_tool_result
        )
        final_response_type = "rag+mcp"

    save_assistant_message(db, conversation_id, final_answer)
    logger.info("[ANSWER] Response generated (type=%s)", final_response_type)

    # ── Automatic memory extraction (background, non-blocking) ─
    if user_id:
        recent_msgs = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(6)
            .all()
        )
        context_for_memory = [
            {"role": m.role, "content": m.content}
            for m in reversed(recent_msgs)
        ]
        background_tasks.add_task(
            run_memory_pipeline,
            db=db,
            user_id=user_id,
            user_message=request.question,
            conversation_id=conversation_id,
            source_message_id=user_msg.id,
            conversation_context=context_for_memory,
        )

    return ChatResponse(
        status=result.status,
        answer=final_answer,
        requires_human_approval=False,
        sources=result.sources,
        response_type=final_response_type,
        conversation_id=conversation_id,
        retrieval_stats=result.retrieval_stats,
    )


# ── HITL Action endpoint ──────────────────────────────────────────────────────

@router.post("/chat/hitl-action", response_model=ChatResponse)
def hitl_action(
    body: dict,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """
    Handle human approval/rejection of web search.
    Body: { conversation_id, action: "search_web"|"documents_only", pending_id }
    """
    conversation_id = body.get("conversation_id")
    action = body.get("action")

    if action not in ("search_web", "documents_only"):
        raise HTTPException(status_code=400, detail="action must be 'search_web' or 'documents_only'")

    conversation = get_conversation_history(db, conversation_id, user_id=current_user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    pending = (
        db.query(HITLPending)
        .filter(
            HITLPending.conversation_id == conversation_id,
            HITLPending.user_id == current_user.id,
            HITLPending.status == "pending",
            HITLPending.expires_at > datetime.utcnow(),
        )
        .order_by(HITLPending.created_at.desc())
        .first()
    )
    if not pending:
        raise HTTPException(status_code=404, detail="No pending approval found or it has expired")

    query = pending.pending_query
    memory_context = pending.memory_context or ""
    pending.status = "resolved"
    db.commit()

    # ── User rejected web search ──────────────────────────────
    if action == "documents_only":
        logger.info("[HITL] User rejected web search — documents only")
        answer = (
            "Understood — I'll stick to your uploaded documents. "
            "The answer may not be available in the current documents. "
            "Try uploading relevant files or rephrasing your question."
        )
        save_assistant_message(db, conversation_id, answer)
        return ChatResponse(
            status="documents_only",
            answer=answer,
            requires_human_approval=False,
            conversation_id=conversation_id,
            response_type="documents_only",
        )

    # ── User approved web search ──────────────────────────────
    logger.info("[HITL] User approved web search")
    logger.info("[WEB_SEARCH] Search started")

    try:
        results = web_search_svc.search(query)
        logger.info("[WEB_SEARCH] Results received: %d", len(results))

        if not results:
            answer = "I searched the web but couldn't find relevant results. Try rephrasing your question."
            save_assistant_message(db, conversation_id, answer)
            return ChatResponse(
                status="web_no_results",
                answer=answer,
                conversation_id=conversation_id,
                response_type="web_search",
            )

        web_context = web_search_svc.format_results_for_llm(results)
        full_context = f"{memory_context}\n\n{web_context}".strip() if memory_context else web_context

        from app.prompts.rag_prompt import GENERAL_SYSTEM_PROMPT
        user_prompt = (
            f"{full_context}\n\n"
            f"User question: {query}\n\n"
            f"Answer using the web search results above. "
            f"Cite sources where relevant. "
            f"Treat web content as external and potentially unverified."
        )
        answer = _call_groq(
            messages=[
                {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        sources = [
            {"filename": r.url, "chunk_id": 0, "chunk_index": i,
             "similarity": 0.0, "keyword_score": 0.0, "rrf_score": 0.0, "rank": i + 1}
            for i, r in enumerate(results)
        ]

        save_assistant_message(db, conversation_id, answer)
        logger.info("[ANSWER] Response generated (type=web_search)")

        # Auto memory after web answer too
        user_msg_obj = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id, Message.role == "user")
            .order_by(Message.created_at.desc())
            .first()
        )
        background_tasks.add_task(
            run_memory_pipeline,
            db=db,
            user_id=current_user.id,
            user_message=query,
            conversation_id=conversation_id,
            source_message_id=user_msg_obj.id if user_msg_obj else None,
            conversation_context=[],
        )

        return ChatResponse(
            status="answered",
            answer=answer,
            requires_human_approval=False,
            sources=sources,
            conversation_id=conversation_id,
            response_type="web_search",
        )

    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("[WEB_SEARCH] Error: %s", exc)
        raise HTTPException(status_code=500, detail="Web search failed.") from exc
