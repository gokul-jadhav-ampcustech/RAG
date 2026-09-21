from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    conversation_id: int | None = None  # Optional: use existing conversation or create new


class Source(BaseModel):
    filename: str
    chunk_id: int
    chunk_index: int
    similarity: float
    keyword_score: float = 0.0
    rrf_score: float = 0.0
    rank: int = 0


class ToolResult(BaseModel):
    tool: str  # "calculator" | "weather" | "youtube"
    status: str  # "success" | "error"
    data: dict  # tool-specific result data


class ChatResponse(BaseModel):
    status: str  # "answered" | "general" | "needs_approval" | "tool_result" | "no_documents"
    answer: str
    requires_human_approval: bool = False
    sources: list[Source] = []
    tool_result: ToolResult | None = None
    conversation_id: int  # Return conversation ID for frontend to use
    response_type: str = "document_based"  # "document_based" | "general_ai" | "hybrid"
    retrieval_stats: dict = {}  # Enterprise RAG stats


class ApproveExternalSearchRequest(BaseModel):
    question: str = Field(..., min_length=1)


class ApproveExternalSearchResponse(BaseModel):
    status: str
    message: str
