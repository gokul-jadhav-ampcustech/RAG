"""
Service for managing conversation history with user isolation.
"""
import logging
from datetime import datetime

from sqlalchemy.orm import Session

from app.db.models import Conversation, Message

logger = logging.getLogger(__name__)


def create_conversation(db: Session, user_id: int = None, title: str = None) -> Conversation:
    conversation = Conversation(user_id=user_id, title=title)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    logger.info("Created conversation %d for user %s", conversation.id, user_id or "anonymous")
    return conversation


def update_conversation_title(db: Session, conversation_id: int, title: str, user_id: int = None) -> bool:
    query = db.query(Conversation).filter(Conversation.id == conversation_id)
    if user_id is not None:
        query = query.filter(Conversation.user_id == user_id)
    conv = query.first()
    if conv:
        conv.title = title
        conv.updated_at = datetime.utcnow()
        db.commit()
        return True
    return False


def save_user_message(db: Session, conversation_id: int, content: str) -> Message:
    message = Message(conversation_id=conversation_id, role="user", content=content)
    db.add(message)
    # Update conversation updated_at
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv:
        conv.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(message)
    return message


def save_assistant_message(db: Session, conversation_id: int, content: str) -> Message:
    message = Message(conversation_id=conversation_id, role="assistant", content=content)
    db.add(message)
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv:
        conv.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(message)
    return message


def get_conversation_history(db: Session, conversation_id: int, user_id: int = None) -> Conversation | None:
    query = db.query(Conversation).filter(Conversation.id == conversation_id)
    if user_id is not None:
        query = query.filter(Conversation.user_id == user_id)
    conversation = query.first()
    if conversation:
        messages = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at).all()
        conversation.messages = messages
    return conversation


def get_user_conversations(db: Session, user_id: int, limit: int = 50) -> list[Conversation]:
    """Get all conversations for a specific user, ordered by most recent."""
    return db.query(Conversation).filter(
        Conversation.user_id == user_id
    ).order_by(Conversation.updated_at.desc()).limit(limit).all()


def delete_conversation(db: Session, conversation_id: int, user_id: int) -> bool:
    conv = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id
    ).first()
    if conv:
        db.delete(conv)
        db.commit()
        return True
    return False


def get_all_conversations(db: Session, limit: int = 50) -> list[Conversation]:
    return db.query(Conversation).order_by(
        Conversation.updated_at.desc()
    ).limit(limit).all()
