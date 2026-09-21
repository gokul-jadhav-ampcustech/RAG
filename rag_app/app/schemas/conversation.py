"""
Schemas for conversation history.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class MessageSchema(BaseModel):
    id: int
    conversation_id: int
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationSchema(BaseModel):
    id: int
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    messages: list[MessageSchema] = []

    class Config:
        from_attributes = True


class ConversationSummary(BaseModel):
    id: int
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    class Config:
        from_attributes = True


class ConversationHistoryResponse(BaseModel):
    conversation_id: int
    title: Optional[str] = None
    messages: list[MessageSchema]
    total_messages: int


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary]
    total: int
