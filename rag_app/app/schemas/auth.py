"""
Authentication Schemas
======================

Pydantic models for authentication requests and responses.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = Field(None, max_length=100)


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    created_at: datetime
    is_active: bool

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse
    session_id: str


class MemorySessionResponse(BaseModel):
    session_id: str
    created_at: datetime
    last_accessed: datetime
    expires_at: Optional[datetime]
    is_active: bool

    class Config:
        from_attributes = True