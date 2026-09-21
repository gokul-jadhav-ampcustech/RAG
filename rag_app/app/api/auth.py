"""
Authentication API Endpoints
============================

Handles user registration, login, logout, and session management.
"""
from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services import auth
from app.schemas.auth import (
    UserCreate, UserLogin, UserResponse, TokenResponse,
    MemorySessionResponse
)

router = APIRouter(tags=["authentication"])


@router.post("/register", response_model=UserResponse)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    try:
        user = auth.create_user(
            db=db,
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name
        )
        
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            created_at=user.created_at,
            is_active=user.is_active
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user and return access token."""
    user = auth.authenticate_user(db, user_credentials.username, user_credentials.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=access_token_expires
    )
    
    # Create or get memory session
    memory_session = auth.get_or_create_memory_session(db, user.id)
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            created_at=user.created_at,
            is_active=user.is_active
        ),
        session_id=memory_session.session_id
    )


@router.post("/logout")
def logout(
    session_id: str,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None)
):
    """Logout user and deactivate memory session."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header"
        )
    
    try:
        token = authorization.split(" ")[1]
        payload = auth.verify_token(token)
        
        # Deactivate memory session
        success = auth.deactivate_memory_session(db, session_id)
        
        return {
            "message": "Successfully logged out",
            "session_deactivated": success
        }
    
    except auth.AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


@router.get("/me", response_model=UserResponse)
def get_current_user(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None)
):
    """Get current user information."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header"
        )
    
    try:
        token = authorization.split(" ")[1]
        payload = auth.verify_token(token)
        user_id = payload.get("user_id")
        
        user = auth.get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            created_at=user.created_at,
            is_active=user.is_active
        )
    
    except auth.AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


@router.get("/session", response_model=MemorySessionResponse)
def get_memory_session(
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None)
):
    """Get current memory session information."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header"
        )
    
    try:
        token = authorization.split(" ")[1]
        payload = auth.verify_token(token)
        user_id = payload.get("user_id")
        
        session = auth.get_active_memory_session(db, user_id)
        if not session:
            # Create new session if none exists
            session = auth.create_memory_session(db, user_id)
        
        return MemorySessionResponse(
            session_id=session.session_id,
            created_at=session.created_at,
            last_accessed=session.last_accessed,
            expires_at=session.expires_at,
            is_active=session.is_active
        )
    
    except auth.AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


def get_current_user_from_token(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> auth.User:
    """Dependency to get current authenticated user."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    try:
        token = authorization.split(" ")[1]
        payload = auth.verify_token(token)
        user_id = payload.get("user_id")
        
        user = auth.get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return user
    
    except auth.AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"}
        )