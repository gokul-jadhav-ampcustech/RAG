"""
Authentication Service
======================

Handles user authentication, session management, and security.
"""
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Optional
import logging

from fastapi import HTTPException, status
from sqlalchemy.orm import Session
import jwt

from app.config import get_settings
from app.db.memory_models import User, MemorySession

logger = logging.getLogger(__name__)
settings = get_settings()

# Simple password hashing using hashlib
def hash_password(password: str) -> str:
    """Hash a password for storing using SHA-256 with salt."""
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{password_hash}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    try:
        salt, stored_hash = hashed_password.split(':')
        password_hash = hashlib.sha256((plain_password + salt).encode()).hexdigest()
        return password_hash == stored_hash
    except ValueError:
        return False


# JWT settings
SECRET_KEY = settings.groq_api_key[:32]  # Use first 32 chars as JWT secret
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60  # 30 days


class AuthenticationError(Exception):
    """Raised when authentication fails."""


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> dict:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError as e:
        logger.error(f"Token verification failed: {e}")
        raise AuthenticationError("Invalid token")


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username."""
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email."""
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Get user by ID."""
    return db.query(User).filter(User.id == user_id).first()


def create_user(db: Session, username: str, email: str, password: str, full_name: str = None) -> User:
    """Create a new user."""
    try:
        # Check if username or email already exists
        if get_user_by_username(db, username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        if get_user_by_email(db, email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        hashed_password = hash_password(password)
        user = User(
            username=username,
            email=email,
            full_name=full_name,
            hashed_password=hashed_password,
            created_at=datetime.utcnow()
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        logger.info(f"Created user: {username}")
        return user
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User creation error: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User creation failed"
        )


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate a user with username and password."""
    try:
        user = get_user_by_username(db, username)
        if not user or not verify_password(password, user.hashed_password):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        return user
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None


def create_memory_session(db: Session, user_id: int) -> MemorySession:
    """Create a new memory session for short-term memory."""
    try:
        session = MemorySession(
            user_id=user_id,
            session_id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=24),  # 24-hour session
            is_active=True
        )
        
        db.add(session)
        db.commit()
        db.refresh(session)
        
        return session
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        db.rollback()
        raise


def get_active_memory_session(db: Session, user_id: int) -> Optional[MemorySession]:
    """Get the active memory session for a user."""
    return db.query(MemorySession).filter(
        MemorySession.user_id == user_id,
        MemorySession.is_active == True,
        MemorySession.expires_at > datetime.utcnow()
    ).first()


def get_or_create_memory_session(db: Session, user_id: int) -> MemorySession:
    """Get existing active session or create new one."""
    try:
        session = get_active_memory_session(db, user_id)
        if not session:
            session = create_memory_session(db, user_id)
        else:
            # Update last accessed
            session.last_accessed = datetime.utcnow()
            db.commit()
        
        return session
    except Exception as e:
        logger.error(f"Session get/create error: {e}")
        raise


def deactivate_memory_session(db: Session, session_id: str) -> bool:
    """Deactivate a memory session (logout)."""
    try:
        session = db.query(MemorySession).filter(
            MemorySession.session_id == session_id
        ).first()
        
        if session:
            session.is_active = False
            db.commit()
            return True
        
        return False
    except Exception as e:
        logger.error(f"Session deactivation error: {e}")
        return False