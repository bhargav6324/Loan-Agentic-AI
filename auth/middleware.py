"""
Auth Middleware — JWT Verification for Protected Routes
───────────────────────────────────────────────────────
FastAPI dependency that extracts and validates JWT from Authorization header.
"""

from __future__ import annotations

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from auth.jwt_handler import decode_token

security = HTTPBearer()


class CurrentUser:
    """Represents the authenticated user extracted from JWT."""

    def __init__(self, user_id: str, role: str):
        self.user_id = user_id
        self.role = role

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> CurrentUser:
    """
    FastAPI dependency — extract and validate JWT.
    Usage: current_user: CurrentUser = Depends(get_current_user)
    """
    token = credentials.credentials
    payload = decode_token(token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type — use access token",
        )

    return CurrentUser(user_id=payload["sub"], role=payload["role"])


async def require_admin(
    current_user: CurrentUser = Depends(get_current_user),
) -> CurrentUser:
    """
    FastAPI dependency — require admin role.
    Usage: admin: CurrentUser = Depends(require_admin)
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user
