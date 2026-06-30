"""
Password Hashing — bcrypt
─────────────────────────
Direct bcrypt usage (passlib removed due to incompatibility with bcrypt>=4.x).
"""

import bcrypt


def hash_password(password: str) -> str:
    """Hash a plaintext password."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its hash."""
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
