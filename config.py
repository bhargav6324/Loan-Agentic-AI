"""
Application Configuration
─────────────────────────
Loads settings from .env file using pydantic-settings.
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://loanai:loanai123@localhost:5432/loanai_db"

    # ── JWT Auth ──────────────────────────────────────
    JWT_SECRET_KEY: str = "dev-secret-key-not-for-production-use"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── LLM ───────────────────────────────────────────
    GROQ_API_KEY: Optional[str] = None
    GEMINI_API_KEY: str = ""

    # ── App ───────────────────────────────────────────
    APP_ENV: str = "development"
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE_MB: int = 10

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        p = Path(self.UPLOAD_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
