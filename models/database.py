"""
SQLAlchemy ORM Models — Database Schema
────────────────────────────────────────
Tables: users, applications, agent_logs, decisions, documents
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, String, Integer, Float, Text, Boolean,
    DateTime, ForeignKey, JSON, Enum as SAEnum,
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
import enum

from config import settings


# ── Base ──────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Enums ─────────────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    USER = "user"
    ADMIN = "admin"


class ApplicationStatus(str, enum.Enum):
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"


# ── Users Table ───────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(SAEnum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True)
    last_login_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    applications = relationship("Application", back_populates="user")


# ── Applications Table ───────────────────────────────────────────────────

class Application(Base):
    __tablename__ = "applications"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    status = Column(SAEnum(ApplicationStatus), default=ApplicationStatus.SUBMITTED)

    # Loan details
    loan_amount = Column(Float, nullable=False)
    loan_purpose = Column(String(255), nullable=False)
    loan_tenure_months = Column(Integer, default=12)

    # Applicant info (submitted by user)
    applicant_name = Column(String(255), nullable=False)
    applicant_age = Column(Integer)
    applicant_gender = Column(String(20))
    applicant_email = Column(String(255))
    applicant_phone = Column(String(20))
    applicant_address = Column(Text)
    applicant_pan = Column(String(10))
    applicant_aadhaar = Column(String(12))
    monthly_income = Column(Float)
    employment_type = Column(String(50))  # salaried, self-employed, business
    employer_name = Column(String(255))

    # Shared state (populated by agents)
    shared_state = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="applications")
    documents = relationship("Document", back_populates="application")
    agent_logs = relationship("AgentLog", back_populates="application", order_by="AgentLog.step_number")
    decision = relationship("Decision", back_populates="application", uselist=False)


# ── Documents Table ──────────────────────────────────────────────────────

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    application_id = Column(String, ForeignKey("applications.id"), nullable=False, index=True)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)  # pdf, image
    file_size_bytes = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    parsed_data = Column(JSON)  # Populated by Data Agent

    # Relationships
    application = relationship("Application", back_populates="documents")


# ── Agent Logs Table ─────────────────────────────────────────────────────

class AgentLog(Base):
    __tablename__ = "agent_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    application_id = Column(String, ForeignKey("applications.id"), nullable=False, index=True)
    agent_name = Column(String(50), nullable=False)  # data_agent, risk_agent, etc.
    step_number = Column(Integer, nullable=False)
    action = Column(String(255), nullable=False)  # What the agent did
    input_data = Column(JSON)
    output_data = Column(JSON)
    status = Column(String(20), default="success")  # success, error, skipped
    error_message = Column(Text)
    duration_ms = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    application = relationship("Application", back_populates="agent_logs")


# ── Decisions Table ──────────────────────────────────────────────────────

class Decision(Base):
    __tablename__ = "decisions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    application_id = Column(String, ForeignKey("applications.id"), nullable=False, unique=True, index=True)
    status = Column(SAEnum(ApplicationStatus), nullable=False)
    risk_score = Column(Float)
    risk_category = Column(String(20))  # LOW, BORDERLINE, HIGH

    # Reasoning
    reasoning = Column(Text)
    risk_factors = Column(JSON)  # List of top risk factors

    # 3-audience explanations (populated in Phase 4)
    explanation_applicant = Column(Text)
    explanation_reviewer = Column(Text)
    explanation_regulator = Column(Text)

    # Audit
    policy_violations = Column(JSON)
    fairness_report = Column(JSON)

    decided_at = Column(DateTime, default=datetime.utcnow)
    decided_by = Column(String(50), default="system")  # system or admin username

    # Relationships
    application = relationship("Application", back_populates="decision")


# ── Database Engine & Session ────────────────────────────────────────────

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=(settings.APP_ENV == "development"),
    pool_size=5,
    max_overflow=10,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    """Dependency: yields a database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def create_tables():
    """Create all tables (development only — use Alembic in production)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables():
    """Drop all tables (testing only)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
