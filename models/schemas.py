"""
Pydantic Schemas — Request / Response Models
─────────────────────────────────────────────
Used for API validation and serialisation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field


# ── Auth Schemas ─────────────────────────────────────────────────────────

class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)
    full_name: str = Field(..., min_length=1, max_length=255)
    role: str = Field(default="user", pattern="^(user|admin)$")


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class TokenRefresh(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    is_active: bool
    last_login_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ── Application Schemas ──────────────────────────────────────────────────

class LoanApplicationCreate(BaseModel):
    loan_amount: float = Field(..., gt=0, le=50_000_000, description="Loan amount in INR")
    loan_purpose: str = Field(..., min_length=3, max_length=255)
    loan_tenure_months: int = Field(default=12, ge=1, le=360)

    applicant_name: str = Field(..., min_length=1, max_length=255)
    applicant_age: Optional[int] = Field(None, ge=18, le=100)
    applicant_gender: Optional[str] = Field(None, pattern="^(male|female|other)$")
    applicant_email: Optional[EmailStr] = None
    applicant_phone: Optional[str] = Field(None, max_length=15)
    applicant_address: Optional[str] = Field(None, max_length=1000)
    applicant_pan: Optional[str] = Field(None, pattern="^[A-Z]{5}[0-9]{4}[A-Z]$")
    applicant_aadhaar: Optional[str] = Field(None, pattern="^[0-9]{12}$")
    monthly_income: Optional[float] = Field(None, ge=0)
    employment_type: Optional[str] = Field(None, pattern="^(salaried|self_employed|business)$")
    employer_name: Optional[str] = Field(None, max_length=255)


class LoanApplicationResponse(BaseModel):
    id: str
    user_id: str
    status: str
    loan_amount: float
    loan_purpose: str
    loan_tenure_months: int
    applicant_name: str
    applicant_pan: Optional[str] = None
    shared_state: Optional[dict] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ApplicationListResponse(BaseModel):
    total: int
    applications: List[LoanApplicationResponse]


# ── Decision Schemas ─────────────────────────────────────────────────────

class DecisionResponse(BaseModel):
    id: str
    application_id: str
    status: str
    risk_score: Optional[float] = None
    risk_category: Optional[str] = None
    reasoning: Optional[str] = None
    risk_factors: Optional[list] = None
    explanation_applicant: Optional[str] = None
    explanation_reviewer: Optional[str] = None
    explanation_regulator: Optional[str] = None
    decided_at: datetime
    decided_by: str

    class Config:
        from_attributes = True



# ── Escalation Schemas ───────────────────────────────────────────────────

class EscalationResolve(BaseModel):
    decision: str = Field(..., pattern="^(approved|rejected)$")
    reviewer_notes: str = Field(default="", max_length=2000)


# ── Agent Log Schemas ────────────────────────────────────────────────────

class AgentLogResponse(BaseModel):
    id: str
    agent_name: str
    step_number: int
    action: str
    status: str
    duration_ms: Optional[int] = None
    timestamp: datetime

    class Config:
        from_attributes = True


# ── Document Schemas ─────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: str
    file_name: str
    file_type: str
    file_size_bytes: Optional[int] = None
    uploaded_at: datetime

    class Config:
        from_attributes = True


# ── Health Check ─────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    service: str
    version: str
    status: str
    environment: str
