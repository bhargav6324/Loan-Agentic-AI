"""
Admin Routes — Application Management & Escalation Review
──────────────────────────────────────────────────────────
Admin-only endpoints. Requires admin JWT token.
"""


from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from auth.middleware import require_admin, CurrentUser
from models.database import Application, Decision, ApplicationStatus, get_db
from models.schemas import (
    LoanApplicationResponse,
    ApplicationListResponse,
    EscalationResolve,
    DecisionResponse,
    UserResponse,
)
from models.database import User
from agents.audit_agent import audit_agent

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/users")
async def list_all_users(
    admin: CurrentUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all users in the system (admin only)."""
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    users = result.scalars().all()
    
    return {
        "total": len(users),
        "users": [UserResponse.model_validate(u) for u in users]
    }


@router.get("/applications", response_model=ApplicationListResponse)
async def list_all_applications(
    admin: CurrentUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all applications in the system (admin only)."""
    result = await db.execute(
        select(Application).order_by(Application.created_at.desc())
    )
    apps = result.scalars().all()

    return ApplicationListResponse(
        total=len(apps),
        applications=[LoanApplicationResponse.model_validate(a) for a in apps],
    )


@router.get("/escalations")
async def list_escalations(
    admin: CurrentUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all escalated applications pending human review."""
    result = await db.execute(
        select(Application)
        .where(Application.status == ApplicationStatus.ESCALATED)
        .order_by(Application.created_at.desc())
    )
    apps = result.scalars().all()

    return {
        "total_pending": len(apps),
        "applications": [
            LoanApplicationResponse.model_validate(a) for a in apps
        ],
    }


@router.post("/escalations/{application_id}/resolve", response_model=DecisionResponse)
async def resolve_escalation(
    application_id: str,
    body: EscalationResolve,
    admin: CurrentUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Human reviewer resolves an escalated application and seals the decision in the audit log."""
    result = await db.execute(
        select(Application).where(Application.id == application_id)
    )
    application = result.scalar_one_or_none()

    if not application:
        raise HTTPException(status_code=404, detail="Application not found")

    if application.status != ApplicationStatus.ESCALATED:
        raise HTTPException(
            status_code=400,
            detail=f"Application is not escalated (current status: {application.status.value})",
        )

    # Map decision string to enum
    new_status = (
        ApplicationStatus.APPROVED if body.decision == "approved"
        else ApplicationStatus.REJECTED
    )

    # Update application status
    application.status = new_status
    application.updated_at = datetime.utcnow()

    # Agentic Fix: Re-run Audit Agent to seal human override
    shared_state = application.shared_state or {}
    shared_state["final_decision"] = body.decision
    shared_state["decision_reasoning"] = f"Admin override by {admin.user_id}: {body.reviewer_notes}"
    
    # Append human override log
    agent_logs = list(shared_state.get("agent_logs", []))
    agent_logs.append({
        "agent": "human_admin",
        "step": len(agent_logs) + 1,
        "action": "Escalation Resolution Override",
        "status": "override",
        "details": f"Admin '{admin.user_id}' decided '{body.decision}'. Reason: {body.reviewer_notes}",
        "duration_ms": 0,
    })
    shared_state["agent_logs"] = agent_logs
    
    # Cryptographically seal the new state with the human decision
    sealed_state = audit_agent(shared_state)
    application.shared_state = sealed_state

    # Create or update decision record
    result = await db.execute(
        select(Decision).where(Decision.application_id == application_id)
    )
    decision = result.scalar_one_or_none()

    if decision:
        decision.status = new_status
        decision.reasoning = f"Human review: {body.reviewer_notes}"
        decision.decided_by = f"admin:{admin.user_id}"
        decision.decided_at = datetime.utcnow()
    else:
        decision = Decision(
            application_id=application_id,
            status=new_status,
            reasoning=f"Human review: {body.reviewer_notes}",
            decided_by=f"admin:{admin.user_id}",
        )
        db.add(decision)

    await db.commit()
    await db.refresh(decision)

    return decision


@router.get("/applications/{application_id}/audit")
async def get_audit_trail(
    application_id: str,
    admin: CurrentUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get the full agent audit trail for an application (admin only)."""
    result = await db.execute(
        select(Application).where(Application.id == application_id)
    )
    application = result.scalar_one_or_none()

    if not application:
        raise HTTPException(status_code=404, detail="Application not found")

    return {
        "application_id": application.id,
        "applicant_name": application.applicant_name,
        "status": application.status.value,
        "shared_state": application.shared_state or {},
        "created_at": application.created_at,
        "updated_at": application.updated_at,
    }
