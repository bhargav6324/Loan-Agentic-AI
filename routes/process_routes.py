"""
Process Routes — Trigger the Agentic AI Pipeline
─────────────────────────────────────────────────
POST /applications/{id}/process — Runs the full LangGraph pipeline
"""


from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from auth.middleware import get_current_user, CurrentUser
from models.database import (
    Application, Decision, AgentLog,
    ApplicationStatus, get_db,
)
from orchestrator.graph import loan_pipeline

router = APIRouter(prefix="/applications", tags=["Processing"])


def _build_shared_state(final_state: dict) -> dict:
    """
    Extract ALL agent outputs from the final LangGraph state
    into a single dict that gets persisted to the database.
    The frontend diagnostic UI reads this to render every agent's internals.
    """
    return {
        # ── Planner Agent ──
        "plan": final_state.get("plan"),
        "plan_reasoning": final_state.get("plan_reasoning"),

        # ── Data Agent ──
        "data_agent_summary": final_state.get("data_agent_summary"),
        "data_agent_reasoning_trace": final_state.get("data_agent_reasoning_trace"),
        "data_agent_tools_called": final_state.get("data_agent_tools_called"),
        "data_agent_early_stop": final_state.get("data_agent_early_stop"),
        "pan_verification": final_state.get("pan_verification"),
        "aadhaar_verification": final_state.get("aadhaar_verification"),
        "credit_report": final_state.get("credit_report"),

        # ── Fraud Agent ──
        "fraud_detected": final_state.get("fraud_detected"),
        "fraud_reason": final_state.get("fraud_reason"),
        "requires_manual_kyc": final_state.get("requires_manual_kyc"),
        "fraud_agent_summary": final_state.get("fraud_agent_summary"),

        # ── Risk Agent ──
        "risk_score": final_state.get("risk_score"),
        "risk_category": final_state.get("risk_category"),
        "risk_factors": final_state.get("risk_factors"),
        "risk_agent_summary": final_state.get("risk_agent_summary"),
        "risk_agent_reflection": final_state.get("risk_agent_reflection"),

        # ── Policy Agent ──
        "policy_violations": final_state.get("policy_violations"),
        "policy_passed": final_state.get("policy_passed"),
        "policy_agent_summary": final_state.get("policy_agent_summary"),
        "policy_rag_reasoning": final_state.get("policy_rag_reasoning"),

        # ── Fairness Agent ──
        "fairness_report": final_state.get("fairness_report"),
        "fairness_debate_log": final_state.get("fairness_debate_log"),
        "bias_detected": final_state.get("bias_detected"),
        "fairness_agent_summary": final_state.get("fairness_agent_summary"),

        # ── Explainability Agent ──
        "explanation": final_state.get("explanation"),
        "explainability_agent_summary": final_state.get("explainability_agent_summary"),

        # ── Audit Agent ──
        "audit_trail": final_state.get("audit_trail"),
        "audit_anomalies": final_state.get("audit_anomalies"),
        "audit_execution_summary": final_state.get("audit_execution_summary"),
        "audit_agent_summary": final_state.get("audit_agent_summary"),

        # ── Decision ──
        "final_decision": final_state.get("final_decision"),
        "decision_reasoning": final_state.get("decision_reasoning"),

        # ── Learning Agent ──
        "learning_insights": final_state.get("learning_insights"),

        # ── Complete Agent Logs (full timeline) ──
        "agent_logs": final_state.get("agent_logs"),
        "errors": final_state.get("errors"),
    }


@router.post("/{application_id}/process")
async def process_application(
    application_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger the full Agentic AI pipeline on a submitted application.

    Flow: Data Agent → Risk Agent → (conditional) → Policy Agent → Fairness Agent → Decision
    """
    # ── 1. Fetch application ─────────────────────────────
    result = await db.execute(
        select(Application).where(
            Application.id == application_id,
            Application.user_id == current_user.user_id,
        )
    )
    application = result.scalar_one_or_none()

    if not application:
        raise HTTPException(status_code=404, detail="Application not found")

    if application.status != ApplicationStatus.SUBMITTED:
        raise HTTPException(
            status_code=400,
            detail=f"Application is already {application.status.value}. Only 'submitted' applications can be processed.",
        )

    # ── 2. Update status to PROCESSING ───────────────────
    application.status = ApplicationStatus.PROCESSING
    application.updated_at = datetime.utcnow()
    await db.commit()

    # ── 3. Build initial state for the pipeline ──────────
    initial_state = {
        "application_id": application.id,
        "user_id": application.user_id,
        "applicant_data": {
            "loan_amount": application.loan_amount,
            "loan_purpose": application.loan_purpose,
            "loan_tenure_months": application.loan_tenure_months,
            "applicant_name": application.applicant_name,
            "applicant_age": application.applicant_age,
            "applicant_gender": application.applicant_gender,
            "applicant_email": application.applicant_email,
            "applicant_phone": application.applicant_phone,
            "applicant_address": application.applicant_address,
            "applicant_pan": application.applicant_pan,
            "applicant_aadhaar": application.applicant_aadhaar,
            "monthly_income": application.monthly_income,
            "employment_type": application.employment_type,
            "employer_name": application.employer_name,
        },
        "agent_logs": [],
        "errors": [],
    }

    # ── 4. Run the LangGraph pipeline ────────────────────
    try:
        final_state = loan_pipeline.invoke(initial_state)
    except Exception as e:
        application.status = ApplicationStatus.SUBMITTED
        await db.commit()
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {str(e)}",
        )

    # ── 5. Map decision to application status ────────────
    decision_str = final_state.get("final_decision", "escalated")
    status_map = {
        "approved": ApplicationStatus.APPROVED,
        "rejected": ApplicationStatus.REJECTED,
        "escalated": ApplicationStatus.ESCALATED,
    }
    new_status = status_map.get(decision_str, ApplicationStatus.ESCALATED)

    application.status = new_status
    application.shared_state = _build_shared_state(final_state)
    application.updated_at = datetime.utcnow()

    # ── 6. Create Decision record ────────────────────────
    # Check if decision already exists
    existing = await db.execute(
        select(Decision).where(Decision.application_id == application_id)
    )
    decision_record = existing.scalar_one_or_none()

    explanation_data = final_state.get("explanation", {}) or {}

    if decision_record:
        decision_record.status = new_status
        decision_record.risk_score = final_state.get("risk_score")
        decision_record.risk_category = final_state.get("risk_category")
        decision_record.reasoning = final_state.get("decision_reasoning")
        decision_record.risk_factors = final_state.get("risk_factors")
        decision_record.policy_violations = final_state.get("policy_violations")
        decision_record.fairness_report = final_state.get("fairness_report")
        decision_record.explanation_applicant = explanation_data.get("applicant_message")
        decision_record.explanation_reviewer = explanation_data.get("admin_summary")
        decision_record.explanation_regulator = explanation_data.get("compliance_report")
        decision_record.decided_at = datetime.utcnow()
        decision_record.decided_by = "agentic_pipeline"
    else:
        decision_record = Decision(
            application_id=application_id,
            status=new_status,
            risk_score=final_state.get("risk_score"),
            risk_category=final_state.get("risk_category"),
            reasoning=final_state.get("decision_reasoning"),
            risk_factors=final_state.get("risk_factors"),
            policy_violations=final_state.get("policy_violations"),
            fairness_report=final_state.get("fairness_report"),
            explanation_applicant=explanation_data.get("applicant_message"),
            explanation_reviewer=explanation_data.get("admin_summary"),
            explanation_regulator=explanation_data.get("compliance_report"),
            decided_by="agentic_pipeline",
        )
        db.add(decision_record)

    # ── 7. Save agent logs ───────────────────────────────
    agent_logs = final_state.get("agent_logs", [])
    for i, log_entry in enumerate(agent_logs):
        agent_log = AgentLog(
            application_id=application_id,
            agent_name=log_entry.get("agent", "unknown"),
            step_number=i + 1,
            action=log_entry.get("action", ""),
            status=log_entry.get("status", "success"),
            duration_ms=log_entry.get("duration_ms", 0),
            output_data={"details": log_entry.get("details", "")},
        )
        db.add(agent_log)

    await db.commit()
    await db.refresh(application)

    # ── 8. Return result ─────────────────────────────────
    return {
        "application_id": application.id,
        "status": application.status.value,
        "decision": decision_str,
        "reasoning": final_state.get("decision_reasoning"),
        "risk_score": final_state.get("risk_score"),
        "risk_category": final_state.get("risk_category"),
        "risk_factors": final_state.get("risk_factors"),
        "policy_violations": final_state.get("policy_violations"),
        "fairness_report": final_state.get("fairness_report"),
        "agents_executed": len(agent_logs),
        "summaries": {
            "data_agent": final_state.get("data_agent_summary"),
            "risk_agent": final_state.get("risk_agent_summary"),
            "policy_agent": final_state.get("policy_agent_summary"),
            "fairness_agent": final_state.get("fairness_agent_summary"),
        },
    }

from fastapi.responses import StreamingResponse
import json
import asyncio

@router.post("/{application_id}/process_stream")
async def process_application_stream(
    application_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger the full Agentic AI pipeline with Server-Sent Events (SSE) streaming.
    """
    # ── 1. Fetch application ─────────────────────────────
    result = await db.execute(
        select(Application).where(
            Application.id == application_id,
            Application.user_id == current_user.user_id,
        )
    )
    application = result.scalar_one_or_none()

    if not application:
        raise HTTPException(status_code=404, detail="Application not found")

    if application.status != ApplicationStatus.SUBMITTED:
        raise HTTPException(
            status_code=400,
            detail=f"Application is already {application.status.value}. Only 'submitted' applications can be processed.",
        )

    # ── 2. Update status to PROCESSING ───────────────────
    application.status = ApplicationStatus.PROCESSING
    application.updated_at = datetime.utcnow()
    await db.commit()

    # ── 3. Build initial state ───────────────────────────
    initial_state = {
        "application_id": application.id,
        "user_id": application.user_id,
        "applicant_data": {
            "loan_amount": application.loan_amount,
            "loan_purpose": application.loan_purpose,
            "loan_tenure_months": application.loan_tenure_months,
            "applicant_name": application.applicant_name,
            "applicant_age": application.applicant_age,
            "applicant_gender": application.applicant_gender,
            "applicant_email": application.applicant_email,
            "applicant_phone": application.applicant_phone,
            "applicant_address": application.applicant_address,
            "applicant_pan": application.applicant_pan,
            "applicant_aadhaar": application.applicant_aadhaar,
            "monthly_income": application.monthly_income,
            "employment_type": application.employment_type,
            "employer_name": application.employer_name,
        },
        "agent_logs": [],
        "errors": [],
    }

    async def event_generator():
        final_state = initial_state
        try:
            yield f"data: {json.dumps({'event': 'started', 'message': 'Initializing LangGraph pipeline...'})}\n\n"
            
            # Use LangGraph astream to yield after each node completes
            async for output in loan_pipeline.astream(initial_state):
                node_name = list(output.keys())[0]
                state_update = output[node_name]
                final_state.update(state_update)
                
                msgs = {
                    "data_agent": "Data Agent: Running ReAct loop for KYC and data extraction...",
                    "fraud_agent": "Fraud Agent: Checking identity and anti-fraud signals...",
                    "risk_agent": "Risk Agent: Analyzing financial profile and calculating risk score...",
                    "planner_agent": "Planner Agent: Autonomously generating execution plan via LLM...",
                    "policy_agent": "Policy Agent: Verifying compliance with lending policies...",
                    "fairness_agent": "Fairness Agent: Debating counterfactuals and auditing bias...",
                    "risk_adjustment_agent": "Risk Adjustment: Negotiating score with Fairness Agent...",
                    "risk_reeval": "Risk Agent (Re-eval): Re-scoring after fairness mitigation...",
                    "fraud_recheck": "Fraud Agent (Re-check): Re-verifying identity signals...",
                    "auto_approve": "Orchestrator: Auto-approving LOW risk application...",
                    "auto_reject": "Orchestrator: Auto-rejecting application...",
                    "final_decision": "Orchestrator: Making final decision...",
                    "explainability_agent": "Explainability Agent: Generating auditor process trace...",
                    "learning_agent": "Learning Agent: Saving outcomes to systemic memory...",
                    "audit_agent": "Audit Agent: Finalizing logs...",
                    "advance_step": "Orchestrator: Advancing to next step in dynamic plan..."
                }
                msg = msgs.get(node_name, f"Executing {node_name}...")
                yield f"data: {json.dumps({'event': 'node_update', 'node': node_name, 'message': msg})}\n\n"
                await asyncio.sleep(0.5)  # Small visual delay to appreciate the agent transitions
                
            # ── 5. Map decision to application status ────────────
            decision_str = final_state.get("final_decision", "escalated")
            status_map = {
                "approved": ApplicationStatus.APPROVED,
                "rejected": ApplicationStatus.REJECTED,
                "escalated": ApplicationStatus.ESCALATED,
            }
            new_status = status_map.get(decision_str, ApplicationStatus.ESCALATED)

            # Re-fetch application to ensure it is attached to the current session context
            app_result = await db.execute(select(Application).where(Application.id == application_id))
            app_to_update = app_result.scalar_one()

            app_to_update.status = new_status
            app_to_update.shared_state = _build_shared_state(final_state)
            app_to_update.updated_at = datetime.utcnow()

            # ── 6. Create Decision record ─────────
            existing = await db.execute(select(Decision).where(Decision.application_id == application_id))
            decision_record = existing.scalar_one_or_none()
            explanation_data = final_state.get("explanation", {}) or {}

            if decision_record:
                decision_record.status = new_status
                decision_record.risk_score = final_state.get("risk_score")
                decision_record.risk_category = final_state.get("risk_category")
                decision_record.reasoning = final_state.get("decision_reasoning")
                decision_record.risk_factors = final_state.get("risk_factors")
                decision_record.policy_violations = final_state.get("policy_violations")
                decision_record.fairness_report = final_state.get("fairness_report")
                decision_record.explanation_applicant = explanation_data.get("applicant_message")
                decision_record.explanation_reviewer = explanation_data.get("admin_summary")
                decision_record.explanation_regulator = explanation_data.get("compliance_report")
                decision_record.decided_at = datetime.utcnow()
                decision_record.decided_by = "agentic_pipeline"
            else:
                decision_record = Decision(
                    application_id=application_id,
                    status=new_status,
                    risk_score=final_state.get("risk_score"),
                    risk_category=final_state.get("risk_category"),
                    reasoning=final_state.get("decision_reasoning"),
                    risk_factors=final_state.get("risk_factors"),
                    policy_violations=final_state.get("policy_violations"),
                    fairness_report=final_state.get("fairness_report"),
                    explanation_applicant=explanation_data.get("applicant_message"),
                    explanation_reviewer=explanation_data.get("admin_summary"),
                    explanation_regulator=explanation_data.get("compliance_report"),
                    decided_by="agentic_pipeline",
                )
                db.add(decision_record)

            # ── 7. Save agent logs ─────────────
            agent_logs = final_state.get("agent_logs", [])
            for i, log_entry in enumerate(agent_logs):
                agent_log = AgentLog(
                    application_id=application_id,
                    agent_name=log_entry.get("agent", "unknown"),
                    step_number=i + 1,
                    action=log_entry.get("action", ""),
                    status=log_entry.get("status", "success"),
                    duration_ms=log_entry.get("duration_ms", 0),
                    output_data={"details": log_entry.get("details", "")},
                )
                db.add(agent_log)

            await db.commit()
            
            yield f"data: {json.dumps({'event': 'complete', 'application_id': application_id})}\n\n"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                # If application is detached, this might fail, so wrap it
                app_result = await db.execute(select(Application).where(Application.id == application_id))
                app_to_revert = app_result.scalar_one_or_none()
                if app_to_revert:
                    app_to_revert.status = ApplicationStatus.SUBMITTED
                    await db.commit()
            except Exception as inner_e:
                print(f"Failed to rollback application status: {inner_e}")
            yield f"data: {json.dumps({'event': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
