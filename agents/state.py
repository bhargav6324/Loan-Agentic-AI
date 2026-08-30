"""
Agent Shared State — TypedDict for LangGraph Pipeline (V7)
──────────────────────────────────────────────────────
V7: All scalar fields use a last_value reducer so parallel nodes that
re-spread **state don't cause "one value per step" conflicts.
agent_logs and errors use operator.add to merge lists from both branches.
"""

from __future__ import annotations
import operator
from typing import TypedDict, Optional, Annotated


def _last(a, b):
    """Last-write-wins: idempotent for unchanged values, later-write wins on conflict."""
    return b


class AgentState(TypedDict, total=False):
    """Shared state flowing through the agent pipeline."""

    # ── Identifiers ──────────────────────────────────────
    application_id: Annotated[str, _last]
    user_id: Annotated[str, _last]

    # ── Raw Application Data ─────────────────────────────
    applicant_data: Annotated[dict, _last]

    # ── Data Agent Outputs ───────────────────────────────
    pan_verification: Annotated[Optional[dict], _last]
    aadhaar_verification: Annotated[Optional[dict], _last]
    credit_report: Annotated[Optional[dict], _last]
    data_agent_summary: Annotated[Optional[str], _last]
    data_agent_reasoning_trace: Annotated[Optional[list], _last]
    data_agent_tools_called: Annotated[Optional[list], _last]
    data_agent_early_stop: Annotated[Optional[bool], _last]

    # ── Fraud Agent Outputs ──────────────────────────────
    fraud_detected: Annotated[Optional[bool], _last]
    requires_manual_kyc: Annotated[Optional[bool], _last]
    fraud_reason: Annotated[Optional[str], _last]
    fraud_agent_summary: Annotated[Optional[str], _last]
    fraud_match_confidence: Annotated[Optional[float], _last]

    # ── Risk Agent Outputs ───────────────────────────────
    risk_score: Annotated[Optional[float], _last]
    risk_category: Annotated[Optional[str], _last]
    risk_factors: Annotated[Optional[list], _last]
    risk_agent_summary: Annotated[Optional[str], _last]
    risk_ml_confidence: Annotated[Optional[float], _last]
    risk_agent_reflection: Annotated[Optional[dict], _last]

    # ── Policy Agent Outputs ─────────────────────────────
    policy_violations: Annotated[Optional[list], _last]
    policy_passed: Annotated[Optional[bool], _last]
    policy_agent_summary: Annotated[Optional[str], _last]
    policy_rag_reasoning: Annotated[Optional[str], _last]
    policy_retrieval_score: Annotated[Optional[float], _last]

    # ── Fairness Agent Outputs ───────────────────────────
    fairness_report: Annotated[Optional[dict], _last]
    fairness_debate_log: Annotated[Optional[list], _last]
    bias_detected: Annotated[Optional[bool], _last]
    negotiation_turns: Annotated[Optional[int], _last]
    fairness_agent_summary: Annotated[Optional[str], _last]
    debate_transcript: Annotated[Optional[list], _last]

    # ── Explainability Agent Outputs ─────────────────────
    explanation: Annotated[Optional[dict], _last]
    explainability_agent_summary: Annotated[Optional[str], _last]

    # ── Audit Agent Outputs ──────────────────────────────
    audit_trail: Annotated[Optional[list], _last]
    audit_anomalies: Annotated[Optional[list], _last]
    audit_execution_summary: Annotated[Optional[dict], _last]
    audit_agent_summary: Annotated[Optional[str], _last]

    # ── Final Decision ───────────────────────────────────
    final_decision: Annotated[Optional[str], _last]
    decision_reasoning: Annotated[Optional[str], _last]
    
    # ── Decision Commitment (Phase 2) ────────────────────
    risk_certificate: Annotated[Optional[dict], _last]
    policy_certificate: Annotated[Optional[dict], _last]
    audit_certificate: Annotated[Optional[dict], _last]
    decision_proposal: Annotated[Optional[str], _last]
    commit_status: Annotated[Optional[str], _last]
    commit_reason: Annotated[Optional[str], _last]

    # ── Learning Agent Outputs ───────────────────────────
    learning_insights: Annotated[Optional[dict], _last]

    # ── Orchestrator Agent Outputs ───────────────────────
    orchestrator_next_agent: Annotated[Optional[str], _last]
    orchestrator_history: Annotated[Optional[list], _last]
    orchestrator_iterations: Annotated[Optional[int], _last]

    # ── Legacy Fields (backward compat) ──────────────────
    plan: Annotated[Optional[list], _last]
    plan_reasoning: Annotated[Optional[str], _last]
    current_step_index: Annotated[Optional[int], _last]

    # ── Audit Trail (merged from parallel branches) ───────
    # operator.add concatenates lists so both branches' logs are preserved
    agent_logs: Annotated[list, operator.add]
    errors: Annotated[list, operator.add]
