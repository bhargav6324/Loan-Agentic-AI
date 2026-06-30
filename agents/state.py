"""
Agent Shared State — TypedDict for LangGraph Pipeline
──────────────────────────────────────────────────────
All agents read from and write to this shared state as it flows
through the LangGraph orchestrator.
"""

from __future__ import annotations
from typing import TypedDict, Optional


class AgentState(TypedDict, total=False):
    """Shared state flowing through the agent pipeline."""

    # ── Identifiers ──────────────────────────────────────
    application_id: str
    user_id: str

    # ── Raw Application Data ─────────────────────────────
    applicant_data: dict  # Full form data from the application

    # ── Data Agent Outputs ───────────────────────────────
    pan_verification: Optional[dict]
    aadhaar_verification: Optional[dict]
    credit_report: Optional[dict]
    data_agent_summary: Optional[str]
    # Agentic: ReAct reasoning trace
    data_agent_reasoning_trace: Optional[list]   # [{thought, action, observation}, ...]
    data_agent_tools_called: Optional[list]      # Which tools the agent chose to invoke
    data_agent_early_stop: Optional[bool]        # Did the agent stop early (e.g. KYC fail)?

    # ── Fraud Agent Outputs ──────────────────────────────
    fraud_detected: Optional[bool]
    requires_manual_kyc: Optional[bool]
    fraud_reason: Optional[str]
    fraud_agent_summary: Optional[str]

    # ── Risk Agent Outputs ───────────────────────────────
    risk_score: Optional[float]        # 0-100 (lower = safer)
    risk_category: Optional[str]       # LOW, MEDIUM, HIGH
    risk_factors: Optional[list]       # Top risk factor descriptions
    risk_agent_summary: Optional[str]
    # Agentic: Self-reflection / self-critique results
    risk_agent_reflection: Optional[dict]  # {contradictions, revised, reason, ...}

    # ── Policy Agent Outputs ─────────────────────────────
    policy_violations: Optional[list]  # List of violated rules
    policy_passed: Optional[bool]
    policy_agent_summary: Optional[str]
    policy_rag_reasoning: Optional[str]

    # ── Fairness Agent Outputs ───────────────────────────
    fairness_report: Optional[dict]    # Counterfactual analysis
    fairness_debate_log: Optional[list] # Textual inter-agent debate records
    bias_detected: Optional[bool]
    negotiation_turns: Optional[int]   # Number of times Fairness and Risk debated
    fairness_agent_summary: Optional[str]
    debate_transcript: Optional[list]  # V4: Full LLM-to-LLM semantic debate exchanges

    # ── Explainability Agent Outputs ─────────────────────
    explanation: Optional[dict]        # {applicant_message, admin_summary, compliance_report}
    explainability_agent_summary: Optional[str]

    # ── Audit Agent Outputs ──────────────────────────────
    audit_trail: Optional[list]        # Formal audit records for DB
    audit_anomalies: Optional[list]    # Detected execution anomalies
    audit_execution_summary: Optional[dict]  # Pipeline execution stats
    audit_agent_summary: Optional[str]

    # ── Final Decision ───────────────────────────────────
    final_decision: Optional[str]      # approved, rejected, escalated
    decision_reasoning: Optional[str]

    # ── Learning Agent Outputs ───────────────────────────
    learning_insights: Optional[dict]   # Systemic bias analysis from learning agent

    # ── Orchestrator Agent Outputs (V5) ──────────────────
    orchestrator_next_agent: Optional[str]   # Next agent to execute (set by orchestrator_decide)
    orchestrator_history: Optional[list]     # [{step, agent, reasoning}, ...] — full decision trace
    orchestrator_iterations: Optional[int]   # Current orchestrator iteration count

    # ── Legacy Planner Fields (backward compat) ──────────
    plan: Optional[list[str]]          # Populated by orchestrator for frontend display
    plan_reasoning: Optional[str]      # Populated by orchestrator for frontend display
    current_step_index: Optional[int]  # Deprecated — kept for schema compatibility

    # ── Audit Trail ──────────────────────────────────────
    agent_logs: list   # List of {agent, action, status, details, duration_ms}
    errors: list       # Any errors encountered
