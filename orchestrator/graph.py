"""
LangGraph Orchestrator — Parallel Architecture (V7)
──────────────────────────────────────────────────────────────
V7 Architecture (fan-out after intake):

  START → intake_agent ─┬─ (fail) → auto_reject ──────────────────┐
                        │                                          │
              (pass) ───┼─── risk_agent → orchestrator_decide ─┐  │
                        │         ↕ (loop: fairness/reeval)    │  │
                        └─── policy_agent ──────────────────────┤  │
                                                                 ↓  │
                                                           policy_join
                                                                 │
                             (auto_approve / final_decision / auto_reject)
                                                                 │
                                                         reporting_agent
                                                                 │
                                                               END

Key changes from V6:
- policy_agent runs in parallel with risk_agent (no serial dependency)
- decision_commit acts as a strict barrier ensuring policy and risk certificates exist before terminating.
- reporting_agent no longer blocks on learning (background thread in V7)
"""

from __future__ import annotations
from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.intake_agent import intake_agent, policy_agent_node
from agents.risk_agent import risk_agent
from agents.fairness_agent import fairness_agent
from agents.orchestrator_agent import orchestrator_decide, generate_initial_plan
from agents.reporting_agent import reporting_agent

def decision_commit(state: AgentState) -> AgentState:
    """
    Strict barrier node for committing final decisions.
    Requires both risk and policy certificates.

    Governance invariant:
      - Fairness may allow the original deterministic decision to continue.
      - Fairness may escalate for human review.
      - Fairness must NEVER approve a deterministic rejection.
      - Fairness must NEVER override a policy rejection.
      - Policy rejection is terminal and does NOT require a fairness audit.
    """
    logs = list(state.get("agent_logs", []))
    proposal = state.get("decision_proposal")
    policy_passed = state.get("policy_passed")
    risk_cert = state.get("risk_certificate")
    policy_cert = state.get("policy_certificate")

    # The certificate is authoritative when a graph branch returns it without
    # separately materialising the derived state fields.
    if policy_passed is None and policy_cert:
        policy_passed = policy_cert.get("passed")
    policy_violations = state.get("policy_violations")
    if policy_violations is None and policy_cert:
        policy_violations = policy_cert.get("violations", [])

    # ── Certificate recovery: parallel fan-out race ──────────────────────────
    # Certificates must come from their producer agents. Ordinary state fields
    # are never used to reconstruct or synthesise missing certificates.
    # ── Gate 1: certificate presence ──
    if not risk_cert or not policy_cert:
        status = "blocked_missing_certificate"
        reasoning = f"Commitment blocked: missing certificates. Risk: {bool(risk_cert)}, Policy: {bool(policy_cert)}"

    # ── Gate 2: policy rejection is terminal — no audit required ──
    elif policy_passed is False:
        if proposal == "approve":
            status = "blocked_policy_violation"
            reasoning = f"Commitment blocked: Orchestrator proposed approve but policy failed. Violations: {policy_violations}"
        else:
            status = "committed_reject"
            reasoning = "Application rejected due to policy violations."

    # ── Gate 3: MEDIUM risk requires fairness audit (only when policy passed) ──
    elif state.get("risk_category") == "MEDIUM" and not state.get("fairness_report"):
        status = "blocked_missing_audit"
        reasoning = "Commitment blocked: medium-risk application has no fairness audit."

    # ── Gate 4: proposal-based commit ──
    elif proposal == "approve":
        status = "committed_approve"
        reasoning = "Application approved. All checks passed."
    elif proposal == "reject":
        status = "committed_reject"
        reasoning = "Application rejected based on risk/fraud evaluation."
    elif proposal == "escalate":
        status = "automated_escalated_status"
        reasoning = "Application routed to automated escalated status due to counterfactual audit."
    else:
        status = "blocked_branch_conflict"
        reasoning = f"Commitment blocked: unknown or missing decision proposal '{proposal}'."

    logs.append({"agent": "decision_commit", "step": len(logs) + 1, "action": f"Decision Commit: {status}", "status": "success", "details": reasoning, "duration_ms": 0})
    
    return {
        **state,
        "commit_status": status,
        "commit_reason": reasoning,
        "final_decision": "approved" if status == "committed_approve" else ("rejected" if status == "committed_reject" else "escalated"),
        "decision_reasoning": reasoning,
        "agent_logs": logs
    }


def policy_join(state: AgentState) -> AgentState:
    """
    Merge point after parallel policy_agent + orchestrator branches.
    Routes to decision_commit.
    """
    next_agent = state.get("orchestrator_next_agent", "decision_commit")
    return {**state, "orchestrator_next_agent": "decision_commit"}


# ── Fan-out Node ─────────────────────────────────────────

def fan_out(state: AgentState) -> AgentState:
    """No-op router: LangGraph fans out from here to risk + policy in parallel."""
    return state


# ── Routing Functions ────────────────────────────────────

def route_after_intake(state: AgentState) -> str:
    if state.get("fraud_detected") or state.get("data_agent_early_stop"):
        return "decision_commit"
    return "planner"


def route_orchestrator(state: AgentState) -> str:
    next_agent = state.get("orchestrator_next_agent", "policy_join")
    valid_targets = {"fairness_agent", "risk_reeval", "decision_commit", "policy_join"}
    return next_agent if next_agent in valid_targets else "policy_join"


def route_after_join(state: AgentState) -> str:
    return "decision_commit"


# ── Build the Graph ──────────────────────────────────────

def build_loan_graph(disable_fairness: bool = False, disable_policy: bool = False) -> StateGraph:
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("intake_agent", intake_agent)
    graph.add_node("planner", generate_initial_plan)
    graph.add_node("fan_out", fan_out)
    graph.add_node("risk_agent", risk_agent)
    graph.add_node("policy_agent", (lambda state: state) if disable_policy else policy_agent_node)
    graph.add_node("orchestrator_decide", orchestrator_decide)
    graph.add_node("fairness_agent", (lambda state: state) if disable_fairness else fairness_agent)
    graph.add_node("risk_reeval", risk_agent)
    graph.add_node("policy_join", policy_join)
    graph.add_node("decision_commit", decision_commit)
    graph.add_node("reporting_agent", reporting_agent)

    # Entry
    graph.set_entry_point("intake_agent")

    # Intake → decision_commit (hard fail) OR planner
    graph.add_conditional_edges("intake_agent", route_after_intake, {
        "decision_commit": "decision_commit",
        "planner": "planner",
    })
    
    # Planner → fan_out
    graph.add_edge("planner", "fan_out")

    # Fan-out: both edges fire concurrently
    graph.add_edge("fan_out", "risk_agent")
    graph.add_edge("fan_out", "policy_agent")

    # True fan-in barrier: wait for both parallel producers before allowing
    # orchestration or commitment to continue.
    graph.add_edge(["risk_agent", "policy_agent"], "orchestrator_decide")

    routing_map = {
        "fairness_agent": "fairness_agent",
        "risk_reeval": "risk_reeval",
        "decision_commit": "decision_commit",
        "policy_join": "policy_join",
    }
    graph.add_conditional_edges("orchestrator_decide", route_orchestrator, routing_map)

    # Non-terminal agents loop back to orchestrator
    graph.add_edge("fairness_agent", "orchestrator_decide")
    graph.add_edge("risk_reeval", "orchestrator_decide")

    # Terminal orchestrator routes pass through this no-op node. Parallel
    # synchronization is handled by the waiting edge above.
    graph.add_conditional_edges("policy_join", route_after_join, {
        "decision_commit": "decision_commit",
    })

    # decision_commit → reporting
    graph.add_edge("decision_commit", "reporting_agent")
    graph.add_edge("reporting_agent", END)

    return graph.compile()


# Singleton compiled graph
loan_pipeline = build_loan_graph()
