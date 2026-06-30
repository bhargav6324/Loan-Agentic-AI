"""
LangGraph Orchestrator — Dynamic Continuous Orchestration (V5)
──────────────────────────────────────────────────────────────
Replaces the static Plan-and-Execute array with a continuous
LLM orchestrator that re-evaluates state after every agent.

V5 Architecture:

  START → data_agent → fraud_agent ─┬─ fraud → auto_reject ──┐
                                     └─ clean → risk_agent     │
                                                  │            │
                                        orchestrator_decide    │
                                         ↕ (continuous loop)   │
                                     ┌───┴───┐                 │
                                     │       │                 │
                               [any agent]  [terminal]         │
                                     │       │                 │
                                     ↓       ↓                 │
                               orchestrator  ┌─────────────────┘
                                  _decide    │
                                             ↓
                                    explainability_agent
                                             │
                                       learning_agent
                                             │
                                        audit_agent
                                             │
                                            END

Key difference from V2:
  • NO plan[] array — the orchestrator decides the next action
    after observing the result of each agent
  • NO advance_step — there is no index to increment
  • The orchestrator can react to NEW information mid-pipeline
    (e.g., skip fairness if policy already rejected)
  • The fairness debate sub-loop is preserved unchanged

This is the "brain" of the system — it demonstrates:
  • Genuine adaptive orchestration (LLM decides in real-time)
  • Multi-agent collaboration via shared state
  • Agent debate (Fairness challenges Risk Adjustment)
  • Runtime replanning (no pre-generated plan)
"""

from __future__ import annotations
from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.data_agent import data_agent
from agents.risk_agent import risk_agent
from agents.policy_agent import policy_agent
from agents.fairness_agent import fairness_agent
from agents.fraud_agent import fraud_agent
from agents.risk_adjustment_agent import risk_adjustment_agent
from agents.learning_agent import learning_agent
from agents.orchestrator_agent import orchestrator_decide


# ── Decision Nodes ───────────────────────────────────────

def auto_approve(state: AgentState) -> AgentState:
    """Auto-approve LOW risk applications."""
    logs = list(state.get("agent_logs", []))
    logs.append({
        "agent": "orchestrator",
        "step": len(logs) + 1,
        "action": "Auto-approved (LOW risk)",
        "status": "success",
        "details": f"Risk score {state.get('risk_score')}/100 is below threshold. Application approved.",
        "duration_ms": 0,
    })
    return {
        **state,
        "final_decision": "approved",
        "decision_reasoning": f"Application auto-approved. Risk score: {state.get('risk_score')}/100 (LOW). Strong financial profile with no policy concerns.",
        "agent_logs": logs,
    }


def auto_reject(state: AgentState) -> AgentState:
    """Auto-reject HIGH risk applications or KYC failures."""
    logs = list(state.get("agent_logs", []))

    # ── Check for Identity Verification Failure ──────────────
    identity_failed = state.get("fraud_detected", False)
    identity_reason = state.get("fraud_reason", "Fraud or identity mismatch detected.")

    if identity_failed:
        logs.append({
            "agent": "orchestrator",
            "step": len(logs) + 1,
            "action": "Auto-rejected (KYC Failure)",
            "status": "success",
            "details": identity_reason,
            "duration_ms": 0,
        })
        return {
            **state,
            "final_decision": "rejected",
            "decision_reasoning": f"Application rejected. Reason: {identity_reason}.",
            "agent_logs": logs,
        }

    # ── Standard High Risk Rejection ─────────────────────────
    risk_factors = state.get("risk_factors", [])
    policy_violations = state.get("policy_violations", [])

    reasons = []
    if risk_factors:
        reasons.append(f"Risk factors: {', '.join(risk_factors[:3])}")
    if policy_violations:
        reasons.append(f"Policy violations: {', '.join(policy_violations[:2])}")

    detail = "; ".join(reasons) or f"Risk score {state.get('risk_score')}/100 exceeds threshold"

    logs.append({
        "agent": "orchestrator",
        "step": len(logs) + 1,
        "action": "Auto-rejected (HIGH risk / Policy failure)",
        "status": "success",
        "details": detail,
        "duration_ms": 0,
    })
    return {
        **state,
        "final_decision": "rejected",
        "decision_reasoning": f"Application rejected. {detail}",
        "agent_logs": logs,
    }


def final_decision(state: AgentState) -> AgentState:
    """
    Make final decision for MEDIUM-risk cases after all checks.
    """
    logs = list(state.get("agent_logs", []))
    policy_passed = state.get("policy_passed", True)
    bias_detected = state.get("bias_detected", False)
    policy_violations = state.get("policy_violations", [])

    if not policy_passed:
        decision = "rejected"
        reasoning = f"Application rejected due to policy violations: {'; '.join(policy_violations)}"
    elif bias_detected:
        decision = "escalated"
        reasoning = f"Application escalated to human reviewer. Risk score: {state.get('risk_score')}/100 (MEDIUM). Potential bias detected in automated assessment — requires human judgment."
    else:
        decision = "approved"
        reasoning = f"Application approved after full review. Risk score: {state.get('risk_score')}/100 (MEDIUM). Passed all policy checks and fairness analysis."

    logs.append({
        "agent": "orchestrator",
        "step": len(logs) + 1,
        "action": f"Final decision: {decision.upper()}",
        "status": "success",
        "details": reasoning,
        "duration_ms": 0,
    })

    return {
        **state,
        "final_decision": decision,
        "decision_reasoning": reasoning,
        "agent_logs": logs,
    }


# ── Routing Functions ────────────────────────────────────

def route_after_fraud(state: AgentState) -> str:
    """After fraud agent: reject immediately if fraud detected, else continue."""
    if state.get("fraud_detected"):
        return "auto_reject"
    return "risk_agent"


def route_orchestrator(state: AgentState) -> str:
    """Route based on the orchestrator's decision."""
    next_agent = state.get("orchestrator_next_agent", "final_decision")
    # Safety: ensure it's a valid routing target
    valid_targets = {
        "policy_agent", "fairness_agent", "risk_reeval", "fraud_recheck",
        "auto_approve", "auto_reject", "final_decision",
    }
    if next_agent in valid_targets:
        return next_agent
    return "final_decision"


def route_fairness(state: AgentState) -> str:
    """After fairness: debate with risk adjustment if bias detected."""
    turns = state.get("negotiation_turns", 0)
    bias = state.get("bias_detected", False)
    if bias and turns < 3:  # V5: increased from 2 to 3 for richer debates
        return "risk_adjustment_agent"
    return "orchestrator_decide"


# ── Build the Graph ──────────────────────────────────────

def build_loan_graph() -> StateGraph:
    """
    Construct the LangGraph pipeline with continuous dynamic orchestration.

    V5: The orchestrator_decide node replaces both planner_agent and
    advance_step. After every agent execution, it observes the state
    and decides what to do next in real-time.
    """
    graph = StateGraph(AgentState)

    # Import remaining agents
    from agents.explainability_agent import explainability_agent
    from agents.audit_agent import audit_agent

    # ── Add all nodes ────────────────────────────────────────
    graph.add_node("data_agent", data_agent)
    graph.add_node("fraud_agent", fraud_agent)
    graph.add_node("risk_agent", risk_agent)
    graph.add_node("orchestrator_decide", orchestrator_decide)   # V5: replaces planner + advance_step
    graph.add_node("policy_agent", policy_agent)
    graph.add_node("fairness_agent", fairness_agent)
    graph.add_node("risk_adjustment_agent", risk_adjustment_agent)
    graph.add_node("auto_approve", auto_approve)
    graph.add_node("auto_reject", auto_reject)
    graph.add_node("final_decision", final_decision)
    graph.add_node("explainability_agent", explainability_agent)
    graph.add_node("learning_agent", learning_agent)
    graph.add_node("audit_agent", audit_agent)

    # Wrapper nodes — allow orchestrator to re-invoke risk/fraud
    graph.add_node("risk_reeval", risk_agent)
    graph.add_node("fraud_recheck", fraud_agent)

    # ── Entry point ──────────────────────────────────────────
    graph.set_entry_point("data_agent")

    # ── Fixed initial pipeline ───────────────────────────────
    # data_agent and fraud_agent ALWAYS run first (prerequisites)
    graph.add_edge("data_agent", "fraud_agent")

    graph.add_conditional_edges("fraud_agent", route_after_fraud, {
        "auto_reject": "auto_reject",
        "risk_agent": "risk_agent",
    })

    # risk_agent ALWAYS runs before the orchestrator takes over
    graph.add_edge("risk_agent", "orchestrator_decide")

    # ── Continuous orchestration loop ────────────────────────
    # The orchestrator decides which agent to run next.
    # After that agent completes, it returns to the orchestrator.
    routing_map = {
        "policy_agent": "policy_agent",
        "fairness_agent": "fairness_agent",
        "risk_reeval": "risk_reeval",
        "fraud_recheck": "fraud_recheck",
        "auto_approve": "auto_approve",
        "auto_reject": "auto_reject",
        "final_decision": "final_decision",
    }
    graph.add_conditional_edges("orchestrator_decide", route_orchestrator, routing_map)

    # Non-terminal agents route BACK to orchestrator for the next decision
    graph.add_edge("policy_agent", "orchestrator_decide")
    graph.add_edge("risk_reeval", "orchestrator_decide")
    graph.add_edge("fraud_recheck", "orchestrator_decide")

    # ── Fairness debate sub-loop (preserved) ─────────────────
    # If bias detected: fairness → risk_adjustment → fairness (debate)
    # When debate resolves: fairness → orchestrator_decide
    graph.add_conditional_edges("fairness_agent", route_fairness, {
        "risk_adjustment_agent": "risk_adjustment_agent",
        "orchestrator_decide": "orchestrator_decide",
    })
    graph.add_edge("risk_adjustment_agent", "fairness_agent")

    # ── Terminal decisions → explainability pipeline ──────────
    # All terminal actions flow into the fixed tail:
    # explainability → learning → audit → END
    graph.add_edge("auto_approve", "explainability_agent")
    graph.add_edge("auto_reject", "explainability_agent")
    graph.add_edge("final_decision", "explainability_agent")

    graph.add_edge("explainability_agent", "learning_agent")
    graph.add_edge("learning_agent", "audit_agent")
    graph.add_edge("audit_agent", END)

    return graph.compile()


# Singleton compiled graph
loan_pipeline = build_loan_graph()
