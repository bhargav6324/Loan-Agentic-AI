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
- policy_join merges policy result + orchestrator decision
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
    """Auto-reject HIGH risk applications or Intake failures."""
    logs = list(state.get("agent_logs", []))

    if state.get("fraud_detected", False) or state.get("data_agent_early_stop", False):
        reason = state.get("fraud_reason") or state.get("data_agent_summary") or "KYC/Fraud Check Failed"
        logs.append({"agent": "orchestrator", "step": len(logs) + 1, "action": "Auto-rejected (Intake Failure)", "status": "success", "details": reason, "duration_ms": 0})
        return {**state, "final_decision": "rejected", "decision_reasoning": f"Application rejected. Reason: {reason}.", "agent_logs": logs}

    if not state.get("policy_passed", True):
        violations = state.get("policy_violations", [])
        detail = f"Policy violations: {', '.join(violations[:2])}"
        logs.append({"agent": "orchestrator", "step": len(logs) + 1, "action": "Auto-rejected (Policy Failure)", "status": "success", "details": detail, "duration_ms": 0})
        return {**state, "final_decision": "rejected", "decision_reasoning": f"Application rejected. {detail}", "agent_logs": logs}

    risk_factors = state.get("risk_factors", [])
    detail = f"Risk factors: {', '.join(risk_factors[:3])}" if risk_factors else f"Risk score {state.get('risk_score')}/100 exceeds threshold"
    logs.append({"agent": "orchestrator", "step": len(logs) + 1, "action": "Auto-rejected (HIGH risk)", "status": "success", "details": detail, "duration_ms": 0})
    return {**state, "final_decision": "rejected", "decision_reasoning": f"Application rejected. {detail}", "agent_logs": logs}


def final_decision(state: AgentState) -> AgentState:
    """Make final decision for MEDIUM-risk cases after all checks."""
    logs = list(state.get("agent_logs", []))
    bias_detected = state.get("bias_detected", False)

    # If parallel policy check failed, override to reject
    if not state.get("policy_passed", True):
        violations = state.get("policy_violations", [])
        decision = "rejected"
        reasoning = f"Application rejected. Policy violations (detected in parallel): {', '.join(violations[:2])}"
    elif bias_detected:
        decision = "escalated"
        reasoning = f"Application escalated to human reviewer. Risk score: {state.get('risk_score')}/100 (MEDIUM). Potential bias detected."
    else:
        decision = "approved"
        reasoning = f"Application approved after full review. Risk score: {state.get('risk_score')}/100 (MEDIUM). Passed all policy and fairness checks."

    logs.append({"agent": "orchestrator", "step": len(logs) + 1, "action": f"Final decision: {decision.upper()}", "status": "success", "details": reasoning, "duration_ms": 0})
    return {**state, "final_decision": decision, "decision_reasoning": reasoning, "agent_logs": logs}


# ── Join Node ─────────────────────────────────────────────

def policy_join(state: AgentState) -> AgentState:
    """
    Merge point after parallel policy_agent + orchestrator branches.
    Pure state inspection — no LLM call.
    If policy failed AND orchestrator wanted to approve → override to reject.
    """
    logs = list(state.get("agent_logs", []))
    next_agent = state.get("orchestrator_next_agent", "final_decision")
    policy_passed = state.get("policy_passed", True)

    if not policy_passed and next_agent in ("auto_approve", "final_decision"):
        logs.append({
            "agent": "policy_join",
            "step": len(logs) + 1,
            "action": "Policy join: override to reject (parallel policy failed)",
            "status": "policy_override",
            "details": f"Orchestrator wanted {next_agent} but policy violations: {state.get('policy_violations', [])}",
            "duration_ms": 0,
        })
        return {**state, "orchestrator_next_agent": "auto_reject", "agent_logs": logs}

    logs.append({
        "agent": "policy_join",
        "step": len(logs) + 1,
        "action": f"Policy join: routing to {next_agent}",
        "status": "success",
        "details": f"Policy passed={policy_passed}, orchestrator decision={next_agent}",
        "duration_ms": 0,
    })
    return {**state, "agent_logs": logs}


# ── Fan-out Node ─────────────────────────────────────────

def fan_out(state: AgentState) -> AgentState:
    """No-op router: LangGraph fans out from here to risk + policy in parallel."""
    return state


# ── Routing Functions ────────────────────────────────────

def route_after_intake(state: AgentState) -> str:
    if state.get("fraud_detected") or state.get("data_agent_early_stop"):
        return "auto_reject"
    return "planner"


def route_orchestrator(state: AgentState) -> str:
    next_agent = state.get("orchestrator_next_agent", "final_decision")
    valid_targets = {"fairness_agent", "risk_reeval", "auto_approve", "auto_reject", "final_decision", "policy_join"}
    return next_agent if next_agent in valid_targets else "policy_join"


def route_after_join(state: AgentState) -> str:
    next_agent = state.get("orchestrator_next_agent", "final_decision")
    valid = {"auto_approve", "auto_reject", "final_decision"}
    return next_agent if next_agent in valid else "final_decision"


# ── Build the Graph ──────────────────────────────────────

def build_loan_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("intake_agent", intake_agent)
    graph.add_node("planner", generate_initial_plan)
    graph.add_node("fan_out", fan_out)
    graph.add_node("risk_agent", risk_agent)
    graph.add_node("policy_agent", policy_agent_node)
    graph.add_node("orchestrator_decide", orchestrator_decide)
    graph.add_node("fairness_agent", fairness_agent)
    graph.add_node("risk_reeval", risk_agent)
    graph.add_node("policy_join", policy_join)
    graph.add_node("auto_approve", auto_approve)
    graph.add_node("auto_reject", auto_reject)
    graph.add_node("final_decision", final_decision)
    graph.add_node("reporting_agent", reporting_agent)

    # Entry
    graph.set_entry_point("intake_agent")

    # Intake → auto_reject (hard fail) OR planner
    graph.add_conditional_edges("intake_agent", route_after_intake, {
        "auto_reject": "auto_reject",
        "planner": "planner",
    })
    
    # Planner → fan_out
    graph.add_edge("planner", "fan_out")

    # Fan-out: both edges fire concurrently
    graph.add_edge("fan_out", "risk_agent")
    graph.add_edge("fan_out", "policy_agent")

    # Risk branch → orchestrator loop
    graph.add_edge("risk_agent", "orchestrator_decide")

    routing_map = {
        "fairness_agent": "fairness_agent",
        "risk_reeval": "risk_reeval",
        "auto_approve": "auto_approve",
        "auto_reject": "auto_reject",
        "final_decision": "final_decision",
        "policy_join": "policy_join",
    }
    graph.add_conditional_edges("orchestrator_decide", route_orchestrator, routing_map)

    # Non-terminal agents loop back to orchestrator
    graph.add_edge("fairness_agent", "orchestrator_decide")
    graph.add_edge("risk_reeval", "orchestrator_decide")

    # Policy branch always joins before terminal decisions
    graph.add_edge("policy_agent", "policy_join")

    # Join → terminal decision
    graph.add_conditional_edges("policy_join", route_after_join, {
        "auto_approve": "auto_approve",
        "auto_reject": "auto_reject",
        "final_decision": "final_decision",
    })

    # Terminal decisions → reporting
    graph.add_edge("auto_approve", "reporting_agent")
    graph.add_edge("auto_reject", "reporting_agent")
    graph.add_edge("final_decision", "reporting_agent")
    graph.add_edge("reporting_agent", END)

    return graph.compile()


# Singleton compiled graph
loan_pipeline = build_loan_graph()
