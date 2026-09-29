"""Full adapted graph reusing compatible original project agents."""
from __future__ import annotations

import json
from typing import Annotated, Optional

from langgraph.graph import END, StateGraph

from agents.orchestrator_agent import generate_initial_plan, orchestrator_decide
from agents.reporting_agent import reporting_agent
from agents.state import AgentState
from orchestrator.graph import decision_commit, policy_join, route_orchestrator
from research_v2.external_validation.loan45k_full_agents.agents import (
    critic_agent_45k, fairness_agent_45k, intake_agent_45k,
    policy_agent_45k, risk_agent_45k,
)


def _last(_old, new):
    return new


def _merge_unique(old, new):
    """Merge branch lists without re-adding the full history returned by nodes."""
    merged = []
    seen = set()
    for item in [*(old or []), *(new or [])]:
        marker = json.dumps(item, sort_keys=True, default=str)
        if marker not in seen:
            seen.add(marker)
            merged.append(item)
    return merged


class FullAgentState(AgentState, total=False):
    source_record: Annotated[dict, _last]
    approval_probability: Annotated[float, _last]
    counterfactual_probability: Annotated[float, _last]
    state_version: Annotated[int, _last]
    state_hash: Annotated[str, _last]
    critic_findings: Annotated[list, _last]
    agent_logs: Annotated[list, _merge_unique]
    errors: Annotated[list, _merge_unique]


def fan_out(state):
    return state


def planner_node(state):
    return generate_initial_plan(state)


def orchestrator_node(state):
    return orchestrator_decide(state)


def join_node(state):
    return policy_join(state)


def commit_node(state):
    return decision_commit(state)


def report_node(state):
    return reporting_agent(state)


def route_after_intake(state):
    return "planner" if not state.get("data_agent_early_stop") else "decision_commit"


def route_after_join(_state):
    return "next"


def route_orchestrator_node(state):
    return route_orchestrator(state)


def build_full_agent_graph(enable_critic: bool = False):
    graph = StateGraph(FullAgentState)
    graph.add_node("intake_agent", intake_agent_45k)
    graph.add_node("planner", planner_node)
    graph.add_node("fan_out", fan_out)
    graph.add_node("risk_agent", risk_agent_45k)
    graph.add_node("policy_agent", policy_agent_45k)
    graph.add_node("orchestrator_decide", orchestrator_node)
    graph.add_node("fairness_agent", fairness_agent_45k)
    graph.add_node("risk_reeval", risk_agent_45k)
    graph.add_node("policy_join", join_node)
    if enable_critic:
        graph.add_node("critic", critic_agent_45k)
    graph.add_node("decision_commit", commit_node)
    graph.add_node("reporting_agent", report_node)
    graph.set_entry_point("intake_agent")
    graph.add_conditional_edges("intake_agent", route_after_intake,
                                {"planner": "planner", "decision_commit": "decision_commit"})
    graph.add_edge("planner", "fan_out")
    graph.add_edge("fan_out", "risk_agent")
    graph.add_edge("fan_out", "policy_agent")
    graph.add_edge(["risk_agent", "policy_agent"], "orchestrator_decide")
    graph.add_conditional_edges("orchestrator_decide", route_orchestrator_node, {
        "fairness_agent": "fairness_agent", "risk_reeval": "risk_reeval",
        "decision_commit": "decision_commit", "policy_join": "policy_join"})
    graph.add_edge("fairness_agent", "orchestrator_decide")
    graph.add_edge("risk_reeval", "orchestrator_decide")
    graph.add_conditional_edges("policy_join", route_after_join,
                                {"next": "critic" if enable_critic else "decision_commit"})
    if enable_critic:
        graph.add_edge("critic", "decision_commit")
    graph.add_edge("decision_commit", "reporting_agent")
    graph.add_edge("reporting_agent", END)
    return graph.compile()


FULL_F2 = build_full_agent_graph(False)
FULL_F3 = build_full_agent_graph(True)
