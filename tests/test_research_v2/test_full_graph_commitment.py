"""
Full-graph commitment tests.

These tests compile and invoke the actual LangGraph pipeline to prove that:
1. No terminal decision (approved/rejected) can be created outside decision_commit.
2. Missing certificates block commitment.
3. Policy hard-reject blocks an approval proposal.
4. MEDIUM-risk approval is blocked until fairness_report is present.
5. After fairness_report is provided, MEDIUM-risk approval succeeds.

All agent nodes are mocked so no LLM calls or ML inference occurs.
Mocks are applied BEFORE build_loan_graph() so the graph captures mock references.
"""

import pytest
import time
from orchestrator.graph import build_loan_graph, decision_commit
from agents.state import AgentState


# ── Helpers ─────────────────────────────────────────────────

def _base_state():
    return {
        "messages": [],
        "application_id": "test_1",
        "user_id": "user_1",
        "applicant_data": {
            "credit_score": 750, "dti_ratio": 0.3, "loan_to_income": 2.0,
            "employment_score": 3, "age": 35, "defaults": 0,
            "credit_utilisation": 20, "gender": "male", "location": "urban",
        },
        "risk_score": 0.0,
        "policy_results": [],
        "agent_logs": [],
        "errors": [],
    }


def _apply_common_mocks(mp, risk_ret, policy_ret, orchestrator_decisions):
    """
    Mock every node so the graph is fully deterministic and requires no LLM/model.
    orchestrator_decisions is a list of dicts; each call to orchestrator_decide pops the first.
    """
    mp.setattr("orchestrator.graph.intake_agent",
               lambda s: {"fraud_detected": False, "data_agent_early_stop": False})
    mp.setattr("orchestrator.graph.generate_initial_plan", lambda s: {})
    mp.setattr("orchestrator.graph.fan_out", lambda s: {})
    mp.setattr("orchestrator.graph.risk_agent", lambda s: risk_ret)
    mp.setattr("orchestrator.graph.policy_agent_node", lambda s: policy_ret)
    mp.setattr("orchestrator.graph.reporting_agent", lambda s: {})

    call_idx = [0]

    def mock_orchestrator(state):
        idx = call_idx[0]
        call_idx[0] += 1
        if idx < len(orchestrator_decisions):
            d = orchestrator_decisions[idx]
        else:
            # Safety: force terminal
            d = {"orchestrator_next_agent": "policy_join", "decision_proposal": "reject"}
        ret = {**state}
        ret["orchestrator_next_agent"] = d.get("orchestrator_next_agent", "policy_join")
        if "decision_proposal" in d:
            ret["decision_proposal"] = d["decision_proposal"]
        if "risk_category" in d:
            ret["risk_category"] = d["risk_category"]
        history = list(state.get("orchestrator_history", []))
        history.append({"step": idx + 1, "agent": d.get("orchestrator_next_agent", "policy_join"),
                        "reasoning": "mock"})
        ret["orchestrator_history"] = history
        ret["orchestrator_iterations"] = idx + 1
        ret["agent_logs"] = list(state.get("agent_logs", []))
        ret["agent_logs"].append({"agent": "orchestrator", "step": idx + 1,
                                   "action": f"Orchestrator → {d.get('orchestrator_next_agent', 'policy_join')}",
                                   "status": "mock", "details": "mock", "duration_ms": 0})
        return ret

    mp.setattr("orchestrator.graph.orchestrator_decide", mock_orchestrator)


# ── Tests ───────────────────────────────────────────────────

def test_full_graph_missing_certificate(monkeypatch):
    """Risk cert present but policy cert missing → blocked_missing_certificate."""
    risk_ret = {
        "risk_certificate": {"timestamp": "123", "risk_score": 10.0},
        "risk_category": "LOW",
    }
    policy_ret = {}  # Missing policy_certificate

    # Orchestrator routes straight to policy_join then decision_commit
    orch = [{"orchestrator_next_agent": "policy_join", "decision_proposal": "approve"}]

    _apply_common_mocks(monkeypatch, risk_ret, policy_ret, orch)
    graph = build_loan_graph(disable_fairness=True)
    final = graph.invoke(_base_state())

    assert final["commit_status"] == "blocked_missing_certificate"
    assert final.get("final_decision") != "approved"


@pytest.mark.parametrize(
    "risk_delay,policy_delay,expected_first",
    [(0.03, 0.0, "policy"), (0.0, 0.03, "risk")],
)
def test_parallel_completion_orders_still_require_both_certificates(
    monkeypatch, risk_delay, policy_delay, expected_first
):
    """Exercise the compiled graph with risk-first and policy-first completion."""
    completions = []

    def risk_node(state):
        time.sleep(risk_delay)
        completions.append("risk")
        return {
            "risk_certificate": {"timestamp": "r", "risk_score": 10.0},
            "risk_category": "LOW",
            "risk_ml_confidence": 0.9,
        }

    def policy_node(state):
        time.sleep(policy_delay)
        completions.append("policy")
        return {
            "policy_certificate": {"timestamp": "p", "passed": True, "violations": []},
            "policy_passed": True,
            "policy_violations": [],
        }

    monkeypatch.setattr("orchestrator.graph.intake_agent", lambda s: {"fraud_detected": False, "data_agent_early_stop": False})
    monkeypatch.setattr("orchestrator.graph.generate_initial_plan", lambda s: {})
    monkeypatch.setattr("orchestrator.graph.fan_out", lambda s: {})
    monkeypatch.setattr("orchestrator.graph.risk_agent", risk_node)
    monkeypatch.setattr("orchestrator.graph.policy_agent_node", policy_node)
    monkeypatch.setattr("orchestrator.graph.orchestrator_decide", lambda s: {
        "orchestrator_next_agent": "policy_join", "decision_proposal": "approve"
    })
    monkeypatch.setattr("orchestrator.graph.reporting_agent", lambda s: {})

    final = build_loan_graph(disable_fairness=True).invoke(_base_state())
    assert completions[0] == expected_first
    assert final["risk_certificate"]
    assert final["policy_certificate"]
    assert final["commit_status"] == "committed_approve"


def test_full_graph_policy_reject_conflict(monkeypatch):
    """Low risk proposal=approve but policy says passed=False → blocked_policy_violation."""
    risk_ret = {
        "risk_certificate": {"timestamp": "123", "risk_score": 10.0},
        "risk_category": "LOW",
    }
    policy_ret = {
        "policy_certificate": {"timestamp": "123", "passed": False, "violations": ["dti_ratio"]},
    }

    orch = [{"orchestrator_next_agent": "policy_join", "decision_proposal": "approve"}]

    _apply_common_mocks(monkeypatch, risk_ret, policy_ret, orch)
    graph = build_loan_graph(disable_fairness=True)
    final = graph.invoke(_base_state())

    assert final["commit_status"] == "blocked_policy_violation"
    assert final.get("final_decision") != "approved"


def test_full_graph_medium_risk_blocked_without_audit(monkeypatch):
    """MEDIUM risk with no fairness_report → blocked_missing_audit."""
    risk_ret = {
        "risk_certificate": {"timestamp": "123", "risk_score": 50.0},
        "risk_category": "MEDIUM",
    }
    policy_ret = {
        "policy_certificate": {"timestamp": "123", "passed": True, "violations": []},
    }

    # Orchestrator goes straight to policy_join without running fairness
    orch = [{"orchestrator_next_agent": "policy_join", "decision_proposal": "approve",
             "risk_category": "MEDIUM"}]

    _apply_common_mocks(monkeypatch, risk_ret, policy_ret, orch)
    graph = build_loan_graph(disable_fairness=True)
    final = graph.invoke(_base_state())

    assert final["commit_status"] == "blocked_missing_audit"
    assert final.get("final_decision") != "approved"


def test_full_graph_medium_risk_approved_with_audit(monkeypatch):
    """MEDIUM risk with fairness_report present → committed_approve."""
    risk_ret = {
        "risk_certificate": {"timestamp": "123", "risk_score": 50.0},
        "risk_category": "MEDIUM",
    }
    policy_ret = {
        "policy_certificate": {"timestamp": "123", "passed": True, "violations": []},
    }

    def mock_fairness(state):
        return {
            "fairness_report": {
                "counterfactual_delta_summary": {
                    "gender": {"worst_signed_delta": 2.0, "bias_flagged": False}
                }
            },
            "bias_detected": False,
            "agent_logs": list(state.get("agent_logs", [])) + [
                {"agent": "fairness_agent", "step": 1,
                 "action": "Gender counterfactual analysis", "status": "success",
                 "details": "mock", "duration_ms": 0}
            ],
        }

    # Step 1: orchestrator routes to fairness_agent
    # Step 2: after fairness returns, orchestrator routes to policy_join with approve
    orch = [
        {"orchestrator_next_agent": "fairness_agent", "risk_category": "MEDIUM"},
        {"orchestrator_next_agent": "policy_join", "decision_proposal": "approve",
         "risk_category": "MEDIUM"},
    ]

    _apply_common_mocks(monkeypatch, risk_ret, policy_ret, orch)
    monkeypatch.setattr("orchestrator.graph.fairness_agent", mock_fairness)
    graph = build_loan_graph(disable_fairness=False)
    final = graph.invoke(_base_state())

    assert final["commit_status"] == "committed_approve"
    assert final["final_decision"] == "approved"

    # Verify fairness ran BEFORE the successful decision_commit.
    # In a parallel architecture, decision_commit may be invoked multiple times
    # (blocked from the policy branch first, then succeeding after fairness).
    # We check that the SUCCESSFUL commit happens after fairness.
    log_actions = [log["action"] for log in final.get("agent_logs", [])]
    successful_commit_idx = next(
        i for i, a in enumerate(log_actions) if "Decision Commit: committed_approve" in a
    )
    fairness_idx = next(
        i for i, a in enumerate(log_actions) if "Gender counterfactual analysis" in a
    )
    assert fairness_idx < successful_commit_idx, \
        f"Fairness (idx={fairness_idx}) must precede successful commit (idx={successful_commit_idx})"


def test_full_graph_clean_low_risk_approve(monkeypatch):
    """Low risk, policy passed, no audit needed → committed_approve."""
    risk_ret = {
        "risk_certificate": {"timestamp": "123", "risk_score": 10.0},
        "risk_category": "LOW",
    }
    policy_ret = {
        "policy_certificate": {"timestamp": "123", "passed": True, "violations": []},
    }

    orch = [{"orchestrator_next_agent": "policy_join", "decision_proposal": "approve"}]

    _apply_common_mocks(monkeypatch, risk_ret, policy_ret, orch)
    graph = build_loan_graph(disable_fairness=True)
    final = graph.invoke(_base_state())

    assert final["commit_status"] == "committed_approve"
    assert final["final_decision"] == "approved"


def test_no_terminal_decision_outside_decision_commit(monkeypatch):
    """Prove that only the decision_commit node sets final_decision."""
    risk_ret = {
        "risk_certificate": {"timestamp": "123", "risk_score": 10.0},
        "risk_category": "LOW",
    }
    policy_ret = {
        "policy_certificate": {"timestamp": "123", "passed": True, "violations": []},
    }

    orch = [{"orchestrator_next_agent": "policy_join", "decision_proposal": "approve"}]

    _apply_common_mocks(monkeypatch, risk_ret, policy_ret, orch)
    graph = build_loan_graph(disable_fairness=True)

    # Stream through the graph and check that final_decision is only set
    # by the decision_commit step
    final_decision_sources = []
    prev_final = None
    for step in graph.stream(_base_state()):
        node_name = list(step.keys())[0]
        node_output = step[node_name]
        cur_final = node_output.get("final_decision") if isinstance(node_output, dict) else None
        if cur_final and cur_final != prev_final:
            final_decision_sources.append(node_name)
            prev_final = cur_final

    # final_decision should only appear from decision_commit
    assert final_decision_sources == ["decision_commit"], \
        f"final_decision was set by nodes: {final_decision_sources}"
