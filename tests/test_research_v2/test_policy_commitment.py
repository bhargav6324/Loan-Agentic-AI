import pytest
from agents.state import AgentState
from orchestrator.graph import decision_commit

def test_decision_commit_blocks_missing_certificates():
    # Attempt to approve without certificates
    state: AgentState = {
        "messages": [],
        "risk_score": 10.0,
        "policy_results": [],
        "final_decision": None,
        "decision_proposal": "approve",
        "errors": []
    }
    
    new_state = decision_commit(state)
    
    # Should block and result in a fallback error or escalation because certificates are missing
    assert new_state["commit_status"] == "blocked_missing_certificate"
    assert "Commitment blocked: missing certificates" in new_state["commit_reason"]
    
def test_decision_commit_allows_with_certificates():
    # Provide valid certificates
    state: AgentState = {
        "messages": [],
        "risk_score": 10.0,
        "policy_results": [],
        "final_decision": None,
        "decision_proposal": "approve",
        "risk_certificate": {"timestamp": "123", "risk_score": 10.0},
        "policy_certificate": {"timestamp": "123", "passed": True, "violations": []},
        "errors": []
    }
    
    new_state = decision_commit(state)
    
    assert new_state["commit_status"] == "committed_approve"
    assert new_state["final_decision"] == "approved"
