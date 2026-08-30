import pytest
from agents.fairness_agent import _run_fairness_math
from agents.state import AgentState

def test_counterfactual_direction():
    """Test that a higher counterfactual score means the original score was lenient."""
    
    # Original score is 40. Counterfactual (flipped) score is 50.
    # The flipped demographic received a higher risk score. This means the original demographic was treated leniently.
    # Delta should be +10 (Positive).
    
    state: AgentState = {
        "messages": [],
        "risk_score": 40.0,
        "applicant_data": {"applicant_gender": "male", "monthly_income": 5000},
        "credit_report": {"credit_score": 700},
        "errors": []
    }
    
    # We must mock the re-evaluation to return 50.0
    def mock_eval(features, state):
        return {"risk_score": 50.0, "risk_category": "MEDIUM", "agent_justification": "Mocked", "bias_source": "mock"}
        
    import agents.risk_agent
    original_eval = agents.risk_agent.evaluate_profile
    agents.risk_agent.evaluate_profile = mock_eval
    
    try:
        new_state = _run_fairness_math(state)
        # Delta = Counterfactual - Original = 50.0 - 40.0 = +10.0
        assert new_state["bias_detected"] is True
        assert new_state["fairness_report"]["counterfactual_delta_summary"]["gender"]["worst_signed_delta"] == 10.0
        
    finally:
        agents.risk_agent.evaluate_profile = original_eval
