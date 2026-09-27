import pytest
from agents.fairness_agent import fairness_agent
from config import settings

def test_paper_mode_fairness_math():
    """
    Prove that in PAPER_MODE:
    - Fairness math executes
    - fairness_report is retained
    - gender, age and location deltas are present
    - LLM debate is not invoked
    - Risk score is not mutated by debate
    """
    # Setup test state for a MEDIUM risk case (score ~ 55)
    initial_score = 55.0
    state = {
        "risk_score": initial_score,
        "risk_category": "MEDIUM",
        "applicant_data": {
            "applicant_name": "Test Applicant",
            "applicant_gender": "female",
            "applicant_age": 35,
            "applicant_location": "urban",
        },
        "credit_report": {
            "credit_score": 650,
            "total_defaults": 0,
        },
        "agent_logs": [],
        "fairness_debate_log": []
    }
    
    # Enable paper mode
    original_paper_mode = getattr(settings, "PAPER_MODE", False)
    settings.PAPER_MODE = True
    
    try:
        final_state = fairness_agent(state)
    finally:
        settings.PAPER_MODE = original_paper_mode

    # 1. Fairness math executes and report is retained
    assert "fairness_report" in final_state
    report = final_state["fairness_report"]
    assert report is not None
    
    # 2. Gender, age, and location deltas are present
    delta_summary = report.get("counterfactual_delta_summary", {})
    assert "gender" in delta_summary
    assert "age" in delta_summary
    assert "location" in delta_summary
    
    # 3. LLM debate is not invoked
    # Debate log should not have any "outcome" keys or semantic debate turns
    # In PAPER_MODE, _run_fairness_debate is skipped.
    debate_log = final_state.get("fairness_debate_log", [])
    for entry in debate_log:
        assert "outcome" not in entry, "Semantic debate outcome should not be present in PAPER_MODE"
    
    # 4. Risk score is not mutated by debate
    assert final_state["risk_score"] == initial_score, "Risk score should not be mutated in PAPER_MODE"
