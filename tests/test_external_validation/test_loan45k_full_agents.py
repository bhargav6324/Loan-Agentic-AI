from config import settings
from research_v2.external_validation.loan45k_full_agents.graph import FULL_F2, FULL_F3


def record(**changes):
    base = {"benchmark_id": 1, "person_age": 35, "person_gender": "female",
            "person_income": 60000, "monthly_income": 5000, "person_emp_exp": 8,
            "person_home_ownership": "RENT", "loan_amnt": 10000,
            "loan_intent": "EDUCATION", "loan_int_rate": 10,
            "loan_percent_income": .17, "cb_person_cred_hist_length": 10,
            "credit_score": 720, "previous_loan_defaults_on_file": "No", "loan_status": 1}
    base.update(changes)
    return base


def invoke(graph, row=None, p=.8, cf=.79):
    old_key, old_mode = settings.GROQ_API_KEY, settings.PAPER_MODE
    settings.GROQ_API_KEY, settings.PAPER_MODE = "gsk_placeholder", True
    try:
        return graph.invoke({"source_record": row or record(),
                             "approval_probability": p,
                             "counterfactual_probability": cf,
                             "messages": [], "agent_logs": [], "errors": [],
                             "audit_epsilon": .06})
    finally:
        settings.GROQ_API_KEY, settings.PAPER_MODE = old_key, old_mode


def test_f2_reuses_original_planner_orchestrator_commit_and_reporting():
    result = invoke(FULL_F2)
    agents = {x.get("agent") for x in result["agent_logs"]}
    assert {"intake_agent", "risk_agent", "policy_agent", "orchestrator",
            "decision_commit", "explainability_agent", "audit_agent"} <= agents
    assert result["commit_status"].startswith("committed_")
    assert result["pan_verification"]["status"] == "not_evaluated"


def test_f3_executes_critic_and_preserves_policy_rejection():
    result = invoke(FULL_F3, record(credit_score=350), .9, .9)
    assert any(x.get("agent") == "critic" for x in result["agent_logs"])
    assert result["final_decision"] == "rejected"


def test_no_identity_values_are_fabricated():
    result = invoke(FULL_F2)
    assert "applicant_pan" not in result["applicant_data"]
    assert "applicant_aadhaar" not in result["applicant_data"]
