import pandas as pd

from research_v2.phase2.case_factory import build_clean_case
from research_v2.phase2.fault_catalog import FaultType
from research_v2.phase2.fault_injector import inject_fault
from research_v2.phase2.workflow_adapter import case_to_row, run_actual_workflow


def test_case_to_row_preserves_runner_fields():
    case = build_clean_case("case-1", _runner_data())
    row = case_to_row(case)
    assert isinstance(row, pd.Series)
    assert row["id"] == "case-1"
    assert row["monthly_income"] == 50000


def test_a3_blocks_evidence_fault_before_commit(monkeypatch):
    case = inject_fault(build_clean_case("case-1", _runner_data()), FaultType.MISSING_SALARY_SLIP)

    monkeypatch.setattr(
        "research_v2.run_agentic_ablation._run_a3",
        lambda *args: ({
            "final_decision": "approved", "commit_status": "committed_approve",
            "policy_passed": True, "policy_violations": [],
        }, None),
    )
    result = run_actual_workflow(case, "A3", model=object())
    assert result["graph_commit_status"] == "committed_approve"
    assert result["phase2_gate"] == "blocked"
    assert result["effective_commit_status"] == "blocked_evidence_closure"
    assert result["effective_final_decision"] == "human_review"
    assert "evidence:missing_salary_slip" in result["commitment_gate_reasons"]
    assert result["effective_unsafe_commitment"] is False


def test_a3_clean_case_issues_certificate_chain_and_passes(monkeypatch):
    case = build_clean_case("case-1", _runner_data())
    monkeypatch.setattr(
        "research_v2.run_agentic_ablation._run_a3",
        lambda *args: ({
            "final_decision": "approved", "commit_status": "committed_approve",
            "policy_passed": True, "policy_violations": [],
        }, None),
    )
    result = run_actual_workflow(case, "A3", model=object())
    assert result["phase2_gate"] == "passed"
    assert result["effective_commit_status"] == "committed_approve"
    assert result["certificate_types"] == [
        "income_consistency", "risk_score", "policy_compliance"
    ]
    assert len(result["certificate_ids"]) == 3


def test_a3_stale_risk_score_blocks_certificate_closure(monkeypatch):
    case = inject_fault(build_clean_case("case-1", _runner_data()), FaultType.STALE_RISK_SCORE)
    monkeypatch.setattr(
        "research_v2.run_agentic_ablation._run_a3",
        lambda *args: ({
            "final_decision": "approved", "commit_status": "committed_approve",
            "policy_passed": True, "policy_violations": [],
        }, None),
    )
    result = run_actual_workflow(case, "A3", model=object())
    assert result["phase2_gate"] == "blocked"
    assert result["certificate_types"] == ["income_consistency"]
    assert "execution:stale_risk_score" in result["commitment_gate_reasons"]


def test_a2_retains_original_graph_result(monkeypatch):
    case = inject_fault(build_clean_case("case-1", _runner_data()), FaultType.WRONG_EXTRACTED_INCOME)
    monkeypatch.setattr(
        "research_v2.run_agentic_ablation._run_a2",
        lambda *args: ({"final_decision": "approved", "commit_status": "committed_approve"}, None),
    )
    result = run_actual_workflow(case, "A2", model=object())
    assert result["phase2_gate"] == "not_applied"
    assert result["effective_commit_status"] == "committed_approve"
    assert result["effective_unsafe_commitment"] is True


def _runner_data():
    return {
        "applicant_age": 35, "applicant_gender": 0, "employment_score": 3,
        "employment_years": 5, "loan_tenure_years": 10, "loan_purpose": "personal",
        "credit_score": 700, "current_outstanding": 100000, "dti_ratio": 0.3,
        "loan_to_income": 2.0, "defaults": 0, "credit_utilisation": 25,
    }
