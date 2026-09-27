import pytest
from core.canonical_contracts import (
    canonical_composite_score,
    canonical_risk_reassessment,
    canonical_policy_check,
    canonical_final_decision
)
from ml.canonical_model_contract import compute_heuristic_score

def test_heuristic_score_golden():
    features = {
        "credit_score": 650,
        "dti_ratio": 0.5,
        "loan_to_income": 3.0,
        "employment_score": 2.0,
        "defaults": 0.0,
        "credit_utilisation": 40.0,
        "loan_tenure_years": 5.0,
        "employment_years": 5.0,
        "loan_purpose": "personal"
    }
    score = compute_heuristic_score(features)
    assert score == 58.0

def test_composite_score_golden():
    # p = 0.8 => confidence = 0.8
    # weight = 0.4 + 0.32 = 0.72
    # heur_weight = 0.28
    # ml_score = 80
    # heuristic = 50
    # composite = 80 * 0.72 + 50 * 0.28 = 57.6 + 14 = 71.6
    ml, h, c = canonical_composite_score(0.8, 50.0)
    assert ml == 80.0
    assert h == 50.0
    assert c == 71.6

def test_risk_reassessment_golden():
    # Gap > 25
    ml_score = 80.0
    h_score = 40.0 # gap = 40 > 25
    original = 70.0
    # conservative = (80 + 40)/2 = 60.0. conservative is NOT > original. So no revision.
    res = canonical_risk_reassessment(original, ml_score, h_score, {})
    assert not res["revised"]
    assert res["revised_score"] == 70.0
    
    # Gap > 25, conservative > original
    original = 50.0
    res = canonical_risk_reassessment(original, ml_score, h_score, {})
    assert res["revised"]
    assert res["revised_score"] == 60.0

def test_policy_boundaries():
    valid = {
        "applicant_age": 21,
        "loan_to_income": 5.0,
        "defaults": 2,
        "credit_score": 400,
        "loan_amount": 2_500_000,
        "employment_score": 1
    }
    passed, vio = canonical_policy_check(valid)
    assert passed is True
    
    invalid = {
        "applicant_age": 20,
        "loan_to_income": 5.1,
        "defaults": 3,
        "credit_score": 399,
        "loan_amount": 2_500_001,
        "employment_score": 1
    }
    passed, vio = canonical_policy_check(invalid)
    assert passed is False
    assert len(vio) == 5

def test_final_decision_golden():
    status, reason = canonical_final_decision("LOW", True, [], False, False, False, "approve")
    assert status == "committed_approve"
    
    status, reason = canonical_final_decision("MEDIUM", True, [], True, False, False, "approve")
    assert status == "blocked_missing_audit"

    # Governance invariant: fairness MUST NOT convert a deterministic rejection
    # into an approval. When audit flags bias (recommend=approve) but the
    # decision_proposal is reject, the correct outcome is escalation for human
    # review — never a committed_approve.
    status, _ = canonical_final_decision(
        "MEDIUM", True, [], True, True, True, "reject",
        audit_recommended_action="approve",
    )
    assert status == "automated_escalated_status", (
        f"Fairness must not unilaterally approve a deterministic rejection. Got {status}"
    )

    status, _ = canonical_final_decision(
        "MEDIUM", True, [], True, True, True, "approve",
        audit_recommended_action="reject",
    )
    assert status == "committed_reject"
