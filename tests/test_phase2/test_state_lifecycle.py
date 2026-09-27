from research_v2.phase2.case_factory import build_clean_case
from research_v2.phase2.certificates import is_current, issue_certificate
from research_v2.phase2.commitment_gate import evaluate_commitment
from research_v2.phase2.state_lifecycle import mark_risk_reassessed, update_applicant_state


def test_state_change_increments_version_and_invalidates_dependencies():
    case = build_clean_case("case-1", {"credit_score": 700})
    risk = issue_certificate(case, "risk_score", ["credit_score"])
    policy = issue_certificate(
        case, "policy_compliance", ["policy_passed"], [risk.certificate_id]
    )

    transition = update_applicant_state(
        case, {"credit_score": 650}, [risk, policy]
    )
    assert transition.state_changed is True
    assert transition.case.state_version == 2
    assert transition.case.execution.risk_score_state_version == 1
    assert [certificate.status for certificate in transition.certificates] == [
        "stale", "stale"
    ]
    assert not is_current(transition.certificates[0], transition.case)


def test_noop_update_does_not_increment_version():
    case = build_clean_case("case-1", {"credit_score": 700})
    transition = update_applicant_state(case, {"credit_score": 700})
    assert transition.state_changed is False
    assert transition.case.state_version == 1


def test_reassessment_rebinds_risk_to_current_state():
    case = build_clean_case("case-1", {"credit_score": 700})
    transition = update_applicant_state(case, {"credit_score": 650})
    stale_gate = evaluate_commitment(transition.case, [], [], True)
    assert "execution:stale_risk_score" in stale_gate.reasons

    reassessed = mark_risk_reassessed(transition.case)
    recovered_gate = evaluate_commitment(reassessed, [], [], True)
    assert recovered_gate.allowed is True
    assert reassessed.execution.risk_score_state_version == 2

