from research_v2.phase2.case_factory import build_clean_case
from research_v2.phase2.certificates import issue_certificate
from research_v2.phase2.commitment_gate import evaluate_commitment
from research_v2.phase2.fault_catalog import FaultType
from research_v2.phase2.fault_injector import inject_fault


def _certificates(case):
    return [
        issue_certificate(case, "income_consistency", ["monthly_income"]),
        issue_certificate(case, "risk_score", ["credit_score"], []),
    ]


def test_clean_case_with_current_certificates_is_allowed():
    case = build_clean_case("case-1", {"credit_score": 700})
    result = evaluate_commitment(
        case, _certificates(case), ["income_consistency", "risk_score"], True
    )
    assert result.allowed is True
    assert result.status == "commit_allowed"


def test_changed_state_blocks_old_certificates():
    case = build_clean_case("case-1", {"credit_score": 700})
    certificates = _certificates(case)
    changed = inject_fault(case, FaultType.WRONG_EXTRACTED_INCOME)
    changed.state_version = 2
    result = evaluate_commitment(
        changed, certificates, ["income_consistency", "risk_score"], True
    )
    assert result.allowed is False
    assert "certificate:stale:income_consistency" in result.reasons


def test_gate_blocks_policy_and_premature_commitment():
    case = build_clean_case("case-1")
    case.execution.policy_node_executed = False
    case.execution.commitment_attempted = True
    result = evaluate_commitment(case, [], ["income_consistency"], False)
    assert result.allowed is False
    assert "execution:policy_node_not_executed" in result.reasons
    assert "execution:premature_commitment" in result.reasons
    assert "policy:failed" in result.reasons


def test_gate_blocks_stale_risk_score():
    case = build_clean_case("case-1")
    case.execution.risk_score_state_version = 0
    result = evaluate_commitment(case, [], [], True)
    assert result.allowed is False
    assert "execution:stale_risk_score" in result.reasons
