from research_v2.phase2.case_factory import build_clean_case
from research_v2.phase2.certificates import (
    compute_state_hash,
    invalidate_dependents,
    is_current,
    issue_certificate,
    next_state_version,
)
from research_v2.phase2.fault_catalog import FaultType
from research_v2.phase2.fault_injector import inject_fault


def test_state_hash_is_deterministic_and_changes_on_evidence_change():
    case = build_clean_case("case-1")
    original = compute_state_hash(case)
    assert original == compute_state_hash(case.model_copy(deep=True))

    changed = inject_fault(case, FaultType.WRONG_EXTRACTED_INCOME)
    assert compute_state_hash(changed) != original
    assert next_state_version(case, changed) == 2
    assert next_state_version(case, case.model_copy(deep=True)) == 1


def test_certificate_is_bound_to_case_version_and_hash():
    case = build_clean_case("case-1")
    certificate = issue_certificate(case, "income_consistency", ["monthly_income"])
    assert is_current(certificate, case)

    changed = inject_fault(case, FaultType.CONTRADICTORY_INCOME)
    changed.state_version = 2
    assert not is_current(certificate, changed)


def test_transitive_invalidation_marks_children_stale():
    case = build_clean_case("case-1")
    income = issue_certificate(case, "income_consistency", ["monthly_income"])
    risk = issue_certificate(
        case, "risk_score", ["credit_score", "monthly_income"], [income.certificate_id]
    )
    decision = issue_certificate(
        case, "decision_readiness", ["policy_status"], [risk.certificate_id]
    )

    updated = invalidate_dependents(
        [income, risk, decision], ["monthly_income"]
    )
    assert all(c.status == "stale" for c in updated)


def test_unrelated_change_does_not_invalidate_certificate():
    case = build_clean_case("case-1")
    income = issue_certificate(case, "income_consistency", ["monthly_income"])
    updated = invalidate_dependents([income], ["employer_name"])
    assert updated[0].status == "valid"

