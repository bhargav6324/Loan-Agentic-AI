from research_v2.phase2.case_factory import build_clean_case
from research_v2.phase2.fault_catalog import FaultType
from research_v2.phase2.fault_injector import inject_fault
from research_v2.phase2.evidence_validator import validate_evidence


def test_clean_case_is_reproducible_and_valid():
    first = build_clean_case("case-001")
    second = build_clean_case("case-001")
    assert first.model_dump() == second.model_dump()
    assert validate_evidence(first).valid is True


def test_injection_does_not_mutate_clean_case():
    clean = build_clean_case("case-001")
    inject_fault(clean, FaultType.WRONG_EXTRACTED_INCOME)
    assert clean.extracted_evidence["monthly_income"].value == 50000


def test_missing_salary_slip_fault():
    faulty = inject_fault(build_clean_case("case-001"), FaultType.MISSING_SALARY_SLIP)
    assert faulty.fault_labels == [FaultType.MISSING_SALARY_SLIP.value]
    assert validate_evidence(faulty).valid is False


def test_wrong_income_fault():
    faulty = inject_fault(build_clean_case("case-001"), FaultType.WRONG_EXTRACTED_INCOME)
    assert faulty.extracted_evidence["monthly_income"].value == 35000


def test_contradictory_income_fault():
    faulty = inject_fault(build_clean_case("case-001"), FaultType.CONTRADICTORY_INCOME)
    result = validate_evidence(faulty)
    assert result.valid is False
    assert "income_contradiction" in result.errors


def test_skipped_policy_node_fault():
    faulty = inject_fault(build_clean_case("case-001"), FaultType.SKIPPED_POLICY_NODE)
    assert faulty.execution.policy_node_executed is False


def test_stale_risk_score_fault():
    faulty = inject_fault(build_clean_case("case-001"), FaultType.STALE_RISK_SCORE)
    assert faulty.execution.risk_score_state_version == 0
    assert faulty.execution.risk_score_state_version != faulty.state_version


def test_premature_commitment_fault():
    faulty = inject_fault(build_clean_case("case-001"), FaultType.PREMATURE_COMMITMENT)
    assert faulty.execution.commitment_attempted is True
