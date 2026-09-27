from research_v2.phase2.case_factory import build_clean_case
from research_v2.phase2.deterministic_critic import critique_case
from research_v2.phase2.fault_catalog import FaultType
from research_v2.phase2.fault_injector import inject_fault


def test_clean_case_is_allowed():
    result = critique_case(build_clean_case("case-001"))
    assert result.action == "allow"
    assert result.fault_detected is False
    assert result.findings == []


def test_missing_salary_slip_requests_reassessment():
    case = inject_fault(build_clean_case("case-001"), FaultType.MISSING_SALARY_SLIP)
    result = critique_case(case)
    assert result.action == "reassess"
    assert "missing_salary_slip" in result.detected_faults


def test_wrong_income_requests_reassessment():
    case = inject_fault(build_clean_case("case-001"), FaultType.WRONG_EXTRACTED_INCOME)
    result = critique_case(case)
    assert result.action == "reassess"
    assert "income_contradiction" in result.detected_faults


def test_contradiction_requests_reassessment():
    case = inject_fault(build_clean_case("case-001"), FaultType.CONTRADICTORY_INCOME)
    result = critique_case(case)
    assert result.action == "reassess"
    assert "income_contradiction" in result.detected_faults


def test_employer_contradiction_requests_reassessment():
    case = build_clean_case("case-001")
    case.extracted_evidence["employer_name"] = case.extracted_evidence[
        "monthly_income"
    ].model_copy(
        update={"field_name": "employer_name", "value": "Other Industries Ltd"}
    )
    result = critique_case(case)
    assert result.action == "reassess"
    assert "employer_contradiction" in result.detected_faults


def test_skipped_policy_node_blocks():
    case = inject_fault(build_clean_case("case-001"), FaultType.SKIPPED_POLICY_NODE)
    result = critique_case(case)
    assert result.action == "block"
    assert "skipped_policy_node" in result.detected_faults


def test_stale_risk_score_requests_reassessment():
    case = inject_fault(build_clean_case("case-001"), FaultType.STALE_RISK_SCORE)
    result = critique_case(case)
    assert result.action == "reassess"
    assert "stale_risk_score" in result.detected_faults


def test_premature_commitment_blocks():
    case = inject_fault(build_clean_case("case-001"), FaultType.PREMATURE_COMMITMENT)
    result = critique_case(case)
    assert result.action == "block"
    assert "premature_commitment" in result.detected_faults
