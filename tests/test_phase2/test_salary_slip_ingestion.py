from research_v2.phase2.case_factory import build_clean_case
from research_v2.phase2.commitment_gate import evaluate_commitment
from research_v2.phase2.state_lifecycle import (
    extract_and_ingest_salary_slip,
    mark_risk_reassessed,
)
from research_v2.phase2.workflow_adapter import (
    REQUIRED_CERTIFICATE_TYPES,
    build_workflow_certificates,
)


TEXT = "Employee: Asha Rao | Employer: Example Pvt Ltd | Net Pay: ₹50,000 | Period: August 2026"
GRAPH_RESULT = {"policy_passed": True, "policy_violations": []}


def _valid_generator(_prompt):
    return {
        "monthly_income": 50000,
        "employer_name": "Example Pvt Ltd",
        "employee_name": "Asha Rao",
        "pay_period": "August 2026",
        "evidence": {
            "monthly_income": "₹50,000",
            "employer_name": "Employer: Example Pvt Ltd",
            "employee_name": "Employee: Asha Rao",
            "pay_period": "Period: August 2026",
        },
        "missing_fields": [],
        "confidence": 0.98,
    }


def test_valid_llm_evidence_versions_state_and_invalidates_old_chain():
    case = build_clean_case("case-1", {"credit_score": 700})
    old_certificates = build_workflow_certificates(case, GRAPH_RESULT)
    result = extract_and_ingest_salary_slip(
        case, "uploaded-salary-1", TEXT, old_certificates, _valid_generator
    )

    assert result.accepted is True
    assert result.transition.state_changed is True
    assert result.transition.case.state_version == 2
    assert result.transition.case.extracted_evidence["monthly_income"].value == 50000.0
    assert result.transition.case.extracted_evidence["monthly_income"].source_document == "uploaded-salary-1"
    assert all(certificate.status == "stale" for certificate in result.transition.certificates)


def test_commitment_recovers_only_after_reassessment_and_fresh_certificates():
    case = build_clean_case("case-1", {"credit_score": 700})
    old_certificates = build_workflow_certificates(case, GRAPH_RESULT)
    ingestion = extract_and_ingest_salary_slip(
        case, "uploaded-salary-1", TEXT, old_certificates, _valid_generator
    )
    updated = ingestion.transition.case

    blocked = evaluate_commitment(
        updated, ingestion.transition.certificates,
        REQUIRED_CERTIFICATE_TYPES, policy_passed=True,
    )
    assert blocked.allowed is False
    assert "execution:stale_risk_score" in blocked.reasons

    reassessed = mark_risk_reassessed(updated)
    fresh_certificates = build_workflow_certificates(reassessed, GRAPH_RESULT)
    recovered = evaluate_commitment(
        reassessed, fresh_certificates,
        REQUIRED_CERTIFICATE_TYPES, policy_passed=True,
    )
    assert recovered.allowed is True
    assert len(fresh_certificates) == 3


def test_invalid_llm_output_does_not_change_state():
    case = build_clean_case("case-1")

    def invalid_generator(_prompt):
        return {
            "monthly_income": 90000,
            "employer_name": None,
            "employee_name": None,
            "pay_period": None,
            "evidence": {"monthly_income": "₹50,000"},
            "missing_fields": [],
            "confidence": 0.5,
        }

    result = extract_and_ingest_salary_slip(
        case, "uploaded-salary-1", TEXT, generator=invalid_generator
    )
    assert result.accepted is False
    assert result.transition.state_changed is False
    assert result.transition.case.state_version == 1


def test_conflicting_employer_remains_blocked_after_risk_reassessment():
    case = build_clean_case("case-1", {"credit_score": 700})

    def conflicting_generator(_prompt):
        response = _valid_generator(_prompt)
        response["employer_name"] = "Other Industries Ltd"
        response["evidence"]["employer_name"] = "Employer: Other Industries Ltd"
        return response

    text = TEXT.replace("Example Pvt Ltd", "Other Industries Ltd")
    ingestion = extract_and_ingest_salary_slip(
        case, "conflicting-salary", text, generator=conflicting_generator
    )
    assert ingestion.accepted is True

    reassessed = mark_risk_reassessed(ingestion.transition.case)
    certificates = build_workflow_certificates(reassessed, GRAPH_RESULT)
    gate = evaluate_commitment(
        reassessed, certificates, REQUIRED_CERTIFICATE_TYPES, policy_passed=True
    )
    assert certificates == []
    assert gate.allowed is False
    assert "evidence:employer_contradiction" in gate.reasons
