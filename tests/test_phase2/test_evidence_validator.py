import pytest

from research_v2.phase2.evidence_validator import (
    normalize_entity_name,
    normalize_numeric,
    validate_evidence,
)
from research_v2.phase2.schemas import DocumentEvidence, EvidenceField, LoanEvidenceCase


def _field(name, value, source="salary-slip-1"):
    return EvidenceField(
        field_name=name,
        value=value,
        source_document=source,
        source_reference="page-1:net-pay",
        confidence=0.95,
    )


def _clean_case(**overrides):
    values = {
        "case_id": "case-001",
        "applicant_data": {"monthly_income": 50000},
        "documents": [
            DocumentEvidence(
                document_id="salary-slip-1",
                document_type="salary_slip",
                fields={"monthly_income": _field("monthly_income", 50000)},
            )
        ],
        "extracted_evidence": {
            "monthly_income": _field("monthly_income", 50000),
        },
    }
    values.update(overrides)
    return LoanEvidenceCase(**values)


@pytest.mark.parametrize(
    "raw,expected",
    [("₹50,000", 50000.0), ("INR 42,500", 42500.0), (50000, 50000.0)],
)
def test_normalize_numeric(raw, expected):
    assert normalize_numeric(raw) == expected


def test_clean_case_is_valid():
    result = validate_evidence(_clean_case())
    assert result.valid is True
    assert result.normalized_evidence["monthly_income"].value == 50000.0


def test_missing_salary_slip_is_invalid():
    case = _clean_case(documents=[])
    result = validate_evidence(case)
    assert result.valid is False
    assert "missing_salary_slip" in result.errors


def test_wrong_numeric_extraction_is_invalid():
    case = _clean_case(
        extracted_evidence={"monthly_income": _field("monthly_income", "not-a-number")}
    )
    result = validate_evidence(case)
    assert result.valid is False
    assert "invalid_numeric_value:monthly_income" in result.errors


def test_contradictory_income_is_invalid():
    case = _clean_case(
        applicant_data={"monthly_income": 50000},
        extracted_evidence={
            "monthly_income": _field("monthly_income", 50000),
            "average_salary_credit": _field(
                "average_salary_credit", 35000, source="bank-statement-1"
            ),
        },
    )
    result = validate_evidence(case)
    assert result.valid is False
    assert "income_contradiction" in result.errors


def test_contradictory_employer_is_invalid():
    case = _clean_case(
        applicant_data={"monthly_income": 50000, "employer_name": "Example Pvt Ltd"},
        extracted_evidence={
            "monthly_income": _field("monthly_income", 50000),
            "employer_name": _field("employer_name", "Other Industries Ltd"),
        },
    )
    result = validate_evidence(case)
    assert result.valid is False
    assert "employer_contradiction" in result.errors


def test_employer_punctuation_does_not_create_false_contradiction():
    case = _clean_case(
        applicant_data={"monthly_income": 50000, "employer_name": "Example Pvt Ltd"},
        extracted_evidence={
            "monthly_income": _field("monthly_income", 50000),
            "employer_name": _field("employer_name", "Example Pvt. Ltd."),
        },
    )
    assert normalize_entity_name("Example Pvt. Ltd.") == "example pvt ltd"
    assert validate_evidence(case).valid is True
