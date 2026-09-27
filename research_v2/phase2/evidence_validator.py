"""Deterministic validation and normalization for Phase 2 evidence."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, Optional

from .schemas import EvidenceField, EvidenceValidationResult, LoanEvidenceCase


NUMERIC_FIELDS = {
    "monthly_income",
    "average_salary_credit",
    "loan_amount",
    "existing_emi",
}


def normalize_numeric(value: object) -> Optional[float]:
    """Convert common currency/number strings to a numeric value."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    cleaned = value.strip().replace(",", "")
    cleaned = re.sub(r"(?:₹|INR|Rs\.?|\s)", "", cleaned, flags=re.IGNORECASE)
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_entity_name(value: object) -> Optional[str]:
    """Normalize organization names for deterministic comparison."""

    if not isinstance(value, str) or not value.strip():
        return None
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split()) or None


def _document_present(case: LoanEvidenceCase, document_type: str) -> bool:
    return any(
        document.document_type == document_type and document.present
        for document in case.documents
    )


def _income_values(case: LoanEvidenceCase) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, field in case.extracted_evidence.items():
        if key in {"monthly_income", "average_salary_credit"}:
            number = normalize_numeric(field.value)
            if number is not None:
                values[key] = number

    declared = normalize_numeric(case.applicant_data.get("monthly_income"))
    if declared is not None:
        values["declared_monthly_income"] = declared
    return values


def validate_evidence(case: LoanEvidenceCase) -> EvidenceValidationResult:
    """Validate evidence completeness, provenance, numeric fields, and income consistency."""

    errors = []
    warnings = []
    normalized: Dict[str, EvidenceField] = {}

    for key, field in case.extracted_evidence.items():
        if not field.source_document or not field.source_reference:
            errors.append(f"missing_provenance:{key}")

        value = field.value
        if key in NUMERIC_FIELDS:
            number = normalize_numeric(value)
            if number is None or number < 0:
                errors.append(f"invalid_numeric_value:{key}")
                continue
            field = field.model_copy(update={"value": number})
        normalized[key] = field

    if not _document_present(case, "salary_slip"):
        errors.append("missing_salary_slip")

    income_values = _income_values(case)
    if "monthly_income" in normalized:
        income_values["monthly_income"] = float(normalized["monthly_income"].value)
    if "average_salary_credit" in normalized:
        income_values["average_salary_credit"] = float(
            normalized["average_salary_credit"].value
        )

    if len(income_values) >= 2:
        reference = next(iter(income_values.values()))
        for name, value in income_values.items():
            if reference == 0 and value != 0:
                errors.append("income_contradiction")
                break
            if reference != 0 and abs(value - reference) / reference > 0.05:
                errors.append("income_contradiction")
                break

    declared_employer = normalize_entity_name(case.applicant_data.get("employer_name"))
    extracted_employer_field = normalized.get("employer_name")
    extracted_employer = normalize_entity_name(
        extracted_employer_field.value if extracted_employer_field else None
    )
    if (
        declared_employer is not None
        and extracted_employer is not None
        and declared_employer != extracted_employer
    ):
        errors.append("employer_contradiction")

    return EvidenceValidationResult(
        valid=not errors,
        errors=sorted(set(errors)),
        warnings=sorted(set(warnings)),
        normalized_evidence=normalized,
    )
