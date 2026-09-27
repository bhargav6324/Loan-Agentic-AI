"""Factories for deterministic clean synthetic loan-evidence cases."""

from __future__ import annotations

from typing import Dict, Optional

from .schemas import DocumentEvidence, EvidenceField, LoanEvidenceCase


def _field(
    name: str,
    value,
    document_id: str,
    reference: str,
    confidence: float = 1.0,
) -> EvidenceField:
    return EvidenceField(
        field_name=name,
        value=value,
        source_document=document_id,
        source_reference=reference,
        confidence=confidence,
    )


def build_clean_case(
    case_id: str,
    applicant_data: Optional[Dict[str, object]] = None,
) -> LoanEvidenceCase:
    """Build a internally consistent case suitable for paired experiments."""

    data = {
        "monthly_income": 50000,
        "loan_amount": 800000,
        "employer_name": "Example Pvt Ltd",
        "applicant_name": "Synthetic Applicant",
    }
    if applicant_data:
        data.update(applicant_data)

    income = data["monthly_income"]
    loan_amount = data["loan_amount"]

    salary_id = f"{case_id}-salary-slip"
    bank_id = f"{case_id}-bank-statement"
    application_id = f"{case_id}-application"

    salary_income = _field("monthly_income", income, salary_id, "page-1:net-pay")
    bank_income = _field(
        "average_salary_credit", income, bank_id, "page-2:salary-credits"
    )
    application_income = _field(
        "monthly_income", income, application_id, "page-1:declared-income"
    )

    return LoanEvidenceCase(
        case_id=case_id,
        applicant_data=data,
        documents=[
            DocumentEvidence(
                document_id=application_id,
                document_type="loan_application",
                fields={
                    "monthly_income": application_income,
                    "loan_amount": _field(
                        "loan_amount", loan_amount, application_id, "page-1:loan-amount"
                    ),
                },
            ),
            DocumentEvidence(
                document_id=salary_id,
                document_type="salary_slip",
                fields={"monthly_income": salary_income},
            ),
            DocumentEvidence(
                document_id=bank_id,
                document_type="bank_statement",
                fields={"average_salary_credit": bank_income},
            ),
        ],
        extracted_evidence={
            "monthly_income": salary_income,
            "average_salary_credit": bank_income,
            "loan_amount": _field(
                "loan_amount", loan_amount, application_id, "page-1:loan-amount"
            ),
        },
        execution={
            "policy_node_executed": True,
            "risk_node_executed": True,
            "risk_score_state_version": 1,
            "critic_node_executed": False,
            "commitment_attempted": False,
        },
    )
