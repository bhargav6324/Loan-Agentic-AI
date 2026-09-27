"""Controlled applicant-state transitions and certificate invalidation."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, Iterable, List

from pydantic import BaseModel, ConfigDict, Field

from .certificates import EvidenceCertificate, compute_state_hash, invalidate_dependents
from .llm_document_extractor import (
    SalarySlipExtractionResult,
    extract_salary_slip_text,
)
from .schemas import EvidenceScalar, LoanEvidenceCase


class StateTransitionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case: LoanEvidenceCase
    certificates: List[EvidenceCertificate] = Field(default_factory=list)
    changed_fields: List[str] = Field(default_factory=list)
    state_changed: bool


class SalarySlipIngestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    extraction: SalarySlipExtractionResult
    transition: StateTransitionResult


def update_applicant_state(
    case: LoanEvidenceCase,
    updates: Dict[str, EvidenceScalar],
    certificates: Iterable[EvidenceCertificate] = (),
) -> StateTransitionResult:
    """Apply applicant updates and invalidate certificates that depend on them."""

    updated = deepcopy(case)
    changed_fields = sorted(
        key for key, value in updates.items()
        if updated.applicant_data.get(key) != value
    )
    if not changed_fields:
        return StateTransitionResult(
            case=updated,
            certificates=[certificate.model_copy(deep=True) for certificate in certificates],
            changed_fields=[],
            state_changed=False,
        )

    for key in changed_fields:
        updated.applicant_data[key] = updates[key]
    updated.state_version = case.state_version + 1

    return StateTransitionResult(
        case=updated,
        certificates=invalidate_dependents(certificates, changed_fields),
        changed_fields=changed_fields,
        state_changed=True,
    )


def mark_risk_reassessed(case: LoanEvidenceCase) -> LoanEvidenceCase:
    """Bind a newly computed risk score to the current applicant-state version."""

    updated = deepcopy(case)
    updated.execution.risk_node_executed = True
    updated.execution.risk_score_state_version = updated.state_version
    return updated


def ingest_salary_slip_extraction(
    case: LoanEvidenceCase,
    extraction: SalarySlipExtractionResult,
    certificates: Iterable[EvidenceCertificate] = (),
) -> SalarySlipIngestionResult:
    """Insert validated salary evidence and invalidate dependent certificates."""

    current_certificates = [
        certificate.model_copy(deep=True) for certificate in certificates
    ]
    if not extraction.valid or extraction.document is None:
        return SalarySlipIngestionResult(
            accepted=False,
            extraction=extraction,
            transition=StateTransitionResult(
                case=case.model_copy(deep=True),
                certificates=current_certificates,
                changed_fields=[],
                state_changed=False,
            ),
        )

    updated = case.model_copy(deep=True)
    removed_salary_ids = {
        document.document_id
        for document in updated.documents
        if document.document_type == "salary_slip"
    }
    old_salary_fields = {
        key
        for key, field in updated.extracted_evidence.items()
        if field.source_document in removed_salary_ids
    }
    updated.documents = [
        document for document in updated.documents
        if document.document_type != "salary_slip"
    ]
    updated.documents.append(extraction.document.model_copy(deep=True))

    for key in old_salary_fields:
        updated.extracted_evidence.pop(key, None)
    for key, field in extraction.document.fields.items():
        updated.extracted_evidence[key] = field.model_copy(deep=True)

    changed_fields = sorted(
        old_salary_fields.union(extraction.document.fields.keys()).union({"salary_slip"})
    )
    if compute_state_hash(updated) == compute_state_hash(case):
        transition = StateTransitionResult(
            case=updated,
            certificates=current_certificates,
            changed_fields=[],
            state_changed=False,
        )
    else:
        updated.state_version = case.state_version + 1
        transition = StateTransitionResult(
            case=updated,
            certificates=invalidate_dependents(current_certificates, changed_fields),
            changed_fields=changed_fields,
            state_changed=True,
        )
    return SalarySlipIngestionResult(
        accepted=True,
        extraction=extraction,
        transition=transition,
    )


def extract_and_ingest_salary_slip(
    case: LoanEvidenceCase,
    document_id: str,
    text: str,
    certificates: Iterable[EvidenceCertificate] = (),
    generator: Callable[[str], dict] = None,
) -> SalarySlipIngestionResult:
    """Run bounded LLM extraction and apply it through the versioned state gate."""

    if generator is None:
        extraction = extract_salary_slip_text(document_id, text)
    else:
        extraction = extract_salary_slip_text(document_id, text, generator)
    return ingest_salary_slip_extraction(case, extraction, certificates)
