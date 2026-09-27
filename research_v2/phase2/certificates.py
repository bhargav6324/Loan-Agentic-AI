"""State-versioned evidence certificates for the Phase-2 governance layer."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, List, Literal, Set

from pydantic import BaseModel, ConfigDict, Field

from .schemas import LoanEvidenceCase


CertificateStatus = Literal["valid", "stale", "revoked"]


class EvidenceCertificate(BaseModel):
    """A signed-by-hash claim about one exact applicant-state version."""

    model_config = ConfigDict(extra="forbid")

    certificate_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    certificate_type: str = Field(min_length=1)
    state_version: int = Field(ge=1)
    state_hash: str = Field(min_length=64, max_length=64)
    input_fields: List[str] = Field(default_factory=list)
    parent_certificates: List[str] = Field(default_factory=list)
    status: CertificateStatus = "valid"


def canonical_case_payload(case: LoanEvidenceCase) -> Dict[str, object]:
    """Return the state covered by a hash, excluding mutable hash metadata."""

    return {
        "case_id": case.case_id,
        "applicant_data": case.applicant_data,
        "documents": [document.model_dump(mode="json") for document in case.documents],
        "extracted_evidence": {
            key: value.model_dump(mode="json")
            for key, value in sorted(case.extracted_evidence.items())
        },
        # Risk-score version is derived execution metadata, not applicant
        # evidence. It is checked independently by the commitment gate.
        "execution": case.execution.model_dump(
            mode="json", exclude={"risk_score_state_version"}
        ),
    }


def compute_state_hash(case: LoanEvidenceCase) -> str:
    """Compute a deterministic SHA-256 hash of the applicant evidence state."""

    encoded = json.dumps(
        canonical_case_payload(case), sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def next_state_version(
    previous_case: LoanEvidenceCase, current_case: LoanEvidenceCase
) -> int:
    """Increment the version only when the canonical state actually changes."""

    previous_hash = compute_state_hash(previous_case)
    current_hash = compute_state_hash(current_case)
    return previous_case.state_version + (1 if previous_hash != current_hash else 0)


def issue_certificate(
    case: LoanEvidenceCase,
    certificate_type: str,
    input_fields: Iterable[str],
    parent_certificates: Iterable[str] = (),
) -> EvidenceCertificate:
    """Issue a deterministic certificate bound to the current case state."""

    fields = sorted(set(input_fields))
    parents = sorted(set(parent_certificates))
    state_hash = compute_state_hash(case)
    material = {
        "case_id": case.case_id,
        "certificate_type": certificate_type,
        "state_version": case.state_version,
        "state_hash": state_hash,
        "input_fields": fields,
        "parent_certificates": parents,
    }
    certificate_id = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return EvidenceCertificate(
        certificate_id=certificate_id,
        case_id=case.case_id,
        certificate_type=certificate_type,
        state_version=case.state_version,
        state_hash=state_hash,
        input_fields=fields,
        parent_certificates=parents,
    )


def is_current(certificate: EvidenceCertificate, case: LoanEvidenceCase) -> bool:
    """Check state binding and status without making a loan decision."""

    return (
        certificate.status == "valid"
        and certificate.case_id == case.case_id
        and certificate.state_version == case.state_version
        and certificate.state_hash == compute_state_hash(case)
    )


def invalidate_dependents(
    certificates: Iterable[EvidenceCertificate], changed_fields: Iterable[str]
) -> List[EvidenceCertificate]:
    """Stale direct consumers and all transitive child certificates."""

    result = [certificate.model_copy(deep=True) for certificate in certificates]
    changed: Set[str] = set(changed_fields)
    invalidated: Set[str] = set()
    by_id = {certificate.certificate_id: certificate for certificate in result}

    progress = True
    while progress:
        progress = False
        for certificate in result:
            if certificate.certificate_id in invalidated:
                continue
            direct = bool(changed.intersection(certificate.input_fields))
            parent_stale = any(parent in invalidated for parent in certificate.parent_certificates)
            if direct or parent_stale:
                certificate.status = "stale"
                invalidated.add(certificate.certificate_id)
                progress = True

    return result
