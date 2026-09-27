"""Deterministic pre-commitment gate for certificate-closed loan evidence."""

from __future__ import annotations

from typing import Iterable, List

from pydantic import BaseModel, ConfigDict, Field

from .certificates import EvidenceCertificate, is_current
from .evidence_validator import validate_evidence
from .schemas import LoanEvidenceCase


class CommitmentGateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed: bool
    status: str
    reasons: List[str] = Field(default_factory=list)


def evaluate_commitment(
    case: LoanEvidenceCase,
    certificates: Iterable[EvidenceCertificate],
    required_certificate_types: Iterable[str],
    policy_passed: bool,
) -> CommitmentGateResult:
    """Allow commitment only when evidence and required certificates are closed."""

    reasons: List[str] = []
    validation = validate_evidence(case)
    if not validation.valid:
        reasons.extend("evidence:" + error for error in validation.errors)

    if not case.execution.risk_node_executed:
        reasons.append("execution:risk_node_not_executed")
    elif case.execution.risk_score_state_version != case.state_version:
        reasons.append("execution:stale_risk_score")
    if not case.execution.policy_node_executed:
        reasons.append("execution:policy_node_not_executed")
    if case.execution.commitment_attempted:
        reasons.append("execution:premature_commitment")
    if not policy_passed:
        reasons.append("policy:failed")

    by_type = {certificate.certificate_type: certificate for certificate in certificates}
    for certificate_type in sorted(set(required_certificate_types)):
        certificate = by_type.get(certificate_type)
        if certificate is None:
            reasons.append("certificate:missing:" + certificate_type)
        elif not is_current(certificate, case):
            reasons.append("certificate:stale:" + certificate_type)

    reasons = sorted(set(reasons))
    return CommitmentGateResult(
        allowed=not reasons,
        status="commit_allowed" if not reasons else "commit_blocked",
        reasons=reasons,
    )
