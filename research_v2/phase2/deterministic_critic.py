"""Deterministic pre-commitment critic for Phase 2 experiments."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .evidence_validator import validate_evidence
from .schemas import LoanEvidenceCase


CriticAction = Literal["allow", "block", "reassess", "escalate"]


class CriticFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    severity: Literal["low", "medium", "high"]
    message: str = Field(min_length=1)
    evidence_refs: List[str] = Field(default_factory=list)


class CriticResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: CriticAction
    fault_detected: bool
    detected_faults: List[str] = Field(default_factory=list)
    findings: List[CriticFinding] = Field(default_factory=list)


def _finding_for_error(error: str) -> CriticFinding:
    if error == "missing_salary_slip":
        return CriticFinding(
            code=error,
            severity="high",
            message="Mandatory salary-slip evidence is missing.",
            evidence_refs=["salary_slip"],
        )
    if error == "income_contradiction":
        return CriticFinding(
            code=error,
            severity="high",
            message="Income values disagree across the available evidence.",
            evidence_refs=["monthly_income", "average_salary_credit"],
        )
    if error == "employer_contradiction":
        return CriticFinding(
            code=error,
            severity="high",
            message="The salary-slip employer conflicts with the declared employer.",
            evidence_refs=["applicant_data.employer_name", "employer_name"],
        )
    return CriticFinding(
        code=error,
        severity="high",
        message="Evidence failed deterministic validation.",
    )


def critique_case(case: LoanEvidenceCase) -> CriticResult:
    """Return a safe workflow action without making a loan decision."""

    findings: List[CriticFinding] = []
    detected_faults = list(case.fault_labels)

    validation = validate_evidence(case)
    for error in validation.errors:
        findings.append(_finding_for_error(error))
        if error not in detected_faults:
            detected_faults.append(error)

    if not case.execution.policy_node_executed:
        findings.append(
            CriticFinding(
                code="skipped_policy_node",
                severity="high",
                message="The mandatory deterministic policy node did not execute.",
                evidence_refs=["execution.policy_node_executed"],
            )
        )
        if "skipped_policy_node" not in detected_faults:
            detected_faults.append("skipped_policy_node")

    if (
        case.execution.risk_node_executed
        and case.execution.risk_score_state_version != case.state_version
    ):
        findings.append(
            CriticFinding(
                code="stale_risk_score",
                severity="high",
                message="The risk score was computed from an older applicant-state version.",
                evidence_refs=["execution.risk_score_state_version", "state_version"],
            )
        )
        if "stale_risk_score" not in detected_faults:
            detected_faults.append("stale_risk_score")

    if case.execution.commitment_attempted:
        findings.append(
            CriticFinding(
                code="premature_commitment",
                severity="high",
                message="Commitment was attempted before critic clearance.",
                evidence_refs=["execution.commitment_attempted"],
            )
        )
        if "premature_commitment" not in detected_faults:
            detected_faults.append("premature_commitment")

    finding_codes = {finding.code for finding in findings}
    if "premature_commitment" in finding_codes or "skipped_policy_node" in finding_codes:
        action: CriticAction = "block"
    elif "invalid_numeric_value:monthly_income" in finding_codes:
        action = "escalate"
    elif findings:
        action = "reassess"
    else:
        action = "allow"

    return CriticResult(
        action=action,
        fault_detected=bool(findings),
        detected_faults=sorted(set(detected_faults)),
        findings=findings,
    )
