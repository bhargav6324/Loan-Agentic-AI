"""Adapter between Phase-2 evidence cases and the frozen A2/A3 runner.

The Phase-1 LangGraph consumes structured applicant rows, while Phase-2 cases
also contain document evidence and injected faults.  This module keeps those
concerns separate: it runs the original graph unchanged, then applies the
Phase-2 evidence/commitment gate at the boundary.  This makes the provenance
of every result explicit and avoids claiming that a document fault magically
changed the Random Forest features.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .certificates import EvidenceCertificate, compute_state_hash, issue_certificate
from .commitment_gate import evaluate_commitment
from .deterministic_critic import critique_case
from .evidence_validator import validate_evidence
from .schemas import LoanEvidenceCase


REQUIRED_CERTIFICATE_TYPES = (
    "income_consistency",
    "risk_score",
    "policy_compliance",
)


def build_workflow_certificates(
    case: LoanEvidenceCase, graph_result: Dict[str, Any]
) -> List[EvidenceCertificate]:
    """Issue certificates only after each deterministic prerequisite passes."""

    certificates: List[EvidenceCertificate] = []
    validation = validate_evidence(case)
    if not validation.valid:
        return certificates

    income = issue_certificate(
        case,
        "income_consistency",
        ["monthly_income", "average_salary_credit", "loan_amount"],
    )
    certificates.append(income)

    if (
        not case.execution.risk_node_executed
        or case.execution.risk_score_state_version != case.state_version
        or not graph_result
    ):
        return certificates
    risk = issue_certificate(
        case,
        "risk_score",
        ["credit_score", "dti_ratio", "loan_to_income", "monthly_income"],
        [income.certificate_id],
    )
    certificates.append(risk)

    if (
        not case.execution.policy_node_executed
        or graph_result.get("policy_passed") is not True
    ):
        return certificates
    certificates.append(
        issue_certificate(
            case,
            "policy_compliance",
            ["policy_passed", "policy_violations"],
            [risk.certificate_id],
        )
    )
    return certificates


def case_to_row(case: LoanEvidenceCase) -> pd.Series:
    """Convert the structured applicant portion to the existing runner row."""

    required = [
        "applicant_age", "applicant_gender", "monthly_income", "loan_amount",
        "employment_score", "employment_years", "loan_tenure_years",
        "loan_purpose", "credit_score", "current_outstanding", "dti_ratio",
        "loan_to_income", "defaults", "credit_utilisation",
    ]
    missing = [key for key in required if key not in case.applicant_data]
    if missing:
        raise ValueError("Case is missing runner fields: " + ", ".join(missing))

    values = dict(case.applicant_data)
    values["id"] = case.case_id
    return pd.Series(values)


def run_actual_workflow(
    case: LoanEvidenceCase,
    configuration: str,
    model: Any,
    model_hash: str = "phase2",
    epsilon: float = 6.0,
) -> Dict[str, Any]:
    """Run the real A2/A3 graph and record the Phase-2 gate outcome.

    A2 is the historical comparison path and does not apply the new gate.
    A3 runs the existing structural critic and additionally applies the
    deterministic evidence critic before commitment.  The original graph
    output is retained under ``graph_*`` fields for auditability.
    """

    if configuration not in {"A2", "A3"}:
        raise ValueError("configuration must be A2 or A3")

    from research_v2.run_agentic_ablation import _run_a2, _run_a3

    row = case_to_row(case)
    runner = _run_a3 if configuration == "A3" else _run_a2
    graph_result, graph_error = runner(row, model, model_hash, epsilon)

    output: Dict[str, Any] = {
        "case_id": case.case_id,
        "configuration": configuration,
        "fault_labels": list(case.fault_labels),
        "graph_error": graph_error,
        "graph_result": graph_result,
        "graph_final_decision": graph_result.get("final_decision") if graph_result else None,
        "graph_commit_status": graph_result.get("commit_status") if graph_result else None,
        "phase2_gate": "not_applied",
        "phase2_action": "not_applied",
        "effective_final_decision": graph_result.get("final_decision") if graph_result else None,
        "effective_commit_status": graph_result.get("commit_status") if graph_result else None,
    }

    if configuration == "A3":
        critic = critique_case(case)
        certificates = build_workflow_certificates(case, graph_result or {})
        gate = evaluate_commitment(
            case=case,
            certificates=certificates,
            required_certificate_types=REQUIRED_CERTIFICATE_TYPES,
            policy_passed=bool(graph_result and graph_result.get("policy_passed") is True),
        )
        output["state_version"] = case.state_version
        output["state_hash"] = compute_state_hash(case)
        output["certificate_ids"] = [c.certificate_id for c in certificates]
        output["certificate_types"] = [c.certificate_type for c in certificates]
        output["commitment_gate_reasons"] = list(gate.reasons)
        output["phase2_gate"] = "passed" if gate.allowed else "blocked"
        output["phase2_action"] = (
            critic.action if critic.action != "allow"
            else ("allow" if gate.allowed else "block")
        )
        output["detected_faults"] = list(critic.detected_faults)
        output["critic_findings"] = [finding.model_dump() for finding in critic.findings]
        if not gate.allowed:
            output["effective_final_decision"] = "human_review"
            output["effective_commit_status"] = "blocked_evidence_closure"
    else:
        output["detected_faults"] = []
        output["critic_findings"] = []
        output["state_version"] = case.state_version
        output["state_hash"] = compute_state_hash(case)
        output["certificate_ids"] = []
        output["certificate_types"] = []
        output["commitment_gate_reasons"] = []

    output["effective_unsafe_commitment"] = bool(
        case.fault_labels
        and output["effective_commit_status"]
        and str(output["effective_commit_status"]).startswith("committed_approve")
    )
    return output
