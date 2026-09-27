"""Deterministic fault injection for paired A2/A3 experiments."""

from __future__ import annotations

from copy import deepcopy

from .fault_catalog import FaultType
from .schemas import LoanEvidenceCase


def inject_fault(case: LoanEvidenceCase, fault: FaultType) -> LoanEvidenceCase:
    """Return a deep-copied faulty case without mutating the clean case."""

    faulty = deepcopy(case)
    faulty.fault_labels = sorted(set(faulty.fault_labels + [fault.value]))

    if fault == FaultType.MISSING_SALARY_SLIP:
        for document in faulty.documents:
            if document.document_type == "salary_slip":
                document.present = False

    elif fault == FaultType.WRONG_EXTRACTED_INCOME:
        field = faulty.extracted_evidence["monthly_income"]
        faulty.extracted_evidence["monthly_income"] = field.model_copy(
            update={"value": 35000}
        )

    elif fault == FaultType.CONTRADICTORY_INCOME:
        field = faulty.extracted_evidence["average_salary_credit"]
        faulty.extracted_evidence["average_salary_credit"] = field.model_copy(
            update={"value": 35000}
        )
        for document in faulty.documents:
            if document.document_type == "bank_statement":
                bank_field = document.fields["average_salary_credit"]
                document.fields["average_salary_credit"] = bank_field.model_copy(
                    update={"value": 35000}
                )

    elif fault == FaultType.SKIPPED_POLICY_NODE:
        faulty.execution.policy_node_executed = False

    elif fault == FaultType.STALE_RISK_SCORE:
        faulty.execution.risk_score_state_version = max(0, faulty.state_version - 1)

    elif fault == FaultType.PREMATURE_COMMITMENT:
        faulty.execution.commitment_attempted = True

    else:
        raise ValueError(f"Unsupported fault type: {fault}")

    return faulty
