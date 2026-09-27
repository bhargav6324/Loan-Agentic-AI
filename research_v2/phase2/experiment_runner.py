"""Paired A2/A3 Phase 2 fault-injection evaluation.

This first harness isolates the contribution of the Phase 2 critic. It does
not rerun the existing Phase 1 model experiment and does not write to its
result directories.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Literal

from pydantic import BaseModel, ConfigDict, Field

from .deterministic_critic import CriticAction, critique_case
from .schemas import LoanEvidenceCase


Configuration = Literal["A2", "A3"]


class CaseEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    configuration: Configuration
    faulty: bool
    fault_labels: List[str] = Field(default_factory=list)
    fault_detected: bool
    unsafe_commitment: bool
    commitment_status: Literal["committed", "blocked_or_escalated"]
    critic_action: CriticAction
    detected_faults: List[str] = Field(default_factory=list)


class PairedMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    configuration: Configuration
    fault_detection_rate: float
    unsafe_commitment_rate: float
    false_alarm_rate: float
    clean_case_agreement: float
    total_cases: int
    total_faulty_cases: int
    total_clean_cases: int


def evaluate_case(case: LoanEvidenceCase, configuration: Configuration) -> CaseEvaluation:
    """Evaluate one case under the unprotected A2 or critic-protected A3 path."""

    faulty = bool(case.fault_labels)

    if configuration == "A2":
        # A2 is intentionally the unprotected Phase 2 comparison path. It
        # commits without consulting the new Phase 2 critic.
        return CaseEvaluation(
            case_id=case.case_id,
            configuration=configuration,
            faulty=faulty,
            fault_labels=list(case.fault_labels),
            fault_detected=False,
            unsafe_commitment=faulty,
            commitment_status="committed",
            critic_action="allow",
            detected_faults=[],
        )

    critic = critique_case(case)
    safe_to_commit = critic.action == "allow"
    return CaseEvaluation(
        case_id=case.case_id,
        configuration=configuration,
        faulty=faulty,
        fault_labels=list(case.fault_labels),
        fault_detected=critic.fault_detected,
        unsafe_commitment=faulty and safe_to_commit,
        commitment_status=("committed" if safe_to_commit else "blocked_or_escalated"),
        critic_action=critic.action,
        detected_faults=critic.detected_faults,
    )


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def summarize(
    evaluations: Iterable[CaseEvaluation], configuration: Configuration
) -> PairedMetrics:
    """Calculate governance metrics for one configuration."""

    rows = [row for row in evaluations if row.configuration == configuration]
    faulty = [row for row in rows if row.faulty]
    clean = [row for row in rows if not row.faulty]
    return PairedMetrics(
        configuration=configuration,
        fault_detection_rate=_rate(
            sum(row.fault_detected for row in faulty), len(faulty)
        ),
        unsafe_commitment_rate=_rate(
            sum(row.unsafe_commitment for row in faulty), len(faulty)
        ),
        false_alarm_rate=_rate(
            sum(row.commitment_status != "committed" for row in clean), len(clean)
        ),
        # Clean-case agreement is measured against the unprotected A2 path;
        # A3 should preserve its clean commitment outcome.
        clean_case_agreement=_rate(
            sum(row.commitment_status == "committed" for row in clean), len(clean)
        ),
        total_cases=len(rows),
        total_faulty_cases=len(faulty),
        total_clean_cases=len(clean),
    )


def run_paired_experiment(
    cases: Iterable[LoanEvidenceCase],
) -> Dict[str, object]:
    """Run A2 and A3 on the exact same ordered cases."""

    case_list = list(cases)
    evaluations: List[CaseEvaluation] = []
    for case in case_list:
        evaluations.append(evaluate_case(case, "A2"))
        evaluations.append(evaluate_case(case, "A3"))

    return {
        "evaluations": evaluations,
        "metrics": {
            "A2": summarize(evaluations, "A2"),
            "A3": summarize(evaluations, "A3"),
        },
    }

