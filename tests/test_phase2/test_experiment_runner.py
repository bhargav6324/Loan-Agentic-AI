from research_v2.phase2.case_factory import build_clean_case
from research_v2.phase2.experiment_runner import (
    evaluate_case,
    run_paired_experiment,
)
from research_v2.phase2.fault_catalog import FaultType
from research_v2.phase2.fault_injector import inject_fault


def _cases():
    clean = build_clean_case("clean-001")
    return [
        clean,
        inject_fault(clean, FaultType.MISSING_SALARY_SLIP),
        inject_fault(clean, FaultType.WRONG_EXTRACTED_INCOME),
        inject_fault(clean, FaultType.CONTRADICTORY_INCOME),
        inject_fault(clean, FaultType.SKIPPED_POLICY_NODE),
        inject_fault(clean, FaultType.PREMATURE_COMMITMENT),
    ]


def test_a2_and_a3_receive_the_same_cases():
    result = run_paired_experiment(_cases())
    rows = result["evaluations"]
    a2_ids = [row.case_id for row in rows if row.configuration == "A2"]
    a3_ids = [row.case_id for row in rows if row.configuration == "A3"]
    assert a2_ids == a3_ids


def test_a2_has_unsafe_commitments_without_phase2_critic():
    result = run_paired_experiment(_cases())
    metrics = result["metrics"]["A2"]
    assert metrics.fault_detection_rate == 0.0
    assert metrics.unsafe_commitment_rate == 1.0


def test_a3_detects_faults_and_prevents_unsafe_commitments():
    result = run_paired_experiment(_cases())
    metrics = result["metrics"]["A3"]
    assert metrics.fault_detection_rate == 1.0
    assert metrics.unsafe_commitment_rate == 0.0
    assert metrics.false_alarm_rate == 0.0
    assert metrics.clean_case_agreement == 1.0


def test_clean_case_is_committed_by_both_paths():
    clean = build_clean_case("clean-002")
    assert evaluate_case(clean, "A2").commitment_status == "committed"
    assert evaluate_case(clean, "A3").commitment_status == "committed"

