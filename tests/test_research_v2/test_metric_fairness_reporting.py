import pandas as pd

from research_v2.run_agentic_ablation import _compute_metrics


def _rows():
    return pd.DataFrame(
        {
            "final_decision": ["approved", "escalated"],
            "ground_truth": [0, 1],
            "p_default": [0.2, 0.8],
            "gender_encoded": [0, 1],
            "posthoc_policy_passed": [True, True],
            "audit_triggered": [True, True],
            "audit_bias_flagged": [False, False],
            "fairness_bias_flagged": [False, True],
            "deterministic_reassessment_count": [0, 0],
            "commitment_block_events": [0, 0],
            "had_commitment_block": [False, False],
            "commit_status": ["committed_approve", "escalated"],
            "decision": ["approve", "escalate"],
            "critique_event": [None, None],
            "latency_ms": [1.0, 1.0],
        }
    )


def test_graph_metrics_use_fairness_bias_field():
    metrics = _compute_metrics(_rows(), seed=42, config="A2")
    assert metrics["audit_bias_flagged_count"] == 1


def test_static_metrics_fall_back_to_legacy_audit_field():
    rows = _rows()
    rows["fairness_bias_flagged"] = None
    rows["audit_bias_flagged"] = [True, False]
    metrics = _compute_metrics(rows, seed=42, config="A1R")
    assert metrics["audit_bias_flagged_count"] == 1
