"""Build the formal 5K-versus-45K comparison from preserved result files."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROLES_5K = {
    "C0a": "approve-all control", "A0": "Random Forest",
    "A1": "Random Forest + policy", "A1R": "policy + reassessment",
    "A2": "full graph", "A3": "full graph + critic",
}
ROLES_45K = {
    "F0": "approve-all control", "F1": "Random Forest",
    "F1P": "policy + reassessment", "F2": "full adapted graph",
    "F3": "full adapted graph + critic",
}


def summarize(path_5k: Path, path_45k: Path) -> pd.DataFrame:
    a = pd.read_csv(path_5k)
    f = pd.read_csv(path_45k)
    rows = []
    for config, group in a.groupby("configuration", sort=False):
        rows.append({
            "dataset": "Original 5K synthetic", "configuration": config,
            "role": ROLES_5K[config], "n_seeds": group.seed.nunique(),
            "test_cases_per_seed": int(group.n_test.iloc[0]),
            "accuracy_mean": group.accuracy.mean(), "accuracy_std": group.accuracy.std(),
            "f1_mean": group.f1_approve.mean(), "f1_std": group.f1_approve.std(),
            "mcc_mean": group.mcc.mean(), "mcc_std": group.mcc.std(),
            "policy_violations_mean": group.policy_violation_count.mean(),
            "reassessments_mean": group.deterministic_reassessment_count.mean(),
            "audit_or_fairness_flags_mean": group.audit_bias_flagged_count.mean(),
            "automation_coverage_mean": None, "selective_accuracy_mean": None,
            "runtime_note": f"{group.mean_latency_ms.mean():.3f} mean ms/applicant",
        })
    for config, group in f.groupby("configuration", sort=False):
        rows.append({
            "dataset": "Published 45K synthetic-enhanced", "configuration": config,
            "role": ROLES_45K[config], "n_seeds": group.seed.nunique(),
            "test_cases_per_seed": int(group.n_test.iloc[0]),
            "accuracy_mean": group.accuracy.mean(), "accuracy_std": group.accuracy.std(),
            "f1_mean": group.f1_approve.mean(), "f1_std": group.f1_approve.std(),
            "mcc_mean": group.mcc.mean(), "mcc_std": group.mcc.std(),
            "policy_violations_mean": group.policy_violating_approvals.mean(),
            "reassessments_mean": group.reassessments.mean(),
            "audit_or_fairness_flags_mean": group.fairness_flags.mean(),
            "automation_coverage_mean": group.automation_coverage.mean(),
            "selective_accuracy_mean": group.selective_accuracy.mean(),
            "runtime_note": f"{group.execution_seconds.mean():.3f} seconds/configuration/seed",
        })
    return pd.DataFrame(rows)


def markdown_table(summary: pd.DataFrame) -> str:
    display = summary[["dataset", "configuration", "role", "accuracy_mean",
                       "f1_mean", "mcc_mean", "policy_violations_mean",
                       "automation_coverage_mean", "selective_accuracy_mean"]].copy()
    for col in ["accuracy_mean", "f1_mean", "mcc_mean",
                "automation_coverage_mean", "selective_accuracy_mean"]:
        display[col] = display[col].map(lambda x: "—" if pd.isna(x) else f"{x:.4f}")
    display["policy_violations_mean"] = display["policy_violations_mean"].map(lambda x: f"{x:.2f}")
    return display.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--five-k", required=True, type=Path)
    parser.add_argument("--forty-five-k", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(args.five_k, args.forty_five_k)
    summary.to_csv(args.output_dir / "cross_dataset_summary.csv", index=False)
    report = f"""# Formal cross-dataset comparison

{markdown_table(summary)}

## Interpretation constraints

- Compare governance trends, not raw accuracy: the datasets use different
  populations, labels, feature contracts, and class balances.
- The 45K source is a published, synthetically enhanced benchmark, not
  independent real-bank validation.
- F2/F3 escalations are unresolved outcomes. Their selective accuracy and
  automation coverage must accompany binary metrics.
- F3 and A3 do not improve predictive metrics over their graph counterparts;
  their purpose is structural critique.
- Runtime fields use different measurement units and are not directly compared.
"""
    (args.output_dir / "CROSS_DATASET_COMPARISON.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
