"""Run a small Phase-2 sample through the real A2/A3 LangGraph workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from research_v2.phase2.batch_factory import build_cases_from_dataframe, build_paired_cases
from research_v2.phase2.workflow_adapter import run_actual_workflow


FEATURES = [
    "credit_score", "dti_ratio", "loan_to_income", "employment_score",
    "applicant_age", "defaults", "credit_utilisation", "applicant_gender",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="research_v2/phase2_results/actual_workflow")
    args = parser.parse_args()

    dataset_path = os.path.join("research", "data", "credit_risk_dataset.csv")
    dataframe = pd.read_csv(dataset_path)
    X_train, _, y_train, _ = train_test_split(
        dataframe[FEATURES], dataframe["target_default"],
        test_size=0.30, random_state=args.seed,
    )
    model = RandomForestClassifier(n_estimators=100, random_state=args.seed)
    model.fit(X_train, y_train)
    model_hash = hashlib.sha256(pickle.dumps(model, protocol=5)).hexdigest()

    clean = build_cases_from_dataframe(dataframe, limit=args.limit, seed=args.seed)
    cases = build_paired_cases(clean)
    rows = []
    for case in cases:
        for configuration in ("A2", "A3"):
            result = run_actual_workflow(case, configuration, model, model_hash)
            rows.append({
                key: value for key, value in result.items()
                if key not in {"graph_result", "critic_findings"}
            })

    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "workflow_results.csv"), index=False)
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "seed": args.seed,
            "n_clean_cases": len(clean),
            "n_cases": len(cases),
            "configurations": ["A2", "A3"],
            "interpretation": "A3 applies the Phase-2 evidence gate after the unchanged graph; A2 is the historical graph path.",
        }, handle, indent=2)
    print("Wrote", len(rows), "workflow observations to", args.output_dir)


if __name__ == "__main__":
    main()
