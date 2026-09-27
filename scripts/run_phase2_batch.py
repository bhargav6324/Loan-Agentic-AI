"""Run a reproducible Phase 2 A2/A3 batch without touching Phase 1 results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_v2.phase2.batch_factory import build_cases_from_dataframe, build_paired_cases
from research_v2.phase2.experiment_runner import run_paired_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", default="research_v2/phase2_results"
    )
    args = parser.parse_args()

    dataset_path = ROOT / "research/data/credit_risk_dataset.csv"
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(dataset_path)
    clean_cases = build_cases_from_dataframe(dataframe, args.limit, args.seed)
    cases = build_paired_cases(clean_cases)
    result = run_paired_experiment(cases)

    rows = [row.model_dump() for row in result["evaluations"]]
    with (output_dir / "fault_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        name: value.model_dump() for name, value in result["metrics"].items()
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    manifest = {
        "dataset": str(dataset_path.relative_to(ROOT)),
        "seed": args.seed,
        "clean_case_count": len(clean_cases),
        "total_case_count": len(cases),
        "configurations": ["A2", "A3"],
        "fault_types": sorted({label for case in cases for label in case.fault_labels}),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    for name, value in metrics.items():
        print(name, json.dumps(value, sort_keys=True))
    print(f"Wrote Phase 2 results to {output_dir}")


if __name__ == "__main__":
    main()

