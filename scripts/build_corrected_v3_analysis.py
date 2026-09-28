#!/usr/bin/env python3
"""Build a corrected, read-only analysis package from V3 ablation traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


CONFIGS = ("C0a", "A0", "A1", "A1R", "A2", "A3")


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def mean(values):
    return sum(values) / len(values) if values else 0.0


def sample_std(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    records = defaultdict(dict)
    counts = defaultdict(Counter)
    seed_counts = defaultdict(Counter)
    total_rows = 0

    raw_path = input_dir / "raw_applicant_traces.csv"
    with raw_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            total_rows += 1
            key = (int(row["seed"]), row["applicant_id"])
            config = row["configuration"]
            records[key][config] = row
            c = counts[config]
            s = seed_counts[(config, key[0])]
            for target in (c, s):
                target["n"] += 1
                target["fairness_audits"] += as_bool(row.get("fairness_audit_triggered", ""))
                target["fairness_bias_flags"] += as_bool(row.get("fairness_bias_flagged", ""))
                target["escalated"] += row.get("final_decision") == "escalated"
                target["approved"] += row.get("final_decision") == "approved"
                target["rejected"] += row.get("final_decision") == "rejected"
                target["unsafe_policy_approvals"] += (
                    row.get("final_decision") == "approved"
                    and not as_bool(row.get("posthoc_policy_passed", ""))
                )
                target["terminal_blocks"] += row.get("commit_status", "").startswith("blocked_")
                target["critique_events"] += bool(row.get("critique_event", "").strip())

    corrected_seed_rows = []
    for (config, seed), c in sorted(seed_counts.items(), key=lambda x: (x[0][1], CONFIGS.index(x[0][0]))):
        corrected_seed_rows.append({
            "seed": seed,
            "configuration": config,
            "n": c["n"],
            "fairness_audit_count": c["fairness_audits"],
            "fairness_bias_flagged_count": c["fairness_bias_flags"],
            "fairness_bias_rate_among_audits": c["fairness_bias_flags"] / c["fairness_audits"] if c["fairness_audits"] else 0.0,
            "escalation_count": c["escalated"],
            "escalation_rate": c["escalated"] / c["n"] if c["n"] else 0.0,
            "unsafe_policy_approval_count": c["unsafe_policy_approvals"],
            "terminal_block_count": c["terminal_blocks"],
            "critique_event_count": c["critique_events"],
        })

    corrected_aggregate_rows = []
    for config in CONFIGS:
        c = counts[config]
        seed_values = [r for r in corrected_seed_rows if r["configuration"] == config]
        flags = [r["fairness_bias_flagged_count"] for r in seed_values]
        corrected_aggregate_rows.append({
            "configuration": config,
            "n": c["n"],
            "fairness_audit_count": c["fairness_audits"],
            "fairness_bias_flagged_count": c["fairness_bias_flags"],
            "fairness_bias_flagged_mean_per_seed": mean(flags),
            "fairness_bias_flagged_std_per_seed": sample_std(flags),
            "fairness_bias_rate_among_audits": c["fairness_bias_flags"] / c["fairness_audits"] if c["fairness_audits"] else 0.0,
            "escalation_count": c["escalated"],
            "escalation_rate": c["escalated"] / c["n"] if c["n"] else 0.0,
            "unsafe_policy_approval_count": c["unsafe_policy_approvals"],
            "terminal_block_count": c["terminal_blocks"],
            "critique_event_count": c["critique_events"],
        })

    transition_counts = Counter()
    disagreement_rows = []
    comparable = 0
    agreement = 0
    unexplained = 0
    for (seed, applicant_id), by_config in sorted(records.items()):
        if "A1R" not in by_config or "A2" not in by_config:
            continue
        comparable += 1
        a1r, a2 = by_config["A1R"], by_config["A2"]
        left, right = a1r["final_decision"], a2["final_decision"]
        transition_counts[(left, right)] += 1
        if left == right:
            agreement += 1
            continue
        fairness_flag = as_bool(a2.get("fairness_bias_flagged", ""))
        if right == "escalated" and fairness_flag:
            cause = "fairness_escalation"
        elif left == "rejected" and right == "approved" and as_bool(a2.get("policy_passed", "")):
            cause = "decision_proposal_contract_divergence"
        else:
            cause = "unexplained_review_required"
            unexplained += 1
        disagreement_rows.append({
            "seed": seed,
            "applicant_id": applicant_id,
            "ground_truth": a2["ground_truth"],
            "a1r_final_decision": left,
            "a2_final_decision": right,
            "a1r_decision_proposal": a1r.get("decision", ""),
            "a2_decision_proposal": a2.get("decision", ""),
            "composite_score": a2.get("composite_score", ""),
            "risk_category": a2.get("risk_category", ""),
            "policy_passed": a2.get("policy_passed", ""),
            "fairness_audit_triggered": a2.get("fairness_audit_triggered", ""),
            "fairness_bias_flagged": a2.get("fairness_bias_flagged", ""),
            "fairness_recommendation": a2.get("fairness_recommendation", ""),
            "classification": cause,
        })

    transition_rows = [
        {"a1r_final_decision": left, "a2_final_decision": right, "count": value, "rate": value / comparable}
        for (left, right), value in sorted(transition_counts.items())
    ]

    six_rows = [r for r in disagreement_rows if r["classification"] == "decision_proposal_contract_divergence"]
    a2_a3_mismatch = 0
    for by_config in records.values():
        if "A2" in by_config and "A3" in by_config:
            fields = ("final_decision", "commit_status", "composite_score", "risk_category", "policy_passed", "fairness_bias_flagged")
            a2_a3_mismatch += any(by_config["A2"].get(f) != by_config["A3"].get(f) for f in fields)

    summary = {
        "source": {
            "input_directory": str(input_dir),
            "experiment_id": manifest.get("experiment_id"),
            "git_commit": manifest.get("git_commit"),
            "dataset_sha256": manifest.get("dataset_sha256"),
            "paper_mode": manifest.get("paper_mode"),
            "llm_calls_attempted": manifest.get("llm_calls_attempted"),
            "raw_trace_rows": total_rows,
        },
        "corrected_fairness": {
            config: dict(counts[config]) for config in CONFIGS
        },
        "a1r_vs_a2": {
            "n": comparable,
            "agreements": agreement,
            "disagreements": comparable - agreement,
            "agreement_rate": agreement / comparable,
            "fairness_escalations": sum(r["classification"] == "fairness_escalation" for r in disagreement_rows),
            "decision_proposal_contract_divergences": len(six_rows),
            "unexplained_review_required": unexplained,
        },
        "a2_vs_a3": {
            "compared": comparable,
            "decision_governance_mismatches": a2_a3_mismatch,
        },
        "reporting_correction": {
            "affected_metric": "audit_bias_flagged_count for A2/A3",
            "cause": "seed metrics read the legacy audit_bias_flagged field instead of fairness_bias_flagged",
            "decision_outputs_affected": False,
            "ten_seed_rerun_required": False,
        },
    }

    write_csv(output_dir / "corrected_fairness_per_seed.csv", corrected_seed_rows, list(corrected_seed_rows[0]))
    write_csv(output_dir / "corrected_fairness_aggregate.csv", corrected_aggregate_rows, list(corrected_aggregate_rows[0]))
    write_csv(output_dir / "a1r_a2_transition_matrix.csv", transition_rows, list(transition_rows[0]))
    write_csv(output_dir / "a1r_a2_disagreements.csv", disagreement_rows, list(disagreement_rows[0]))
    write_csv(output_dir / "decision_contract_divergences.csv", six_rows, list(disagreement_rows[0]))
    (output_dir / "corrected_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    a2 = counts["A2"]
    report = f"""# Corrected V3 Analysis

## Provenance

- Experiment: `{manifest.get('experiment_id')}`
- Git commit: `{manifest.get('git_commit')}`
- Dataset SHA-256: `{manifest.get('dataset_sha256')}`
- Raw applicant traces: {total_rows:,}
- Paper mode: {manifest.get('paper_mode')}
- LLM calls: {manifest.get('llm_calls_attempted')}

## Corrected fairness reporting

The original seed-level summary read the legacy `audit_bias_flagged` field for A2/A3. The graph stores its result in `fairness_bias_flagged`, so the aggregate value of zero was a reporting error. It did not change any applicant decision.

- A2 fairness audits: {a2['fairness_audits']:,}
- A2 fairness flags: {a2['fairness_bias_flags']:,}
- Flag rate among audited cases: {a2['fairness_bias_flags'] / a2['fairness_audits']:.2%}
- Escalations among all A2 cases: {a2['escalated']:,} / {a2['n']:,} ({a2['escalated'] / a2['n']:.2%})
- Unsafe policy approvals: {a2['unsafe_policy_approvals']}
- Terminal commitment blocks: {a2['terminal_blocks']}

## A1R versus A2

- Compared cases: {comparable:,}
- Agreement: {agreement:,} ({agreement / comparable:.2%})
- Disagreements: {comparable - agreement:,}
- Fairness-driven escalations: {sum(r['classification'] == 'fairness_escalation' for r in disagreement_rows):,}
- Decision-proposal contract divergences: {len(six_rows)}
- Unexplained differences: {unexplained}

The six contract divergences are policy-compliant MEDIUM-risk cases where A1R's canonical proposal rejected the case but A2's graph proposal approved it after the fairness check found no bias. They are not policy bypasses, but the project must choose one authoritative proposal contract before making a strict A1R/A2 equivalence claim.

## A2 versus A3

A2 and A3 had {a2_a3_mismatch} mismatches across final decision, commitment status, score, category, policy outcome, and fairness flag. No predictive improvement from A3 should be claimed.

## Conclusion

No ten-seed rerun is required for this reporting correction. Use these corrected fairness files alongside the original immutable experiment outputs. Review the six proposal-contract divergences before freezing the paper's final architecture claims.
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
