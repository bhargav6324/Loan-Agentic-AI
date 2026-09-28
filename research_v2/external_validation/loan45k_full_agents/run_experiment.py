"""Server runner for the isolated 45K full-agent experiment."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import train_test_split

from config import settings
from research_v2.external_validation.loan45k_adapter import prepare_loan45k
from research_v2.external_validation.loan45k_full_agents.graph import FULL_F2, FULL_F3
from research_v2.external_validation.loan45k_full_agents.model import FEATURES, fit_model
from research_v2.external_validation.loan45k_governance import (
    deterministic_reassessment, native_risk_score, research_policy, risk_category,
)

CONFIGURATIONS = ("F0", "F1", "F1P", "F2", "F3")


def policy_outcome(record, probability):
    passed, _ = research_policy(record)
    score = native_risk_score(record, 1 - float(probability))
    score, reassessed = deterministic_reassessment(score, record)
    final = int(float(probability) >= .5 and passed and risk_category(score) != "HIGH")
    return {"final_approval": final, "policy_passed": passed,
            "fairness_flagged": False, "reassessed": reassessed,
            "commit_status": "committed", "critic_findings": 0,
            "agent_steps": 0, "audit_entries": 0}


def graph_outcome(graph, record, probability, counterfactual):
    state = graph.invoke({"source_record": record,
                          "approval_probability": float(probability),
                          "counterfactual_probability": float(counterfactual),
                          "messages": [], "agent_logs": [], "errors": [],
                          "audit_epsilon": .06})
    return {"final_approval": 1 if state["final_decision"] == "approved" else 0,
            "policy_passed": state.get("policy_passed") is True,
            "fairness_flagged": state.get("bias_detected", False),
            "reassessed": bool((state.get("risk_agent_reflection") or {}).get("revised")),
            "commit_status": state.get("commit_status", "missing"),
            "critic_findings": len(state.get("critic_findings", [])),
            "agent_steps": len(state.get("agent_logs", [])),
            "audit_entries": len(state.get("audit_trail", []))}


def run_seed(data, seed, configurations, limit=None):
    train, test = train_test_split(data, test_size=.30, random_state=seed,
                                   stratify=data["loan_status"])
    if limit:
        test = test.sample(n=min(limit, len(test)), random_state=seed).sort_index()
    model = fit_model(train, seed)
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    flipped = test[FEATURES].copy()
    flipped["person_gender"] = flipped["person_gender"].map({"male": "female", "female": "male"})
    counterfactuals = model.predict_proba(flipped)[:, 1]
    records, y = test.to_dict("records"), test["loan_status"].to_numpy()
    metrics, traces = [], []
    for configuration in configurations:
        started = time.perf_counter()
        outcomes = []
        for row, p, cf in zip(records, probabilities, counterfactuals):
            if configuration == "F0":
                outcome = {"final_approval": 1, "policy_passed": True,
                           "fairness_flagged": False, "reassessed": False,
                           "commit_status": "committed", "critic_findings": 0,
                           "agent_steps": 0, "audit_entries": 0}
            elif configuration == "F1":
                outcome = {"final_approval": int(p >= .5), "policy_passed": True,
                           "fairness_flagged": False, "reassessed": False,
                           "commit_status": "committed", "critic_findings": 0,
                           "agent_steps": 0, "audit_entries": 0}
            elif configuration == "F1P":
                outcome = policy_outcome(row, p)
            else:
                outcome = graph_outcome(FULL_F2 if configuration == "F2" else FULL_F3,
                                        row, p, cf)
            outcomes.append(outcome)
        elapsed = time.perf_counter() - started
        pred = np.array([x["final_approval"] for x in outcomes])
        raw_policy_fail = np.array([not research_policy(r)[0] for r in records])
        metrics.append({"seed": seed, "configuration": configuration, "n_test": len(test),
                        "accuracy": accuracy_score(y, pred),
                        "precision_approve": precision_score(y, pred, zero_division=0),
                        "recall_approve": recall_score(y, pred, zero_division=0),
                        "f1_approve": f1_score(y, pred, zero_division=0),
                        "mcc": matthews_corrcoef(y, pred),
                        "policy_violating_approvals": int((raw_policy_fail & (pred == 1)).sum()),
                        "fairness_flags": sum(x["fairness_flagged"] for x in outcomes),
                        "reassessments": sum(x["reassessed"] for x in outcomes),
                        "commitment_blocks": sum(x["commit_status"].startswith("blocked_") for x in outcomes),
                        "automated_escalations": sum(x["commit_status"] == "automated_escalated_status" for x in outcomes),
                        "critic_findings": sum(x["critic_findings"] for x in outcomes),
                        "mean_agent_steps": float(np.mean([x["agent_steps"] for x in outcomes])),
                        "mean_audit_entries": float(np.mean([x["audit_entries"] for x in outcomes])),
                        "execution_seconds": elapsed})
        traces.append(pd.DataFrame({"benchmark_id": test["benchmark_id"].to_numpy(),
                                    "seed": seed, "configuration": configuration,
                                    "ground_truth_approval": y, "final_approval": pred}))
    return metrics, traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--configurations", default=",".join(CONFIGURATIONS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    configurations = tuple(args.configurations.split(","))
    if set(configurations) - set(CONFIGURATIONS):
        parser.error("Unknown configuration")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    prepared = prepare_loan45k(args.input)
    seeds = tuple(int(x) for x in args.seeds.split(","))
    old_key, old_mode = settings.GROQ_API_KEY, settings.PAPER_MODE
    settings.GROQ_API_KEY, settings.PAPER_MODE = "gsk_placeholder", True
    try:
        metrics, traces = [], []
        for seed in seeds:
            print(f"Starting seed {seed}...", flush=True)
            m, t = run_seed(prepared.data, seed, configurations, args.limit)
            metrics.extend(m); traces.extend(t)
            print(f"Completed seed {seed}.", flush=True)
    finally:
        settings.GROQ_API_KEY, settings.PAPER_MODE = old_key, old_mode
    pd.DataFrame(metrics).to_csv(output / "metrics_per_seed.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(output / "prediction_traces.csv", index=False)
    manifest = {"experiment": "loan45k_full_agent_adapter", "seeds": list(seeds),
                "configurations": list(configurations), "test_limit": args.limit,
                "external_calls_forced_off": True, "paper_mode": True,
                "reused_original_components": ["planner", "orchestrator", "decision_commit", "reporting"],
                "adapted_components": ["intake", "risk", "policy", "fairness", "critic"],
                "unsupported_fields_not_imputed": ["aadhaar", "pan", "salary_slip", "dti_ratio", "credit_utilisation"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
