import json
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from research_v2.run_agentic_ablation import _run_a1r, _run_a2

FEATURES = [
    "credit_score", "dti_ratio", "loan_to_income", "employment_score",
    "applicant_age", "defaults", "credit_utilisation", "applicant_gender",
]
CORE_PARITY_FIELDS = [
    "p_default",
    "heuristic_score",
    "composite_score",
    "risk_category",
    "policy_passed",
    "policy_violations",
    "decision",
    "final_decision",
    "commit_status",
    "deterministic_reassessment_count",
]

def _equal(left, right):
    if left is None and right is None:
        return True
    if isinstance(left, float) and isinstance(right, float):
        import math
        if math.isnan(left) and math.isnan(right):
            return True
        return abs(left - right) <= 1e-9
    return left == right

def main():
    df = pd.read_csv("research/data/credit_risk_dataset.csv")
    X = df[FEATURES]
    y = df["target_default"]
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    test_df = df.loc[X_test.index].copy()
    
    mismatches = []
    
    for _, row in test_df.iterrows():
        static, _ = _run_a1r(row, model, "seed42", 6.0)
        graph, _ = _run_a2(row, model, "seed42", 6.0)
        
        differing = [f for f in CORE_PARITY_FIELDS if not _equal(static.get(f), graph.get(f))]
        
        if graph["audit_triggered"]:
            for field in ("audit_bias_flagged", "audit_signed_delta", "audit_recommended_action"):
                if not _equal(static.get(field), graph.get(field)):
                    differing.append(field)
        elif static["audit_triggered"]:
            if not (graph["policy_passed"] is False and graph["final_decision"] == "rejected"):
                differing.append("unsafe_audit_skip")
                
        if differing:
            mismatches.append(row)
            if len(mismatches) == 5:
                break

    out = []
    for row in mismatches:
        # Run twice to detect nondeterminism
        a1r_1, _ = _run_a1r(row, model, "seed42", 6.0)
        a2_1, _ = _run_a2(row, model, "seed42", 6.0)
        a1r_2, _ = _run_a1r(row, model, "seed42", 6.0)
        a2_2, _ = _run_a2(row, model, "seed42", 6.0)
        
        out.append({
            "applicant_id": int(row["id"]),
            "nondeterministic_a1r": a1r_1 != a1r_2,
            "nondeterministic_a2": a2_1 != a2_2,
            "a1r": a1r_1,
            "a2": a2_1
        })
        
    os.makedirs("research_v2/phase2_results/full_suite_diagnostic", exist_ok=True)
    with open("research_v2/phase2_results/full_suite_diagnostic/parity_mismatch_traces.json", "w") as f:
        json.dump(out, f, indent=2)
        
    print("Diagnosis complete. Traces saved.")

if __name__ == "__main__":
    main()
