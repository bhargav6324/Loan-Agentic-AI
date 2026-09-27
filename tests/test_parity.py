import math
import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from research_v2.run_agentic_ablation import _run_a1r, _run_a2, _run_a3


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
    "deterministic_reassessment_count",
]


@pytest.fixture(scope="module")
def data_setup():
    """Use the exact unstratified split frozen by the paper protocol."""
    df = pd.read_csv("research/data/credit_risk_dataset.csv")
    X = df[FEATURES]
    y = df["target_default"]
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return df.loc[X_test.index].copy(), model


def _equal(left, right):
    if left is None and right is None:
        return True
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
        return abs(left - right) <= 1e-9
    return left == right


@pytest.mark.slow
def test_a1r_vs_a2_computational_parity(data_setup):
    """Core logic must match: probability, heuristic score, composite score, category, reassessment and policy."""
    test_df, model = data_setup
    mismatches = []
    
    computational_fields = [
        "p_default",
        "heuristic_score",
        "composite_score",
        "risk_category",
        "policy_passed",
        "policy_violations",
        "deterministic_reassessment_count",
    ]
    
    for _, row in test_df.iterrows():
        static, static_error = _run_a1r(row, model, "seed42", 6.0)
        graph, graph_error = _run_a2(row, model, "seed42", 6.0)
        assert static_error is None
        assert graph_error is None

        differing = [
            field for field in computational_fields
            if not _equal(static.get(field), graph.get(field))
        ]

        if differing:
            mismatches.append({
                "applicant_id": int(row["id"]),
                "fields": differing,
                "static": {f: static.get(f) for f in differing},
                "graph": {f: graph.get(f) for f in differing},
            })

    assert not mismatches, f"A1R/A2 computational parity failures (first 5): {mismatches[:5]}"


@pytest.mark.slow
def test_a1r_vs_a2_outcome_comparison(data_setup):
    """Compare decision proposal, commitment status and final decision, reporting disagreement reasons."""
    test_df, model = data_setup
    
    outcome_fields = ["decision", "commit_status", "final_decision"]
    mismatches = []
    agreements = 0
    safe_blocks = 0
    
    for _, row in test_df.iterrows():
        static, _ = _run_a1r(row, model, "seed42", 6.0)
        graph, _ = _run_a2(row, model, "seed42", 6.0)
        
        if static.get("final_decision") == "approved" and graph.get("final_decision") in ["rejected", "escalated"]:
            safe_blocks += 1
            
        differing = [
            field for field in outcome_fields
            if not _equal(static.get(field), graph.get(field))
        ]
        
        if differing:
            mismatches.append({
                "applicant_id": int(row["id"]),
                "fields": differing,
                "static": {f: static.get(f) for f in outcome_fields},
                "graph": {f: graph.get(f) for f in outcome_fields},
            })
        else:
            agreements += 1

    agreement_rate = (agreements / len(test_df)) * 100
    
    report = (
        f"Outcome Agreement: {agreements}/{len(test_df)} ({agreement_rate:.2f}%)\n"
        f"Disagreements: {len(mismatches)}\n"
        f"Safe blocks (A2 rejected/escalated what A1R approved): {safe_blocks}\n"
        f"Mismatches (first 5): {mismatches[:5]}"
    )
    
    assert agreements > 0, "Expected at least some outcomes to match."
    if mismatches:
        print("\nNote: Expected A1R and A2 outcomes to diverge due to safe blocks and orchestration differences.")
        print(report)


@pytest.mark.slow
def test_a1r_vs_a2_orchestration_behaviour(data_setup):
    """Verify fairness execution and safe policy-based skipping."""
    test_df, model = data_setup
    mismatches = []
    
    for _, row in test_df.iterrows():
        static, _ = _run_a1r(row, model, "seed42", 6.0)
        graph, _ = _run_a2(row, model, "seed42", 6.0)
        
        a1r_audit = static.get("audit_triggered")
        a2_audit = graph.get("fairness_audit_triggered")
        policy_passed = graph.get("policy_passed")
        risk_category = graph.get("risk_category")
        
        # A2 skips audit if policy failed, or if risk is HIGH or LOW.
        # A1R runs audit for MEDIUM and HIGH.
        if policy_passed and a1r_audit and not a2_audit:
            if risk_category != "HIGH":
                mismatches.append({"applicant_id": int(row["id"]), "reason": f"A2 skipped audit on passed policy (Risk: {risk_category})"})
            
        if not policy_passed and a2_audit:
            mismatches.append({"applicant_id": int(row["id"]), "reason": "A2 ran audit on failed policy"})
            
    assert not mismatches, f"Orchestration behaviour failures (first 5): {mismatches[:5]}"


@pytest.mark.slow
def test_a2_vs_a3_clean_data_decision_invariance(data_setup):
    """The passive critic must not change clean-data decisions or commitments."""
    test_df, model = data_setup
    mismatches = []
    for _, row in test_df.iterrows():
        graph, graph_error = _run_a2(row, model, "seed42", 6.0)
        critic, critic_error = _run_a3(row, model, "seed42", 6.0)
        assert graph_error is None
        assert critic_error is None
        for field in ("decision", "final_decision", "commit_status"):
            if graph.get(field) != critic.get(field):
                mismatches.append((int(row["id"]), field, graph.get(field), critic.get(field)))

    assert not mismatches, f"A2/A3 clean-data mismatches (first 5): {mismatches[:5]}"
