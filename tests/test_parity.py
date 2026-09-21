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
    "decision",
    "final_decision",
    "commit_status",
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
def test_a1r_vs_a2_full_test_split_parity(data_setup):
    """Core logic must match; A2 may safely skip audits after policy rejection."""
    test_df, model = data_setup
    mismatches = []
    safe_audit_short_circuits = 0
    for _, row in test_df.iterrows():
        static, static_error = _run_a1r(row, model, "seed42", 6.0)
        graph, graph_error = _run_a2(row, model, "seed42", 6.0)
        assert static_error is None
        assert graph_error is None

        differing = [
            field for field in CORE_PARITY_FIELDS
            if not _equal(static.get(field), graph.get(field))
        ]

        if graph["audit_triggered"]:
            for field in (
                "audit_bias_flagged", "audit_signed_delta",
                "audit_recommended_action",
            ):
                if not _equal(static.get(field), graph.get(field)):
                    differing.append(field)
        elif static["audit_triggered"]:
            # Dynamic A2 is allowed to avoid an unnecessary audit only when
            # the policy certificate has already made approval impossible.
            if graph["policy_passed"] is False and graph["final_decision"] == "rejected":
                safe_audit_short_circuits += 1
            else:
                differing.append("unsafe_audit_skip")
        if differing:
            mismatches.append({
                "applicant_id": int(row["id"]),
                "fields": differing,
                "static": {f: static.get(f) for f in differing},
                "graph": {f: graph.get(f) for f in differing},
            })

    assert not mismatches, f"A1R/A2 parity failures (first 5): {mismatches[:5]}"
    assert safe_audit_short_circuits > 0, "Expected A2 to demonstrate at least one safe dynamic audit skip"


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
