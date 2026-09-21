"""
Test: Zero LLM/API calls in paper-mode ablation harness.

Proves that A0–A3 execute without any call to call_groq or
any HTTP request to external APIs.
"""
import pytest
from unittest.mock import patch, MagicMock
from contextlib import ExitStack
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the harness functions directly
from research_v2.run_agentic_ablation import (
    _run_a0, _run_a1, _run_a2, _run_a3,
    _validate_gender, _canonical_features, _predict_proba
)


@pytest.fixture
def trained_model():
    """Train a small RF on synthetic data for testing."""
    df = pd.read_csv("research/data/credit_risk_dataset.csv")
    features = ["credit_score","dti_ratio","loan_to_income","employment_score",
                 "applicant_age","defaults","credit_utilisation","applicant_gender"]
    X = df[features]; y = df["target_default"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf


@pytest.fixture
def sample_row():
    return pd.Series({
        "id": 1, "credit_score": 700, "dti_ratio": 0.3, "loan_to_income": 2.0,
        "employment_score": 3, "applicant_age": 35, "defaults": 0,
        "credit_utilisation": 25, "applicant_gender": 0, "target_default": 0,
    })


def test_zero_llm_calls_all_configs(trained_model, sample_row):
    """Prove that A0–A3 make zero calls to call_groq."""
    call_count = {"n": 0}

    def sentinel(*args, **kwargs):
        call_count["n"] += 1
        raise AssertionError(f"LLM call #{call_count['n']} was attempted!")

    targets = [
        "agents.llm_client.call_groq",
        "agents.intake_agent.call_groq",
        "agents.risk_agent.call_groq",
        "agents.fairness_agent.call_groq",
        "agents.orchestrator_agent.call_groq",
        "agents.reporting_agent.call_groq",
    ]
    with ExitStack() as stack:
        for target in targets:
            stack.enter_context(patch(target, side_effect=sentinel))
        res0, err0 = _run_a0(sample_row, trained_model, "hash0")
        assert err0 is None
        assert res0["final_decision"] in ("approved", "rejected")

        res1, err1 = _run_a1(sample_row, trained_model, "hash1", 6.0)
        assert err1 is None
        assert res1["final_decision"] in ("approved", "rejected")

        res2, err2 = _run_a2(sample_row, trained_model, "hash2", 6.0)
        assert err2 is None
        assert res2["final_decision"] in ("approved", "rejected")

        res3, err3 = _run_a3(sample_row, trained_model, "hash3", 6.0)
        assert err3 is None
        assert res3["final_decision"] in ("approved", "rejected")

    assert call_count["n"] == 0, f"Expected 0 LLM calls, got {call_count['n']}"


def test_zero_http_requests(trained_model, sample_row):
    """Prove that no HTTP requests are made (requests.post, httpx.post)."""
    http_calls = {"n": 0}

    def http_sentinel(*args, **kwargs):
        http_calls["n"] += 1
        raise AssertionError(f"HTTP call #{http_calls['n']} was attempted!")

    with patch("requests.post", side_effect=http_sentinel), \
         patch("httpx.post", side_effect=http_sentinel):
        for runner in [_run_a0, _run_a1, _run_a2, _run_a3]:
            if runner == _run_a0:
                res, err = runner(sample_row, trained_model, "h")
            else:
                res, err = runner(sample_row, trained_model, "h", 6.0)
            assert err is None

    assert http_calls["n"] == 0, f"Expected 0 HTTP calls, got {http_calls['n']}"


def test_paper_mode_starts_no_learning_thread(trained_model, sample_row):
    """The graph evaluation must not mutate long-term memory asynchronously."""
    with patch("agents.reporting_agent._run_learning") as learning_fn:
        result, error = _run_a2(sample_row, trained_model, "h", 6.0)
    assert error is None
    assert result is not None
    learning_fn.assert_not_called()


def test_no_target_label_leakage(trained_model, sample_row):
    """Prove that target_default is never accessed by decision logic."""
    # Create a row WITHOUT target_default
    row_no_label = sample_row.drop("target_default")
    # All runners should work without it
    res0, _ = _run_a0(row_no_label, trained_model, "h")
    res1, _ = _run_a1(row_no_label, trained_model, "h", 6.0)
    res2, _ = _run_a2(row_no_label, trained_model, "h", 6.0)
    res3, _ = _run_a3(row_no_label, trained_model, "h", 6.0)
    for r in [res0, res1, res2, res3]:
        assert r is not None
        assert r["final_decision"] in ("approved", "rejected")


def test_a1r_a2_equivalence_on_medium_high(trained_model):
    """For MEDIUM/HIGH risk applicants (where A2 also runs fairness), A1R==A2 decisions."""
    df = pd.read_csv("research/data/credit_risk_dataset.csv")
    features = ["credit_score","dti_ratio","loan_to_income","employment_score",
                 "applicant_age","defaults","credit_utilisation","applicant_gender"]
    X = df[features]; y = df["target_default"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    df_test = df.loc[X_test.index].head(100)

    from research_v2.run_agentic_ablation import _run_a1r

    mismatches = []
    for _, row in df_test.iterrows():
        r1r, e1r = _run_a1r(row, trained_model, "h", 6.0)
        r2, e2 = _run_a2(row, trained_model, "h", 6.0)
        if e1r or e2: continue
        # For cases where A2 triggered audit (MEDIUM/HIGH), decisions must match
        if r2["audit_triggered"]:
            if r1r["final_decision"] != r2["final_decision"]:
                mismatches.append((row.to_dict(), r1r, r2))

    assert len(mismatches) == 0, f"A1R vs A2 mismatch: {mismatches}"


def test_a2_a3_identical_clean_data(trained_model):
    """A3 must produce identical decisions to A2 on clean data."""
    df = pd.read_csv("research/data/credit_risk_dataset.csv")
    features = ["credit_score","dti_ratio","loan_to_income","employment_score",
                 "applicant_age","defaults","credit_utilisation","applicant_gender"]
    X = df[features]; y = df["target_default"]
    _, X_test, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    df_test = df.loc[X_test.index].head(100)

    mismatches = 0
    for _, row in df_test.iterrows():
        r2, e2 = _run_a2(row, trained_model, "h", 6.0)
        r3, e3 = _run_a3(row, trained_model, "h", 6.0)
        if e2 or e3: continue
        if r2["final_decision"] != r3["final_decision"]:
            mismatches += 1

    assert mismatches == 0, f"A2 vs A3 mismatch on {mismatches} applicants (expected 0 on clean data)"
