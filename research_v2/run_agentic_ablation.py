"""
Phase 4 Ablation Harness — Paper Mode (Zero LLM Calls)
═══════════════════════════════════════════════════════
All configurations (A0–A3) run with LLM functionality structurally disabled.
A sentinel on call_groq ensures zero API calls are attempted.
"""
import os, sys, json, time, hashlib, argparse, pickle, importlib
from numbers import Real
from datetime import datetime
from copy import deepcopy
from unittest.mock import patch
from contextlib import ExitStack

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             matthews_corrcoef, balanced_accuracy_score)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

V2_DIR = os.path.dirname(__file__)
FINAL_RESULTS_DIR = os.path.join(V2_DIR, "final_results")
DRY_RUN_RESULTS_DIR = os.path.join(V2_DIR, "dry_run_results")
VALIDATION_RESULTS_DIR = os.path.join(V2_DIR, "validation_results")
MANIFEST_DIR = os.path.join(V2_DIR, "manifests")
CONFIG_DIR = os.path.join(V2_DIR, "configs")

# ── LLM Call Counter & Sentinel ────────────────────────────
_llm_call_count = 0

def _llm_sentinel(*args, **kwargs):
    """Sentinel: any LLM call attempt is an error in paper mode."""
    global _llm_call_count
    _llm_call_count += 1
    raise AssertionError(
        f"PAPER MODE VIOLATION: call_groq was invoked (call #{_llm_call_count}). "
        "All agent logic must use deterministic fallback paths."
    )

# ── Deterministic Agent Functions ──────────────────────────
# These replicate the exact deterministic-fallback logic from the agents
# WITHOUT importing the full agent modules (which try to connect to DBs/LLMs).

def _canonical_features(row):
    """Extract 8 canonical features from a dataset row."""
    return {
        "credit_score": float(row.get("credit_score", 0)),
        "dti_ratio": float(row.get("dti_ratio", 0)),
        "loan_to_income": float(row.get("loan_to_income", 0)),
        "employment_score": float(row.get("employment_score", 0)),
        "age": float(row.get("applicant_age", row.get("age", 0))),
        "defaults": float(row.get("defaults", 0)),
        "credit_utilisation": float(row.get("credit_utilisation", 0)),
        "gender_encoded": float(row.get("applicant_gender", row.get("gender_encoded", -1))),
        "employment_years": row.get("employment_years", 3),
        "loan_tenure_years": row.get("loan_tenure_years", 10),
        "loan_purpose": row.get("loan_purpose", "personal"),
    }

def _validate_gender(row):
    """Return (valid: bool, encoded: float). Raises nothing."""
    g = row.get("applicant_gender", row.get("gender_encoded", row.get("gender")))
    if g is None:
        return False, -1.0
    if isinstance(g, Real) and not isinstance(g, bool):
        gv = float(g)
        if gv in (0.0, 1.0):
            return True, gv
        return False, -1.0
    gl = str(g).lower()
    if gl == "male": return True, 0.0
    if gl == "female": return True, 1.0
    return False, -1.0

# Dataset columns → internal canonical names mapping
_DS_TO_INTERNAL = {
    "credit_score": "credit_score", "dti_ratio": "dti_ratio",
    "loan_to_income": "loan_to_income", "employment_score": "employment_score",
    "applicant_age": "age", "defaults": "defaults",
    "credit_utilisation": "credit_utilisation", "applicant_gender": "gender_encoded",
}
_INTERNAL_TO_DS = {v: k for k, v in _DS_TO_INTERNAL.items()}

# The RF was fitted on dataset column names; prediction must use those same names.
_DS_FEATURE_ORDER = ["credit_score","dti_ratio","loan_to_income","employment_score",
                     "applicant_age","defaults","credit_utilisation","applicant_gender"]

def _predict_proba(model, features_dict):
    """Run RF prediction through 8-feature vector. Returns P(default).
    features_dict uses internal canonical names; we map back to dataset names for sklearn."""
    row = {_INTERNAL_TO_DS.get(k, k): v for k, v in features_dict.items()}
    X = pd.DataFrame([{col: row[col] for col in _DS_FEATURE_ORDER}])
    if list(model.classes_) != [0, 1]:
        raise ValueError(f"Model classes must be exactly [0, 1], got {list(model.classes_)}")
    return float(model.predict_proba(X)[0][1])

from ml.canonical_model_contract import compute_heuristic_score
from core.canonical_contracts import (
    canonical_composite_score,
    canonical_risk_category,
    canonical_policy_check,
    canonical_audit_flip,
    canonical_risk_reassessment,
    canonical_final_decision,
    canonical_audit_eligibility,
    canonical_hard_overrides,
    canonical_hard_override_score,
)

# ── Metrics ────────────────────────────────────────────────

def _compute_metrics(df, seed, config):
    if len(df) == 0:
        return {}
    y_pred = (df["final_decision"] == "approved").astype(int)
    y_true = (df["ground_truth"] == 0).astype(int)  # non-default=approve-worthy
    
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try: roc = roc_auc_score(df["ground_truth"], df["p_default"])
    except: roc = float('nan')
    try: pr = average_precision_score(df["ground_truth"], df["p_default"])
    except: pr = float('nan')
    try: mcc = matthews_corrcoef(y_true, y_pred)
    except: mcc = 0.0
    
    # Fairness metrics
    g1_mask = df["gender_encoded"] == 1
    g0_mask = df["gender_encoded"] == 0
    
    g1_approve_rate = y_pred[g1_mask].mean() if g1_mask.sum() > 0 else 0
    g0_approve_rate = y_pred[g0_mask].mean() if g0_mask.sum() > 0 else 0
    demographic_parity_diff = abs(g1_approve_rate - g0_approve_rate)
    
    # TPR (Recall) per group
    y_true_g1, y_pred_g1 = y_true[g1_mask], y_pred[g1_mask]
    y_true_g0, y_pred_g0 = y_true[g0_mask], y_pred[g0_mask]
    tpr_g1 = recall_score(y_true_g1, y_pred_g1, zero_division=0)
    tpr_g0 = recall_score(y_true_g0, y_pred_g0, zero_division=0)
    equal_opportunity_diff = abs(tpr_g1 - tpr_g0)
    
    # FPR per group
    tn_g1 = ((y_true_g1 == 0) & (y_pred_g1 == 0)).sum()
    fp_g1 = ((y_true_g1 == 0) & (y_pred_g1 == 1)).sum()
    fpr_g1 = fp_g1 / (tn_g1 + fp_g1) if (tn_g1 + fp_g1) > 0 else 0.0
    
    tn_g0 = ((y_true_g0 == 0) & (y_pred_g0 == 0)).sum()
    fp_g0 = ((y_true_g0 == 0) & (y_pred_g0 == 1)).sum()
    fpr_g0 = fp_g0 / (tn_g0 + fp_g0) if (tn_g0 + fp_g0) > 0 else 0.0
    
    equalized_odds_diff = max(abs(tpr_g1 - tpr_g0), abs(fpr_g1 - fpr_g0))

    audit_exec = int(df["audit_triggered"].sum()) if "audit_triggered" in df else 0
    # Same governance definition for every configuration: count only final
    # approvals that violate the canonical policy contract.
    pol_vio = 0
    if "posthoc_policy_passed" in df:
        pol_vio = int(
            ((df["final_decision"] == "approved") & (~df["posthoc_policy_passed"].astype(bool))).sum()
        )
        
    audit_exec = int(df["audit_triggered"].sum()) if "audit_triggered" in df else 0
    audit_bias = int(df["audit_bias_flagged"].sum()) if "audit_bias_flagged" in df else 0
    det_reass = int(df["deterministic_reassessment_count"].sum()) if "deterministic_reassessment_count" in df else 0
    
    blocked_count = int(df.get("commitment_block_events", pd.Series(dtype=float)).fillna(0).sum())
    blocked_applicant_count = int(df.get("had_commitment_block", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    terminal_block_count = int(df["commit_status"].fillna("").str.startswith("blocked_").sum()) if "commit_status" in df else 0
    
    override_approve = int(((df["decision"] == "reject") & (df["final_decision"] == "approved")).sum()) if "decision" in df else 0
    override_reject = int(((df["decision"] == "approve") & (df["final_decision"] == "rejected")).sum()) if "decision" in df else 0

    return {
        "seed": seed, "configuration": config, "n_test": len(df),
        "n_approve": int(y_pred.sum()), "n_reject": int(len(df) - y_pred.sum()),
        "approval_rate": float(y_pred.mean()),
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float((rec + spec) / 2.0),
        "precision_approve": float(prec), "recall_approve": float(rec),
        "f1_approve": float(f1), "specificity": float(spec), "mcc": float(mcc),
        "roc_auc_default": float(roc), "pr_auc_default": float(pr),
        
        # New telemetry fields
        "audit_executed_count": audit_exec,
        "audit_bias_flagged_count": audit_bias,
        "downstream_override_approve_count": override_approve,
        "downstream_override_reject_count": override_reject,
        "deterministic_reassessment_count": det_reass,
        "policy_violation_count": pol_vio,
        "commitment_block_event_count": blocked_count,
        "commitment_block_applicant_count": blocked_applicant_count,
        "terminal_commitment_block_count": terminal_block_count,
        "critique_event_count": int(df["critique_event"].notna().sum() if "critique_event" in df else 0),
        
        # Fairness
        "demographic_parity_diff": float(demographic_parity_diff),
        "equal_opportunity_diff": float(equal_opportunity_diff),
        "equalized_odds_diff": float(equalized_odds_diff),

        
        "mean_latency_ms": float(df["latency_ms"].mean()),
        "median_latency_ms": float(df["latency_ms"].median()),
    }


def _aggregate_metrics(metrics_df):
    """Mean, sample SD and t-based 95% CI across experimental seeds."""
    from scipy.stats import t

    excluded = {"seed", "configuration"}
    numeric = [
        c for c in metrics_df.select_dtypes(include=[np.number]).columns
        if c not in excluded
    ]
    rows = []
    for config, group in metrics_df.groupby("configuration", sort=False):
        for metric in numeric:
            values = group[metric].dropna().astype(float)
            n = len(values)
            mean = float(values.mean()) if n else float("nan")
            std = float(values.std(ddof=1)) if n > 1 else float("nan")
            if n > 1:
                half_width = float(t.ppf(0.975, n - 1) * std / np.sqrt(n))
                ci_low, ci_high = mean - half_width, mean + half_width
            else:
                ci_low = ci_high = float("nan")
            rows.append({
                "configuration": config,
                "metric": metric,
                "n_seeds": n,
                "mean": mean,
                "std": std,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "min": float(values.min()) if n else float("nan"),
                "max": float(values.max()) if n else float("nan"),
            })
    return pd.DataFrame(rows)


def _holm_adjust(p_values):
    """Holm step-down adjustment, preserving the original row order."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    finite = np.where(np.isfinite(p))[0]
    order = finite[np.argsort(p[finite])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def _paired_comparisons(metrics_df):
    """Predefined paired seed-level tests; no applicant-level pseudo-replication."""
    from scipy.stats import ttest_rel, wilcoxon, t

    comparisons = [
        ("A0", "A1", "pipeline_vs_rf"),
        ("A0", "A2", "agentic_vs_rf"),
        ("A1", "A1R", "reassessment_effect"),
    ]
    metrics = [
        "f1_approve", "specificity", "balanced_accuracy", "mcc",
        "demographic_parity_diff", "equal_opportunity_diff",
        "equalized_odds_diff", "policy_violation_count",
    ]
    rows = []
    for baseline, treatment, role in comparisons:
        for metric in metrics:
            wide = metrics_df.pivot(index="seed", columns="configuration", values=metric)
            if baseline not in wide or treatment not in wide:
                continue
            paired = wide[[baseline, treatment]].dropna()
            diff = paired[treatment] - paired[baseline]
            n = len(diff)
            if n < 2:
                continue
            mean_diff = float(diff.mean())
            sd_diff = float(diff.std(ddof=1))
            half_width = float(t.ppf(0.975, n - 1) * sd_diff / np.sqrt(n))
            t_stat, p_value = ttest_rel(paired[treatment], paired[baseline])
            if np.allclose(diff, 0):
                w_stat, w_p = float("nan"), 1.0
            else:
                try:
                    w_stat, w_p = wilcoxon(diff)
                except ValueError:
                    w_stat, w_p = float("nan"), float("nan")
            rows.append({
                "comparison": f"{treatment} - {baseline}",
                "comparison_role": role,
                "metric": metric,
                "n_pairs": n,
                "mean_difference": mean_diff,
                "ci95_low": mean_diff - half_width,
                "ci95_high": mean_diff + half_width,
                "cohen_dz": mean_diff / sd_diff if sd_diff > 0 else float("nan"),
                "paired_t_statistic": float(t_stat),
                "raw_p_value": float(p_value),
                "wilcoxon_statistic": float(w_stat),
                "wilcoxon_p_value": float(w_p),
                "direction_consistency": float((np.sign(diff) == np.sign(mean_diff)).mean()),
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["holm_adjusted_p_value"] = _holm_adjust(result["raw_p_value"].to_numpy())
        result["holm_significant_0_05"] = result["holm_adjusted_p_value"] < 0.05
    return result

# ── Per-Applicant Simulation ───────────────────────────────

def _run_c0a(row, model, model_hash):
    """C0a: Always-approve baseline for class-imbalance context."""
    start = time.time()
    gender_valid, gender_enc = _validate_gender(row)
    if not gender_valid:
        return None, {"error_type": "ValueError", "error_message": "Missing/invalid gender"}
    return {
        "model_hash": None, "threshold": None, "p_default": 0.0,
        "heuristic_score": None, "composite_score": None, "risk_category": None,
        "policy_passed": None, "policy_violations": None,
        "audit_triggered": False, "audit_bias_flagged": False,
        "audit_signed_delta": None, "audit_recommended_action": None,
        "decision": "approved", "final_decision": "approved",
        "commit_status": "committed_approve",
        "latency_ms": (time.time() - start) * 1000,
        "critique_event": None, "gender_encoded": gender_enc, "gender_missing": False,
        "commitment_block_events": 0, "had_commitment_block": False,
        "nodes_executed": "always_approve",
    }, None

def _run_a0(row, model, model_hash):
    """A0: Pure ML threshold. No heuristic, policy, audit, or critique."""
    start = time.time()
    gender_valid, gender_enc = _validate_gender(row)
    if not gender_valid:
        return None, {"error_type": "ValueError", "error_message": "Missing/invalid gender"}
    feats = _canonical_features(row)
    feats["gender_encoded"] = gender_enc
    p = _predict_proba(model, feats)
    decision = "approved" if p <= 0.5 else "rejected"
    return {
        "model_hash": model_hash, "threshold": 0.5, "p_default": p,
        "heuristic_score": None, "composite_score": None, "risk_category": None,
        "policy_passed": None, "policy_violations": None,
        "audit_triggered": False, "audit_bias_flagged": False,
        "audit_signed_delta": None, "audit_recommended_action": None,
        "decision": decision, "final_decision": decision,
        "commit_status": f"committed_{'approve' if decision=='approved' else 'reject'}",
        "latency_ms": (time.time()-start)*1000,
        "critique_event": None, "gender_encoded": gender_enc, "gender_missing": False,
        "commitment_block_events": 0, "had_commitment_block": False,
        "nodes_executed": "threshold_only",
    }, None

def _run_a1(row, model, model_hash, epsilon):
    """A1: Fixed sequential pipeline. intake→risk→policy→fairness→commit."""
    start = time.time()
    gender_valid, gender_enc = _validate_gender(row)
    if not gender_valid:
        return None, {"error_type": "ValueError", "error_message": "Missing/invalid gender"}
    
    feats = _canonical_features(row)
    feats["gender_encoded"] = gender_enc
    p = _predict_proba(model, feats)
    h = compute_heuristic_score(feats)
    
    # ── Hard Overrides ──
    hard_override = canonical_hard_overrides(feats)
    if hard_override == "HARD_REJECT":
        composite = canonical_hard_override_score(hard_override)
    elif hard_override == "HARD_FAST_TRACK":
        composite = canonical_hard_override_score(hard_override)
    else:
        _, _, composite = canonical_composite_score(p, h)
        
    cat = canonical_risk_category(composite)
    
    pol_passed, pol_violations = canonical_policy_check(row.to_dict() if hasattr(row, 'to_dict') else row)
    from core.canonical_contracts import canonical_audit_eligibility
    audit_eligible = canonical_audit_eligibility(cat)
    
    if audit_eligible:
        audit = canonical_audit_flip(
            lambda f: _predict_proba(model, f), feats, composite, epsilon,
            apply_reassessment=False,
        )
        bias_flagged = audit["flagged"]
        delta = audit["signed_delta"]
        audit_action = audit["recommended_action"]
    else:
        bias_flagged = False
        delta = None
        audit_action = None
    
    # Calculate ML confidence identically to risk_agent for Ambiguity-Gating
    ml_confidence = max(p, 1.0 - p)
    
    # Proposal logic (before final decision)
    from core.canonical_contracts import canonical_decision_proposal
    proposal = canonical_decision_proposal(cat, pol_passed, ml_confidence)
        
    commit, reasoning = canonical_final_decision(
        risk_category=cat,
        policy_passed=pol_passed,
        policy_violations=pol_violations,
        audit_required=audit_eligible,
        audit_executed=audit_eligible,
        audit_flagged=bias_flagged,
        decision_proposal=proposal,
        audit_recommended_action=audit_action,
    )
    
    final = "approved" if commit == "committed_approve" else "rejected"
    return {
        "model_hash": model_hash, "threshold": 0.5, "p_default": p,
        "heuristic_score": h, "composite_score": composite, "risk_category": cat,
        "policy_passed": pol_passed, "policy_violations": ",".join(pol_violations),
        "audit_triggered": audit_eligible, "audit_bias_flagged": bias_flagged,
        "audit_signed_delta": delta, "audit_recommended_action": audit_action,
        "decision": proposal, "final_decision": final,
        "commit_status": commit, "latency_ms": (time.time()-start)*1000,
        "critique_event": None, "gender_encoded": gender_enc, "gender_missing": False,
        "commitment_block_events": 0, "had_commitment_block": False,
        "nodes_executed": "intake,risk,policy,fairness,commit",
    }, None

def _run_a1r(row, model, model_hash, epsilon):
    """A1R: Fixed sequential pipeline WITH deterministic risk reassessment."""
    start = time.time()
    gender_valid, gender_enc = _validate_gender(row)
    if not gender_valid:
        return None, {"error_type": "ValueError", "error_message": "Missing/invalid gender"}
    
    feats = _canonical_features(row)
    feats["gender_encoded"] = gender_enc
    p = _predict_proba(model, feats)
    h = compute_heuristic_score(feats)
    
    # ── Hard Overrides ──
    hard_override = canonical_hard_overrides(feats)
    if hard_override == "HARD_REJECT":
        composite = canonical_hard_override_score(hard_override)
        reassess = {"revised": False}
    elif hard_override == "HARD_FAST_TRACK":
        composite = canonical_hard_override_score(hard_override)
        reassess = {"revised": False}
    else:
        _, _, composite = canonical_composite_score(p, h)
        # ── Deterministic Risk Reassessment (Same as Risk Agent) ──
        reassess = canonical_risk_reassessment(composite, p * 100.0, h, feats)
        if reassess["revised"]:
            composite = reassess["revised_score"]
            
    cat = canonical_risk_category(composite)
    # ────────────────────────────────────────────────────────
    
    pol_passed, pol_violations = canonical_policy_check(row.to_dict() if hasattr(row, 'to_dict') else row)
    
    audit_eligible = canonical_audit_eligibility(cat)
    if audit_eligible:
        # A1R: static sequential path — runs same audit contract as A2
        audit = canonical_audit_flip(lambda f: _predict_proba(model, f), feats, composite, epsilon)
        bias_flagged = audit["flagged"]
        delta = audit["signed_delta"]
        audit_action = audit["recommended_action"]
    else:
        bias_flagged = False
        delta = None
        audit_action = None

    # Calculate ML confidence identically to risk_agent for Ambiguity-Gating
    ml_confidence = max(p, 1.0 - p)

    # Proposal logic (before final decision)
    from core.canonical_contracts import canonical_decision_proposal
    proposal = canonical_decision_proposal(cat, pol_passed, ml_confidence)

    commit, reasoning = canonical_final_decision(
        risk_category=cat,
        policy_passed=pol_passed,
        policy_violations=pol_violations,
        audit_required=audit_eligible,
        audit_executed=audit_eligible,
        audit_flagged=bias_flagged,
        decision_proposal=proposal,
        audit_recommended_action=audit_action,
    )

    final = "approved" if commit == "committed_approve" else "rejected"

    nodes = "intake,risk,policy,commit"
    if audit_eligible:
        nodes = "intake,risk,policy,fairness,commit"

    return {
        "model_hash": model_hash, "threshold": 0.5, "p_default": p,
        "heuristic_score": h, "composite_score": composite, "risk_category": cat,
        "policy_passed": pol_passed, "policy_violations": ",".join(pol_violations),
        "audit_triggered": audit_eligible, "audit_bias_flagged": bias_flagged,
        "audit_signed_delta": delta, "audit_recommended_action": audit_action,
        "decision": proposal, "final_decision": final,
        "commit_status": commit, "latency_ms": (time.time()-start)*1000,
        "critique_event": None, "gender_encoded": gender_enc, "gender_missing": False,
        "commitment_block_events": 0, "had_commitment_block": False,
        "nodes_executed": nodes,
        "deterministic_reassessment_count": 1 if reassess["revised"] else 0,
    }, None


import ml.canonical_model_contract

_graph_a2 = None


def _get_paper_graph():
    """Import and compile LangGraph only when A2/A3 is actually requested."""
    global _graph_a2
    if _graph_a2 is None:
        from orchestrator.graph import build_loan_graph
        _graph_a2 = build_loan_graph(disable_fairness=False, disable_policy=False)
    return _graph_a2

def _invoke_graph_runner(row, model, model_hash, epsilon, use_critic=False):
    from agents.state import AgentState
    from config import settings

    original_paper_mode = getattr(settings, "PAPER_MODE", False)
    settings.PAPER_MODE = True
    
    # Global injection: unavoidable because graph loads ml.canonical_model_contract._model internally.
    # Safe because ablation script is strictly single-threaded.
    original_model = getattr(ml.canonical_model_contract, "_model", None)
    ml.canonical_model_contract._model = model
    
    try:
        start = time.time()
        gender_valid, gender_enc = _validate_gender(row)
        if not gender_valid:
            return None, {"error_type": "ValueError", "error_message": "Missing/invalid gender"}
    
        monthly_income = row.get("monthly_income", 50000)
        annual_income = monthly_income * 12
        loan_amount = row.get("loan_amount", row["loan_to_income"] * annual_income)
        current_outstanding = row.get("current_outstanding", row["dti_ratio"] * annual_income)
        emp_map = {3: "salaried", 2: "business", 1: "self_employed"}
        emp_type = emp_map.get(int(row["employment_score"]), "salaried")
        
        fake_report = {
            "credit_score": row["credit_score"],
            "current_outstanding": current_outstanding,
            "total_defaults": row["defaults"],
            "credit_utilisation_pct": row["credit_utilisation"]
        }
        
        initial_state = AgentState(
            applicant_data={
                "applicant_name": "Test User",
                "applicant_pan": "ABCDE1234F",
                "applicant_aadhaar": "123456789012",
                "applicant_age": row["applicant_age"],
                "applicant_gender": gender_enc,
                "loan_amount": loan_amount,
                "monthly_income": monthly_income,
                "employment_type": emp_type,
                "employment_score": float(row["employment_score"]),
                "employment_years": row.get("employment_years", 3),
                "loan_tenure_years": row.get("loan_tenure_years", 10),
                "loan_purpose": row.get("loan_purpose", "personal"),
                "dti_ratio": row.get("dti_ratio"),
                "loan_to_income": row.get("loan_to_income"),
            },
            messages=[],
            agent_logs=[],
            errors=[],
            audit_epsilon=float(epsilon),
        )
        
        with patch("agents.intake_agent.get_credit_report", return_value=fake_report), \
             patch("agents.intake_agent.verify_pan", return_value={"verified": True, "name_match": True, "status": "ACTIVE"}), \
             patch("agents.intake_agent.verify_aadhaar", return_value={"verified": True, "name_match": True, "status": "ACTIVE"}):
             
            final_state = _get_paper_graph().invoke(initial_state)
             
        feats = _canonical_features(row)
        feats["gender_encoded"] = gender_enc
        p = _predict_proba(model, feats)
        h = compute_heuristic_score(feats)
        _, _, composite = canonical_composite_score(p, h)
        
        # Pull revised score from state if risk agent did a reassessment
        if "risk_score" in final_state:
            composite = final_state["risk_score"]
            
        cat = canonical_risk_category(composite)        
        commit_status = final_state.get("commit_status", "")
        decision = final_state.get("decision_proposal", "approve")
        final_decision = "approved" if commit_status == "committed_approve" else "rejected"
        
        fairness_report = final_state.get("fairness_report")
        audit_triggered = fairness_report is not None
        if audit_triggered:
            audit_delta = fairness_report.get("signed_delta", 0.0)
            audit_flagged = fairness_report.get("bias_detected", False)
            audit_action = fairness_report.get("recommended_action")
        else:
            audit_delta = None
            audit_flagged = False
            audit_action = None
            
        critique = None
        if use_critic:
            if decision == "approve" and final_state.get("policy_passed") is False:
                critique = "STRUCTURAL INCONSISTENCY: Proposed 'approve' but policy_passed is False."
            elif audit_triggered and audit_delta is None:
                critique = "STRUCTURAL INCONSISTENCY: Audit triggered but delta is missing."
            elif audit_triggered is False and cat == "MEDIUM" and final_state.get("policy_passed") is True:
                critique = "STRUCTURAL INCONSISTENCY: MEDIUM risk did not trigger audit."
                
        nodes_executed = ",".join(list(dict.fromkeys([log["agent"] for log in final_state.get("agent_logs", []) if "agent" in log])))
        # Parallel reducers can duplicate prior log entries. Count unique block
        # reasons per applicant rather than raw merged log rows.
        block_signatures = {
            (log.get("action", ""), log.get("details", ""))
            for log in final_state.get("agent_logs", [])
            if log.get("agent") == "decision_commit"
            and "Decision Commit: blocked_" in log.get("action", "")
        }
        commitment_block_events = len(block_signatures)
        if use_critic:
            nodes_executed += ",critic"
            
        det_reass = 1 if final_state.get("risk_agent_reassessment", {}).get("revised") else 0
            
        return {
            "model_hash": model_hash, "threshold": 0.5, "p_default": p,
            "heuristic_score": h, "composite_score": composite, "risk_category": cat,
            "policy_passed": final_state.get("policy_passed"), 
            "policy_violations": ",".join(final_state.get("policy_violations", [])),
            "audit_triggered": audit_triggered, "audit_bias_flagged": audit_flagged,
            "audit_signed_delta": audit_delta, "audit_recommended_action": audit_action,
            "decision": decision, "final_decision": final_decision,
            "commit_status": commit_status, 
            "latency_ms": (time.time()-start)*1000,
            "critique_event": critique, 
            "gender_encoded": gender_enc, "gender_missing": False,
            "commitment_block_events": commitment_block_events,
            "had_commitment_block": commitment_block_events > 0,
            "nodes_executed": nodes_executed,
            "deterministic_reassessment_count": det_reass,
        }, None
    finally:
        ml.canonical_model_contract._model = original_model
        settings.PAPER_MODE = original_paper_mode

def _run_a2(row, model, model_hash, epsilon):
    """A2: Runs the actual LangGraph, which natively skips fairness for LOW risk."""
    return _invoke_graph_runner(row, model, model_hash, epsilon, use_critic=False)

def _run_a3(row, model, model_hash, epsilon):
    """A3: Runs the LangGraph + structural critic."""
    return _invoke_graph_runner(row, model, model_hash, epsilon, use_critic=True)

# ── Main ───────────────────────────────────────────────────

def main():
    global _llm_call_count
    _llm_call_count = 0

    parser = argparse.ArgumentParser(description="Phase 4 Ablation Harness (Paper Mode)")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--configurations", type=str, default="A0,A1,A1R,A2,A3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test-split", type=float, default=0.30)
    parser.add_argument("--epsilon", type=float, default=6.0)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    configs = args.configurations.split(",")
    allowed_configs = {"C0a", "A0", "A1", "A1R", "A2", "A3"}
    unknown = set(configs) - allowed_configs
    if unknown:
        parser.error(f"Unknown configurations: {sorted(unknown)}")

    frozen_final = (
        seeds == list(range(42, 52))
        and configs == ["C0a", "A0", "A1", "A1R", "A2", "A3"]
        and not args.dry_run
    )
    if args.dry_run:
        results_dir = DRY_RUN_RESULTS_DIR
    elif frozen_final:
        results_dir = FINAL_RESULTS_DIR
    else:
        results_dir = VALIDATION_RESULTS_DIR
    for directory in (results_dir, MANIFEST_DIR, CONFIG_DIR):
        os.makedirs(directory, exist_ok=True)

    print(f"{'='*60}")
    print(f"  ABLATION HARNESS — PAPER MODE (ZERO LLM)")
    print(f"{'='*60}")
    print(f"  Seeds:    {seeds}")
    print(f"  Configs:  {configs}")
    print(f"  Split:    {args.test_split}")
    print(f"  Epsilon:  {args.epsilon}")
    print(f"  Dry run:  {args.dry_run}")
    print(f"{'='*60}\n")

    # Patch the client and every directly imported alias. This proves that
    # paper mode performs zero LLM calls rather than merely surviving failures.
    llm_targets = []
    if {"A2", "A3"}.intersection(configs):
        llm_targets.extend([
            "agents.llm_client.call_groq",
            "agents.intake_agent.call_groq",
            "agents.risk_agent.call_groq",
            "agents.fairness_agent.call_groq",
            "agents.orchestrator_agent.call_groq",
            "agents.reporting_agent.call_groq",
    ])
    with ExitStack() as llm_stack:
        for target in llm_targets:
            importlib.import_module(target.rsplit(".", 1)[0])
            llm_stack.enter_context(patch(target, side_effect=_llm_sentinel))

        dataset_path = os.path.join(os.path.dirname(__file__), "..", "research", "data", "credit_risk_dataset.csv")
        df = pd.read_csv(dataset_path)
        dataset_hash = hashlib.sha256(open(dataset_path, "rb").read()).hexdigest()

        FEATURES = ["credit_score","dti_ratio","loan_to_income","employment_score",
                     "applicant_age","defaults","credit_utilisation","applicant_gender"]

        all_traces, all_metrics, failure_logs, telemetry = [], [], [], []

        for seed in seeds:
            print(f"[Seed {seed}] Training RF (n_estimators=100, random_state={seed})...")
            X = df[FEATURES]; y = df["target_default"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_split, random_state=seed)

            print(f"  Train: {len(X_train)} | Test: {len(X_test)} | Default prevalence (test): {y_test.mean():.4f}")

            rf = RandomForestClassifier(n_estimators=100, random_state=seed)
            rf.fit(X_train, y_train)
            model_hash = hashlib.sha256(pickle.dumps(rf, protocol=5)).hexdigest()

            test_indices = X_test.index
            df_test = df.loc[test_indices].copy()
            if args.dry_run:
                df_test = df_test.head(50)
                print(f"  Dry run: trimmed to {len(df_test)} samples")

            runners = {"C0a": _run_c0a, "A0": _run_a0, "A1": _run_a1, "A1R": _run_a1r, "A2": _run_a2, "A3": _run_a3}

            for config in configs:
                print(f"  Config {config}...", end=" ", flush=True)
                t0 = time.time()
                traces = []
                n_fail = 0
                for idx, row in df_test.iterrows():
                    if config in {"C0a", "A0"}:
                        res, err = runners[config](row, rf, model_hash)
                    else:
                        res, err = runners[config](row, rf, model_hash, args.epsilon)
                    if err:
                        n_fail += 1
                        failure_logs.append({
                            "seed": seed, "configuration": config,
                            "applicant_id": int(row.get("id", idx)),
                            "node_name": "canonical_features",
                            "error_type": err["error_type"],
                            "error_message": err["error_message"],
                            "recovery_action": "abort",
                        })
                        continue
                    trace = {
                        "seed": seed, "configuration": config, "split": "test",
                        "applicant_id": int(row.get("id", idx)),
                        "ground_truth": int(row["target_default"]),
                        "posthoc_policy_passed": bool(canonical_policy_check(row.to_dict())[0]),
                        "correct": (res["final_decision"] == "approved") == (row["target_default"] == 0),
                        **res,
                    }
                    traces.append(trace)

                elapsed = (time.time() - t0) * 1000
                print(f"{len(traces)} ok, {n_fail} fail, {elapsed:.0f}ms")
                all_traces.extend(traces)

                if traces:
                    df_tr = pd.DataFrame(traces)
                    all_metrics.append(_compute_metrics(df_tr, seed, config))

                telemetry.append({
                    "seed": seed, "configuration": config,
                    "n_evaluated": len(traces), "n_failed": n_fail,
                    "total_ms": round(elapsed, 1),
                    "mean_latency_ms": round(sum(t["latency_ms"] for t in traces)/max(len(traces),1), 3),
                })

        # ── Save Outputs ───────────────────────────────────────
        code_hash = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
        manifest = {
            "experiment_id": f"ablation_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "created_at": datetime.now().isoformat(),
            "paper_mode": True,
            "llm_calls_attempted": _llm_call_count,
            "configurations": configs,
            "seeds": seeds,
            "test_split": args.test_split,
            "fairness_epsilon": args.epsilon,
            "rf_params": {"n_estimators": 100, "random_state": "per-seed"},
            "dataset_columns": FEATURES,
            "canonical_internal_fields": ["credit_score","dti_ratio","loan_to_income",
                "employment_score","age","defaults","credit_utilisation","gender_encoded"],
            "dataset_sha256": dataset_hash,
            "code_sha256": code_hash,
            "threshold": 0.5,
            "final_binary_mapping": "only committed_approve → approved; all other terminal states → rejected",
            "git_commit": os.popen("git rev-parse HEAD 2>/dev/null").read().strip(),
            "python_version": sys.version.split()[0],
            "dry_run": args.dry_run,
        }

        manifest_name = f"manifest_{manifest['experiment_id']}.json"
        with open(os.path.join(MANIFEST_DIR, manifest_name), "w") as f:
            json.dump(manifest, f, indent=2)
        with open(os.path.join(results_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        frozen_config = {
            "seeds": seeds,
            "configurations": configs,
            "test_split": args.test_split,
            "epsilon": args.epsilon,
            "random_forest": {"n_estimators": 100, "random_state": "seed"},
            "split_strategy": "unstratified train_test_split with random_state=seed",
            "approval_positive_metrics": ["precision_approve", "recall_approve", "f1_approve"],
            "default_positive_ranking_metrics": ["roc_auc_default", "pr_auc_default"],
        }
        config_path = os.path.join(CONFIG_DIR, "ablation_config.json") if frozen_final else os.path.join(results_dir, "run_config.json")
        with open(config_path, "w") as f:
            json.dump(frozen_config, f, indent=2)
        if all_traces:
            traces_df = pd.DataFrame(all_traces)
            traces_df.to_csv(os.path.join(results_dir, "raw_applicant_traces.csv"), index=False)
            trace_dir = os.path.join(results_dir, "traces")
            os.makedirs(trace_dir, exist_ok=True)
            for (config, seed), group in traces_df.groupby(["configuration", "seed"], sort=False):
                group.to_csv(
                    os.path.join(trace_dir, f"raw_applicant_traces_{config}_{seed}.csv"),
                    index=False,
                )
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            metrics_df.to_csv(os.path.join(results_dir, "metrics_per_seed.csv"), index=False)
            _aggregate_metrics(metrics_df).to_csv(
                os.path.join(results_dir, "aggregate_metrics.csv"), index=False
            )
            _paired_comparisons(metrics_df).to_csv(
                os.path.join(results_dir, "paired_comparisons.csv"), index=False
            )
        if failure_logs:
            pd.DataFrame(failure_logs).to_csv(os.path.join(results_dir, "failure_log.csv"), index=False)
        else:
            pd.DataFrame(columns=["seed","configuration","applicant_id","node_name",
                                   "error_type","error_message","recovery_action"]).to_csv(
                os.path.join(results_dir, "failure_log.csv"), index=False)
        pd.DataFrame(telemetry).to_csv(os.path.join(results_dir, "telemetry.csv"), index=False)

        # ── Summary ────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"  LLM calls attempted: {_llm_call_count}")
        print(f"  Failures logged: {len(failure_logs)}")
        for m in all_metrics:
            print(f"\n  [{m['configuration']}] seed={m['seed']}")
            print(f"    n={m['n_test']}  approve={m['n_approve']}  reject={m['n_reject']}")
            print(f"    accuracy={m['accuracy']:.4f}  prec={m['precision_approve']:.4f}  rec={m['recall_approve']:.4f}  f1={m['f1_approve']:.4f}")
            print(f"    specificity={m['specificity']:.4f}  mcc={m['mcc']:.4f}")
            print(f"    roc_auc_default={m['roc_auc_default']:.4f}  pr_auc_default={m['pr_auc_default']:.4f}")
            print(
                f"    policy_violations={m['policy_violation_count']}  "
                f"audit_triggers={m['audit_executed_count']}  "
                f"temporary_blocked_applicants={m['commitment_block_applicant_count']}  "
                f"terminal_blocks={m['terminal_commitment_block_count']}"
            )
        print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
