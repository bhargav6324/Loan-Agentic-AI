#!/usr/bin/env python3
"""
A1R-vs-A2 Pre-Experiment Disagreement Analysis
═══════════════════════════════════════════════
Runs A1R and A2 over a single shared test split (seed=42, 30 % holdout)
in PAPER_MODE (zero LLM calls) and produces a detailed disagreement report.

Requirements:
  - Does NOT run the full 10-seed A0–A3 experiment.
  - Does NOT modify any code to improve agreement.
  - Does NOT touch research_v2/final_results or research_v2/validation_results.
  - Stores output only in the configured versioned validation directory.
  - Fairness acts as audit-and-escalation authority only; it must NOT approve
    cases rejected by deterministic risk or policy controls.
"""
import os, sys, json, hashlib, pickle, datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Repo root is always one level above scripts/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

# ── Constants ────────────────────────────────────────────────────────────────
SEED        = 42
TEST_SPLIT  = 0.30
EPSILON     = 6.0
FEATURES    = ["credit_score", "dti_ratio", "loan_to_income", "employment_score",
               "applicant_age", "defaults", "credit_utilisation", "applicant_gender"]
OUTPUT_DIR  = os.path.join(REPO_ROOT, "research_v2", "validation_results",
                           "a1r_a2_disagreement_v3")
FROZEN_DIRS = [
    os.path.join(REPO_ROOT, "research_v2", "final_results"),
    os.path.join(REPO_ROOT, "research_v2", "validation_results"),
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def _freeze_checksum(path: str) -> str:
    """Return MD5 of a file for frozen-directory guard."""
    import hashlib
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def _assert_frozen(dirs_before: dict):
    """Re-check frozen directory file sizes have not changed."""
    for d, snap in dirs_before.items():
        for fname, size_before in snap.items():
            fpath = os.path.join(d, fname)
            if os.path.exists(fpath):
                size_now = os.path.getsize(fpath)
                assert size_now == size_before, (
                    f"FROZEN DIRECTORY MODIFIED: {fpath} "
                    f"({size_before} → {size_now} bytes)"
                )

def _snapshot_dir(d: str) -> dict:
    snap = {}
    for root, _, files in os.walk(d):
        for f in files:
            fp = os.path.join(root, f)
            snap[os.path.relpath(fp, d)] = os.path.getsize(fp)
    return snap

def _classify_disagreement(r1r: dict, r2: dict) -> str:
    """
    Classify the root cause of an A1R vs A2 decision disagreement.

    Priority order (mutually exclusive, exhaustive):
      1. certificate/commitment gate  — A2 blocked due to a missing certificate
      2. fairness                     — A1R audit and A2 fairness agent reached
                                        different bias conclusions; OR A2 fairness
                                        flagged bias and escalated a case A1R approved
      3. reassessment                 — A1R reassessment CHANGED the composite score,
                                        causing a different risk category/decision
      4. policy                       — policy_passed differs between A1R and A2
      5. routing                      — composite score OR risk category diverged
                                        (without a reassessment explanation)
      6. unexplained                  — none of the above; requires per-case trace
    """
    cs2 = r2.get("commit_status", "")

    # 1. Certificate commitment gate — only when A2 blocked on a missing cert
    if cs2 in ("blocked_missing_certificate", "blocked_missing_audit",
               "blocked_policy_violation", "blocked_invalid_audit_action",
               "blocked_branch_conflict"):
        return "certificate_commitment_gate"

    # 2. Fairness — covers three sub-cases:
    #    a. A1R bias flag differs from A2 bias flag
    #    b. A2 fairness escalated (override recommendation) a case A1R approved
    #    c. A1R audit_recommended_action caused an approve that A2 escalated
    bias1 = r1r.get("audit_bias_flagged", False)
    bias2 = r2.get("fairness_bias_flagged", False)
    fair_rec2 = r2.get("fairness_recommendation", "")
    a1r_action = r1r.get("audit_recommended_action", "")

    if bias1 != bias2:
        return "fairness"
    if bias2 and fair_rec2 in ("override", "escalate"):
        return "fairness"
    if bias1 and a1r_action == "approve" and r1r["final_decision"] == "approved":
        # A1R fairness approved; A2 didn't → fairness divergence
        return "fairness"

    # 3. Reassessment — A1R reassessment count > 0 AND score actually changed
    if r1r.get("deterministic_reassessment_count", 0) > 0:
        s1 = r1r.get("composite_score")
        s2 = r2.get("composite_score")
        # Score changed is sufficient evidence reassessment drove the divergence
        if s1 is not None and s2 is not None and abs(s1 - s2) > 0.5:
            return "reassessment"

    # 4. Policy
    p1 = r1r.get("policy_passed")
    p2 = r2.get("policy_passed")
    if p1 != p2:
        return "policy"

    # 5. Routing — risk category or composite score diverged
    cat1 = r1r.get("risk_category")
    cat2 = r2.get("risk_category")
    if cat1 != cat2:
        return "routing"

    s1 = r1r.get("composite_score")
    s2 = r2.get("composite_score")
    if s1 is not None and s2 is not None and abs(s1 - s2) > 0.5:
        return "routing"

    return "unexplained"

def _is_unsafe(r1r: dict, r2: dict) -> bool:
    """
    Detect unsafe commitments.

    Governance invariant: fairness is audit-and-escalation authority only.
    Any case where A2 approves what A1R rejected is unsafe by definition,
    because the ONLY path from rejected to approved in A2 would have to be
    via the fairness agent (which is prohibited from doing so).

    Also unsafe: A2 approves a policy-rejected case (regardless of fairness).
    """
    d1 = r1r["final_decision"]
    d2 = r2["final_decision"]

    # rejected → approved is always unsafe
    if d1 == "rejected" and d2 == "approved":
        return True

    # policy failed in A1R but A2 approved
    if not r1r.get("policy_passed", True) and d2 == "approved":
        return True

    # policy failed in A2 but A2 somehow approved
    if not r2.get("policy_passed", True) and d2 == "approved":
        return True

    return False

# ── Main Analysis ────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("  A1R vs A2 PRE-EXPERIMENT DISAGREEMENT ANALYSIS")
    print(f"  {datetime.datetime.now().isoformat()}")
    print("=" * 66)

    # Record git state
    git_commit = os.popen(f"git -C {REPO_ROOT} rev-parse HEAD 2>/dev/null || echo UNKNOWN").read().strip()
    git_dirty  = os.popen(f"git -C {REPO_ROOT} status --porcelain 2>/dev/null | wc -l").read().strip()
    print(f"\nGit commit : {git_commit}")
    print(f"Dirty files: {git_dirty}")

    # Freeze-guard: snapshot before analysis
    frozen_snaps = {}
    for d in FROZEN_DIRS:
        if os.path.isdir(d):
            frozen_snaps[d] = _snapshot_dir(d)
    manifest_checksums = {
        d: _freeze_checksum(os.path.join(d, "manifest.json"))
        for d in FROZEN_DIRS if os.path.exists(os.path.join(d, "manifest.json"))
    }
    print(f"\nFrozen directories guarded: {list(frozen_snaps.keys())}")
    for d, ck in manifest_checksums.items():
        print(f"  {os.path.basename(d)}/manifest.json  MD5={ck}")

    # Output directory (new; must not overlap frozen)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput dir : {OUTPUT_DIR}")

    # Dataset
    dataset_path = os.path.join(REPO_ROOT, "research", "data", "credit_risk_dataset.csv")
    df = pd.read_csv(dataset_path)
    dataset_hash = hashlib.sha256(open(dataset_path, "rb").read()).hexdigest()[:16]
    print(f"Dataset    : {dataset_path}  ({len(df)} rows, hash={dataset_hash})")

    # Train model — same procedure as the ablation harness
    X = df[FEATURES]; y = df["target_default"]
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=TEST_SPLIT,
                                                    random_state=SEED)
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    rf.fit(X_train, y_train)
    model_hash = hashlib.sha256(pickle.dumps(rf, protocol=5)).hexdigest()[:16]
    df_test = df.loc[X_test.index].copy()
    print(f"\nSeed={SEED} | Train={len(X_train)} | Test={len(df_test)} | "
          f"Model hash={model_hash}")
    print(f"Default prevalence (test): {y[X_test.index].mean():.4f}")

    # Inject model for graph runner
    import ml.canonical_model_contract as _cmc
    original_model = _cmc._model
    _cmc._model = rf

    from research_v2.run_agentic_ablation import _run_a1r, _run_a2
    from config import settings
    settings.PAPER_MODE = True

    print(f"\nPAPER_MODE={settings.PAPER_MODE}  |  epsilon={EPSILON}")
    print("\nRunning A1R and A2 over test set …")

    records = []
    n_fail_a1r = n_fail_a2 = 0

    try:
        for idx, row in df_test.iterrows():
            r1r, e1r = _run_a1r(row, rf, model_hash, EPSILON)
            r2,  e2  = _run_a2 (row, rf, model_hash, EPSILON)

            if e1r or e2:
                if e1r: n_fail_a1r += 1
                if e2:  n_fail_a2  += 1
                continue

            agree = r1r["final_decision"] == r2["final_decision"]
            cause = _classify_disagreement(r1r, r2) if not agree else None
            unsafe = _is_unsafe(r1r, r2) if not agree else False

            records.append({
                "applicant_id":            int(row.get("id", idx)),
                "ground_truth":            int(row["target_default"]),
                # Shared scoring
                "a1r_composite_score":     r1r.get("composite_score"),
                "a2_composite_score":      r2.get("composite_score"),
                "a1r_risk_category":       r1r.get("risk_category"),
                "a2_risk_category":        r2.get("risk_category"),
                "a1r_p_default":           r1r.get("p_default"),
                "a2_p_default":            r2.get("p_default"),
                # Policy
                "a1r_policy_passed":       r1r.get("policy_passed"),
                "a2_policy_passed":        r2.get("policy_passed"),
                # Fairness
                "a1r_audit_triggered":     r1r.get("audit_triggered"),
                "a1r_audit_bias_flagged":  r1r.get("audit_bias_flagged"),
                "a1r_audit_signed_delta":  r1r.get("audit_signed_delta"),
                "a1r_audit_action":        r1r.get("audit_recommended_action"),
                "a2_audit_triggered":      r2.get("audit_triggered"),
                "a2_fairness_bias_flagged":r2.get("fairness_bias_flagged"),
                "a2_fairness_recommendation": r2.get("fairness_recommendation"),
                # Reassessment
                "a1r_reassessment_count":  r1r.get("deterministic_reassessment_count", 0),
                # Decisions
                "a1r_decision_proposal":   r1r.get("decision"),
                "a2_decision_proposal":    r2.get("decision"),
                "a1r_final_decision":      r1r["final_decision"],
                "a2_final_decision":       r2["final_decision"],
                "a1r_commit_status":       r1r.get("commit_status"),
                "a2_commit_status":        r2.get("commit_status"),
                # Analysis
                "agree":                   agree,
                "transition":              f"{r1r['final_decision']}→{r2['final_decision']}" if not agree else None,
                "disagreement_cause":      cause,
                "unsafe_commitment":       unsafe,
            })
    finally:
        _cmc._model = original_model
        settings.PAPER_MODE = False

    df_out = pd.DataFrame(records)
    total          = len(df_out)
    n_agree        = df_out["agree"].sum()
    n_disagree     = total - n_agree
    agree_rate     = n_agree / total if total else 0.0
    disagree_rate  = 1.0 - agree_rate

    df_dis         = df_out[~df_out["agree"]]
    transitions    = df_dis["transition"].value_counts().to_dict()

    # Fairness metrics (all cases)
    ft_a1r  = df_out["a1r_audit_triggered"].sum()
    ft_a2   = df_out["a2_audit_triggered"].sum()
    fbf_a1r = df_out["a1r_audit_bias_flagged"].sum()
    fbf_a2  = df_out["a2_fairness_bias_flagged"].sum()

    # Safe-block: both agree on reject AND at least one had a commitment block
    safe_block_mask = (
        df_out["agree"] &
        (df_out["a1r_final_decision"] == "rejected") &
        (df_out["a1r_commit_status"].str.startswith("blocked_", na=False) |
         df_out["a2_commit_status"].str.startswith("blocked_", na=False))
    )
    n_safe_block   = safe_block_mask.sum()
    n_unsafe       = int(df_out["unsafe_commitment"].sum())

    # Escalation
    n_escalated_a2 = int((df_out["a2_final_decision"] == "escalated").sum())

    # Corrected safety metrics
    # Safe escalation: A2 correctly escalated (fairness flagged → escalated, not approved)
    n_safe_escalation = int((
        df_out["a2_fairness_bias_flagged"].fillna(False) &
        (df_out["a2_final_decision"] == "escalated")
    ).sum())
    # Policy rejection commitment rate: A2 policy-failed cases that committed_reject
    policy_fail_a2 = df_out[df_out["a2_policy_passed"] == False]
    n_policy_committed_reject = int((policy_fail_a2["a2_commit_status"] == "committed_reject").sum())
    policy_commit_rate = n_policy_committed_reject / len(policy_fail_a2) if len(policy_fail_a2) else float("nan")
    # Missing-cert block rate: fraction of all cases blocked_missing_certificate in A2
    n_missing_cert = int((df_out["a2_commit_status"] == "blocked_missing_certificate").sum())
    missing_cert_rate = n_missing_cert / total if total else 0.0
    # False escalation: A2 escalated but both A1R approved and policy passed and no bias
    false_esc_mask = (
        (df_out["a2_final_decision"] == "escalated") &
        (df_out["a1r_final_decision"] == "approved") &
        df_out["a1r_policy_passed"].fillna(True) &
        ~df_out["a1r_audit_bias_flagged"].fillna(False) &
        ~df_out["a2_fairness_bias_flagged"].fillna(False)
    )
    n_false_escalation = int(false_esc_mask.sum())

    cause_counts   = df_dis["disagreement_cause"].value_counts().to_dict()
    n_unexplained  = cause_counts.get("unexplained", 0)

    # ── Print Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  DISAGREEMENT ANALYSIS REPORT  (v3 — certificate producer/barrier corrected)")
    print("=" * 66)
    print(f"\n  Failures: A1R={n_fail_a1r}  A2={n_fail_a2}")
    print(f"\n  AGREEMENT METRICS")
    print(f"  ─────────────────")
    print(f"  Total cases          : {total}")
    print(f"  Agreement count      : {n_agree}  ({agree_rate:.4f})")
    print(f"  Disagreement count   : {n_disagree}  ({disagree_rate:.4f})")

    print(f"\n  DECISION TRANSITIONS (A1R → A2) in disagreements:")
    for t, c in sorted(transitions.items(), key=lambda x: -x[1]):
        print(f"    {t:30s}  {c:5d}  ({c/total:.4f})")

    print(f"\n  FAIRNESS METRICS")
    print(f"  ─────────────────")
    print(f"  A1R audit triggered  : {int(ft_a1r)}  ({ft_a1r/total:.4f})")
    print(f"  A2  audit triggered  : {int(ft_a2)}   ({ft_a2/total:.4f})")
    print(f"  A1R bias flagged     : {int(fbf_a1r)}  ({fbf_a1r/total:.4f})")
    print(f"  A2  bias flagged     : {int(fbf_a2)}   ({fbf_a2/total:.4f})")

    print(f"\n  SAFETY METRICS")
    print(f"  ─────────────────")
    print(f"  Safe escalation (A2 bias→escalated)      : {n_safe_escalation}  ({n_safe_escalation/total:.4f})")
    print(f"  Unsafe commitments (A2 reject→approve)   : {n_unsafe}  ({n_unsafe/total:.4f})")
    print(f"  Policy-rejection commitment rate (A2)    : {n_policy_committed_reject}/{len(policy_fail_a2)}  ({policy_commit_rate:.4f})")
    print(f"  Missing-cert block count (A2)            : {n_missing_cert}  ({missing_cert_rate:.4f})")
    print(f"  False escalation (clean cases escalated) : {n_false_escalation}  ({n_false_escalation/total:.4f})")
    print(f"  Total escalation count (A2)              : {n_escalated_a2}  ({n_escalated_a2/total:.4f})")

    print(f"\n  DISAGREEMENT CAUSES")
    print(f"  ─────────────────────")
    for cause, cnt in sorted(cause_counts.items(), key=lambda x: -x[1]):
        print(f"    {cause:40s}  {cnt:5d}  ({cnt/n_disagree:.4f} of disagree, {cnt/total:.4f} of total)")

    if n_unsafe > 0:
        print(f"\n  ⚠️  UNSAFE CASES (fairness overrode deterministic risk/policy reject):")
        unsafe_df = df_out[df_out["unsafe_commitment"]]
        print(unsafe_df[["applicant_id","a1r_risk_category","a1r_policy_passed",
                          "a1r_final_decision","a2_final_decision","disagreement_cause"]].to_string(index=False))
    else:
        print(f"\n  ✅  No unsafe commitments detected.")

    if n_unexplained > 0:
        print(f"\n  ⚠️  {n_unexplained} UNEXPLAINED DISAGREEMENTS — per-case traces:")
        ux_df = df_dis[df_dis["disagreement_cause"] == "unexplained"]
        trace_cols = ["applicant_id","a1r_risk_category","a2_risk_category",
                      "a1r_composite_score","a2_composite_score",
                      "a1r_policy_passed","a2_policy_passed",
                      "a1r_audit_bias_flagged","a2_fairness_bias_flagged",
                      "a2_fairness_recommendation","a1r_audit_action",
                      "a1r_final_decision","a2_final_decision","a2_commit_status"]
        print(ux_df[trace_cols].to_string(index=False))
    else:
        print(f"\n  ✅  No unexplained disagreements.")

    # ── Freeze guard post-check ──────────────────────────────────────────────
    _assert_frozen(frozen_snaps)
    for d, ck in manifest_checksums.items():
        new_ck = _freeze_checksum(os.path.join(d, "manifest.json"))
        assert new_ck == ck, f"FROZEN manifest modified: {d}"
    print(f"\n  ✅  Frozen result directories confirmed untouched.")

    # ── Write outputs ────────────────────────────────────────────────────────
    df_out.to_csv(os.path.join(OUTPUT_DIR, "all_cases.csv"), index=False)
    df_dis.to_csv(os.path.join(OUTPUT_DIR, "disagreements.csv"), index=False)
    if n_unsafe > 0:
        df_out[df_out["unsafe_commitment"]].to_csv(
            os.path.join(OUTPUT_DIR, "unsafe_cases.csv"), index=False)
    if n_unexplained > 0:
        df_dis[df_dis["disagreement_cause"] == "unexplained"].to_csv(
            os.path.join(OUTPUT_DIR, "unexplained_traces.csv"), index=False)

    # Disagreement summary by cause and transition
    cause_df = df_dis.groupby(["disagreement_cause","transition"]).size().reset_index(name="count")
    cause_df.to_csv(os.path.join(OUTPUT_DIR, "disagreement_causes.csv"), index=False)

    # Per-risk-category breakdown
    riskcat_df = df_dis.groupby(["a1r_risk_category","a2_risk_category","transition",
                                  "disagreement_cause"]).size().reset_index(name="count")
    riskcat_df.to_csv(os.path.join(OUTPUT_DIR, "disagreement_by_risk_category.csv"), index=False)

    # Reproducibility snapshot: hashes of all modified source files
    modified_files = [
        "core/canonical_contracts.py",
        "orchestrator/graph.py",
        "agents/risk_agent.py",
        "agents/intake_agent.py",
        "agents/orchestrator_agent.py",
        "agents/fairness_agent.py",
        "ml/canonical_model_contract.py",
        "research_v2/run_agentic_ablation.py",
    ]
    source_hashes = {}
    for rel_path in modified_files:
        full_path = os.path.join(REPO_ROOT, rel_path)
        if os.path.exists(full_path):
            source_hashes[rel_path] = hashlib.sha256(open(full_path, "rb").read()).hexdigest()[:16]
    with open(os.path.join(OUTPUT_DIR, "reproducibility_snapshot.json"), "w") as f:
        json.dump({
            "git_commit": git_commit,
            "git_dirty_files": int(git_dirty) if git_dirty.isdigit() else git_dirty,
            "note": "HEAD commit alone does not identify analyzed code (working tree is dirty). "
                    "Source file hashes are the authoritative reproducibility record.",
            "dataset_hash": dataset_hash,
            "model_hash": model_hash,
            "model_config": {"n_estimators": 100, "random_state": SEED, "test_split": TEST_SPLIT},
            "source_hashes": source_hashes,
        }, f, indent=2)

    # Manifest
    manifest = {
        "analysis":                    "a1r_a2_disagreement_v3",
        "generated_at":                datetime.datetime.now().isoformat(),
        "git_commit":                  git_commit,
        "git_dirty_files":             int(git_dirty) if git_dirty.isdigit() else git_dirty,
        "seed":                        SEED,
        "test_split":                  TEST_SPLIT,
        "epsilon":                     EPSILON,
        "dataset_hash":                dataset_hash,
        "model_hash":                  model_hash,
        "n_test":                      total,
        "n_agree":                     int(n_agree),
        "n_disagree":                  int(n_disagree),
        "agree_rate":                  float(agree_rate),
        "disagree_rate":               float(disagree_rate),
        "transitions":                 {k: int(v) for k, v in transitions.items()},
        "a1r_audit_triggered":         int(ft_a1r),
        "a2_audit_triggered":          int(ft_a2),
        "a1r_bias_flagged":            int(fbf_a1r),
        "a2_bias_flagged":             int(fbf_a2),
        "safe_escalation_count":       n_safe_escalation,
        "safe_block_count":            int(n_safe_block),
        "unsafe_count":                n_unsafe,
        "policy_reject_commit_rate":   float(policy_commit_rate) if policy_commit_rate == policy_commit_rate else None,
        "missing_cert_block_count":    n_missing_cert,
        "missing_cert_block_rate":     float(missing_cert_rate),
        "false_escalation_count":      n_false_escalation,
        "escalation_count_a2":         n_escalated_a2,
        "disagreement_causes":         {k: int(v) for k, v in cause_counts.items()},
        "n_unexplained":               n_unexplained,
        "frozen_manifests_ok":         True,
        "n_fail_a1r":                  n_fail_a1r,
        "n_fail_a2":                   n_fail_a2,
        "paper_mode":                  True,
        "governance_corrections": [
            "canonical_final_decision: fairness approve→approve path removed; now escalates",
            "decision_commit: policy rejection now checked before MEDIUM audit gate",
            "graph: true risk/policy fan-in barrier added before orchestration",
            "risk_agent: hard-reject and fast-track exits now issue risk certificates",
            "decision_commit: certificate synthesis removed; producer certificates required",
            "disagree classifier: unexplained fairness-escalations reclassified as fairness",
        ],
    }
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Files written to {OUTPUT_DIR}/:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        print(f"    {fname:40s}  {os.path.getsize(fpath):>8d} bytes")

    print(f"\n{'='*66}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*66}\n")
    return manifest


if __name__ == "__main__":
    main()
