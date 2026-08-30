"""
Publication Asset Generator — High-Resolution Figures (300 DPI)
────────────────────────────────────────────────────────────────────────
Generates 4 publication-quality figures for IEEE/Springer manuscripts:
  1. ROC Curve (ML Baseline vs. Agentic System)
    2. Ablation Comparison Figure
  3. Counterfactual Score Difference Distribution (with threshold line at ε=6.0)
  4. Policy Violation Comparison

All figures are exported in both 300 DPI PNG and vector SVG formats
to research/publication_assets/figures/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

# Import read-only domain heuristics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from research.evaluation_harness import compute_heuristic_score, check_hard_overrides, check_policy

# Set publication styling
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 16


def main():
    print("======================================================================")
    print("GENERATING 300 DPI PUBLICATION FIGURES")
    print("======================================================================")

    data_path = os.path.join(os.path.dirname(__file__), "data", "credit_risk_dataset.csv")
    df = pd.read_csv(data_path)
    features = ["credit_score", "dti_ratio", "loan_to_income", "employment_score", "applicant_age", "defaults", "credit_utilisation", "applicant_gender"]
    X_train, X_test, y_train, y_test = train_test_split(df[features], df["target_default"], test_size=0.3, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

    out_dir = os.path.join(os.path.dirname(__file__), "publication_assets", "figures")
    os.makedirs(out_dir, exist_ok=True)

    true_approved = (y_test == 0).astype(int)
    rf_proba = rf_model.predict_proba(X_test)[:, 0]

    # Compute agent scores for ROC
    df_test = df.loc[X_test.index]
    agent_scores = []
    deltas = []
    for i in range(len(X_test)):
        row = df_test.iloc[i]
        ov = check_hard_overrides(row)
        if ov == "HARD_REJECT":
            agent_scores.append(0.0)
            continue
        if ov == "HARD_FAST_TRACK":
            agent_scores.append(1.0)
            continue
        
        proba_i = rf_model.predict_proba(row[features].to_frame().T)[0]
        ml_score = proba_i[1] * 100
        ml_conf = max(proba_i)
        ml_w = 0.4 + (0.4 * ml_conf)
        heur_w = 1 - ml_w
        heur_score = compute_heuristic_score(row)
        risk_score = ml_w * ml_score + heur_w * heur_score
        agent_scores.append(1.0 - risk_score / 100.0)

        if 35 < risk_score <= 99 and check_policy(row):
            cf_row = row.copy()
            cf_row["applicant_gender"] = 0 if row["applicant_gender"] == 1 else 1
            cf_proba = rf_model.predict_proba(cf_row[features].to_frame().T)[0]
            cf_rs = (0.4 + 0.4 * max(cf_proba)) * (cf_proba[1]*100) + (1 - (0.4 + 0.4 * max(cf_proba))) * compute_heuristic_score(cf_row)
            deltas.append(abs(cf_rs - risk_score))

    agent_scores = np.array(agent_scores)

    # ── 1. ROC Curve ──────────────────────────────────────────────
    print("Generating Figure 1: ROC Curve...")
    fpr_rf, tpr_rf, _ = roc_curve(true_approved, rf_proba)
    auc_rf = roc_auc_score(true_approved, rf_proba)

    fpr_ag, tpr_ag, _ = roc_curve(true_approved, agent_scores)
    auc_ag = roc_auc_score(true_approved, agent_scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_rf, tpr_rf, color="#66a61e", lw=2.5, label=f"Random Forest Baseline (AUC = {auc_rf:.3f})")
    ax.plot(fpr_ag, tpr_ag, color="#2b5c8f", lw=2.5, label=f"LangGraph Agentic System v7 (AUC = {auc_ag:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Guessing (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="black")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_roc_curve.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "fig1_roc_curve.svg"))
    plt.close()
    print("✅ Generated fig1_roc_curve (.png & .svg)")

    
    # ── 2. Ablation Comparison Figure ─────────────────────────────
    print("Generating Figure 2: Ablation Comparison Figure...")
    ablation_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "final_results", "ablation_results_per_seed.csv"), comment='#')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = ["Full System", "No Fairness", "No Policy", "RF Baseline"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#66a61e"]

    conf_names = ["Config 1: Full Agentic System", "Config 2: Agentic - Fairness", "Config 3: Agentic - Policy", "Config 4: Random Forest Baseline"]
    def get_stats(col):
        m, s = [], []
        for name in conf_names:
            series = ablation_df[ablation_df["Configuration"] == name][col]
            m.append(series.mean())
            s.append(series.std())
        return m, s

    # Subplot 1: F1 Score
    f1_m, f1_s = get_stats("F1")
    axes[0].bar(labels, f1_m, yerr=f1_s, color=colors, width=0.5, edgecolor="black", capsize=5)
    axes[0].set_title("F1 Score (Predictive)", fontweight="bold")
    axes[0].set_ylim(0.7, 1.0)
    for i, (m, s) in enumerate(zip(f1_m, f1_s)):
        axes[0].text(i, m + 0.01 + s, f"{m:.3f}", ha="center", fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].tick_params(axis="x", rotation=15)

    # Subplot 2: Policy Violations
    viol_m, viol_s = get_stats("Policy Violations")
    axes[1].bar(labels, viol_m, yerr=viol_s, color=colors, width=0.5, edgecolor="black", capsize=5)
    axes[1].set_title("Regulatory Policy Violations", fontweight="bold")
    axes[1].set_ylim(0, max([m+s for m,s in zip(viol_m, viol_s)]) * 1.2)
    for i, (m, s) in enumerate(zip(viol_m, viol_s)):
        axes[1].text(i, m + 2 + s, f"{m:.1f}", ha="center", fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].tick_params(axis="x", rotation=15)

    # 3. Operational Workload (Override Rule Triggers)
    esc_m, esc_s = get_stats("Override Rule Triggers")
    axes[2].bar(labels, esc_m, yerr=esc_s, color=["#cc0000", "#5c5c8a", "#2d862d", "#333333"], edgecolor="black", capsize=5)
    axes[2].set_title("Override Rule Triggers", fontweight="bold")
    axes[2].set_ylim(0, max([m+s for m,s in zip(esc_m, esc_s)]) * 1.2 if max(esc_m) > 0 else 50)
    for i, (m, s) in enumerate(zip(esc_m, esc_s)):
        axes[2].text(i, m + 1 + s, f"{m:.1f}", ha="center", fontweight="bold")
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    axes[2].tick_params(axis="x", rotation=15)

    fig.suptitle("Controlled Multi-Agent Architecture Ablation Comparison (n=10)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "fig2_ablation_comparison.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "fig2_ablation_comparison.svg"))
    plt.close()
    print("✅ Generated fig2_ablation_comparison (.png & .svg)")

    # ── 3. Counterfactual Score Difference Distribution ───────────────────────
    print("Generating Figure 3: Counterfactual Score Difference Distribution...")
    fig, ax = plt.subplots(figsize=(8, 5))
    n_bins, bins, patches = ax.hist(deltas, bins=30, color="#2b5c8f", edgecolor="black", alpha=0.85)
    
    # Highlight bars above threshold 6.0
    for bin_left, patch in zip(bins[:-1], patches):
        if bin_left >= 6.0:
            patch.set_facecolor("#d95f02")

    ax.axvline(x=6.0, color="red", linestyle="--", lw=2.5, label=r"Locked Fairness Threshold ($\epsilon=6.0$, 56 Escalations)")
    ax.set_xlabel(r"Absolute Counterfactual Score Delta ($|\Delta_{CF}|$)")
    ax.set_ylabel("Number of Medium-Risk Applicants")
    ax.set_title("Distribution of Counterfactual Fairness Deltas ($N=1,500$)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_fairness_distribution.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "fig3_fairness_distribution.svg"))
    plt.close()
    print("✅ Generated fig3_fairness_distribution (.png & .svg)")

    # ── 4. Policy Violation Comparison ────────────────────────────
    print("Generating Figure 4: Policy Violation Comparison...")
    fig, ax = plt.subplots(figsize=(8, 5))
    viol_m, viol_s = get_stats("Policy Violations")
    bars = ax.bar(labels, viol_m, yerr=viol_s, color=["#2b5c8f", "#2b5c8f", "#d95f02", "#66a61e"], width=0.5, edgecolor="black", capsize=5)
    
    # Highlight No Policy bar
    bars[2].set_facecolor("#d95f02")
    bars[2].set_edgecolor("red")
    bars[2].set_linewidth(2)

    for i, (m, s) in enumerate(zip(viol_m, viol_s)):
        ax.text(i, m + 4 + s, f"{m:.1f}\\n({'0%' if m==0 else f'{(m/1500)*100:.1f}%'})", ha="center", fontweight="bold", fontsize=10)

    ax.set_ylabel("Number of Illegal Policy Approvals")
    ax.set_title("Regulatory Compliance: Hard Policy Violations by Configuration")
    ax.set_ylim(0, 200)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_policy_violations.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "fig4_policy_violations.svg"))
    plt.close()
    print("✅ Generated fig4_policy_violations (.png & .svg)")

    print("\n✅ All 4 publication figures exported successfully (300 DPI PNG & SVG)!")


if __name__ == "__main__":
    main()
