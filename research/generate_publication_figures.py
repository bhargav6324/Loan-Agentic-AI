"""
Publication Asset Generator — High-Resolution Figures (300 DPI)
────────────────────────────────────────────────────────────────────────
Generates 5 publication-quality figures for IEEE/Springer manuscripts:
  1. ROC Curve (ML Baseline vs. Agentic System)
  2. Pareto Frontier (Fairness vs. Predictive Accuracy Trade-off)
  3. Ablation Comparison Figure
  4. Fairness Escalation Distribution (with threshold line at ε=6.0)
  5. Policy Violation Comparison

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

    # ── 2. Pareto Frontier ────────────────────────────────────────
    print("Generating Figure 2: Pareto Frontier...")
    fig, ax = plt.subplots(figsize=(8, 6))
    # Points: (DP Diff, F1 Score)
    points = [
        ("RF Baseline", 0.1794, 0.9167, "#66a61e", "s"),
        ("No Policy", 0.0062, 0.8487, "#7570b3", "^"),
        ("No Fairness", 0.0634, 0.8748, "#d95f02", "o"),
        ("Full System (ε=6.0)", 0.0626, 0.8748, "#2b5c8f", "D"),
    ]
    
    for name, dp, f1, color, marker in points:
        ax.scatter(dp, f1, color=color, s=140, marker=marker, edgecolors="black", zorder=5, label=name)
        offset_y = 0.005 if name != "No Fairness" else -0.012
        ax.annotate(name, (dp, f1), xytext=(0, 12), textcoords="offset points", ha="center", fontweight="bold", fontsize=10)

    # Plot frontier curve connecting non-dominated points
    f_dp = [0.0062, 0.0626, 0.1794]
    f_f1 = [0.8487, 0.8748, 0.9167]
    ax.plot(f_dp, f_f1, linestyle="--", color="gray", lw=1.5, zorder=3, label="Optimal Pareto Frontier")

    ax.set_xlabel(r"Demographic Parity Disparity ($\Delta_{DP}$ — Lower is Better)")
    ax.set_ylabel("Predictive Performance (F1 Score — Higher is Better)")
    ax.set_title("Pareto Trade-off Frontier: Fairness vs. Predictive Performance")
    ax.set_xlim(-0.01, 0.20)
    ax.set_ylim(0.83, 0.93)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_pareto_frontier.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "fig2_pareto_frontier.svg"))
    plt.close()
    print("✅ Generated fig2_pareto_frontier (.png & .svg)")

    # ── 3. Ablation Comparison Figure ─────────────────────────────
    print("Generating Figure 3: Ablation Comparison Figure...")
    ablation_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "final_results", "ablation_results.csv"))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = ["Full System", "No Fairness", "No Policy", "RF Baseline"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#66a61e"]

    # Subplot 1: F1 Score
    f1_vals = ablation_df["F1"].values
    axes[0].bar(labels, f1_vals, color=colors, width=0.5, edgecolor="black")
    axes[0].set_title("F1 Score (Predictive)", fontweight="bold")
    axes[0].set_ylim(0.7, 1.0)
    for i, v in enumerate(f1_vals):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].tick_params(axis="x", rotation=15)

    # Subplot 2: Policy Violations
    viol_vals = ablation_df["Policy Violations"].values
    axes[1].bar(labels, viol_vals, color=colors, width=0.5, edgecolor="black")
    axes[1].set_title("Regulatory Policy Violations", fontweight="bold")
    axes[1].set_ylim(0, max(viol_vals) * 1.2)
    for i, v in enumerate(viol_vals):
        axes[1].text(i, v + 3, str(v), ha="center", fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].tick_params(axis="x", rotation=15)

    # Subplot 3: Escalations
    esc_vals = ablation_df["Human Escalations"].values
    axes[2].bar(labels, esc_vals, color=colors, width=0.5, edgecolor="black")
    axes[2].set_title("Human Review Escalations", fontweight="bold")
    axes[2].set_ylim(0, max(esc_vals) * 1.2 if max(esc_vals) > 0 else 50)
    for i, v in enumerate(esc_vals):
        axes[2].text(i, v + 2, str(v), ha="center", fontweight="bold")
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    axes[2].tick_params(axis="x", rotation=15)

    fig.suptitle("Controlled Multi-Agent Architecture Ablation Comparison", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "fig3_ablation_comparison.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "fig3_ablation_comparison.svg"))
    plt.close()
    print("✅ Generated fig3_ablation_comparison (.png & .svg)")

    # ── 4. Fairness Escalation Distribution ───────────────────────
    print("Generating Figure 4: Fairness Escalation Distribution...")
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
    plt.savefig(os.path.join(out_dir, "fig4_fairness_distribution.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "fig4_fairness_distribution.svg"))
    plt.close()
    print("✅ Generated fig4_fairness_distribution (.png & .svg)")

    # ── 5. Policy Violation Comparison ────────────────────────────
    print("Generating Figure 5: Policy Violation Comparison...")
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, viol_vals, color=["#2b5c8f", "#2b5c8f", "#d95f02", "#66a61e"], width=0.5, edgecolor="black")
    
    # Highlight No Policy bar
    bars[2].set_facecolor("#d95f02")
    bars[2].set_edgecolor("red")
    bars[2].set_linewidth(2)

    for i, v in enumerate(viol_vals):
        ax.text(i, v + 4, f"{v}\n({'0%' if v==0 else f'{(v/1500)*100:.1f}%'})", ha="center", fontweight="bold", fontsize=10)

    ax.set_ylabel("Number of Illegal Policy Approvals")
    ax.set_title("Regulatory Compliance: Hard Policy Violations by Configuration")
    ax.set_ylim(0, 200)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_policy_violations.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "fig5_policy_violations.svg"))
    plt.close()
    print("✅ Generated fig5_policy_violations (.png & .svg)")

    print("\n✅ All 5 publication figures exported successfully (300 DPI PNG & SVG)!")


if __name__ == "__main__":
    main()
