import os
import re
import pandas as pd

# --- 1. Modify run_ablation_study.py ---
f_path = "research/run_ablation_study.py"
with open(f_path, "r") as f:
    ab_content = f.read()

# Fix Config 0a Policy Violations to NaN
if '"Policy Violations": 0' in ab_content:
    ab_content = ab_content.replace('"Policy Violations": 0', '"Policy Violations": np.nan')

# Restore markdown report and figure
new_report_code = """
    # Restore markdown report and figure
    md_path = os.path.join(output_dir, "ablation_summary.md")
    
    c1 = agg_df[agg_df["Configuration"] == "Config 1: Full Agentic System"].iloc[0]
    c2 = agg_df[agg_df["Configuration"] == "Config 2: Agentic - Fairness"].iloc[0]
    c3 = agg_df[agg_df["Configuration"] == "Config 3: Agentic - Policy"].iloc[0]
    c4 = agg_df[agg_df["Configuration"] == "Config 4: Random Forest Baseline"].iloc[0]
    c0a = agg_df[agg_df["Configuration"] == "Config 0a: Always-Approve Constant"].iloc[0]

    paired_df = pd.read_csv(os.path.join(output_dir, "paired_comparisons.csv"))
    paired_md = "\\n| Contrast | Metric | Mean Diff | 95% CI | p-value |\\n| :--- | :--- | :---: | :---: | :---: |\\n"
    for _, row in paired_df.iterrows():
        paired_md += f"| {row['contrast']} | {row['metric']} | {row['mean_diff']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] | {row['p_value']:.4e} |\\n"

    md_content = rf\"\"\"# Controlled Ablation Study Summary — Multi-Seed Evaluation

**Dataset:** `credit_risk_dataset.csv` ($N=5,000$, Held-out Test Split $N=1,500$)  
**Counterfactual Fairness Threshold:** `6.0`  
**Model Architecture:** Random Forest (`n_estimators=100`) coupled with V7 LangGraph Multi-Agent Orchestration

---

## 1. Ablation Comparative Matrix (Mean ± Std, n=10)

| Configuration | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Demog. Parity Diff ($\Delta_{{DP}}$) | Equalized Odds Diff ($\Delta_{{EO}}$) | Policy Violations | Override Rule Triggers |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **{c1['Configuration']}** | {c1['Accuracy']} | {c1['Precision']} | {c1['Recall']} | {c1['F1']} | {c1['ROC-AUC']} | {c1['Demographic Parity Diff']} | {c1['Equalized Odds Diff']} | {c1['Policy Violations']} | {c1['Override Rule Triggers']} |
| **{c2['Configuration']}** | {c2['Accuracy']} | {c2['Precision']} | {c2['Recall']} | {c2['F1']} | {c2['ROC-AUC']} | {c2['Demographic Parity Diff']} | {c2['Equalized Odds Diff']} | {c2['Policy Violations']} | {c2['Override Rule Triggers']} |
| **{c3['Configuration']}** | {c3['Accuracy']} | {c3['Precision']} | {c3['Recall']} | {c3['F1']} | {c3['ROC-AUC']} | {c3['Demographic Parity Diff']} | {c3['Equalized Odds Diff']} | {c3['Policy Violations']} | {c3['Override Rule Triggers']} |
| **{c4['Configuration']}** | {c4['Accuracy']} | {c4['Precision']} | {c4['Recall']} | {c4['F1']} | {c4['ROC-AUC']} | {c4['Demographic Parity Diff']} | {c4['Equalized Odds Diff']} | {c4['Policy Violations']} | {c4['Override Rule Triggers']} |
| **{c0a['Configuration']}** | {c0a['Accuracy']} | {c0a['Precision']} | {c0a['Recall']} | {c0a['F1']} | {c0a['ROC-AUC']} | {c0a['Demographic Parity Diff']} | {c0a['Equalized Odds Diff']} | {c0a['Policy Violations']} | {c0a['Override Rule Triggers']} |

---

## 2. Key Architectural Takeaways

### A. Impact of Removing the Fairness Agent (`Config 1` vs. `Config 2`)
* **Demographic Parity Disparity:** When the Fairness Agent is disabled, $\Delta_{{DP}}$ increases from **{c1['Demographic Parity Diff']}** to **{c2['Demographic Parity Diff']}**.
* **Equalized Odds Disparity:** When disabled, $\Delta_{{EO}}$ changes from **{c1['Equalized Odds Diff']}** to **{c2['Equalized Odds Diff']}**.
* **Operational Cost:** The active Fairness Agent routes **{c1['Override Rule Triggers']}** ambiguous applications to the override rule. 

### B. Impact of Removing the Policy Agent (`Config 1` vs. `Config 3`)
* **Regulatory Compliance Leaks:** In the Full System (`Config 1`), the Policy Agent blocks non-compliant loan applications (**{c1['Policy Violations']}** policy violations). When the Policy Agent is ablated (`Config 3`), **{c3['Policy Violations']}** applications that violate hard RBI/regulatory rules leak through and receive approval.

### C. Comparison Against Random Forest Baseline (`Config 1` vs. `Config 4`)
* **Fairness vs. Baseline:** The raw Random Forest model (`Config 4`) exhibits demographic disparity ($\Delta_{{DP}} = {c4['Demographic Parity Diff']}$ and $\Delta_{{EO}} = {c4['Equalized Odds Diff']}$). 
* **Trade-Off Analysis:** The Agentic System F1 score is **{c1['F1']}** vs. **{c4['F1']}** for raw RF, while preventing **{c4['Policy Violations']}** regulatory violations compared to **{c1['Policy Violations']}**.

### D. Always-Approve Constant Baseline (`Config 1` vs. `Config 0a`)
* **Baseline Accuracy:** Config 0a (always-approve) achieves Accuracy {c0a['Accuracy']} and F1 {c0a['F1']}, exceeding Config 1 on both.

## 3. Paired Comparisons
{paired_md}
---
*Report automatically generated by `run_ablation_study.py`.*
\"\"\"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"✅ Saved summary report → {md_path}")

    # Generate Graphical Plot
    png_path = os.path.join(output_dir, "ablation_comparison.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Controlled LangGraph V7 Ablation Study — Multi-Metric Comparison (n=10 seeds)", fontsize=16, fontweight="bold", y=0.98)

    labels = ["Full System", "No Fairness", "No Policy", "RF Baseline"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#66a61e"]

    def extract_stats(config_names, col):
        means, stds = [], []
        for name in config_names:
            series = results_df[results_df["Configuration"] == name][col]
            means.append(series.mean())
            stds.append(series.std())
        return means, stds

    conf_names = ["Config 1: Full Agentic System", "Config 2: Agentic - Fairness", "Config 3: Agentic - Policy", "Config 4: Random Forest Baseline"]
    
    # 1. Predictive F1 Score
    f1_m, f1_s = extract_stats(conf_names, "F1")
    axes[0, 0].bar(labels, f1_m, yerr=f1_s, color=colors, width=0.5, edgecolor="black", capsize=5)
    axes[0, 0].set_title("Predictive Performance (F1 Score)", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylim(0.7, 1.0)
    for i, (m, s) in enumerate(zip(f1_m, f1_s)):
        axes[0, 0].text(i, m + 0.01 + s, f"{m:.3f}", ha="center", fontweight="bold")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Demographic Parity Diff
    dp_m, dp_s = extract_stats(conf_names, "Demographic Parity Diff")
    axes[0, 1].bar(labels, dp_m, yerr=dp_s, color=colors, width=0.5, edgecolor="black", capsize=5)
    axes[0, 1].set_title(r"Fairness Disparity ($\Delta_{DP}$ — Lower is Better)", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylim(0, max([m+s for m,s in zip(dp_m,dp_s)]) * 1.2 if max(dp_m) > 0 else 0.2)
    for i, (m, s) in enumerate(zip(dp_m, dp_s)):
        axes[0, 1].text(i, m + 0.005 + s, f"{m:.3f}", ha="center", fontweight="bold")
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)

    # 3. Policy Violations
    viol_m, viol_s = extract_stats(conf_names, "Policy Violations")
    axes[1, 0].bar(labels, viol_m, yerr=viol_s, color=colors, width=0.5, edgecolor="black", capsize=5)
    axes[1, 0].set_title("Regulatory Policy Violations (Lower is Better)", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylim(0, max([m+s for m,s in zip(viol_m, viol_s)]) * 1.2 if max(viol_m) > 0 else 10)
    for i, (m, s) in enumerate(zip(viol_m, viol_s)):
        axes[1, 0].text(i, m + 0.5 + s, f"{m:.1f}", ha="center", fontweight="bold")
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)

    # 4. Override Rule Triggers
    esc_m, esc_s = extract_stats(conf_names, "Override Rule Triggers")
    axes[1, 1].bar(labels, esc_m, yerr=esc_s, color=colors, width=0.5, edgecolor="black", capsize=5)
    axes[1, 1].set_title("Override Rule Triggers", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylim(0, max([m+s for m,s in zip(esc_m, esc_s)]) * 1.2 if max(esc_m) > 0 else 50)
    for i, (m, s) in enumerate(zip(esc_m, esc_s)):
        axes[1, 1].text(i, m + 1 + s, f"{m:.1f}", ha="center", fontweight="bold")
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"✅ Saved comparison figure → {png_path}\\n")

    print("======================================================================")
    print("ALL ABLATION EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("======================================================================")
"""

if "# Task 3: Stability check" in ab_content:
    ab_content = ab_content.replace('    # Task 3: Stability check', new_report_code + '\n    # Task 3: Stability check')
with open(f_path, "w") as f:
    f.write(ab_content)


# --- 2. Modify generate_publication_assets.py ---
gpa_path = "research/generate_publication_assets.py"
with open(gpa_path, "r") as f:
    gpa = f.read()

new_table3_code = """
    # ── 4. Table 3: Random Forest vs Agentic System ───────────────
    seed_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "final_results", "ablation_results_per_seed.csv"))
    def get_mean(conf, metric):
        return seed_df[seed_df["Configuration"] == conf][metric].mean()

    c1 = "Config 1: Full Agentic System"
    c4 = "Config 4: Random Forest Baseline"
    
    rf_f1 = get_mean(c4, "F1")
    ag_f1 = get_mean(c1, "F1")
    diff_f1 = (ag_f1 - rf_f1) / rf_f1 * 100 if rf_f1 > 0 else 0

    rf_auc = get_mean(c4, "ROC-AUC")
    ag_auc = get_mean(c1, "ROC-AUC")
    diff_auc = (ag_auc - rf_auc) / rf_auc * 100 if rf_auc > 0 else 0

    rf_dp = get_mean(c4, "Demographic Parity Diff")
    ag_dp = get_mean(c1, "Demographic Parity Diff")
    diff_dp = (ag_dp - rf_dp) / rf_dp * 100 if rf_dp > 0 else 0

    rf_eo = get_mean(c4, "Equalized Odds Diff")
    ag_eo = get_mean(c1, "Equalized Odds Diff")
    diff_eo = (ag_eo - rf_eo) / rf_eo * 100 if rf_eo > 0 else 0

    rf_pol = get_mean(c4, "Policy Violations")
    ag_pol = get_mean(c1, "Policy Violations")
    diff_pol = (ag_pol - rf_pol) / rf_pol * 100 if rf_pol > 0 else 0

    rf_esc = get_mean(c4, "Override Rule Triggers")
    rf_esc_rate = get_mean(c4, "Override Trigger Rate")
    ag_esc = get_mean(c1, "Override Rule Triggers")
    ag_esc_rate = get_mean(c1, "Override Trigger Rate")
    diff_esc = (ag_esc_rate - rf_esc_rate) * 100

    t3_df = pd.DataFrame([
        {"Metric Category": "Predictive Accuracy", "Metric": "F1 Score", "Random Forest Baseline": f"{rf_f1:.4f}", "Agentic System v7": f"{ag_f1:.4f}", "Relative Difference": f"{diff_f1:+.2f}%"},
        {"Metric Category": "Predictive Accuracy", "Metric": "ROC-AUC", "Random Forest Baseline": f"{rf_auc:.4f}", "Agentic System v7": f"{ag_auc:.4f}", "Relative Difference": f"{diff_auc:+.2f}%"},
        {"Metric Category": "Demographic Fairness", "Metric": "Demographic Parity Diff (ΔDP)", "Random Forest Baseline": f"{rf_dp:.4f}", "Agentic System v7": f"{ag_dp:.4f}", "Relative Difference": f"{diff_dp:+.2f}%"},
        {"Metric Category": "Demographic Fairness", "Metric": "Equalized Odds Diff (ΔEO)", "Random Forest Baseline": f"{rf_eo:.4f}", "Agentic System v7": f"{ag_eo:.4f}", "Relative Difference": f"{diff_eo:+.2f}%"},
        {"Metric Category": "Regulatory Compliance", "Metric": "Hard Policy Violations", "Random Forest Baseline": f"{rf_pol:.1f}", "Agentic System v7": f"{ag_pol:.1f}", "Relative Difference": f"{diff_pol:+.2f}%"},
        {"Metric Category": "Operational Workload", "Metric": "Override Rule Triggers", "Random Forest Baseline": f"{rf_esc:.1f} ({rf_esc_rate:.2%})", "Agentic System v7": f"{ag_esc:.1f} ({ag_esc_rate:.2%})", "Relative Difference": f"{diff_esc:+.2f}%"},
    ])
"""
gpa = re.sub(r'# ── 4\. Table 3: Random Forest vs Agentic System ───────────────.*?\]\)', new_table3_code.strip(), gpa, flags=re.DOTALL)
with open(gpa_path, "w") as f:
    f.write(gpa)


# --- 3. Modify generate_publication_figures.py ---
gpf_path = "research/generate_publication_figures.py"
with open(gpf_path, "r") as f:
    gpf = f.read()

new_figure3_code = """
    # ── 3. Ablation Comparison Figure ─────────────────────────────
    print("Generating Figure 3: Ablation Comparison Figure...")
    ablation_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "final_results", "ablation_results_per_seed.csv"))
    
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
"""
gpf = re.sub(r'# ── 3\. Ablation Comparison Figure ─────────────────────────────.*?fig\.suptitle\("Controlled Multi-Agent Architecture Ablation Comparison", fontsize=15, fontweight="bold", y=0.98\)', new_figure3_code.strip(), gpf, flags=re.DOTALL)


# Fix figure 5
new_figure5_code = """
    # ── 5. Policy Violation Comparison ────────────────────────────
    print("Generating Figure 5: Policy Violation Comparison...")
    fig, ax = plt.subplots(figsize=(8, 5))
    viol_m, viol_s = get_stats("Policy Violations")
    bars = ax.bar(labels, viol_m, yerr=viol_s, color=["#2b5c8f", "#2b5c8f", "#d95f02", "#66a61e"], width=0.5, edgecolor="black", capsize=5)
    
    # Highlight No Policy bar
    bars[2].set_facecolor("#d95f02")
    bars[2].set_edgecolor("red")
    bars[2].set_linewidth(2)

    for i, (m, s) in enumerate(zip(viol_m, viol_s)):
        ax.text(i, m + 4 + s, f"{m:.1f}\\n({'0%' if m==0 else f'{(m/1500)*100:.1f}%'})", ha="center", fontweight="bold", fontsize=10)
"""
gpf = re.sub(r'# ── 5\. Policy Violation Comparison ────────────────────────────.*?for i, v in enumerate\(viol_vals\):\n\s+ax\.text\(i, v \+ 4, f"\{v\}\\n\(\{\'0%\' if v==0 else f\'\{\(v/1500\)\*100:\.1f\}%\'\}\)", ha="center", fontweight="bold", fontsize=10\)', new_figure5_code.strip(), gpf, flags=re.DOTALL)


# Fix Pareto frontier points
new_pareto_code = """
    # ── 2. Pareto Frontier ────────────────────────────────────────
    print("Generating Figure 2: Pareto Frontier...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ablation_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "final_results", "ablation_results_per_seed.csv"))
    def get_stats_p(col):
        conf_names = ["Config 1: Full Agentic System", "Config 2: Agentic - Fairness", "Config 3: Agentic - Policy", "Config 4: Random Forest Baseline"]
        m = []
        for name in conf_names:
            series = ablation_df[ablation_df["Configuration"] == name][col]
            m.append(series.mean())
        return m
    dp_m = get_stats_p("Demographic Parity Diff")
    f1_m = get_stats_p("F1")
    
    # Points: (DP Diff, F1 Score)
    points = [
        ("RF Baseline", dp_m[3], f1_m[3], "#66a61e", "s"),
        ("No Policy", dp_m[2], f1_m[2], "#7570b3", "^"),
        ("No Fairness", dp_m[1], f1_m[1], "#d95f02", "o"),
        ("Full System (ε=6.0)", dp_m[0], f1_m[0], "#2b5c8f", "D"),
    ]
"""
gpf = re.sub(r'# ── 2\. Pareto Frontier ────────────────────────────────────────.*?\]', new_pareto_code.strip(), gpf, flags=re.DOTALL)

with open(gpf_path, "w") as f:
    f.write(gpf)
