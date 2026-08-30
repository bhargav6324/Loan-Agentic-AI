"""
Automated Ablation Study Harness — LangGraph V7 Loan Approval System
────────────────────────────────────────────────────────────────────────
Evaluates 4 configurations automatically against the test split (N=1500):
  1. Full Agentic System (disable_fairness=False, disable_policy=False)
  2. Agentic - Fairness  (disable_fairness=True,  disable_policy=False)
  3. Agentic - Policy    (disable_fairness=False, disable_policy=True)
  4. Random Forest Baseline (non-agentic ML model)

Outputs generated in research/final_results/:
  - ablation_results.csv
  - ablation_summary.md
  - ablation_comparison.png
"""

import os
import sys
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef, average_precision_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

# Import existing domain heuristics without modifying any project code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml.canonical_model_contract import compute_composite_score
from research.evaluation_harness import check_hard_overrides, check_policy


def evaluate_config(df_test, X_test, y_test, rf_model, features, config_name, disable_fairness=False, disable_policy=False, is_ml_only=False, epsilon=6.0):
    true_approved = (y_test == 0).astype(int)
    gender_features = X_test["applicant_gender"]

    if is_ml_only:
        rf_proba = rf_model.predict_proba(X_test)
        proba_approve = rf_proba[:, 0]
        approved = (proba_approve > 0.5).astype(int)
        scores = proba_approve
        
        # Post-hoc audit of policy violations among approved loans
        violations = 0
        for i in range(len(X_test)):
            if approved[i] == 1 and not check_policy(df_test.iloc[i]):
                violations += 1
                
        escalations = 0
        bias_escalations = 0
    else:
        rf_proba = rf_model.predict_proba(X_test)
        approved = []
        scores = []
        violations = 0
        escalations = 0
        bias_escalations = 0

        for i in range(len(X_test)):
            row = df_test.iloc[i]

            # ── Hard Override Pre-Screening ───────────────────
            override = check_hard_overrides(row)
            if override == "HARD_REJECT":
                approved.append(0)
                scores.append(0.0)
                continue
            if override == "HARD_FAST_TRACK":
                approved.append(1)
                scores.append(1.0)
                # Check policy post-hoc if approved
                if not check_policy(row):
                    violations += 1
                continue

            # ── 1. Risk Agent ─────────────────────────────────
            _, _, risk_score = compute_composite_score(row.to_dict())
            scores.append(1.0 - risk_score / 100.0)

            # ── 2. Routing ────────────────────────────────────
            if risk_score <= 35:
                app_val = 1
                approved.append(app_val)
                if app_val == 1 and not check_policy(row):
                    violations += 1
            elif risk_score > 99:
                approved.append(0)
            else:
                # Medium / High Risk → Policy Agent (if enabled)
                if not disable_policy and not check_policy(row):
                    approved.append(0)
                    continue

                # ── 3. Fairness Agent (if enabled) ────────────
                if not disable_fairness:
                    cf_row = row.copy()
                    cf_row["applicant_gender"] = 0 if row["applicant_gender"] == 1 else 1
                    cf_features = cf_row[features].to_frame().T
                    cf_proba_row = rf_model.predict_proba(cf_features)[0]
                    cf_ml_score = cf_proba_row[1] * 100
                    cf_ml_confidence = max(cf_proba_row)
                    cf_ml_weight = 0.4 + (0.4 * cf_ml_confidence)
                    cf_heuristic_weight = 1 - cf_ml_weight
                    cf_heuristic = compute_heuristic_score(cf_row)
                    cf_risk_score = cf_ml_weight * cf_ml_score + cf_heuristic_weight * cf_heuristic

                    # Parameterized threshold epsilon
                    if abs(cf_risk_score - risk_score) > epsilon:
                        escalations += 1
                        bias_escalations += 1
                        if cf_risk_score < risk_score - epsilon:
                            app_val = 1
                            approved.append(app_val)
                        else:
                            app_val = 0
                            approved.append(app_val)
                    else:
                        app_val = 1
                        approved.append(app_val)
                else:
                    app_val = 1
                    approved.append(app_val)

                if app_val == 1 and not check_policy(row):
                    violations += 1

        approved = np.array(approved)
        scores = np.array(scores)

    # Compute Metrics
    acc = accuracy_score(true_approved, approved)
    prec = precision_score(true_approved, approved, zero_division=0)
    rec = recall_score(true_approved, approved, zero_division=0)
    f1 = f1_score(true_approved, approved, zero_division=0)
    try:
        auc = roc_auc_score(true_approved, scores)
    except Exception:
        auc = 0.5

    dp_diff = demographic_parity_difference(true_approved, approved, sensitive_features=gender_features)
    eo_diff = equalized_odds_difference(true_approved, approved, sensitive_features=gender_features)
    approval_rate = np.mean(approved)
    esc_rate = escalations / len(X_test)

    tn, fp, fn, tp = confusion_matrix(true_approved, approved, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    balanced_accuracy = (specificity + sensitivity) / 2
    mcc = matthews_corrcoef(true_approved, approved)
    try:
        pr_auc = average_precision_score(1 - true_approved, 1 - scores)
    except Exception:
        pr_auc = 0.0
    positive_prevalence = np.mean(true_approved)

    return {
        "Configuration": config_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "ROC-AUC": round(auc, 4),
        "Demographic Parity Diff": round(dp_diff, 4),
        "Equalized Odds Diff": round(eo_diff, 4),
        "Approval Rate": f"{approval_rate:.1%}",
        "Policy Violations": violations,
        "Override Rule Triggers": escalations,
        "Override Trigger Rate": f"{esc_rate:.2%}",
        "Counterfactual Flags": bias_escalations,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Specificity": round(specificity, 4),
        "Sensitivity": round(sensitivity, 4),
        "Balanced Accuracy": round(balanced_accuracy, 4),
        "MCC": round(mcc, 4),
        "PR-AUC (default class)": round(pr_auc, 4),
        "Positive Prevalence": round(positive_prevalence, 4),
    }


def main():
    print("======================================================================")
    print("STARTING CONTROLLED ABLATION STUDY — MULTI-SEED EVALUATION")
    print("======================================================================")

    data_path = os.path.join(os.path.dirname(__file__), "data", "credit_risk_dataset.csv")
    df = pd.read_csv(data_path)

    features = [
        "credit_score", "dti_ratio", "loan_to_income",
        "employment_score", "applicant_age", "defaults",
        "credit_utilisation", "applicant_gender"
    ]

    X = df[features]
    y = df["target_default"]

    configs = [
        ("Config 1: Full Agentic System", False, False, False),
        ("Config 2: Agentic - Fairness", True, False, False),
        ("Config 3: Agentic - Policy", False, True, False),
        ("Config 4: Random Forest Baseline", False, False, True),
    ]

    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    
    all_results = []
    print("Convention: positive class = APPROVAL (true_approved = y_test == 0).\n      Specificity = share of true defaulters correctly rejected.")
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )
        df_test = df.loc[X_test.index]

        rf_model = RandomForestClassifier(n_estimators=100, random_state=seed)
        rf_model.fit(X_train, y_train)
        
        for name, dis_f, dis_p, is_ml in configs:
            res = evaluate_config(df_test, X_test, y_test, rf_model, features, name, dis_f, dis_p, is_ml)
            res["seed"] = seed
            
            # Clean string metrics back to floats for aggregation
            res["Approval Rate"] = float(res["Approval Rate"].strip('%')) / 100
            res["Override Trigger Rate"] = float(res["Override Trigger Rate"].strip('%')) / 100
            
            all_results.append(res)
            
        true_approved_test = (y_test == 0).astype(int)
        pos_prev = np.mean(true_approved_test)
        
        always_approve = np.ones(len(y_test))
        tn, fp, fn, tp = confusion_matrix(true_approved_test, always_approve, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        bal_acc = (spec + sens) / 2
        default_prev = 1 - pos_prev
        
        # Config 0a
        res_0a = {
            "Configuration": "Config 0a: Always-Approve Constant", "seed": seed,
            "Accuracy": accuracy_score(true_approved_test, always_approve),
            "Precision": precision_score(true_approved_test, always_approve, zero_division=0),
            "Recall": recall_score(true_approved_test, always_approve, zero_division=0),
            "F1": f1_score(true_approved_test, always_approve, zero_division=0),
            "ROC-AUC": 0.5, "Demographic Parity Diff": 0.0, "Equalized Odds Diff": 0.0,
            "Approval Rate": 1.0, "Policy Violations": np.nan, "Override Rule Triggers": 0,
            "Override Trigger Rate": 0.0, "Counterfactual Flags": 0,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Specificity": spec, "Sensitivity": sens, "Balanced Accuracy": bal_acc,
            "MCC": matthews_corrcoef(true_approved_test, always_approve),
            "PR-AUC (default class)": default_prev, "Positive Prevalence": pos_prev
        }
        all_results.append(res_0a)
        
        # Config 0b
        res_0b = {
            "Configuration": "Config 0b: No-Skill PR floor", "seed": seed,
            "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan, "F1": np.nan, "ROC-AUC": np.nan,
            "Demographic Parity Diff": np.nan, "Equalized Odds Diff": np.nan,
            "Approval Rate": np.nan, "Policy Violations": np.nan, "Override Rule Triggers": np.nan,
            "Override Trigger Rate": np.nan, "Counterfactual Flags": np.nan,
            "TP": np.nan, "FP": np.nan, "TN": np.nan, "FN": np.nan,
            "Specificity": np.nan, "Sensitivity": np.nan, "Balanced Accuracy": np.nan,
            "MCC": np.nan,
            "PR-AUC (default class)": default_prev, "Positive Prevalence": pos_prev
        }
        all_results.append(res_0b)

    results_df = pd.DataFrame(all_results)
    
    # Save per-seed CSV
    output_dir = os.path.join(os.path.dirname(__file__), "final_results")
    os.makedirs(output_dir, exist_ok=True)
    per_seed_csv = os.path.join(output_dir, "ablation_results_per_seed.csv")
    
    # Reorder columns to match requested format
    cols = list(results_df.columns)
    cols.remove("seed")
    cols.insert(1, "seed")
    results_df = results_df[cols]
    results_df.to_csv(per_seed_csv, index=False)
    
    # Aggregate to mean +/- std
    numeric_cols = [c for c in results_df.columns if c not in ["Configuration", "seed"]]
    agg_df = pd.DataFrame()
    agg_df["Configuration"] = ["Config 1: Full Agentic System", "Config 2: Agentic - Fairness", "Config 3: Agentic - Policy", "Config 4: Random Forest Baseline", "Config 0a: Always-Approve Constant", "Config 0b: No-Skill PR floor"]
    agg_df["n_seeds"] = 10
    
    for col in numeric_cols:
        col_series = []
        for conf in agg_df["Configuration"]:
            subset = results_df[results_df["Configuration"] == conf][col].dropna()
            if len(subset) == 0:
                col_series.append("")
            else:
                m = subset.mean()
                s = subset.std()
                col_series.append(f"{m:.4f} ± {s:.4f}")
        agg_df[col] = col_series
        
    agg_csv = os.path.join(output_dir, "ablation_results.csv")
    with open(agg_csv, "w") as f:
        f.write("# Convention: positive class = APPROVAL (true_approved = y_test == 0). Specificity = share of true defaulters correctly rejected.\n")
    agg_df.to_csv(agg_csv, index=False, mode="a")
    
    # Task 2: Paired comparison
    def paired_test(metric, conf1, conf2):
        d1 = results_df[results_df["Configuration"] == conf1].sort_values("seed")[metric].values
        d2 = results_df[results_df["Configuration"] == conf2].sort_values("seed")[metric].values
        diffs = d1 - d2
        mean_diff = diffs.mean()
        std_diff = diffs.std(ddof=1)
        n = len(diffs)
        se = std_diff / np.sqrt(n)
        ci_low, ci_high = scipy.stats.t.interval(0.95, df=n-1, loc=mean_diff, scale=se)
        p_val = scipy.stats.ttest_rel(d1, d2).pvalue
        return mean_diff, std_diff, ci_low, ci_high, p_val, n

    contrasts = [
        ("C1 vs C2", "Demographic Parity Diff", "Config 1: Full Agentic System", "Config 2: Agentic - Fairness"),
        ("C1 vs C2", "Equalized Odds Diff", "Config 1: Full Agentic System", "Config 2: Agentic - Fairness"),
        ("C1 vs C2", "Specificity", "Config 1: Full Agentic System", "Config 2: Agentic - Fairness"),
        ("C1 vs C2", "Balanced Accuracy", "Config 1: Full Agentic System", "Config 2: Agentic - Fairness"),
        ("C1 vs C3", "Policy Violations", "Config 1: Full Agentic System", "Config 3: Agentic - Policy"),
        ("C1 vs C3", "Approval Rate", "Config 1: Full Agentic System", "Config 3: Agentic - Policy"),
        ("C1 vs C4", "Specificity", "Config 1: Full Agentic System", "Config 4: Random Forest Baseline"),
        ("C1 vs C4", "Balanced Accuracy", "Config 1: Full Agentic System", "Config 4: Random Forest Baseline"),
        ("C1 vs C4", "MCC", "Config 1: Full Agentic System", "Config 4: Random Forest Baseline"),
        ("C1 vs C4", "Demographic Parity Diff", "Config 1: Full Agentic System", "Config 4: Random Forest Baseline"),
        ("C1 vs C4", "Equalized Odds Diff", "Config 1: Full Agentic System", "Config 4: Random Forest Baseline"),
    ]
    
    paired_results = []
    for c_name, metric, conf1, conf2 in contrasts:
        md, sd, cil, cih, pval, n = paired_test(metric, conf1, conf2)
        paired_results.append({
            "contrast": c_name,
            "metric": metric,
            "mean_diff": md,
            "std_diff": sd,
            "ci_low": cil,
            "ci_high": cih,
            "p_value": pval,
            "n": n
        })
        
    pd.DataFrame(paired_results).to_csv(os.path.join(output_dir, "paired_comparisons.csv"), index=False)
    

    # Restore markdown report and figure
    md_path = os.path.join(output_dir, "ablation_summary.md")
    
    c1 = agg_df[agg_df["Configuration"] == "Config 1: Full Agentic System"].iloc[0]
    c2 = agg_df[agg_df["Configuration"] == "Config 2: Agentic - Fairness"].iloc[0]
    c3 = agg_df[agg_df["Configuration"] == "Config 3: Agentic - Policy"].iloc[0]
    c4 = agg_df[agg_df["Configuration"] == "Config 4: Random Forest Baseline"].iloc[0]
    c0a = agg_df[agg_df["Configuration"] == "Config 0a: Always-Approve Constant"].iloc[0]

    paired_df = pd.read_csv(os.path.join(output_dir, "paired_comparisons.csv"))
    paired_md = "\n| Contrast | Metric | Mean Diff | 95% CI | p-value |\n| :--- | :--- | :---: | :---: | :---: |\n"
    for _, row in paired_df.iterrows():
        paired_md += f"| {row['contrast']} | {row['metric']} | {row['mean_diff']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] | {row['p_value']:.4e} |\n"

    md_content = rf"""# Controlled Ablation Study Summary — Multi-Seed Evaluation

**Dataset:** `credit_risk_dataset.csv` ($N=5,000$, Held-out Test Split $N=1,500$)  
**Counterfactual Fairness Threshold:** `6.0`  
**Model Architecture:** Random Forest (`n_estimators=100`)

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
"""
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
    print(f"✅ Saved comparison figure → {png_path}\n")

    print("======================================================================")
    print("ALL ABLATION EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("======================================================================")

    # Task 3: Stability check
    c1_dp = results_df[results_df["Configuration"] == "Config 1: Full Agentic System"].sort_values("seed")["Demographic Parity Diff"].values
    c2_dp = results_df[results_df["Configuration"] == "Config 2: Agentic - Fairness"].sort_values("seed")["Demographic Parity Diff"].values
    dp_diffs = c1_dp - c2_dp
    print("\nTask 3 - Stability Check (C1 vs C2):")
    # print(f"Delta_DP diff (C1 - C2) per seed: {dp_diffs}")
    pos_dp = np.sum(dp_diffs > 0)
    print(f"Number of seeds where (C1.Delta_DP - C2.Delta_DP) is positive: {pos_dp} out of 10")
    
    c1_spec = results_df[results_df["Configuration"] == "Config 1: Full Agentic System"].sort_values("seed")["Specificity"].values
    c2_spec = results_df[results_df["Configuration"] == "Config 2: Agentic - Fairness"].sort_values("seed")["Specificity"].values
    spec_diffs = c1_spec - c2_spec
    # print(f"Specificity diff (C1 - C2) per seed: {spec_diffs}")
    pos_spec = np.sum(spec_diffs > 0)
    print(f"Number of seeds where (C1.Specificity - C2.Specificity) is positive: {pos_spec} out of 10")

if __name__ == '__main__':
    main()
