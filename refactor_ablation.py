import os

file_path = "research/run_ablation_study.py"
with open(file_path, "r") as f:
    content = f.read()

import_idx = content.find("import pandas as pd")
content = content[:import_idx] + "import scipy.stats\n" + content[import_idx:]

main_idx = content.find("def main():")

new_main = """def main():
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
    print("Convention: positive class = APPROVAL (true_approved = y_test == 0).\\n      Specificity = share of true defaulters correctly rejected.")
    
    for seed in seeds:
        print(f"\\n--- Running Seed {seed} ---")
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
            "Approval Rate": 1.0, "Policy Violations": 0, "Override Rule Triggers": 0,
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
        f.write("# Convention: positive class = APPROVAL (true_approved = y_test == 0). Specificity = share of true defaulters correctly rejected.\\n")
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
    
    # Task 3: Stability check
    c1_dp = results_df[results_df["Configuration"] == "Config 1: Full Agentic System"].sort_values("seed")["Demographic Parity Diff"].values
    c2_dp = results_df[results_df["Configuration"] == "Config 2: Agentic - Fairness"].sort_values("seed")["Demographic Parity Diff"].values
    dp_diffs = c1_dp - c2_dp
    print("\\nTask 3 - Stability Check (C1 vs C2):")
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
"""

new_content = content[:main_idx] + new_main
with open(file_path, "w") as f:
    f.write(new_content)
