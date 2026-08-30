import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Import evaluate_config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from research.run_ablation_study import evaluate_config

def main():
    print("======================================================================")
    print("STARTING THRESHOLD SENSITIVITY SWEEP ON VALIDATION SPLIT")
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

    epsilons = np.arange(5.0, 8.5, 0.5)
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

    all_results = []

    for seed in seeds:
        # Identical first split to run_ablation_study.py (70% train, 30% test)
        # Note: We must NOT touch X_test, y_test!
        X_train_full, _, y_train_full, _ = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )
        
        # Second split: carve out validation set from the 70%
        # train 60% of total / validation 10% of total
        # test_size here should be 10/70 = 1/7 of the train_full
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=(1/7.0), random_state=seed
        )

        df_val = df.loc[X_val.index]

        # Train RF on the 60% train split
        rf_model = RandomForestClassifier(n_estimators=100, random_state=seed)
        rf_model.fit(X_train, y_train)

        for eps in epsilons:
            res = evaluate_config(df_val, X_val, y_val, rf_model, features, "Config 1: Full Agentic System", False, False, False, epsilon=eps)
            
            all_results.append({
                "epsilon": eps,
                "seed": seed,
                "F1": res["F1"],
                "balanced_accuracy": res["Balanced Accuracy"],
                "specificity": res["Specificity"],
                "Delta_DP": res["Demographic Parity Diff"],
                "Delta_EO": res["Equalized Odds Diff"],
                "override_rule_triggers": res["Override Rule Triggers"],
                "override_trigger_rate": float(res["Override Trigger Rate"].strip('%')) / 100,
                "policy_violations": res["Policy Violations"]
            })

    results_df = pd.DataFrame(all_results)
    
    output_dir = os.path.join(os.path.dirname(__file__), "final_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Write per_seed csv
    per_seed_csv = os.path.join(output_dir, "threshold_sweep_per_seed.csv")
    results_df.to_csv(per_seed_csv, index=False)
    
    # Aggregate mean +/- std
    agg_df = pd.DataFrame()
    agg_df["epsilon"] = epsilons
    
    cols_to_agg = [c for c in results_df.columns if c not in ["epsilon", "seed"]]
    
    for col in cols_to_agg:
        col_series = []
        for eps in epsilons:
            subset = results_df[results_df["epsilon"] == eps][col]
            col_series.append(f"{subset.mean():.4f} ± {subset.std():.4f}")
        agg_df[col] = col_series
        
    agg_csv = os.path.join(output_dir, "threshold_sweep.csv")
    agg_df.to_csv(agg_csv, index=False)
    
    # Generate Figure
    fig_dir = os.path.join(os.path.dirname(__file__), "publication_assets", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Data for plotting
    eps_m = []
    dp_m, dp_s = [], []
    ba_m, ba_s = [], []
    esc_m, esc_s = [], []
    
    for eps in epsilons:
        subset = results_df[results_df["epsilon"] == eps]
        eps_m.append(eps)
        dp_m.append(subset["Delta_DP"].mean())
        dp_s.append(subset["Delta_DP"].std())
        ba_m.append(subset["balanced_accuracy"].mean())
        ba_s.append(subset["balanced_accuracy"].std())
        esc_m.append(subset["override_trigger_rate"].mean())
        esc_s.append(subset["override_trigger_rate"].std())
        
    # Left panel: Delta_DP vs balanced_accuracy
    axes[0].errorbar(dp_m, ba_m, xerr=dp_s, yerr=ba_s, fmt='o', color='#2b5c8f', ecolor='gray', capsize=4, markersize=8)
    for i, eps in enumerate(eps_m):
        axes[0].annotate(f"ε={eps:.1f}", (dp_m[i], ba_m[i]), xytext=(5, 5), textcoords="offset points", fontweight='bold')
    axes[0].set_xlabel(r"Demographic Parity Disparity ($\Delta_{DP}$ — Lower is Better)", fontweight='bold')
    axes[0].set_ylabel("Balanced Accuracy (Higher is Better)", fontweight='bold')
    axes[0].set_title("Fairness vs. Predictive Performance Trade-off", fontweight='bold')
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Right panel: epsilon vs override_trigger_rate
    axes[1].plot(eps_m, esc_m, 'o-', color='#d95f02', linewidth=2)
    axes[1].fill_between(eps_m, np.array(esc_m) - np.array(esc_s), np.array(esc_m) + np.array(esc_s), color='#d95f02', alpha=0.2)
    axes[1].set_xlabel("Counterfactual Threshold ($\epsilon$)", fontweight='bold')
    axes[1].set_ylabel("Override Trigger Rate", fontweight='bold')
    axes[1].set_title("Operational Escalation Cost", fontweight='bold')
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    fig.suptitle("Threshold sensitivity on the validation split (n=10 seeds)", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    pdf_path = os.path.join(fig_dir, "fig_threshold_sweep.pdf")
    png_path = os.path.join(fig_dir, "fig_threshold_sweep.png")
    
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print("\n--- THRESHOLD SWEEP RESULTS ---")
    print(agg_df.to_markdown(index=False))

if __name__ == "__main__":
    main()
