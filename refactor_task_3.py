import os
import re
import pandas as pd
import scipy.stats
import numpy as np

# Task 1 - Delete the ablation Pareto figure
figures_file = "research/generate_publication_figures.py"
with open(figures_file, "r") as f:
    figures_code = f.read()

# Delete Pareto Frontier section
pareto_pattern = re.compile(r'# ── 2\. Pareto Frontier ────────────────────────────────────────.*?print\("✅ Generated fig2_pareto_frontier \(\.png & \.svg\)"\)\n', re.DOTALL)
figures_code = pareto_pattern.sub('', figures_code)

# Rename figures 3, 4, 5 to 2, 3, 4
figures_code = figures_code.replace('3. Ablation Comparison Figure', '2. Ablation Comparison Figure')
figures_code = figures_code.replace('Figure 3:', 'Figure 2:')
figures_code = figures_code.replace('fig3_ablation_comparison', 'fig2_ablation_comparison')

figures_code = figures_code.replace('4. Fairness Escalation Distribution', '3. Counterfactual Score Difference Distribution')
figures_code = figures_code.replace('Figure 4: Fairness Escalation Distribution', 'Figure 3: Counterfactual Score Difference Distribution')
figures_code = figures_code.replace('fig4_fairness_distribution', 'fig3_fairness_distribution')

figures_code = figures_code.replace('5. Policy Violation Comparison', '4. Policy Violation Comparison')
figures_code = figures_code.replace('Figure 5:', 'Figure 4:')
figures_code = figures_code.replace('fig5_policy_violations', 'fig4_policy_violations')

figures_code = figures_code.replace('All 5 publication figures', 'All 4 publication figures')

# Fix docstring
figures_code = figures_code.replace('2. Pareto Frontier (Fairness vs. Predictive Accuracy Trade-off)\n', '')
figures_code = figures_code.replace('3. Ablation Comparison Figure', '2. Ablation Comparison Figure')
figures_code = figures_code.replace('4. Fairness Escalation Distribution', '3. Counterfactual Score Difference Distribution')
figures_code = figures_code.replace('5. Policy Violation Comparison', '4. Policy Violation Comparison')
figures_code = figures_code.replace('5 publication-quality figures', '4 publication-quality figures')

with open(figures_file, "w") as f:
    f.write(figures_code)

# Remove files
try:
    os.remove("research/publication_assets/figures/fig2_pareto_frontier.png")
    os.remove("research/publication_assets/figures/fig2_pareto_frontier.svg")
except OSError:
    pass

# Task 2 - Paired trend test on the sweep
sweep_path = "research/final_results/threshold_sweep_per_seed.csv"
if os.path.exists(sweep_path):
    df_sweep = pd.read_csv(sweep_path)
    
    eps5 = df_sweep[df_sweep["epsilon"] == 5.0].sort_values("seed")
    eps8 = df_sweep[df_sweep["epsilon"] == 8.0].sort_values("seed")
    
    results = []
    
    metrics = ["Delta_DP", "Delta_EO", "specificity", "balanced_accuracy"]
    for m in metrics:
        # paired t-test
        d1 = eps5[m].values
        d2 = eps8[m].values
        diffs = d1 - d2
        mean_diff = diffs.mean()
        std_diff = diffs.std(ddof=1)
        n = len(diffs)
        se = std_diff / np.sqrt(n) if n > 0 else 0
        
        ci_low, ci_high = scipy.stats.t.interval(0.95, df=n-1, loc=mean_diff, scale=se)
        p_val = scipy.stats.ttest_rel(d1, d2).pvalue
        
        # spearman correlation across all rows
        corr, p_corr = scipy.stats.spearmanr(df_sweep["epsilon"], df_sweep[m])
        
        results.append({
            "contrast": "epsilon 5.0 vs 8.0",
            "metric": m,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value_paired": p_val,
            "n": n,
            "spearman_corr": corr,
            "spearman_p_value": p_corr
        })
        
    trend_df = pd.DataFrame(results)
    trend_df.to_csv("research/final_results/threshold_sweep_trend.csv", index=False)


# Task 3 - Fix stale metadata in run_ablation_study.py
ablation_file = "research/run_ablation_study.py"
with open(ablation_file, "r") as f:
    ab_code = f.read()

ab_code = ab_code.replace("**Random Seed:** `42`", "**Seeds:** 42-51 (n=10)")
ab_code = ab_code.replace("Random Forest (`n_estimators=100`) coupled with V7 LangGraph Multi-Agent Orchestration", "Random Forest (`n_estimators=100`)")

with open(ablation_file, "w") as f:
    f.write(ab_code)
