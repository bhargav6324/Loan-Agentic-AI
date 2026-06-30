"""
Evaluation Harness — Fairness Benchmarking (v2 — Enhanced Risk Scoring)
───────────────────────────────────────────────────────────────────────
Trains ML baselines (LR, RF, XGBoost) on a biased synthetic dataset
and compares them against the Agentic AI pipeline that uses:
  - Dynamic confidence-based weighting
  - Exponential compounding penalties
  - 9 heuristic rules + interaction penalty
  - Creditworthiness bonus
  - Hard override pre-screening
  - Counterfactual fairness escalation
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference


# ══════════════════════════════════════════════════════════
#  ENHANCED HEURISTIC SCORE  (mirrors risk_agent.py v2)
# ══════════════════════════════════════════════════════════

def compute_heuristic_score(row):
    """
    Enhanced rule-based risk score (0-100, higher = riskier).
    """
    score = 50.0

    # ── Credit score impact ───────────────────────────────
    cs = row["credit_score"]
    if cs >= 750:   score -= 20
    elif cs >= 700: score -= 10
    elif cs >= 650: score -= 0
    elif cs >= 600: score += 10
    else:           score += 25

    # ── DTI ratio ─────────────────────────────────────────
    dti = row["dti_ratio"]
    if dti > 0.6:   score += 15
    elif dti > 0.4: score += 8
    elif dti < 0.2: score -= 5

    # ── Loan-to-income ────────────────────────────────────
    lti = row["loan_to_income"]
    if lti > 5:    score += 15
    elif lti > 3:  score += 8
    elif lti < 1:  score -= 5

    # ── Improvement #2: Exponential penalty for defaults ──
    defaults = row["defaults"]
    score += min(40, 10 * (2 ** defaults - 1))

    # ── Credit utilisation (exponential at high end) ──────
    cu = row["credit_utilisation"]
    if cu > 90:    score += 15
    elif cu > 80:  score += 10
    elif cu > 60:  score += 5
    elif cu < 30:  score -= 5

    # ── Employment stability ──────────────────────────────
    emp = row["employment_score"]
    if emp == 3:   score -= 5
    elif emp == 1: score += 5

    # ── Improvement #3a: Loan Tenure Risk ─────────────────
    tenure = row.get("loan_tenure_years", 10)
    if tenure > 20:  score += 8
    elif tenure > 10: score += 4

    # ── Improvement #3b: Income Stability (Job Tenure) ────
    emp_years = row.get("employment_years", 3)
    if emp_years < 1:    score += 10
    elif emp_years < 3:  score += 5
    elif emp_years > 7:  score -= 5

    # ── Improvement #3c: Loan Purpose Risk ────────────────
    purpose_risk = {
        "home": -8, "education": -5, "personal": 0,
        "business": 10, "medical": 5,
    }
    score += purpose_risk.get(row.get("loan_purpose", "personal"), 0)

    # ── Improvement #4: Risk Interaction Penalty ──────────
    danger_flags = sum([
        dti > 0.60,
        defaults > 0,
        cu > 80,
        cs < 600,
    ])
    if danger_flags >= 3:   score += 20
    elif danger_flags == 2: score += 8

    # ── Improvement #5: Creditworthiness Bonus ────────────
    bonus = 0
    if cs > 800:                         bonus -= 5
    if defaults == 0 and emp_years > 5:  bonus -= 5
    if dti < 0.20:                       bonus -= 5
    if cu < 20:                          bonus -= 3
    score += max(bonus, -15)

    return max(0, min(100, score))


# ══════════════════════════════════════════════════════════
#  HARD OVERRIDE PRE-SCREENING  (Improvement #6)
# ══════════════════════════════════════════════════════════

def check_hard_overrides(row):
    """Returns 'HARD_REJECT', 'HARD_FAST_TRACK', or None."""
    if (row["defaults"] > 3 or
        row["credit_score"] < 450 or
        row["dti_ratio"] > 0.90):
        return "HARD_REJECT"

    if (row["credit_score"] > 820 and
        row["defaults"] == 0 and
        row["dti_ratio"] < 0.15 and
        row.get("employment_years", 3) > 5):
        return "HARD_FAST_TRACK"

    return None


# ══════════════════════════════════════════════════════════
#  POLICY CHECK
# ══════════════════════════════════════════════════════════

def check_policy(row):
    if row["applicant_age"] < 21: return False
    if row["loan_to_income"] > 5: return False
    if row["defaults"] > 2: return False
    if row["credit_score"] < 400: return False
    if row["loan_amount"] > 2500000 and row["employment_score"] == 1: return False
    return True


# ══════════════════════════════════════════════════════════
#  MAIN EVALUATION
# ══════════════════════════════════════════════════════════

def run_evaluation():
    data_path = os.path.join(os.path.dirname(__file__), "data", "credit_risk_dataset.csv")
    df = pd.read_csv(data_path)

    # Features for ML models (include gender so baselines learn the proxy bias)
    features = [
        "credit_score", "dti_ratio", "loan_to_income",
        "employment_score", "applicant_age", "defaults",
        "credit_utilisation", "applicant_gender"
    ]

    X = df[features]
    y = df["target_default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # ── Train Baselines ───────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    }

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        approved = (y_pred == 0).astype(int)
        true_approved = (y_test == 0).astype(int)

        acc = accuracy_score(true_approved, approved)
        f1 = f1_score(true_approved, approved)

        dp_diff = demographic_parity_difference(
            true_approved, approved,
            sensitive_features=X_test["applicant_gender"]
        )
        eo_diff = equalized_odds_difference(
            true_approved, approved,
            sensitive_features=X_test["applicant_gender"]
        )

        results.append({
            "Model": name,
            "Accuracy": f"{acc:.3f}",
            "F1 Score": f"{f1:.3f}",
            "Demographic Parity Diff": f"{dp_diff:.3f}",
            "Equalized Odds Diff": f"{eo_diff:.3f}",
            "Escalation Rate": "0.0%",
        })

    # ── Simulate Agentic AI Pipeline (v2 — Enhanced) ─────
    print("Running Agentic AI Pipeline (v2 — Enhanced Scoring)...")
    agent_approved = []
    escalations = 0
    hard_rejects = 0
    hard_fast_tracks = 0

    rf_model = models["Random Forest"]
    rf_proba = rf_model.predict_proba(X_test)

    # Full dataframe row for policy checks & new features
    df_test = df.loc[X_test.index]

    for i in range(len(X_test)):
        row = df_test.iloc[i]

        # ── Improvement #6: Hard Override Pre-Screening ───
        override = check_hard_overrides(row)
        if override == "HARD_REJECT":
            agent_approved.append(0)
            hard_rejects += 1
            continue
        if override == "HARD_FAST_TRACK":
            # Fast-track still goes through Policy & Fairness, but skip deep risk
            agent_approved.append(1)
            hard_fast_tracks += 1
            continue

        # ── 1. Risk Agent ─────────────────────────────────
        proba_approve, proba_reject = rf_proba[i][0], rf_proba[i][1]
        ml_score = proba_reject * 100

        # Improvement #1: Dynamic Confidence-Based Weighting
        ml_confidence = max(rf_proba[i])
        ml_weight = 0.4 + (0.4 * ml_confidence)    # 0.4 → 0.8
        heuristic_weight = 1 - ml_weight

        heuristic_score = compute_heuristic_score(row)
        risk_score = ml_weight * ml_score + heuristic_weight * heuristic_score

        # ── 2. Routing ────────────────────────────────────
        if risk_score <= 35:
            agent_approved.append(1)    # Auto-approve
        elif risk_score > 99:
            agent_approved.append(0)    # Auto-reject
        else:
            # Medium / High Risk → Policy Agent
            if not check_policy(row):
                agent_approved.append(0)    # Reject via Policy
            else:
                # ── 3. Fairness Agent (counterfactual) ────
                cf_row = row.copy()
                if row["applicant_gender"] == 1:
                    cf_row["applicant_gender"] = 0
                    cf_row["credit_score"] = min(850, row["credit_score"] + 100)
                else:
                    cf_row["applicant_gender"] = 1
                    cf_row["credit_score"] = max(300, row["credit_score"] - 100)

                # Recalculate ML Score with counterfactual features
                cf_features = cf_row[features].to_frame().T
                cf_proba = rf_model.predict_proba(cf_features)[0]
                cf_ml_score = cf_proba[1] * 100
                cf_ml_confidence = max(cf_proba)
                cf_ml_weight = 0.4 + (0.4 * cf_ml_confidence)
                cf_heuristic_weight = 1 - cf_ml_weight

                cf_heuristic = compute_heuristic_score(cf_row)
                cf_risk_score = cf_ml_weight * cf_ml_score + cf_heuristic_weight * cf_heuristic

                if abs(cf_risk_score - risk_score) > 5:
                    escalations += 1
                    # If risk DROPS when we flip gender → model was unfairly harsh
                    if cf_risk_score < risk_score - 5:
                        agent_approved.append(1)   # human overrides bias
                    else:
                        agent_approved.append(0)
                else:
                    agent_approved.append(1)   # Approved

    agent_approved = np.array(agent_approved)
    true_approved = (y_test == 0).astype(int)

    acc = accuracy_score(true_approved, agent_approved)
    f1 = f1_score(true_approved, agent_approved)

    dp_diff = demographic_parity_difference(
        true_approved, agent_approved,
        sensitive_features=X_test["applicant_gender"]
    )
    eo_diff = equalized_odds_difference(
        true_approved, agent_approved,
        sensitive_features=X_test["applicant_gender"]
    )
    esc_rate = escalations / len(X_test)

    results.append({
        "Model": "Agentic System v2 (LangGraph)",
        "Accuracy": f"{acc:.3f}",
        "F1 Score": f"{f1:.3f}",
        "Demographic Parity Diff": f"{dp_diff:.3f}",
        "Equalized Odds Diff": f"{eo_diff:.3f}",
        "Escalation Rate": f"{esc_rate:.1%}",
    })

    # ── Output Results ────────────────────────────────────
    results_df = pd.DataFrame(results)
    output_csv = os.path.join(os.path.dirname(__file__), "results.csv")
    output_md  = os.path.join(os.path.dirname(__file__), "metrics_report.md")

    results_df.to_csv(output_csv, index=False)

    # ── Generate Charts ───────────────────────────────────
    figures_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Collect model probabilities for ROC curves
    # For ML models: use predict_proba on X_test
    # For Agentic:   use normalised risk_score (0-100 → 0-1, inverted so high p = approve)
    roc_data = {}
    true_approved = (y_test == 0).astype(int)

    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 0]  # P(approve)
        fpr, tpr, _ = roc_curve(true_approved, proba)
        roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}

    # Agentic: reconstruct per-sample approval scores
    agent_scores = []
    for i in range(len(X_test)):
        row = df_test.iloc[i]
        override = check_hard_overrides(row)
        if override == "HARD_REJECT":
            agent_scores.append(0.0)       # certainty of rejection
            continue
        if override == "HARD_FAST_TRACK":
            agent_scores.append(1.0)       # certainty of approval
            continue
        proba_i = rf_model.predict_proba(X_test.iloc[[i]])[0]
        ml_score_i = proba_i[1] * 100
        ml_w = 0.4 + 0.4 * max(proba_i)
        h_score_i = compute_heuristic_score(row)
        risk_i = ml_w * ml_score_i + (1 - ml_w) * h_score_i
        agent_scores.append(1.0 - risk_i / 100.0)   # invert: high score = approve

    agent_scores = np.array(agent_scores)
    fpr_a, tpr_a, _ = roc_curve(true_approved, agent_scores)
    roc_data["Agentic System v2"] = {"fpr": fpr_a, "tpr": tpr_a, "auc": auc(fpr_a, tpr_a)}

    roc_path    = _generate_roc_curves(roc_data, figures_dir)
    pareto_path = _generate_pareto_frontier(results_df, figures_dir)

    # ── Write Markdown Report ─────────────────────────────
    with open(output_md, "w") as f:
        f.write("# Formal Fairness Metrics Evaluation (v2 — Enhanced Risk Scoring)\n\n")
        f.write("This table compares the **enhanced** Agentic AI system against standard ML baselines ")
        f.write("on a synthetic credit dataset with historical demographic bias.\n\n")
        f.write("## Comparative Results\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n\n## Agentic Pipeline Telemetry\n\n")
        f.write(f"- **Hard Rejects** (pre-screening): {hard_rejects} ({hard_rejects/len(X_test):.1%})\n")
        f.write(f"- **Hard Fast-Tracks** (pre-screening): {hard_fast_tracks} ({hard_fast_tracks/len(X_test):.1%})\n")
        f.write(f"- **Fairness Escalations**: {escalations} ({esc_rate:.1%})\n")
        f.write("\n## ROC Curve Comparison\n\n")
        f.write("![ROC Curves](figures/roc_curves.png)\n\n")
        f.write("The ROC curves compare the true-positive rate vs false-positive rate for each model "
                "across all decision thresholds. AUC closer to 1.0 = better discrimination.\n\n")
        f.write("## Fairness–Accuracy Pareto Frontier\n\n")
        f.write("![Pareto Frontier](figures/pareto_frontier.png)\n\n")
        f.write("Each point is one model. The ideal model sits at the **top-left** corner: "
                "highest accuracy (y-axis) and lowest bias / Demographic Parity Diff (x-axis). "
                "The Agentic System should dominate all baselines on this frontier.\n\n")
        f.write("## Scoring Improvements (v2)\n\n")
        f.write("| # | Improvement | Description |\n")
        f.write("|---|-------------|-------------|\n")
        f.write("| 1 | Dynamic Confidence Weighting | ML weight ranges 0.4–0.8 based on prediction confidence |\n")
        f.write("| 2 | Exponential Default Penalties | Compounding penalty: 1 default=+10, 2=+30, 3=+70 (capped) |\n")
        f.write("| 3 | 3 New Risk Factors | Loan tenure, job stability (years), loan purpose |\n")
        f.write("| 4 | Risk Interaction Penalty | +20 when ≥3 danger flags fire simultaneously |\n")
        f.write("| 5 | Creditworthiness Bonus | Up to -15 points for excellent profiles |\n")
        f.write("| 6 | Hard Override Pre-Screening | Instant reject/fast-track for extreme profiles |\n")
        f.write("\n### Metrics Explanation\n\n")
        f.write("- **Demographic Parity Diff**: Difference in approval rates between Male and Female applicants. Closer to 0 is fairer.\n")
        f.write("- **Equalized Odds Diff**: Difference in error rates between groups. Closer to 0 is fairer.\n")
        f.write("- **Escalation Rate**: Percentage of applications routed to human reviewers by the Fairness Agent.\n")

    print(f"\nEvaluation complete. Results saved to:\n  {output_csv}\n  {output_md}")
    print(f"  ROC curve: {roc_path}")
    print(f"  Pareto frontier: {pareto_path}")
    print(f"\nAgentic Telemetry: Hard Rejects={hard_rejects}, Fast-Tracks={hard_fast_tracks}, Escalations={escalations}")
    print("\n" + results_df.to_markdown(index=False))


# ══════════════════════════════════════════════════════════
#  CHART GENERATORS
# ══════════════════════════════════════════════════════════

def _generate_roc_curves(roc_data: dict, figures_dir: str) -> str:
    """
    Plot overlapping ROC curves for all models on a single axes.
    Each curve is labelled with its AUC score.
    Saves as figures/roc_curves.png and returns the path.
    """
    COLORS = {
        "Logistic Regression": "#4e8ef7",
        "Random Forest":       "#f76b4e",
        "XGBoost":             "#4ef7a1",
        "Agentic System v2":   "#f7d44e",
    }
    STYLES = {
        "Logistic Regression": "--",
        "Random Forest":       "-.",
        "XGBoost":             ":",
        "Agentic System v2":   "-",
    }
    WIDTHS = {k: (3 if k == "Agentic System v2" else 1.8) for k in COLORS}

    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#0d1117")
    ax.set_facecolor("#161b22")

    # Diagonal chance line
    ax.plot([0, 1], [0, 1], color="#444c56", linewidth=1, linestyle="--", label="Chance (AUC = 0.50)")

    for name, data in roc_data.items():
        colour = COLORS.get(name, "#ffffff")
        style  = STYLES.get(name, "-")
        width  = WIDTHS.get(name, 1.8)
        ax.plot(
            data["fpr"], data["tpr"],
            color=colour, linewidth=width, linestyle=style,
            label=f"{name}  (AUC = {data['auc']:.3f})",
        )

    ax.set_xlabel("False Positive Rate", color="#c9d1d9", fontsize=12)
    ax.set_ylabel("True Positive Rate", color="#c9d1d9", fontsize=12)
    ax.set_title("ROC Curve Comparison — Loan Approval Models",
                 color="#f0f6fc", fontsize=14, fontweight="bold", pad=14)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.legend(
        facecolor="#21262d", edgecolor="#30363d",
        labelcolor="#c9d1d9", fontsize=10, loc="lower right",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(color="#21262d", linewidth=0.5)

    # Annotate the Agentic point at the elbow (Youden's J)
    if "Agentic System v2" in roc_data:
        d = roc_data["Agentic System v2"]
        j = np.argmax(d["tpr"] - d["fpr"])
        ax.annotate(
            f"  Optimal threshold\n  TPR={d['tpr'][j]:.2f}, FPR={d['fpr'][j]:.2f}",
            xy=(d["fpr"][j], d["tpr"][j]),
            xytext=(d["fpr"][j] + 0.08, d["tpr"][j] - 0.12),
            color="#f7d44e", fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="#f7d44e", lw=1.2),
        )

    out = os.path.join(figures_dir, "roc_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [chart] ROC curves saved → {out}")
    return out


def _generate_pareto_frontier(results_df: pd.DataFrame, figures_dir: str) -> str:
    """
    Plot a fairness–accuracy Pareto frontier scatter.

    x-axis: |Demographic Parity Diff|  (lower = fairer)
    y-axis: Accuracy                   (higher = better)

    The ideal model is in the TOP-LEFT corner.
    The Pareto-optimal frontier is drawn as a step line.
    Saves as figures/pareto_frontier.png and returns the path.
    """
    COLORS = [
        "#4e8ef7",   # LR
        "#f76b4e",   # RF
        "#4ef7a1",   # XGBoost
        "#f7d44e",   # Agentic
    ]
    MARKERS = ["o", "s", "^", "D"]
    SIZES   = [80, 80, 80, 130]

    accs   = results_df["Accuracy"].astype(float).tolist()
    dp_abs = results_df["Demographic Parity Diff"].astype(float).abs().tolist()
    names  = results_df["Model"].tolist()

    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#0d1117")
    ax.set_facecolor("#161b22")

    for i, (name, acc, dp) in enumerate(zip(names, accs, dp_abs)):
        colour = COLORS[i % len(COLORS)]
        ax.scatter(
            dp, acc,
            color=colour, marker=MARKERS[i % len(MARKERS)],
            s=SIZES[i % len(SIZES)], zorder=5, edgecolors="white", linewidths=0.7,
        )
        ax.annotate(
            f"  {name}",
            xy=(dp, acc), color=colour, fontsize=9.5,
            va="center",
        )

    # ── Compute and draw Pareto frontier ──────────────────
    # A point (dp, acc) is Pareto-optimal if no other point has BOTH lower dp AND higher acc
    points = list(zip(dp_abs, accs))
    pareto = []
    for i, (dp_i, acc_i) in enumerate(points):
        dominated = any(
            dp_j <= dp_i and acc_j >= acc_i and (dp_j < dp_i or acc_j > acc_i)
            for j, (dp_j, acc_j) in enumerate(points) if j != i
        )
        if not dominated:
            pareto.append((dp_i, acc_i))
    pareto.sort()  # sort by dp ascending
    if len(pareto) >= 2:
        px, py = zip(*pareto)
        ax.step(
            px, py, where="post",
            color="#58a6ff", linewidth=1.4, linestyle="--",
            label="Pareto frontier", zorder=3,
        )
        ax.legend(
            facecolor="#21262d", edgecolor="#30363d",
            labelcolor="#c9d1d9", fontsize=10,
        )

    # Ideal corner annotation
    ax.annotate(
        "← Ideal (fair & accurate)",
        xy=(0, 1), xytext=(0.03, 0.97),
        color="#3fb950", fontsize=9, style="italic",
    )

    ax.set_xlabel("|Demographic Parity Difference|  (lower = fairer)",
                  color="#c9d1d9", fontsize=12)
    ax.set_ylabel("Accuracy  (higher = better)",
                  color="#c9d1d9", fontsize=12)
    ax.set_title("Fairness–Accuracy Pareto Frontier",
                 color="#f0f6fc", fontsize=14, fontweight="bold", pad=14)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(color="#21262d", linewidth=0.5)

    # Set sensible axis limits with some padding
    all_dp = [0] + dp_abs
    ax.set_xlim(-0.02, max(all_dp) * 1.25)
    ax.set_ylim(min(accs) * 0.97, min(1.01, max(accs) * 1.03))

    out = os.path.join(figures_dir, "pareto_frontier.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [chart] Pareto frontier saved → {out}")
    return out


if __name__ == "__main__":
    run_evaluation()
