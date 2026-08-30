"""
Comprehensive Metrics Report — ML vs Agentic System
────────────────────────────────────────────────────
Computes ALL standard ML metrics for each model:
  - Accuracy, Precision, Recall, F1, AUC-ROC
  - Confusion Matrix (TP, TN, FP, FN)
  - Specificity, Sensitivity, MCC
  - Demographic Parity Diff, Equalized Odds Diff
  - Per-group (Male/Female) approval rates
  - Escalation / Hard Override telemetry

Run: python research/full_metrics.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef,
    classification_report
)
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import agentic pipeline functions from the existing harness
from research.evaluation_harness import compute_heuristic_score, check_hard_overrides, check_policy


def compute_all_metrics(y_true, y_pred, y_proba, gender_features, model_name):
    """Compute comprehensive metrics for a single model."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # AUC-ROC
    try:
        auc_roc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc_roc = float("nan")

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # same as recall

    # Fairness
    dp_diff = demographic_parity_difference(
        y_true, y_pred, sensitive_features=gender_features
    )
    eo_diff = equalized_odds_difference(
        y_true, y_pred, sensitive_features=gender_features
    )

    # Per-group approval rates
    male_mask = gender_features == 0
    female_mask = gender_features == 1

    male_approval = y_pred[male_mask].mean() if male_mask.sum() > 0 else 0
    female_approval = y_pred[female_mask].mean() if female_mask.sum() > 0 else 0

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall (Sensitivity)": rec,
        "F1 Score": f1,
        "AUC-ROC": auc_roc,
        "Specificity": specificity,
        "MCC": mcc,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Male Approval Rate": male_approval,
        "Female Approval Rate": female_approval,
        "Approval Gap (M-F)": male_approval - female_approval,
        "Demographic Parity Diff": dp_diff,
        "Equalized Odds Diff": eo_diff,
    }


def run_full_metrics():
    data_path = os.path.join(os.path.dirname(__file__), "data", "credit_risk_dataset.csv")
    df = pd.read_csv(data_path)

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

    true_approved = (y_test == 0).astype(int)
    gender_features = X_test["applicant_gender"].values

    # ── Train ML Baselines ────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    }

    all_results = []

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        y_pred_raw = model.predict(X_test)
        approved = (y_pred_raw == 0).astype(int)
        proba = model.predict_proba(X_test)[:, 0]  # P(approve)

        metrics = compute_all_metrics(
            true_approved, approved, proba, gender_features, name
        )
        metrics["Override Trigger Rate"] = 0.0
        metrics["Hard Rejects"] = 0
        metrics["Hard Fast-Tracks"] = 0
        all_results.append(metrics)

    # ── Simulate Agentic AI Pipeline ─────────────────────
    print("Running Agentic AI Pipeline (v2 — Enhanced Scoring)...")
    agent_approved = []
    agent_scores = []
    escalations = 0
    hard_rejects = 0
    hard_fast_tracks = 0

    rf_model = models["Random Forest"]
    rf_proba = rf_model.predict_proba(X_test)
    df_test = df.loc[X_test.index]

    for i in range(len(X_test)):
        row = df_test.iloc[i]

        override = check_hard_overrides(row)
        if override == "HARD_REJECT":
            agent_approved.append(0)
            agent_scores.append(0.0)
            hard_rejects += 1
            continue
        if override == "HARD_FAST_TRACK":
            agent_approved.append(1)
            agent_scores.append(1.0)
            hard_fast_tracks += 1
            continue

        proba_i = rf_proba[i]
        ml_score = proba_i[1] * 100
        ml_confidence = max(proba_i)
        ml_weight = 0.4 + (0.4 * ml_confidence)
        heuristic_weight = 1 - ml_weight

        heuristic_score = compute_heuristic_score(row)
        risk_score = ml_weight * ml_score + heuristic_weight * heuristic_score
        agent_scores.append(1.0 - risk_score / 100.0)

        if risk_score <= 35:
            agent_approved.append(1)
        elif risk_score > 99:
            agent_approved.append(0)
        else:
            if not check_policy(row):
                agent_approved.append(0)
            else:
                cf_row = row.copy()
                if row["applicant_gender"] == 1:
                    cf_row["applicant_gender"] = 0
                else:
                    cf_row["applicant_gender"] = 1

                cf_features_row = cf_row[features].to_frame().T
                cf_proba = rf_model.predict_proba(cf_features_row)[0]
                cf_ml_score = cf_proba[1] * 100
                cf_ml_confidence = max(cf_proba)
                cf_ml_weight = 0.4 + (0.4 * cf_ml_confidence)
                cf_heuristic_weight = 1 - cf_ml_weight

                cf_heuristic = compute_heuristic_score(cf_row)
                cf_risk_score = cf_ml_weight * cf_ml_score + cf_heuristic_weight * cf_heuristic

                if abs(cf_risk_score - risk_score) > 5:
                    escalations += 1
                    if cf_risk_score < risk_score - 5:
                        agent_approved.append(1)
                    else:
                        agent_approved.append(0)
                else:
                    agent_approved.append(1)

    agent_approved = np.array(agent_approved)
    agent_scores = np.array(agent_scores)

    agent_metrics = compute_all_metrics(
        true_approved, agent_approved, agent_scores, gender_features,
        "Agentic System v2 (LangGraph)"
    )
    agent_metrics["Override Trigger Rate"] = escalations / len(X_test)
    agent_metrics["Hard Rejects"] = hard_rejects
    agent_metrics["Hard Fast-Tracks"] = hard_fast_tracks
    all_results.append(agent_metrics)

    # ═══════════════════════════════════════════════════════
    #  OUTPUT REPORT
    # ═══════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("COMPREHENSIVE METRICS REPORT — ML vs AGENTIC SYSTEM")
    print("=" * 70)

    # ── 1. Performance Metrics Table ──────────────────────
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    PERFORMANCE METRICS                         │")
    print("└─────────────────────────────────────────────────────────────────┘")

    perf_df = pd.DataFrame(all_results)[
        ["Model", "Accuracy", "Precision", "Recall (Sensitivity)", "F1 Score", "AUC-ROC", "Specificity", "MCC"]
    ]
    for col in perf_df.columns[1:]:
        perf_df[col] = perf_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    print(perf_df.to_markdown(index=False))

    # ── 2. Confusion Matrix Table ─────────────────────────
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    CONFUSION MATRIX                            │")
    print("└─────────────────────────────────────────────────────────────────┘")

    cm_df = pd.DataFrame(all_results)[["Model", "TP", "TN", "FP", "FN"]]
    print(cm_df.to_markdown(index=False))

    # ── 3. Fairness Metrics Table ─────────────────────────
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    FAIRNESS METRICS                            │")
    print("└─────────────────────────────────────────────────────────────────┘")

    fair_df = pd.DataFrame(all_results)[
        ["Model", "Male Approval Rate", "Female Approval Rate", "Approval Gap (M-F)",
         "Demographic Parity Diff", "Equalized Odds Diff"]
    ]
    for col in fair_df.columns[1:]:
        fair_df[col] = fair_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    print(fair_df.to_markdown(index=False))

    # ── 4. Agentic Telemetry ──────────────────────────────
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    AGENTIC PIPELINE TELEMETRY                  │")
    print("└─────────────────────────────────────────────────────────────────┘")

    agent = all_results[-1]
    total = len(X_test)
    print(f"  Total test samples:        {total}")
    print(f"  Hard Rejects:              {agent['Hard Rejects']:4d} ({agent['Hard Rejects']/total:.1%})")
    print(f"  Hard Fast-Tracks:          {agent['Hard Fast-Tracks']:4d} ({agent['Hard Fast-Tracks']/total:.1%})")
    print(f"  Fairness Escalations:      {escalations:4d} ({escalations/total:.1%})")
    print(f"  Normal Processing:         {total - agent['Hard Rejects'] - agent['Hard Fast-Tracks']:4d}")

    # ── 5. Trade-off Summary ──────────────────────────────
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    TRADE-OFF SUMMARY                           │")
    print("└─────────────────────────────────────────────────────────────────┘")

    rf_result = all_results[1]  # Random Forest baseline
    agent_result = all_results[-1]

    dp_reduction = (1 - abs(agent_result["Demographic Parity Diff"]) / abs(rf_result["Demographic Parity Diff"])) * 100
    eo_reduction = (1 - abs(agent_result["Equalized Odds Diff"]) / abs(rf_result["Equalized Odds Diff"])) * 100
    gap_reduction = (1 - abs(agent_result["Approval Gap (M-F)"]) / abs(rf_result["Approval Gap (M-F)"])) * 100
    acc_cost = (rf_result["Accuracy"] - agent_result["Accuracy"]) * 100
    spec_drop = rf_result["Specificity"] - agent_result["Specificity"]
    mcc_drop = rf_result["MCC"] - agent_result["MCC"]

    print(f"  Demographic Parity improvement: {dp_reduction:.1f}% reduction in bias")
    print(f"  Equalized Odds improvement:     {eo_reduction:.1f}% reduction in bias")
    print(f"  Approval Gap reduction:         {gap_reduction:.1f}% reduction in bias")
    print(f"  --- Costs ---")
    print(f"  Accuracy Cost:                  {acc_cost:.1f}% drop")
    print(f"  Specificity Cost:               {spec_drop:.3f} drop ({rf_result['Specificity']:.2f} → {agent_result['Specificity']:.2f})")
    print(f"  MCC Cost:                       {mcc_drop:.3f} drop ({rf_result['MCC']:.2f} → {agent_result['MCC']:.2f})")

    # ── Save to files ─────────────────────────────────────
    output_dir = os.path.dirname(__file__)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, "full_metrics.csv"), index=False)

    with open(os.path.join(output_dir, "full_metrics_report.md"), "w") as f:
        f.write("# Comprehensive Metrics Report — ML vs Agentic System\n\n")

        f.write("## 1. Performance Metrics\n\n")
        f.write(perf_df.to_markdown(index=False))

        f.write("\n\n## 2. Confusion Matrix\n\n")
        f.write(cm_df.to_markdown(index=False))

        f.write("\n\n## 3. Fairness Metrics\n\n")
        f.write(fair_df.to_markdown(index=False))

        f.write("\n\n## 4. Agentic Pipeline Telemetry\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Total Test Samples | {total} |\n")
        f.write(f"| Hard Rejects | {agent['Hard Rejects']} ({agent['Hard Rejects']/total:.1%}) |\n")
        f.write(f"| Hard Fast-Tracks | {agent['Hard Fast-Tracks']} ({agent['Hard Fast-Tracks']/total:.1%}) |\n")
        f.write(f"| Fairness Escalations | {escalations} ({escalations/total:.1%}) |\n")

        f.write("\n\n## 5. Trade-off Summary\n\n")
        f.write(f"| Metric | Change |\n|---|---|\n")
        f.write(f"| Demographic Parity | {dp_reduction:.1f}% reduction in bias |\n")
        f.write(f"| Equalized Odds | {eo_reduction:.1f}% reduction in bias |\n")
        f.write(f"| Approval Gap (M-F) | {gap_reduction:.1f}% reduction in bias |\n")
        f.write(f"| **Accuracy Cost** | {acc_cost:.1f}% drop |\n")
        f.write(f"| **Specificity Cost** | {spec_drop:.3f} drop ({rf_result['Specificity']:.2f} → {agent_result['Specificity']:.2f}) |\n")
        f.write(f"| **MCC Cost** | {mcc_drop:.3f} drop ({rf_result['MCC']:.2f} → {agent_result['MCC']:.2f}) |\n")

    print(f"\n✅ Full report saved to:")
    print(f"   {os.path.join(output_dir, 'full_metrics.csv')}")
    print(f"   {os.path.join(output_dir, 'full_metrics_report.md')}")


if __name__ == "__main__":
    run_full_metrics()
