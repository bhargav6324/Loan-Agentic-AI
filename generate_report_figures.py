"""
Generate report figures: Feature Importance Bar Chart + Confusion Matrix
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "report_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Feature Importance Bar Chart ──────────────────────
def generate_feature_importance():
    import joblib
    model = joblib.load(os.path.join(os.path.dirname(__file__), "ml", "model", "risk_model.joblib"))
    
    feature_names = [
        "Credit Score", "DTI Ratio", "Loan-to-Income",
        "Employment Score", "Age", "Defaults", "Credit Utilisation"
    ]
    importances = model.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importances)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1a73e8', '#1a73e8', '#1a73e8', '#1a73e8', '#1a73e8', '#1a73e8', '#1a73e8']
    bars = ax.barh(range(len(indices)), importances[indices], color='#2563eb', edgecolor='#1e40af', linewidth=0.5)
    
    # Highlight top feature
    bars[-1].set_color('#dc2626')
    bars[-1].set_edgecolor('#991b1b')
    
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=11)
    ax.set_xlabel('Feature Importance (Gini)', fontsize=12)
    ax.set_title('RandomForest Feature Importance — Risk Model', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, importances[indices])):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{imp:.3f}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Feature importance chart saved: {path}")


# ── 2. Confusion Matrix ─────────────────────────────────
def generate_confusion_matrix():
    from ml.train_model import generate_synthetic_data
    
    np.random.seed(42)
    df = generate_synthetic_data(2000)
    
    feature_cols = ["credit_score", "dti_ratio", "loan_to_income",
                    "employment_score", "age", "defaults", "credit_utilisation"]
    X = df[feature_cols]
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    import joblib
    model = joblib.load(os.path.join(os.path.dirname(__file__), "ml", "model", "risk_model.joblib"))
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Approve", "Review", "Reject"])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix — Risk Classification Model', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {path}")
    
    # Print actual support numbers
    print(f"\n📊 Actual test set support:")
    print(f"   Approve: {(y_test == 0).sum()}")
    print(f"   Review:  {(y_test == 1).sum()}")
    print(f"   Reject:  {(y_test == 2).sum()}")


# ── 3. Real Test Case 3 — Fairness Escalation ───────────
def run_test_case_3():
    from orchestrator.graph import loan_pipeline
    import json
    
    print("\n" + "="*60)
    print("TEST CASE 3: Borderline Applicant — Fairness Check")
    print("="*60)
    
    # Design a MEDIUM-risk profile (score should land 36-60)
    result = loan_pipeline.invoke({
        "application_id": "test-3-fairness",
        "user_id": "u3",
        "applicant_data": {
            "loan_amount": 1500000,
            "loan_purpose": "Business Expansion",
            "applicant_name": "Priya Mehta",
            "applicant_age": 32,
            "applicant_gender": "female",
            "applicant_pan": "ABCDE1234F",
            "applicant_aadhaar": "123456789012",
            "monthly_income": 55000,
            "employment_type": "business",
            "employer_name": "Self-owned Business",
        },
        "agent_logs": [],
        "errors": [],
    })
    
    print(f"\n📋 Applicant: Priya Mehta")
    print(f"   Loan: ₹15,00,000 | Income: ₹55,000/mo | Age: 32")
    print(f"   Employment: Business | Gender: Female")
    
    print(f"\n📊 Risk Assessment:")
    print(f"   Risk Score: {result['risk_score']}/100")
    print(f"   Risk Category: {result['risk_category']}")
    print(f"   Risk Factors: {result.get('risk_factors', [])}")
    
    print(f"\n📜 Policy Check:")
    print(f"   Passed: {result.get('policy_passed', 'N/A')}")
    print(f"   Violations: {result.get('policy_violations', [])}")
    
    print(f"\n⚖️ Fairness Analysis:")
    fr = result.get("fairness_report", {})
    print(f"   Bias Detected: {result.get('bias_detected', 'N/A')}")
    if fr:
        print(f"   Recommendation: {fr.get('recommendation', 'N/A')}")
        print(f"   Summary: {fr.get('summary', 'N/A')}")
        for cr in fr.get("counterfactual_results", []):
            print(f"   - {cr['attribute']}: {cr['original_value']} → {cr['counterfactual_value']}")
            print(f"     Score: {cr['original_score']} → {cr['counterfactual_score']} (diff: {cr['score_difference']})")
            print(f"     Bias flag: {cr['bias_flag']}")
    
    print(f"\n🎯 Final Decision: {result['final_decision'].upper()}")
    print(f"   Reasoning: {result.get('decision_reasoning', 'N/A')}")
    
    print(f"\n📝 Agent Logs ({len(result['agent_logs'])} steps):")
    for i, log in enumerate(result['agent_logs']):
        print(f"   Step {i+1} [{log['agent']}]: {log['action']} → {log['status']}")
        print(f"           {log.get('details', '')}")
    
    print(f"\n❌ Errors: {result.get('errors', [])}")
    
    return result


if __name__ == "__main__":
    generate_feature_importance()
    generate_confusion_matrix()
    run_test_case_3()
