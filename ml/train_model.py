"""
ML Model Training — Bias-Aware Risk Classifier (V3)
────────────────────────────────────────────────────
Generates synthetic Indian loan applicant data WITH demographic
features (gender, location) and trains a RandomForestClassifier.

V3 Change: Gender and location are included as features so the model
learns bias from the data distribution (biased dataset), NOT from
hardcoded if-statements.  The Fairness Agent detects this learned
bias via counterfactual testing — this is how real fairness research
works.

Classes: 0 = approve, 1 = review, 2 = reject

Run: python ml/train_model.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

np.random.seed(42)

N_SAMPLES = 3000  # Increased for better demographic representation


def generate_synthetic_data(n: int) -> pd.DataFrame:
    """Generate realistic Indian loan applicant profiles with demographic bias."""

    # ── Core financial features ────────────────────────────
    data = {
        "credit_score": np.random.normal(680, 80, n).clip(300, 900).astype(int),
        "dti_ratio": np.random.exponential(0.3, n).clip(0, 1.5).round(3),
        "loan_to_income": np.random.exponential(2.0, n).clip(0.1, 15).round(3),
        "employment_score": np.random.choice([1, 2, 3], n, p=[0.25, 0.30, 0.45]),
        "age": np.random.normal(35, 10, n).clip(18, 70).astype(int),
        "defaults": np.random.choice([0, 0, 0, 0, 1, 1, 2, 3], n),
        "credit_utilisation": np.random.normal(40, 20, n).clip(0, 100).round(1),
    }

    # ── Demographic features (bias-carrying) ──────────────
    # Gender: skewed distribution reflecting real-world loan data
    data["gender_encoded"] = np.random.choice(
        [0, 1, 2],  # 0=male, 1=female, 2=other
        n,
        p=[0.65, 0.30, 0.05],  # Male-dominant applicant pool
    )

    # Location: urban/semi-urban/rural distribution
    data["location_encoded"] = np.random.choice(
        [0, 1, 2],  # 0=urban, 1=semi_urban, 2=rural
        n,
        p=[0.40, 0.35, 0.25],
    )

    df = pd.DataFrame(data)

    # ── Generate BIASED labels ────────────────────────────
    # The bias comes from the labeling function, simulating historical
    # human decision-making that discriminated based on demographics.
    labels = []
    for _, row in df.iterrows():
        score = 50.0

        # Credit score impact
        if row["credit_score"] >= 750:
            score -= 20
        elif row["credit_score"] >= 700:
            score -= 10
        elif row["credit_score"] >= 600:
            score += 10
        else:
            score += 25

        # DTI
        if row["dti_ratio"] > 0.6:
            score += 15
        elif row["dti_ratio"] > 0.4:
            score += 8

        # Loan to income
        if row["loan_to_income"] > 5:
            score += 15
        elif row["loan_to_income"] > 3:
            score += 8

        # Defaults
        score += row["defaults"] * 12

        # Credit utilisation
        if row["credit_utilisation"] > 80:
            score += 10
        elif row["credit_utilisation"] > 60:
            score += 5

        # Employment
        if row["employment_score"] == 3:
            score -= 5
        elif row["employment_score"] == 1:
            score += 5

        # ══ INTENTIONAL BIAS IN TRAINING DATA ══════════════
        # This simulates historical discrimination that existed in
        # real lending data.  The ML model will learn these patterns.
        # The Fairness Agent's counterfactual testing will detect them.

        # Gender bias: females historically rejected more often
        if row["gender_encoded"] == 1:   # female
            score += 7  # Bias: penalises female applicants
        elif row["gender_encoded"] == 2:  # other
            score += 10  # Bias: penalises non-binary applicants

        # Location bias: rural applicants historically rejected more
        if row["location_encoded"] == 2:  # rural
            score += 6  # Bias: penalises rural applicants
        elif row["location_encoded"] == 0:  # urban
            score -= 3  # Bias: favours urban applicants

        # Add noise for realistic ML training
        score += np.random.normal(0, 8)

        if score <= 35:
            labels.append(0)  # approve
        elif score <= 60:
            labels.append(1)  # review
        else:
            labels.append(2)  # reject

    df["label"] = labels
    return df


def train_model():
    """Train and save the bias-aware risk model."""
    print("📊 Generating synthetic training data WITH demographic features...")
    df = generate_synthetic_data(N_SAMPLES)

    print(f"   Total samples: {len(df)}")
    print(f"   Class distribution:")
    for label, name in [(0, "approve"), (1, "review"), (2, "reject")]:
        count = (df["label"] == label).sum()
        print(f"     {name}: {count} ({count/len(df)*100:.1f}%)")

    # Show demographic bias in the training data
    print(f"\n📊 Demographic bias in training data:")
    for gender, label in [(0, "Male"), (1, "Female"), (2, "Other")]:
        subset = df[df["gender_encoded"] == gender]
        reject_rate = (subset["label"] == 2).mean()
        approve_rate = (subset["label"] == 0).mean()
        print(f"   {label}: approve={approve_rate:.1%}, reject={reject_rate:.1%} (n={len(subset)})")

    for loc, label in [(0, "Urban"), (1, "Semi-Urban"), (2, "Rural")]:
        subset = df[df["location_encoded"] == loc]
        reject_rate = (subset["label"] == 2).mean()
        approve_rate = (subset["label"] == 0).mean()
        print(f"   {label}: approve={approve_rate:.1%}, reject={reject_rate:.1%} (n={len(subset)})")

    # Split — NOW INCLUDING gender_encoded and location_encoded
    feature_cols = [
        "credit_score", "dti_ratio", "loan_to_income",
        "employment_score", "age", "defaults", "credit_utilisation",
        "gender_encoded", "location_encoded",  # NEW: demographic features
    ]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    print("\n🤖 Training RandomForest classifier (with demographic features)...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n📈 Test Set Performance:")
    print(classification_report(y_test, y_pred, target_names=["approve", "review", "reject"]))

    # Feature importance — shows how much the model relies on demographics
    print("📊 Feature Importance (watch for demographic feature weight):")
    for name, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
        marker = " ⚠ DEMOGRAPHIC" if name in ("gender_encoded", "location_encoded") else ""
        print(f"   {name}: {imp:.3f}{marker}")

    # Save model
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "risk_model.joblib")
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved to {model_path}")
    print(f"   Features: {feature_cols}")
    print(f"   ⚠ Model contains learned demographic bias — Fairness Agent will detect it.")

    return model


if __name__ == "__main__":
    train_model()
