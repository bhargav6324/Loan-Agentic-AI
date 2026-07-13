"""
ML Model Training — Bias-Aware Risk Classifier (V4)
────────────────────────────────────────────────────
V4 Change: Uses the real Kaggle Credit Risk Dataset (~32K rows)
instead of purely synthetic data. Falls back to the V3 synthetic
generator if the CSV is missing.

The Kaggle dataset (by Laotse) has 12 columns from simulated
credit bureau data. We map them to our 9-feature schema:
  credit_score, dti_ratio, loan_to_income, employment_score,
  age, defaults, credit_utilisation, gender_encoded, location_encoded

Gender and location are synthetically injected with intentional bias
(since the Kaggle dataset doesn't include demographics). This preserves
the Fairness Agent's counterfactual testing capability.

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

N_SAMPLES = 3000  # Fallback synthetic sample count

KAGGLE_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "kaggle", "credit_risk_dataset.csv")


# ══════════════════════════════════════════════════════════
#  KAGGLE DATA LOADER (V4)
# ══════════════════════════════════════════════════════════

def load_kaggle_data() -> pd.DataFrame:
    """
    Load and transform the Kaggle Credit Risk Dataset into our
    9-feature model schema.

    Kaggle columns → Our features:
      person_age           → age
      person_income        → (used for loan_to_income)
      person_emp_length    → employment_score (mapped: <2y=1, 2-5y=2, >5y=3)
      loan_amnt            → (used for loan_to_income)
      loan_percent_income  → dti_ratio
      loan_grade           → credit_score (A=800, B=720, C=650, D=580, E=520, F=460, G=400)
      loan_int_rate        → credit_utilisation (proxy: higher rates ≈ higher utilisation)
      loan_status          → label (0=non-default, 1=default → mapped to 3 classes)
      cb_person_default_on_file → defaults (Y=1, N=0)
      cb_person_cred_hist_length → (used as bonus in label engineering)
    """
    print(f"📂 Loading Kaggle Credit Risk Dataset from {KAGGLE_CSV}...")
    raw = pd.read_csv(KAGGLE_CSV)
    print(f"   Raw rows: {len(raw)} | Columns: {list(raw.columns)}")

    # Drop rows with critical NaN values
    raw = raw.dropna(subset=["person_age", "person_income", "loan_amnt", "loan_status"])
    # Cap extreme outlier ages (dataset has some ages > 100)
    raw = raw[raw["person_age"] <= 70]
    raw = raw[raw["person_age"] >= 18]
    # Cap extreme employment lengths
    raw["person_emp_length"] = raw["person_emp_length"].fillna(2).clip(0, 40)
    # Fill missing interest rates with median
    raw["loan_int_rate"] = raw["loan_int_rate"].fillna(raw["loan_int_rate"].median())

    n = len(raw)
    print(f"   After cleanup: {n} rows")

    # ── Map to our feature schema ─────────────────────────

    # 1. Age (direct)
    age = raw["person_age"].astype(int).values

    # 2. Credit Score (derived from loan_grade)
    grade_to_score = {"A": 800, "B": 720, "C": 650, "D": 580, "E": 520, "F": 460, "G": 400}
    credit_score = raw["loan_grade"].map(grade_to_score).fillna(600).astype(int).values
    # Add noise to make it more realistic
    credit_score = (credit_score + np.random.normal(0, 30, n)).clip(300, 900).astype(int)

    # 3. DTI Ratio (from loan_percent_income)
    dti_ratio = raw["loan_percent_income"].clip(0, 1.5).round(3).values

    # 4. Loan-to-Income (derived from loan_amnt / person_income)
    annual_income = raw["person_income"].values
    loan_amount = raw["loan_amnt"].values
    loan_to_income = np.where(
        annual_income > 0,
        (loan_amount / annual_income).clip(0.1, 15),
        10.0
    ).round(3)

    # 5. Employment Score (from person_emp_length)
    emp_length = raw["person_emp_length"].values
    employment_score = np.where(emp_length >= 5, 3, np.where(emp_length >= 2, 2, 1))

    # 6. Defaults (from cb_person_default_on_file)
    defaults = (raw["cb_person_default_on_file"] == "Y").astype(int).values

    # 7. Credit Utilisation (proxy from interest rate)
    # Higher interest rates correlate with higher risk/utilisation
    credit_utilisation = ((raw["loan_int_rate"] / 25.0) * 100).clip(0, 100).round(1).values

    # 8 & 9. Gender and Location (synthetic, with intentional bias)
    # These are injected to enable the Fairness Agent's counterfactual testing
    gender_encoded = np.random.choice(
        [0, 1, 2],  # 0=male, 1=female, 2=other
        n,
        p=[0.65, 0.30, 0.05],
    )
    location_encoded = np.random.choice(
        [0, 1, 2],  # 0=urban, 1=semi_urban, 2=rural
        n,
        p=[0.40, 0.35, 0.25],
    )

    # ── Engineer 3-class labels from the 2-class loan_status ──
    # Kaggle: loan_status = 0 (non-default) / 1 (default)
    # Our model: 0 = approve, 1 = review, 2 = reject
    #
    # Strategy:
    #   - Default (loan_status=1) → reject (2)
    #   - Non-default + borderline credit → review (1)
    #   - Non-default + good credit → approve (0)

    raw_status = raw["loan_status"].values
    cred_hist = raw["cb_person_cred_hist_length"].fillna(3).values

    labels = np.full(n, 1)  # default = review
    for i in range(n):
        if raw_status[i] == 1:
            labels[i] = 2  # default → reject
        elif credit_score[i] >= 700 and dti_ratio[i] < 0.4 and defaults[i] == 0:
            labels[i] = 0  # strong profile → approve
        elif credit_score[i] >= 650 and dti_ratio[i] < 0.3 and cred_hist[i] >= 5:
            labels[i] = 0  # decent profile + long history → approve
        else:
            labels[i] = 1  # everything else → review

    # ── Apply INTENTIONAL DEMOGRAPHIC BIAS ────────────────
    # This simulates historical discrimination in lending data.
    # The ML model learns these biased patterns.
    # The Fairness Agent detects them via counterfactual testing.
    for i in range(n):
        if labels[i] == 0:  # currently approve
            # Females have a chance of being moved to review
            if gender_encoded[i] == 1 and np.random.random() < 0.12:
                labels[i] = 1
            # Rural applicants have a chance of being moved to review
            if location_encoded[i] == 2 and np.random.random() < 0.10:
                labels[i] = 1

    df = pd.DataFrame({
        "credit_score": credit_score,
        "dti_ratio": dti_ratio,
        "loan_to_income": loan_to_income,
        "employment_score": employment_score,
        "age": age,
        "defaults": defaults,
        "credit_utilisation": credit_utilisation,
        "gender_encoded": gender_encoded,
        "location_encoded": location_encoded,
        "label": labels,
    })

    return df


# ══════════════════════════════════════════════════════════
#  SYNTHETIC DATA GENERATOR (V3 FALLBACK)
# ══════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════
#  TRAINING FUNCTION
# ══════════════════════════════════════════════════════════

def train_model():
    """Train and save the bias-aware risk model."""

    # Prefer Kaggle data, fall back to synthetic
    if os.path.exists(KAGGLE_CSV):
        df = load_kaggle_data()
        data_source = "Kaggle Credit Risk Dataset"
    else:
        print("⚠ Kaggle CSV not found, using synthetic fallback...")
        df = generate_synthetic_data(N_SAMPLES)
        data_source = f"Synthetic ({N_SAMPLES} samples)"

    print(f"\n📊 Training data source: {data_source}")
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
    print(f"\n🤖 Training RandomForest classifier on {len(X_train)} samples...")
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
    print(f"   Data source: {data_source}")
    print(f"   ⚠ Model contains learned demographic bias — Fairness Agent will detect it.")

    # Save the processed training data for reproducibility
    processed_path = os.path.join(os.path.dirname(__file__), "..", "data", "kaggle", "processed_training_data.csv")
    df.to_csv(processed_path, index=False)
    print(f"   📁 Processed data saved to {processed_path}")

    return model


if __name__ == "__main__":
    train_model()
