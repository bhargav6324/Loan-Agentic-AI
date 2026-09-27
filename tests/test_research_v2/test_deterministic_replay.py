import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import ml.canonical_model_contract as _cmc


def test_deterministic_replay():
    """Test that canonical evaluation is deterministic across multiple calls with same inputs.

    Trains a small in-memory RF model and injects it as the module singleton so
    the test is fully self-contained and does not require ml/model/risk_model.joblib.
    """
    from ml.canonical_model_contract import compute_composite_score

    # Build a minimal in-memory RF from the research dataset
    dataset_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "research", "data", "credit_risk_dataset.csv"
    )
    df = pd.read_csv(dataset_path)
    features = [
        "credit_score", "dti_ratio", "loan_to_income", "employment_score",
        "applicant_age", "defaults", "credit_utilisation", "applicant_gender",
    ]
    X = df[features]
    y = df["target_default"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)

    # Inject the trained model into the module singleton (safe: test is single-threaded)
    original_model = _cmc._model
    _cmc._model = rf
    try:
        data = {
            "credit_score": 750,
            "dti_ratio": 0.3,
            "loan_to_income": 2.0,
            "employment_score": 3,
            "age": 35,
            "defaults": 0,
            "credit_utilisation": 20,
            "gender": "female",
            "location": "urban",
        }

        # Call 1
        ml1, heur1, comp1 = compute_composite_score(data)

        # Call 2 — must produce bit-identical outputs (RF is deterministic)
        ml2, heur2, comp2 = compute_composite_score(data)

        assert ml1 == ml2, f"ML score not deterministic: {ml1} vs {ml2}"
        assert heur1 == heur2, f"Heuristic score not deterministic: {heur1} vs {heur2}"
        assert comp1 == comp2, f"Composite score not deterministic: {comp1} vs {comp2}"
    finally:
        _cmc._model = original_model
