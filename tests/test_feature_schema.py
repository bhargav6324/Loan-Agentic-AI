import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from ml.canonical_model_contract import predict_ml_probability, compute_composite_score

class MockModel:
    def __init__(self, n_features_in, feature_names_in, classes=[0, 1]):
        self.n_features_in_ = n_features_in
        self.feature_names_in_ = feature_names_in
        self.classes_ = classes

    def predict_proba(self, X):
        return [[0.3, 0.7]]
from unittest.mock import patch

def test_real_model_schema():
    """Verify the actual fitted model matches the canonical schema."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    
    expected_names = [
        "credit_score", "dti_ratio", "loan_to_income",
        "employment_score", "applicant_age", "defaults", "credit_utilisation",
        "applicant_gender"
    ]
    
    X = pd.DataFrame([[700, 0.4, 2.0, 2, 30, 0, 40, 0], [600, 0.6, 3.0, 1, 40, 1, 60, 1]], columns=expected_names)
    y = [0, 1]
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # 1. Eight features
    assert model.n_features_in_ == 8
    
    # 2. Exact feature names and exact order
    assert list(model.feature_names_in_) == expected_names
    
    with patch("ml.canonical_model_contract.get_model", return_value=model):
        # 3. Successful predict_proba
        prob = predict_ml_probability({
            "credit_score": 700,
            "dti_ratio": 0.4,
            "loan_to_income": 2.0,
            "employment_score": 2,
            "applicant_age": 30,
            "defaults": 0,
            "credit_utilisation": 40,
            "applicant_gender": 0,
        })
        
        assert 0.0 <= prob <= 1.0


def test_feature_count_mismatch():
    """Fail when the number of constructed features differs from expected count."""
    mock_model = MockModel(n_features_in=9, feature_names_in=None)
    
    with patch("ml.canonical_model_contract.get_model", return_value=mock_model):
        with pytest.raises(ValueError, match="Schema mismatch: Model expects 9 features"):
            predict_ml_probability({"credit_score": 700})

def test_feature_name_order_mismatch():
    """Fail when feature names or order differ."""
    wrong_names = [
        "credit_score", "dti_ratio", "loan_to_income",
        "employment_score", "defaults", "applicant_age", "credit_utilisation", # Swapped defaults and applicant_age
        "applicant_gender"
    ]
    mock_model = MockModel(n_features_in=8, feature_names_in=wrong_names)
    
    with patch("ml.canonical_model_contract.get_model", return_value=mock_model):
        with pytest.raises(ValueError, match="Schema mismatch: Model feature names"):
            predict_ml_probability({"credit_score": 700})

def test_no_silent_fallback():
    """System must not silently fall back to heuristic-only decisions on schema mismatch."""
    mock_model = MockModel(n_features_in=9, feature_names_in=None)
    
    with patch("ml.canonical_model_contract.get_model", return_value=mock_model):
        # compute_composite_score should raise ValueError, not return heuristic-only silently
        with pytest.raises(ValueError, match="Schema mismatch"):
            compute_composite_score({"credit_score": 700})
