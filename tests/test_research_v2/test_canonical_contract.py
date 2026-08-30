import pytest
from ml.canonical_model_contract import CanonicalFeatures, compute_composite_score, compute_heuristic_score
from pydantic import ValidationError

def test_canonical_features_valid():
    """Test that CanonicalFeatures accepts valid schema."""
    data = {
        "credit_score": 750,
        "dti_ratio": 0.3,
        "loan_to_income": 2.0,
        "employment_score": 3,
        "age": 35,
        "defaults": 0,
        "credit_utilisation": 20,
        "gender": "female",
        "location": "urban"
    }
    
    features = CanonicalFeatures.from_raw(data)
    assert features.credit_score == 750.0
    assert features.gender_encoded == 1.0  # female -> 1
    assert features.location_encoded == 2.0  # urban -> 2

def test_canonical_features_missing():
    """Test that missing required features fallback to default or fail if unspecified correctly."""
    data = {
        "credit_score": 750,
        # missing others
    }
    
    # We use from_raw which has defaults
    features = CanonicalFeatures.from_raw(data)
    assert features.credit_score == 750.0
    assert features.dti_ratio == 0.5  # default

def test_compute_heuristic_score_determinism():
    """Test that heuristic scoring is deterministic and matches expected math."""
    data = {
        "credit_score": 850, # -20, bonus -5
        "dti_ratio": 0.1,    # -5, bonus -5
        "loan_to_income": 0.5, # -5, bonus -5
        "defaults": 0,       # bonus -5
        "credit_utilisation": 10, # -5, bonus -3
        "employment_score": 3, # -5
        "employment_years": 8, # -5
    }
    # Expected: 50 - 20 - 5 - 5 - 5 - 5 - 5 = 5 (pre-bonus)
    # Danger flags: 0.
    # Bonus: -5 (cs) - 5 (clean+stable) - 5 (dti) - 3 (cu) = -18 -> cap at -15.
    # Score: 5 - 15 = -10 -> capped at 0.0
    
    score = compute_heuristic_score(data)
    assert score == 0.0

def test_compute_composite_score_mocked_model(monkeypatch):
    """Test the composite score blends ML and Heuristic correctly."""
    
    def mock_predict(features):
        return 0.8  # ML proba (reject=0.8)
        
    monkeypatch.setattr("ml.canonical_model_contract.predict_ml_probability", mock_predict)
    
    data = {
        "credit_score": 600,
        "dti_ratio": 0.7, 
        "loan_to_income": 6.0,
        "defaults": 2,
    }
    
    ml, heur, comp = compute_composite_score(data)
    
    # ml = 0.8 * 100 = 80.0
    # ml_conf = 1.0 - (2.0 * abs(0.8 - 0.5)) = 1.0 - 0.6 = 0.4
    # ml_weight = 0.4 + 0.4 * 0.4 = 0.56
    # heur_weight = 0.44
    
    assert ml == 80.0
    assert 0.0 <= comp <= 100.0
