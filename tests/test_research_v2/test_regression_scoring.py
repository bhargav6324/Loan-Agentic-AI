import pytest
import os
import sys

# Load old evaluation_harness heuristic function from before refactor
def load_legacy_heuristic():
    import importlib.util
    spec = importlib.util.spec_from_file_location("old_eval", "/tmp/old_eval.py")
    old_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_eval)
    return old_eval.compute_heuristic_score

def test_scoring_regression():
    """Ensure the canonical composite score aligns numerically with the old risk agent math."""
    from ml.canonical_model_contract import compute_composite_score
    old_heuristic = load_legacy_heuristic()
    
    # 1. Test Heuristic Equivalence
    # We test on an applicant record.
    record = {
        "credit_score": 650,
        "dti_ratio": 0.5,
        "loan_to_income": 4.0,
        "defaults": 1,
        "credit_utilisation": 85,
        "employment_score": 1,
        "loan_tenure_years": 15,
        "employment_years": 2,
        "loan_purpose": "business",
        "gender": "male",
        "location": "urban",
        "age": 35
    }
    
    # Old logic used raw row inputs
    # Let's mock a Pandas Series since old logic expected row["..."]
    import pandas as pd
    row = pd.Series(record)
    
    legacy_score = old_heuristic(row)
    
    # Canonical logic uses dict
    _, canonical_heuristic, canonical_composite = compute_composite_score(record)
    
    # Expected differences:
    # None. The canonical compute_heuristic_score was an exact copy of the old eval harness logic.
    assert legacy_score == canonical_heuristic, f"Expected exact numerical match. Old={legacy_score}, Canonical={canonical_heuristic}"
    
    # We will test the weighting logic manually to ensure no unexpected shifts.
    # We mock predict_ml_probability to return a fixed probability
    import ml.canonical_model_contract
    original_predict = ml.canonical_model_contract.predict_ml_probability
    
    try:
        def mock_predict(features):
            return 0.7  # P(reject) = 0.7
        ml.canonical_model_contract.predict_ml_probability = mock_predict
        
        ml_score, heur_score, comp_score = compute_composite_score(record)
        
        # P(reject) = 0.7 -> ml_score = 70.0
        # Confidence = 1 - 2*abs(0.7 - 0.5) = 0.6
        # ml_weight = 0.4 + 0.4 * 0.6 = 0.64
        # heur_weight = 1 - 0.64 = 0.36
        
        expected_comp = (0.64 * 70.0) + (0.36 * canonical_heuristic)
        
        assert ml_score == 70.0
        assert round(comp_score, 2) == round(expected_comp, 2)
    finally:
        ml.canonical_model_contract.predict_ml_probability = original_predict
