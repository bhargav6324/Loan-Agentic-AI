import pytest
import os
import sys

def test_deterministic_replay():
    """Test that canonical evaluation is deterministic across multiple calls with same inputs."""
    from ml.canonical_model_contract import compute_composite_score
    
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
    
    # Call 1
    ml1, heur1, comp1 = compute_composite_score(data)
    
    # Call 2
    ml2, heur2, comp2 = compute_composite_score(data)
    
    assert ml1 == ml2
    assert heur1 == heur2
    assert comp1 == comp2
