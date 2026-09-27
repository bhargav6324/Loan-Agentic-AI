import pytest
import os
import sys

def legacy_heuristic_score(row) -> float:
    score = 50.0
    cs = row.get("credit_score", 0)
    if cs >= 750:   score -= 20
    elif cs >= 700: score -= 10
    elif cs >= 650: score -= 0
    elif cs >= 600: score += 10
    else:           score += 25
    dti = row.get("dti_ratio", 0)
    if dti > 0.6:   score += 15
    elif dti > 0.4: score += 8
    elif dti < 0.2: score -= 5
    lti = row.get("loan_to_income", 0)
    if lti > 5:    score += 15
    elif lti > 3:  score += 8
    elif lti < 1:  score -= 5
    defaults = row.get("defaults", 0)
    score += min(40, 10 * (2 ** defaults - 1))
    cu = row.get("credit_utilisation", 0)
    if cu > 90:    score += 15
    elif cu > 80:  score += 10
    elif cu > 60:  score += 5
    elif cu < 30:  score -= 5
    emp = row.get("employment_score", 0)
    if emp == 3:   score -= 5
    elif emp == 1: score += 5
    tenure = row.get("loan_tenure_years", 10)
    if tenure > 20:  score += 8
    elif tenure > 10: score += 4
    emp_years = row.get("employment_years", 3)
    if emp_years < 1:    score += 10
    elif emp_years < 3:  score += 5
    elif emp_years > 7:  score -= 5
    purpose_risk = {"home": -8, "education": -5, "personal": 0, "business": 10, "medical": 5}
    score += purpose_risk.get(row.get("loan_purpose", "personal"), 0)
    danger_flags = sum([dti > 0.60, defaults > 0, cu > 80, cs < 600])
    if danger_flags >= 3:   score += 20
    elif danger_flags == 2: score += 8
    bonus = 0
    if cs > 800:                         bonus -= 5
    if defaults == 0 and emp_years > 5:  bonus -= 5
    if dti < 0.20:                       bonus -= 5
    if cu < 20:                          bonus -= 3
    score += max(bonus, -15)
    return max(0, min(100, score))


def test_scoring_regression():
    """Ensure canonical composite score aligns numerically with the old risk agent math, without ML dependencies."""
    from ml.canonical_model_contract import compute_composite_score, compute_heuristic_score
    from unittest.mock import patch
    
    record = {
        "credit_score": 650, "dti_ratio": 0.5, "loan_to_income": 4.0,
        "defaults": 1, "credit_utilisation": 85, "employment_score": 1,
        "loan_tenure_years": 15, "employment_years": 2, "loan_purpose": "business",
        "gender": "male", "location": "urban", "age": 35
    }
    
    legacy_score = legacy_heuristic_score(record)
    canonical_heuristic = compute_heuristic_score(record)
    assert legacy_score == canonical_heuristic, f"Expected exact match. Old={legacy_score}, Canonical={canonical_heuristic}"
    
    with patch("ml.canonical_model_contract.predict_ml_probability", return_value=0.7):
        ml_score, heur_score, comp_score = compute_composite_score(record)
        
        # P(reject) = 0.7 -> ml_score = 70.0
        # Confidence = max(0.7, 1-0.7) = 0.7
        # ml_weight = 0.4 + (0.4 * 0.7) = 0.68
        # heur_weight = 1 - 0.68 = 0.32
        expected_comp = (0.68 * 70.0) + (0.32 * canonical_heuristic)
        
        assert ml_score == 70.0
        assert round(comp_score, 2) == round(expected_comp, 2)
