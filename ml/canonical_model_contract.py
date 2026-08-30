import os
import joblib
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

class CanonicalFeatures(BaseModel):
    """
    The strictly verified 8-feature schema for the Random Forest model.
    Any attempt to score a model without exactly these 8 features will fail.
    """
    credit_score: float = Field(..., description="Credit score (300-900)")
    dti_ratio: float = Field(..., description="Debt-to-Income ratio")
    loan_to_income: float = Field(..., description="Loan-to-Income ratio")
    employment_score: float = Field(..., description="Employment stability score (1-3)")
    age: float = Field(..., description="Applicant age")
    defaults: float = Field(..., description="Total previous defaults")
    credit_utilisation: float = Field(..., description="Credit utilisation percentage (0-100)")
    gender_encoded: float = Field(..., description="Encoded gender feature")
    location_encoded: float = Field(..., description="Encoded location feature")

    @classmethod
    def from_raw(cls, data: dict):
        # Handle the encoding mapping centrally so research & live agent use the exact same encoding
        gender = data.get("gender", data.get("applicant_gender", "male")).lower()
        gender_encoded = 1 if gender == "female" else 0
        
        location = data.get("location", data.get("applicant_location", "semi_urban")).lower()
        loc_map = {"rural": 0, "semi_urban": 1, "urban": 2}
        location_encoded = loc_map.get(location, 1)

        return cls(
            credit_score=float(data.get("credit_score", 650)),
            dti_ratio=float(data.get("dti_ratio", 0.5)),
            loan_to_income=float(data.get("loan_to_income", 1.0)),
            employment_score=float(data.get("employment_score", 1.0)),
            age=float(data.get("age", data.get("applicant_age", 30))),
            defaults=float(data.get("defaults", 0.0)),
            credit_utilisation=float(data.get("credit_utilisation", 40.0)),
            gender_encoded=float(gender_encoded),
            location_encoded=float(location_encoded)
        )

# Load the model singleton safely
_model = None
def get_model():
    global _model
    if _model is None:
        model_path = os.path.join(os.path.dirname(__file__), "model", "risk_model.joblib")
        if os.path.exists(model_path):
            _model = joblib.load(model_path)
    return _model

def predict_ml_probability(features: dict) -> float:
    """
    Execute the Random Forest model through the canonical contract.
    Validates features before prediction.
    Returns risk probability (0.0 to 1.0).
    """
    model = get_model()
    if not model:
        raise RuntimeError("Model not found at ml/model/risk_model.joblib")
        
    try:
        valid_features = CanonicalFeatures.from_raw(features)
    except ValidationError as e:
        raise ValueError(f"Feature schema validation failed: {e}")

    # Ensure correct ordering
    feature_names = [
        "credit_score", "dti_ratio", "loan_to_income",
        "employment_score", "age", "defaults", "credit_utilisation",
        "gender_encoded", "location_encoded"
    ]
    
    # Extract as DataFrame (some models expect DataFrame for feature names)
    df = pd.DataFrame([{name: getattr(valid_features, name) for name in feature_names}])
    
    # Assuming positive class is the 1st index
    prob = model.predict_proba(df)[0][1]
    return prob

def compute_heuristic_score(data: dict) -> float:
    """
    Centralized enhanced rule-based risk score (0-100, higher = riskier).
    Guarantees that Research and Live pipelines use the identical heuristic math.
    """
    score = 50.0

    cs = data.get("credit_score", 650)
    if cs >= 750:   score -= 20
    elif cs >= 700: score -= 10
    elif cs >= 650: score -= 0
    elif cs >= 600: score += 10
    else:           score += 25

    dti = data.get("dti_ratio", 0.5)
    if dti > 0.6:   score += 15
    elif dti > 0.4: score += 8
    elif dti < 0.2: score -= 5

    lti = data.get("loan_to_income", 1.0)
    if lti > 5:    score += 15
    elif lti > 3:  score += 8
    elif lti < 1:  score -= 5

    defaults = data.get("defaults", 0)
    score += min(40, 10 * (2 ** defaults - 1))

    cu = data.get("credit_utilisation", 40.0)
    if cu > 90:    score += 15
    elif cu > 80:  score += 10
    elif cu > 60:  score += 5
    elif cu < 30:  score -= 5

    emp = data.get("employment_score", 1.0)
    if emp == 3:   score -= 5
    elif emp == 1: score += 5

    tenure = data.get("loan_tenure_years", 10)
    if tenure > 20:   score += 8
    elif tenure > 10: score += 4

    emp_years = data.get("employment_years", 3)
    if emp_years < 1:    score += 10
    elif emp_years < 3:  score += 5
    elif emp_years > 7:  score -= 5

    purpose_risk = {
        "home": -8, "education": -5, "personal": 0,
        "business": 10, "medical": 5,
    }
    score += purpose_risk.get(data.get("loan_purpose", "personal"), 0)

    danger_flags = sum([
        dti > 0.60,
        defaults > 0,
        cu > 80,
        cs < 600,
    ])
    if danger_flags >= 3:   score += 20
    elif danger_flags == 2: score += 10

    bonus = 0
    if dti < 0.3: bonus -= 5
    if lti < 2:   bonus -= 5
    if defaults == 0 and cs > 720:
        bonus -= 5
    if cu < 30 and defaults == 0:
        bonus -= 3
    score += max(bonus, -15)

    return max(0.0, min(100.0, float(score)))

def compute_composite_score(data: dict) -> tuple[float, float, float]:
    """
    Computes ML score, heuristic score, and the dynamic weighted composite.
    Returns (ml_score, heuristic_score, composite_score)
    """
    try:
        prob = predict_ml_probability(data)
        ml_score = prob * 100.0
        ml_confidence = 1.0 - (2.0 * abs(prob - 0.5))
    except Exception as e:
        # Fallback if model fails (e.g. not found)
        ml_score = None
        ml_confidence = 0.0

    heuristic_score = compute_heuristic_score(data)
    
    if ml_score is not None:
        ml_weight = max(0.1, min(0.9, 0.4 + ml_confidence * 0.4))
        heur_weight = 1.0 - ml_weight
        composite_score = (ml_score * ml_weight) + (heuristic_score * heur_weight)
    else:
        composite_score = heuristic_score
        
    return ml_score if ml_score is not None else heuristic_score, heuristic_score, composite_score
