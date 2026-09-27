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
        if "gender_encoded" in data:
            gender_encoded = float(data["gender_encoded"])
        else:
            gender = data.get("gender", data.get("applicant_gender", "male"))
            if isinstance(gender, str):
                gender_encoded = 1.0 if gender.lower() == "female" else 0.0
            else:
                gender_encoded = float(gender)
            
        if "location_encoded" in data:
            location_encoded = float(data["location_encoded"])
        else:
            location = data.get("location", data.get("applicant_location", "semi_urban"))
            if isinstance(location, str):
                loc_map = {"rural": 0, "semi_urban": 1, "urban": 2}
                location_encoded = float(loc_map.get(location.lower(), 1))
            else:
                location_encoded = float(location)

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

def extract_canonical_features(state: dict) -> dict:
    """
    Unified feature builder used consistently by A1R, A2/A3 risk scoring, 
    and fairness counterfactual rescoring.
    Accepts an AgentState (nested dict) or a flat dataset row.
    """
    applicant = state.get("applicant_data", state)
    credit = state.get("credit_report", state)

    monthly_income = applicant.get("monthly_income") or 50000
    annual_income = monthly_income * 12
    loan_amount = applicant.get("loan_amount", 0)

    # Use dataset values if available (for parity), else calculate
    dti_ratio = applicant.get("dti_ratio")
    if dti_ratio is None:
        current_outstanding = credit.get("current_outstanding", 0) if credit else 0
        dti_ratio = current_outstanding / annual_income if annual_income > 0 else 1.0

    loan_to_income = applicant.get("loan_to_income")
    if loan_to_income is None:
        loan_to_income = loan_amount / annual_income if annual_income > 0 else 10.0

    emp_score = applicant.get("employment_score")
    if emp_score is None:
        emp_type = applicant.get("employment_type", "")
        employment_scores = {"salaried": 3, "business": 2, "self_employed": 1}
        emp_score = employment_scores.get(emp_type, 1)

    gender = applicant.get("applicant_gender", applicant.get("gender"))
    if isinstance(gender, str):
        gender_encoded = 1.0 if gender.lower() == "female" else 0.0
    elif gender is not None:
        gender_encoded = float(gender)
    else:
        gender_encoded = 0.0

    return {
        "credit_score": float(credit.get("credit_score", 650) if credit else 650),
        "dti_ratio": float(dti_ratio),
        "loan_to_income": float(loan_to_income),
        "employment_score": float(emp_score),
        "age": float(applicant.get("applicant_age", applicant.get("age", 30))),
        "defaults": float(credit.get("total_defaults", credit.get("defaults", 0)) if credit else 0),
        "credit_utilisation": float(credit.get("credit_utilisation_pct", credit.get("credit_utilisation", 40)) if credit else 40),
        "gender_encoded": gender_encoded,
        "employment_years": float(applicant.get("employment_years", 3)),
        "loan_tenure_years": float(applicant.get("loan_tenure_years", 10)),
        "loan_purpose": applicant.get("loan_purpose", "personal"),
        # Extra fields needed by some heuristcs/logging
        "monthly_income": monthly_income,
        "loan_amount": loan_amount,
        "annual_income": annual_income,
        "location": applicant.get("applicant_location", applicant.get("location", "semi_urban")),
        "gender": applicant.get("applicant_gender", applicant.get("gender", "male")),
    }

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
        raise FileNotFoundError("Model not found at ml/model/risk_model.joblib")

    if list(getattr(model, "classes_", [])) != [0, 1]:
        raise ValueError(
            f"Model classes must be exactly [0, 1], got {list(getattr(model, 'classes_', []))}"
        )
        
    try:
        valid_features = CanonicalFeatures.from_raw(features)
    except ValidationError as e:
        raise ValueError(f"Feature schema validation failed: {e}")

    # Ensure correct ordering matching the 8-feature RF
    trained_features = [
        "credit_score", "dti_ratio", "loan_to_income",
        "employment_score", "applicant_age", "defaults", "credit_utilisation",
        "applicant_gender"
    ]
    
    row_dict = {
        "credit_score": valid_features.credit_score,
        "dti_ratio": valid_features.dti_ratio,
        "loan_to_income": valid_features.loan_to_income,
        "employment_score": valid_features.employment_score,
        "applicant_age": valid_features.age,
        "defaults": valid_features.defaults,
        "credit_utilisation": valid_features.credit_utilisation,
        "applicant_gender": valid_features.gender_encoded,
    }
    
    expected_n = getattr(model, "n_features_in_", 8)
    if expected_n != len(trained_features):
        raise ValueError(f"Schema mismatch: Model expects {expected_n} features, but contract builds {len(trained_features)}.")
        
    expected_names = getattr(model, "feature_names_in_", None)
    if expected_names is not None:
        if list(expected_names) != trained_features:
            raise ValueError(f"Schema mismatch: Model feature names {list(expected_names)} do not match contract {trained_features}.")
    
    df = pd.DataFrame([row_dict], columns=trained_features)
    
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
    
    Uses the same formula as canonical_composite_score in core/canonical_contracts.py.
    """
    prob = predict_ml_probability(data)
    ml_score = prob * 100.0
    # Must match canonical_composite_score: confidence = max(p, 1-p)
    ml_confidence = max(prob, 1.0 - prob)

    heuristic_score = compute_heuristic_score(data)
    
    ml_weight = 0.4 + 0.4 * ml_confidence
    heur_weight = 1.0 - ml_weight
    composite_score = (ml_score * ml_weight) + (heuristic_score * heur_weight)
        
    return round(ml_score, 1), round(heuristic_score, 1), round(composite_score, 1)
