"""
Risk Agent — ML-Powered Credit Risk Scoring (v2 — Enhanced)
────────────────────────────────────────────────────────────
Uses a trained RandomForest model + advanced rule-based heuristics to compute
a composite risk score. This is NOT simple ML — the agent:
  • Extracts features from enriched data (Data Agent output)
  • Runs the ML model for a base probability
  • Uses DYNAMIC confidence-based weighting (not fixed 60/40)
  • Applies exponential penalties for compounding risk
  • Checks dangerous risk-factor interactions
  • Rewards excellent profiles with a creditworthiness bonus
  • Implements hard override pre-screening
  • Categorises into LOW / MEDIUM / HIGH with reasoning
"""

from __future__ import annotations
import time
import os
import numpy as np
from agents.state import AgentState

# Try to load the trained model; fall back to heuristic-only if not available
_model = None
_feature_names = [
    "credit_score", "dti_ratio", "loan_to_income",
    "employment_score", "age", "defaults", "credit_utilisation",
    "gender_encoded", "location_encoded",  # V3: demographics for emergent bias
]

try:
    import joblib
    model_path = os.path.join(os.path.dirname(__file__), "..", "ml", "model", "risk_model.joblib")
    if os.path.exists(model_path):
        _model = joblib.load(model_path)
except Exception:
    _model = None


def _extract_features(state: AgentState) -> dict:
    """Extract ML features from the enriched state."""
    applicant = state.get("applicant_data", {})
    credit = state.get("credit_report", {})

    monthly_income = applicant.get("monthly_income") or 50000
    loan_amount = applicant.get("loan_amount", 0)
    annual_income = monthly_income * 12

    credit_score = credit.get("credit_score", 650) if credit else 650
    current_outstanding = credit.get("current_outstanding", 0) if credit else 0
    total_defaults = credit.get("total_defaults", 0) if credit else 0
    credit_utilisation = credit.get("credit_utilisation_pct", 40) if credit else 40

    # Debt-to-Income ratio (existing debt / annual income)
    dti_ratio = current_outstanding / annual_income if annual_income > 0 else 1.0

    # Loan-to-Income ratio (requested loan / annual income)
    loan_to_income = loan_amount / annual_income if annual_income > 0 else 10.0

    # Employment score
    emp_type = applicant.get("employment_type", "")
    employment_scores = {"salaried": 3, "business": 2, "self_employed": 1}
    employment_score = employment_scores.get(emp_type, 1)

    age = applicant.get("applicant_age") or 30
    location = applicant.get("applicant_location", "semi_urban")

    # NEW: Additional features for enhanced heuristics
    employment_years = applicant.get("employment_years", 3)
    loan_tenure_years = applicant.get("loan_tenure_years", 10)
    loan_purpose = applicant.get("loan_purpose", "personal")

    return {
        "credit_score": credit_score,
        "dti_ratio": round(dti_ratio, 3),
        "loan_to_income": round(loan_to_income, 3),
        "employment_score": employment_score,
        "age": age,
        "defaults": total_defaults,
        "credit_utilisation": credit_utilisation,
        # NEW enhanced features
        "employment_years": employment_years,
        "loan_tenure_years": loan_tenure_years,
        "loan_purpose": loan_purpose,
        # Extra for reasoning
        "monthly_income": monthly_income,
        "loan_amount": loan_amount,
        "annual_income": annual_income,
        "location": location,
        "gender": applicant.get("applicant_gender", "male"),
    }


def _check_hard_overrides(features: dict) -> str | None:
    """
    Improvement #6: Hard Override Pre-Screening Layer.
    Some conditions should bypass scoring entirely.
    Returns 'HARD_REJECT', 'HARD_FAST_TRACK', or None.
    """
    # Hard Reject — no scoring needed, catastrophic risk
    if (features["defaults"] > 3 or
        features["credit_score"] < 450 or
        features["dti_ratio"] > 0.90):
        return "HARD_REJECT"

    # Hard Fast-Track Candidate — still goes through Policy & Fairness agents
    # but skips deep risk analysis
    if (features["credit_score"] > 820 and
        features["defaults"] == 0 and
        features["dti_ratio"] < 0.15 and
        features["employment_years"] > 5):
        return "HARD_FAST_TRACK"

    return None


def _compute_heuristic_score(features: dict) -> tuple[float, str]:
    """
    Enhanced rule-based risk score (0-100, higher = riskier).
    Returns (score, justification_string)
    """
    score = 50.0  # Start neutral
    justification_parts = ["Started with neutral baseline (50)."]

    # ── Credit score impact (most important) ──────────────
    cs = features["credit_score"]
    if cs >= 750:
        score -= 20
    elif cs >= 700:
        score -= 10
    elif cs >= 650:
        score -= 0
    elif cs >= 600:
        score += 10
    else:
        score += 25

    # ── DTI ratio ─────────────────────────────────────────
    dti = features["dti_ratio"]
    if dti > 0.6:
        score += 15
    elif dti > 0.4:
        score += 8
    elif dti < 0.2:
        score -= 5

    # ── Loan-to-income ────────────────────────────────────
    lti = features["loan_to_income"]
    if lti > 5:
        score += 15
    elif lti > 3:
        score += 8
    elif lti < 1:
        score -= 5

    # ── Improvement #2: Exponential penalty for defaults ──
    defaults = features["defaults"]
    score += min(40, 10 * (2 ** defaults - 1))

    # ── Credit utilisation (exponential at high end) ──────
    cu = features["credit_utilisation"]
    if cu > 90:
        score += 15  # catastrophic utilisation
    elif cu > 80:
        score += 10
    elif cu > 60:
        score += 5
    elif cu < 30:
        score -= 5

    # ── Employment stability ──────────────────────────────
    emp = features["employment_score"]
    if emp == 3:
        score -= 5
    elif emp == 1:
        score += 5

    # ── Improvement #3a: Loan Tenure Risk ─────────────────
    tenure = features.get("loan_tenure_years", 10)
    if tenure > 20:
        score += 8
    elif tenure > 10:
        score += 4

    # ── Improvement #3b: Income Stability (Job Tenure) ────
    emp_years = features.get("employment_years", 3)
    if emp_years < 1:
        score += 10
    elif emp_years < 3:
        score += 5
    elif emp_years > 7:
        score -= 5

    # ── Improvement #3c: Loan Purpose Risk ────────────────
    purpose_risk = {
        "home": -8,        # secured, lower risk
        "education": -5,   # generally good repayment
        "personal": 0,
        "business": 10,    # higher volatility
        "medical": 5,
    }
    loan_purpose = features.get("loan_purpose", "personal")
    score += purpose_risk.get(loan_purpose, 0)

    # ── Improvement #4: Risk Interaction / Stacking Penalty
    danger_flags = sum([
        dti > 0.60,
        defaults > 0,
        cu > 80,
        cs < 600,
    ])
    if danger_flags >= 3:
        score += 20  # compounding risk penalty
    elif danger_flags == 2:
        score += 8

    # ── Improvement #5: Creditworthiness Bonus ────────────
    bonus = 0
    if cs > 800:
        bonus -= 5   # exceptional credit
    if defaults == 0 and emp_years > 5:
        bonus -= 5   # clean + stable
    if dti < 0.20:
        bonus -= 5   # very low debt burden
    if cu < 20:
        bonus -= 3   # responsible credit use
    score += max(bonus, -15)  # cap total bonus at -15 points

    # ── NOTE: Demographic bias REMOVED from heuristics (V3) ─
    # V2 had hardcoded lines like `if gender == "female": score += 6`
    # and `if location == "rural": score += 5`.  This was methodologically
    # invalid: the Fairness Agent was "detecting" bias that was explicitly
    # injected.  In V3, bias is emergent from the ML model trained on
    # biased synthetic data (gender + location are ML features).
    # The heuristic engine is now demographic-blind.
    # The Fairness Agent detects real learned bias via counterfactual testing.

    final_score = max(0.0, min(100.0, float(score)))
    justification = " ".join(justification_parts)
    return final_score, justification


def _identify_risk_factors(features: dict) -> list:
    """Identify the top risk factors for explainability."""
    factors = []

    if features["credit_score"] < 650:
        factors.append(f"Low credit score ({features['credit_score']})")
    if features["dti_ratio"] > 0.5:
        factors.append(f"High debt-to-income ratio ({features['dti_ratio']:.1%})")
    if features["loan_to_income"] > 4:
        factors.append(f"Loan amount is {features['loan_to_income']:.1f}x annual income")
    if features["defaults"] > 0:
        factors.append(f"{features['defaults']} previous default(s)")
    if features["credit_utilisation"] > 70:
        factors.append(f"High credit utilisation ({features['credit_utilisation']:.0f}%)")
    if features["employment_score"] == 1:
        factors.append("Self-employed — variable income")
    # NEW risk factors
    if features.get("employment_years", 3) < 1:
        factors.append("Less than 1 year at current job")
    if features.get("loan_tenure_years", 10) > 20:
        factors.append(f"Long loan tenure ({features.get('loan_tenure_years')} years)")
    if features.get("loan_purpose") == "business":
        factors.append("Business loan — higher default volatility")

    # Risk stacking warning
    danger_flags = sum([
        features["dti_ratio"] > 0.60,
        features["defaults"] > 0,
        features["credit_utilisation"] > 80,
        features["credit_score"] < 600,
    ])
    if danger_flags >= 3:
        factors.append("⚠ Multiple compounding risk factors detected")

    if not factors:
        factors.append("Strong financial profile")

    return factors


# NOTE: ML model trained without demographic features (by design).
# Fairness Agent counterfactuals test heuristic bias only.
# This is architecturally intentional — the ML model is 
# deliberately kept demographic-blind; heuristic rules are 
# the bias vector under test.
def evaluate_profile(features: dict, state: AgentState) -> dict:
    """
    Core risk evaluation logic extracted for inter-agent communication.
    The Fairness Agent can call this to simulate 'debate' and ask for
    re-evaluation of modified features.
    """
    internal_logs = []
    errors = []

    # ── ML Model Prediction ──────────────────────────────
    ml_score = None
    ml_confidence = 0.5  # default: uncertain
    if _model is not None:
        try:
            # Encode demographics for ML model (V3: emergent bias)
            gender_map = {"male": 0, "female": 1, "other": 2}
            location_map = {"urban": 0, "semi_urban": 1, "rural": 2}
            gender_encoded = gender_map.get(features.get("gender", "male"), 0)
            location_encoded = location_map.get(features.get("location", "semi_urban"), 1)

            X = np.array([[
                features["credit_score"],
                features["dti_ratio"],
                features["loan_to_income"],
                features["employment_score"],
                features["age"],
                features["defaults"],
                features["credit_utilisation"],
                gender_encoded,
                location_encoded,
            ]])
            # Model predicts probabilities for [approve, review, reject]
            proba = _model.predict_proba(X)[0]
            # Convert to risk score: weighted sum (approve=low, reject=high)
            ml_score = float(proba[1] * 50 + proba[2] * 100)  # review*50 + reject*100
            # Improvement #1: ML confidence = max probability
            ml_confidence = float(np.max(proba))
            internal_logs.append({
                "action": "ML model prediction",
                "status": "success",
                "details": f"Probabilities: approve={proba[0]:.2f}, review={proba[1]:.2f}, reject={proba[2]:.2f} | Confidence={ml_confidence:.2f}",
            })
        except Exception as e:
            errors.append(f"ML model failed: {e}")
            internal_logs.append({
                "action": "ML model prediction",
                "status": "error",
                "details": str(e),
            })

    # ── Heuristic Score ──────────────────────────────────
    heuristic_score, heuristic_justification = _compute_heuristic_score(features)

    # ── Improvement #1: Dynamic Confidence-Based Weighting
    if ml_score is not None:
        # When ML is confident (e.g., 0.92), trust it more (weight up to 0.8)
        # When ML is uncertain (e.g., 0.51), lean on heuristics (weight down to 0.4)
        ml_weight = 0.4 + (0.4 * ml_confidence)   # ranges from 0.4 to 0.8
        heuristic_weight = 1 - ml_weight
        risk_score = round(float(ml_weight * ml_score + heuristic_weight * heuristic_score), 1)
        internal_logs.append({
            "action": "Dynamic weight blending",
            "status": "success",
            "details": f"ML weight={ml_weight:.2f}, Heuristic weight={heuristic_weight:.2f} (confidence={ml_confidence:.2f})",
        })
    else:
        risk_score = round(float(heuristic_score), 1)

    # Compile the final agent justification text for the debate
    agent_justification = (
        f"Score {risk_score} driven by: {heuristic_justification} "
        f"| ML Confidence: {ml_confidence:.2f} "
        f"| Bias Source: heuristic"
    )

    # ══════════════════════════════════════════════════════
    #  AGENTIC: SELF-REFLECTION / SELF-CRITIQUE LOOP
    # ══════════════════════════════════════════════════════
    # After computing the initial score, the agent reflects on whether
    # the result is internally consistent.  This is what makes it agentic:
    # a pipeline would accept the first score; an agent can challenge itself.

    reflection = _reflect_on_score(
        risk_score=risk_score,
        ml_score=ml_score,
        heuristic_score=heuristic_score,
        ml_confidence=ml_confidence,
        features=features,
        state=state,
    )

    # Apply reflection adjustments
    original_score = risk_score
    if reflection["revised"]:
        risk_score = reflection["revised_score"]
        internal_logs.append({
            "action": "Self-reflection: REVISED score",
            "status": "revised",
            "details": (
                f"Original={original_score} → Revised={risk_score} | "
                f"Reason: {reflection['revision_reason']}"
            ),
        })
    else:
        internal_logs.append({
            "action": "Self-reflection: score CONFIRMED",
            "status": "confirmed",
            "details": reflection["confirmation_reason"],
        })

    # Log each contradiction found
    for contradiction in reflection.get("contradictions", []):
        internal_logs.append({
            "action": "Reflection: contradiction detected",
            "status": "warning",
            "details": contradiction,
        })

    # ── Categorise ───────────────────────────────────────
    if risk_score <= 35:
        risk_category = "LOW"
    elif risk_score <= 60:
        risk_category = "MEDIUM"
    else:
        risk_category = "HIGH"

    # ── Risk Factors ─────────────────────────────────────
    risk_factors = _identify_risk_factors(features)

    return {
        "risk_score": risk_score,
        "risk_category": risk_category,
        "risk_factors": risk_factors,
        "reflection": reflection,
        "internal_logs": internal_logs,
        "errors": errors,
        "original_score": original_score,
        "agent_justification": agent_justification,
        "bias_source": "heuristic",
        "ml_confidence": ml_confidence,
    }


def risk_agent(state: AgentState) -> AgentState:
    """
    Risk Agent node for LangGraph (v2 — Enhanced).

    Computes a composite risk score using:
      - Hard override pre-screening
      - ML model prediction with dynamic confidence weighting
      - Advanced heuristic scoring with exponential penalties
    Then categorises the application.
    """
    logs = list(state.get("agent_logs", []))
    errors = list(state.get("errors", []))
    start = time.time()

    # Extract features
    features = _extract_features(state)

    # ── Improvement #6: Hard Override Pre-Screening ───────
    hard_override = _check_hard_overrides(features)
    if hard_override == "HARD_REJECT":
        duration = int((time.time() - start) * 1000)
        risk_factors = _identify_risk_factors(features)
        summary = f"Risk Score: 95/100 (HIGH — HARD REJECT) | Factors: {', '.join(risk_factors[:3])}"
        logs.append({
            "agent": "risk_agent",
            "step": len(logs) + 1,
            "action": "Hard reject pre-screening triggered",
            "status": "success",
            "details": summary,
            "duration_ms": duration,
        })
        return {
            **state,
            "risk_score": 95.0,
            "risk_category": "HIGH",
            "risk_factors": risk_factors,
            "risk_agent_summary": summary,
            "risk_ml_confidence": 1.0,  # Hard override = full confidence
            "agent_logs": logs,
            "errors": errors,
        }

    if hard_override == "HARD_FAST_TRACK":
        duration = int((time.time() - start) * 1000)
        summary = "Risk Score: 10/100 (LOW — FAST TRACK) | Excellent profile, minimal risk"
        logs.append({
            "agent": "risk_agent",
            "step": len(logs) + 1,
            "action": "Fast-track pre-screening triggered",
            "status": "success",
            "details": summary,
            "duration_ms": duration,
        })
        return {
            **state,
            "risk_score": 10.0,
            "risk_category": "LOW",
            "risk_factors": ["Strong financial profile"],
            "risk_agent_summary": summary,
            "risk_ml_confidence": 1.0,  # Hard override = full confidence
            "agent_logs": logs,
            "errors": errors,
        }

    # ── Core Evaluation (Agentic Extraction) ─────────────
    eval_result = evaluate_profile(features, state)
    risk_score = eval_result["risk_score"]
    risk_category = eval_result["risk_category"]
    risk_factors = eval_result["risk_factors"]
    reflection = eval_result["reflection"]
    original_score = eval_result["original_score"]

    # Append internal logs to main logs
    for log in eval_result["internal_logs"]:
        logs.append({
            "agent": "risk_agent",
            "step": len(logs) + 1,
            "action": log["action"],
            "status": log["status"],
            "details": log["details"],
            "duration_ms": 0,
        })

    for err in eval_result["errors"]:
        errors.append(err)

    duration = int((time.time() - start) * 1000)

    revision_tag = ""
    if reflection["revised"]:
        revision_tag = f" [revised from {original_score}]"
    summary = f"Risk Score: {risk_score}/100 ({risk_category}){revision_tag} | Top factors: {', '.join(risk_factors[:3])}"

    logs.append({
        "agent": "risk_agent",
        "step": len(logs) + 1,
        "action": f"Risk assessment complete",
        "status": "success",
        "details": summary,
        "duration_ms": duration,
    })

    return {
        **state,
        "risk_score": risk_score,
        "risk_category": risk_category,
        "risk_factors": risk_factors,
        "risk_agent_summary": summary,
        "risk_agent_reflection": reflection,
        "risk_ml_confidence": eval_result.get("ml_confidence", 0.5),
        "agent_logs": logs,
        "errors": errors,
    }


# ══════════════════════════════════════════════════════════
#  REFLECTION ENGINE — True LLM Semantic Self-Critique (V3)
# ══════════════════════════════════════════════════════════
#
#  V2 used three hardcoded if-statements disguised as "reflection".
#  V3 uses an actual LLM call: the model autonomously reasons about
#  the score, finds contradictions the rules can't, and decides
#  whether to revise. Deterministic fallback preserved for safety.

import json
import httpx
from agents.state import AgentState
from agents.llm_client import call_groq
from config import settings

REFLECTION_PROMPT = """
You are a Senior Credit Risk Analyst reviewing an AI-generated risk assessment.

Applicant Profile:
{applicant_summary}

ML Model Output: {ml_score}/100 (confidence: {ml_confidence:.2f})
Heuristic Score: {heuristic_score:.1f}/100
Combined Score: {combined_score:.1f}/100
Key Risk Factors Identified: {risk_factors}

Credit Report:
{credit_summary}

Similar Past Cases from Episodic Memory:
{similar_cases}

Find logical inconsistencies or contradictions in this assessment.
Ask yourself:
- Does the score make sense given the credit score and income together?
- Are there contradictions between what the ML model says and what the heuristics say?
- Is the applicant being penalized for correlated factors (e.g., young AND self-employed AND rural)?
- Is any factor being double-counted?
- If the credit report is missing, should we trust this score?
- If the applicant has defaults but the score is LOW, that is suspicious.
- If the applicant has excellent credit (>750) but the score is HIGH, something is wrong.
- How does this applicant compare to the similar past cases? 
- Did similar past cases default or face rejection? Should we adjust the score based on historical precedent?

You MUST respond ONLY with valid JSON (no markdown, no explanation outside JSON):
{{
    "contradictions_found": ["list of specific contradictions, empty list if none"],
    "revised": true or false,
    "final_score": <number between 0 and 100>,
    "revision_reason": "explanation if revised, null if not revised"
}}
"""


def _reflect_on_score(
    risk_score: float,
    ml_score: float | None,
    heuristic_score: float,
    ml_confidence: float,
    features: dict,
    state: AgentState,
) -> dict:
    """
    True LLM-powered self-reflection: sends the computed risk assessment
    to a separate LLM call that autonomously reasons about contradictions
    and decides whether to revise the score.

    Falls back to deterministic safety checks when the LLM is unavailable.
    """
    applicant = state.get("applicant_data", {})
    credit = state.get("credit_report", {}) or {}

    # Build rich applicant summary for the LLM
    monthly_income = applicant.get("monthly_income") or 0
    loan_amount = applicant.get("loan_amount") or 0
    applicant_summary = (
        f"  Name: {applicant.get('applicant_name', 'Unknown')}\n"
        f"  Age: {applicant.get('applicant_age', 'N/A')} | "
        f"Gender: {applicant.get('applicant_gender', 'N/A')} | "
        f"Location: {applicant.get('applicant_location', 'N/A')}\n"
        f"  Employment: {applicant.get('employment_type', 'N/A')} "
        f"({applicant.get('employment_years', 'N/A')} years)\n"
        f"  Monthly Income: ₹{monthly_income:,.0f} | "
        f"Loan Amount: ₹{loan_amount:,.0f}\n"
        f"  Loan Purpose: {applicant.get('loan_purpose', 'N/A')}"
    )

    credit_summary = (
        f"  Credit Score: {credit.get('credit_score', 'MISSING')}\n"
        f"  Total Defaults: {credit.get('total_defaults', 'MISSING')}\n"
        f"  Current Outstanding: ₹{credit.get('current_outstanding', 'MISSING')}\n"
        f"  Credit Utilisation: {credit.get('credit_utilisation_pct', 'MISSING')}%"
        if credit else "  ⚠ NO CREDIT REPORT AVAILABLE — scoring under incomplete data"
    )

    risk_factors = _identify_risk_factors(features)
    ml_score_display = f"{ml_score:.1f}" if ml_score is not None else "N/A (model unavailable)"

    # ── Retrieve Episodic Memory ─────────────────────────
    similar_text = "Episodic memory unavailable."
    try:
        from memory.episodic_memory import retrieve_similar_cases
        similar = retrieve_similar_cases(applicant, credit, top_k=3)
        if similar:
            lines = []
            for idx, case in enumerate(similar):
                outcome = case.get("outcome", {})
                lines.append(f"[{idx+1}] Profile: {case.get('profile')}")
                lines.append(f"    Decision: {outcome.get('decision')} | Risk: {outcome.get('risk_score')} | Bias Detected: {outcome.get('bias_detected')}")
            similar_text = "\n".join(lines)
        else:
            similar_text = "No similar past cases found."
    except Exception as e:
        print(f"Failed to retrieve episodic memory: {e}")

    # ── Try LLM reflection ────────────────────────────────
    if settings.GROQ_API_KEY and settings.GROQ_API_KEY != "gsk_placeholder":
        try:
            prompt = REFLECTION_PROMPT.format(
                applicant_summary=applicant_summary,
                ml_score=ml_score_display,
                ml_confidence=ml_confidence,
                heuristic_score=heuristic_score,
                combined_score=risk_score,
                risk_factors=", ".join(risk_factors[:5]),
                credit_summary=credit_summary,
                similar_cases=similar_text,
            )

            data = call_groq(
                payload={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a Senior Credit Risk Analyst. "
                                "You review AI risk scores for logical inconsistencies. "
                                "Respond only with valid JSON."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
                api_key=settings.GROQ_API_KEY,
                timeout=12.0,
            )
            content = data["choices"][0]["message"]["content"].strip()
            llm_result = json.loads(content)

            contradictions = llm_result.get("contradictions_found", [])
            revised = bool(llm_result.get("revised", False))
            revised_score = None
            revision_reason = None
            confirmation_reason = None

            if revised:
                raw_score = llm_result.get("final_score", risk_score)
                # Safety: clamp to [0, 100] and don't allow extreme swings (>25 pts)
                clamped = max(0.0, min(100.0, float(raw_score)))
                if abs(clamped - risk_score) > 25:
                    clamped = risk_score + (25 if clamped > risk_score else -25)
                    clamped = max(0.0, min(100.0, clamped))
                revised_score = round(clamped, 1)
                revision_reason = llm_result.get("revision_reason") or "LLM revised without explanation"
            else:
                confirmation_reason = (
                    f"LLM reflection confirmed score of {risk_score:.1f} — "
                    f"no logical inconsistencies found in the assessment."
                )

            summary_parts = [f"LLM reflection found {len(contradictions)} issue(s)."]
            if revised:
                summary_parts.append(f"Score revised: {risk_score} → {revised_score}.")
            else:
                summary_parts.append("Score confirmed — no revision needed.")

            return {
                "contradictions": contradictions,
                "revised": revised,
                "revised_score": revised_score,
                "revision_reason": revision_reason,
                "confirmation_reason": confirmation_reason,
                "original_score": risk_score,
                "ml_score": ml_score,
                "heuristic_score": heuristic_score,
                "reflection_summary": " ".join(summary_parts),
                "reflection_source": "llm",  # NEW: indicates LLM-powered reflection
            }

        except Exception as e:
            # Fall through to deterministic fallback
            print(f"LLM reflection failed ({e}), using deterministic fallback")

    # ── Deterministic Fallback ────────────────────────────
    # Used when LLM is unavailable. Preserves the same output schema.
    contradictions = []
    revised = False
    revised_score = None
    revision_reason = None

    # Check 1: ML vs Heuristic Contradiction
    if ml_score is not None:
        gap = abs(ml_score - heuristic_score)
        if gap > 25:
            direction = (
                f"ML says LOW risk ({ml_score:.1f}) but heuristics say HIGH risk ({heuristic_score:.1f})"
                if ml_score < heuristic_score else
                f"ML says HIGH risk ({ml_score:.1f}) but heuristics say LOW risk ({heuristic_score:.1f})"
            )
            contradictions.append(f"ML/Heuristic contradiction: {direction} — gap of {gap:.1f} points.")
            conservative_score = round((ml_score + heuristic_score) / 2.0, 1)
            if conservative_score > risk_score:
                revised = True
                revised_score = conservative_score
                revision_reason = f"ML/heuristic gap={gap:.1f}pts. Re-scored with equal weights: {conservative_score:.1f}"

    # Check 2: Missing Credit Data
    if state.get("credit_report") is None:
        contradictions.append("Credit bureau report is missing — scoring under incomplete data.")
        penalty = 10
        base = revised_score if revised else risk_score
        penalised = min(100, round(base + penalty, 1))
        revised = True
        revised_score = penalised
        prev = (revision_reason + " | ") if revision_reason else ""
        revision_reason = f"{prev}Missing credit report — applied +{penalty} uncertainty penalty."

    # Check 3: LOW score with defaults
    effective = revised_score if revised else risk_score
    if effective <= 35 and features["defaults"] > 0:
        contradictions.append(f"Score is LOW ({effective}) but applicant has {features['defaults']} default(s).")
        if effective < 40:
            revised = True
            revised_score = max(effective, 40.0)
            prev = (revision_reason + " | ") if revision_reason else ""
            revision_reason = f"{prev}LOW score with defaults — bumped to {revised_score}."

    if not contradictions:
        confirmation_reason = (
            f"Score of {risk_score} is internally consistent: "
            f"credit_score={features['credit_score']}, defaults={features['defaults']}, "
            f"DTI={features['dti_ratio']:.2f}. No contradictions detected."
        )
    else:
        confirmation_reason = None

    summary_parts = [f"Fallback reflection found {len(contradictions)} issue(s)."]
    if revised:
        summary_parts.append(f"Score revised: {risk_score} → {revised_score}.")
    else:
        summary_parts.append("Score confirmed — no revision needed.")

    return {
        "contradictions": contradictions,
        "revised": revised,
        "revised_score": revised_score,
        "revision_reason": revision_reason,
        "confirmation_reason": confirmation_reason,
        "original_score": risk_score,
        "ml_score": ml_score,
        "heuristic_score": heuristic_score,
        "reflection_summary": " ".join(summary_parts),
        "reflection_source": "deterministic_fallback",
    }

