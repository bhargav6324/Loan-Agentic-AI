"""
Canonical Deterministic Contracts for Loan Agentic AI
These functions represent the absolute source of truth for business logic.
They must remain PURE (no LangGraph or agent imports).
"""
import copy

# ── 1. Schema ─────────────────────────────────────────────────────────

class PolicyInput:
    """
    Schema for the policy contract. It accepts the core RF features
    and additional fields required for policy enforcement (e.g. loan_amount).
    Does NOT include target_default.
    """
    def __init__(self, applicant_data: dict):
        self.applicant_age = applicant_data.get("applicant_age", applicant_data.get("age", 30))
        self.loan_to_income = applicant_data.get("loan_to_income", 1.0)
        self.defaults = applicant_data.get("defaults", 0)
        self.credit_score = applicant_data.get("credit_score", 650)
        self.loan_amount = applicant_data.get("loan_amount", 0)
        self.employment_score = applicant_data.get("employment_score", 1.0)

# ── 2. Composite Risk Score ──────────────────────────────────────────

def canonical_composite_score(p_default: float, heuristic_score: float) -> tuple[float, float, float]:
    """
    Returns (ml_score, heuristic_score, composite_score)
    """
    ml_score = p_default * 100.0
    ml_confidence = max(p_default, 1.0 - p_default)
    ml_weight = 0.4 + 0.4 * ml_confidence
    heuristic_weight = 1.0 - ml_weight
    
    composite_score = (ml_weight * ml_score) + (heuristic_weight * heuristic_score)
    
    return round(ml_score, 1), round(heuristic_score, 1), round(composite_score, 1)

def canonical_risk_category(score: float) -> str:
    if score <= 35: return "LOW"
    if score <= 65: return "MEDIUM"
    return "HIGH"

# ── 3. Hard Overrides ────────────────────────────────────────────────
def canonical_hard_overrides(features: dict) -> str | None:
    """
    Hard Override Pre-Screening Layer.
    Some conditions should bypass scoring entirely.
    Returns 'HARD_REJECT', 'HARD_FAST_TRACK', or None.
    """
    if (features["defaults"] > 3 or
        features["credit_score"] < 450 or
        features["dti_ratio"] > 0.90):
        return "HARD_REJECT"

    if (features["credit_score"] > 820 and
        features["defaults"] == 0 and
        features["dti_ratio"] < 0.15 and
        features.get("employment_years", 3) > 5):
        return "HARD_FAST_TRACK"

    return None

def canonical_hard_override_score(override: str) -> float:
    """Return the shared score used by both static and graph paths."""
    if override == "HARD_REJECT":
        return 95.0
    if override == "HARD_FAST_TRACK":
        return 10.0
    raise ValueError(f"Unknown hard override: {override}")

# ── 4. Deterministic Risk Reassessment ───────────────────────────────

def canonical_risk_reassessment(original_score: float, ml_score: float, heuristic_score: float, features: dict) -> dict:
    """
    One-pass risk reassessment to handle contradictions.
    """
    revised_score = original_score
    reassessed = False
    revision_reason = ""
    contradictions = []

    if ml_score is not None:
        gap = abs(ml_score - heuristic_score)
        if gap > 25:
            contradictions.append(f"ML ({ml_score:.1f}) and Heuristic ({heuristic_score:.1f}) diverge by {gap:.1f}.")
            conservative_score = round((ml_score + heuristic_score) / 2.0, 1)
            if conservative_score > revised_score:
                revised_score = conservative_score
                reassessed = True
                revision_reason = "Conservative blending applied due to high divergence."

    defaults = features.get("defaults", 0)
    if revised_score <= 35 and defaults > 0:
        contradictions.append(f"LOW score ({revised_score}) despite having {defaults} default(s).")
        if revised_score < 40.0:
            revised_score = 40.0
            reassessed = True
            revision_reason = "Floor applied (40.0) due to previous defaults."
            
    return {
        "revised": reassessed,
        "revised_score": round(revised_score, 1),
        "revision_reason": revision_reason,
        "confirmation_reason": "No structural contradictions found." if not reassessed else "",
        "contradictions": contradictions
    }

# ── 4. Policy Check ──────────────────────────────────────────────────

def canonical_policy_check(applicant_data: dict) -> tuple[bool, list[str]]:
    """
    Shared pure policy evaluation. Returns (passed, violations).
    """
    data = PolicyInput(applicant_data)
    violations = []
    
    if data.applicant_age < 21:
        violations.append(f"Minimum age requirement not met (applicant is {data.applicant_age}, minimum is 21)")
    if data.loan_to_income > 5:
        violations.append(f"Loan-to-income ratio too high ({data.loan_to_income:.1f}x, maximum allowed is 5x)")
    if data.defaults > 2:
        violations.append(f"Too many active defaults ({data.defaults}, maximum allowed is 2)")
    if data.credit_score < 400:
        violations.append(f"Credit score below minimum threshold ({data.credit_score}, minimum is 400)")
    if data.loan_amount > 2_500_000 and data.employment_score == 1:
        violations.append(f"High-value loan (₹{data.loan_amount:,.0f}) requires salaried or business employment, not self-employed")
        
    return len(violations) == 0, violations

# ── 5. Audit Flip ────────────────────────────────────────────────────

def canonical_audit_eligibility(risk_category: str) -> bool:
    """Only MEDIUM and HIGH risk require audit."""
    return risk_category in ["MEDIUM", "HIGH"]

def canonical_audit_flip(
    predict_fn,
    features_dict: dict,
    original_composite_score: float,
    epsilon: float = 6.0,
    apply_reassessment: bool = True,
) -> dict:
    """
    Counterfactual fairness audit computing the composite delta.
    """
    f_flip = copy.deepcopy(features_dict)
    
    # Flip gender (assumes 0/1 encoding)
    if "gender_encoded" in f_flip:
        f_flip["gender_encoded"] = 1.0 - f_flip["gender_encoded"]
    elif "gender" in f_flip and isinstance(f_flip["gender"], (int, float)):
        f_flip["gender"] = 1.0 - f_flip["gender"]
    
    # Evaluate model probability using the provided prediction function
    p_default_flip = predict_fn(f_flip)
    
    from ml.canonical_model_contract import compute_heuristic_score
    heuristic_score_flip = compute_heuristic_score(f_flip)
    
    _, _, counterfactual_composite_score = canonical_composite_score(p_default_flip, heuristic_score_flip)
    
    # Match the configuration being audited: A1 has no reassessment, while
    # A1R/A2/A3 apply the same deterministic reassessment to both worlds.
    if apply_reassessment:
        reassessment = canonical_risk_reassessment(
            counterfactual_composite_score,
            p_default_flip * 100.0,
            heuristic_score_flip,
            f_flip,
        )
        if reassessment["revised"]:
            counterfactual_composite_score = reassessment["revised_score"]
        
    signed_delta = counterfactual_composite_score - original_composite_score
    flagged = abs(signed_delta) > epsilon
    
    # Flag generation and downstream action are deliberately separate.
    # Positive delta: flipped applicant is scored riskier, so the original
    # applicant received the more lenient treatment -> conservative reject.
    # Negative delta: original applicant was scored more harshly -> approve.
    if not flagged:
        recommended_action = "proceed"
    elif signed_delta > 0:
        recommended_action = "reject"
    else:
        recommended_action = "approve"
    
    return {
        "executed": True,
        "original_score": original_composite_score,
        "counterfactual_score": counterfactual_composite_score,
        "signed_delta": round(signed_delta, 2),
        "flagged": flagged,
        "recommended_action": recommended_action
    }

# ── 6. Final Decision Logic ──────────────────────────────────────────

def canonical_decision_proposal(risk_category: str, policy_passed: bool, risk_ml_confidence: float) -> str:
    """
    Core logic for proposing a decision before final commitment.
    Includes Ambiguity-Gating for MEDIUM risk.
    """
    if policy_passed is False or risk_category == "HIGH":
        return "reject"
    elif risk_category == "LOW":
        return "approve"
    else:
        # MEDIUM risk => Ambiguity-gated check based on ML confidence
        if risk_ml_confidence < 0.6:
            return "reject"
        else:
            return "approve"

def canonical_final_decision(
    risk_category: str,
    policy_passed: bool,
    policy_violations: list,
    audit_required: bool,
    audit_executed: bool,
    audit_flagged: bool,
    decision_proposal: str,
    audit_recommended_action: str | None = None,
) -> tuple[str, str]:
    """
    Determines final commit status and decision reasoning.
    Returns (status, reasoning)
    """
    if not policy_passed:
        if decision_proposal == "approve":
            status = "blocked_policy_violation"
            reasoning = f"Commitment blocked: Orchestrator proposed approve but policy failed. Violations: {policy_violations}"
        else:
            status = "committed_reject"
            reasoning = "Application rejected due to policy violations."
    elif audit_required and not audit_executed:
        status = "blocked_missing_audit"
        reasoning = "Commitment blocked: required gender attribute-flip sensitivity audit was not executed."
    elif audit_flagged:
        if audit_recommended_action == "approve":
            status = "committed_approve"
            reasoning = "Automated audit override: original applicant was scored more harshly than the gender-flipped profile."
        elif audit_recommended_action == "reject":
            status = "committed_reject"
            reasoning = "Automated audit override: original applicant was scored more leniently than the gender-flipped profile."
        else:
            status = "blocked_invalid_audit_action"
            reasoning = "Commitment blocked: sensitivity flag lacks a valid deterministic downstream action."
    elif decision_proposal == "approve":
        status = "committed_approve"
        reasoning = "Application approved. All checks passed."
    elif decision_proposal == "reject":
        status = "committed_reject"
        reasoning = "Application rejected based on risk evaluation."
    elif decision_proposal == "escalate":
        status = "automated_escalated_status"
        reasoning = "Application routed to automated escalated status."
    else:
        status = "blocked_branch_conflict"
        reasoning = f"Commitment blocked: unknown or missing decision proposal '{decision_proposal}'."
        
    return status, reasoning
