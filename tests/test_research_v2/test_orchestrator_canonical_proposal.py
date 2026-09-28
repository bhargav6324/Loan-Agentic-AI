from agents.orchestrator_agent import _fallback_decide


def _medium_after_fairness(confidence: float, *, bias_detected: bool = False):
    return {
        "risk_category": "MEDIUM",
        "risk_ml_confidence": confidence,
        "policy_passed": True,
        "bias_detected": bias_detected,
        "orchestrator_history": [
            {"agent": "fairness_agent", "reasoning": "completed"},
            *([{"agent": "risk_reeval", "reasoning": "completed"}] if bias_detected else []),
        ],
    }


def test_medium_low_confidence_remains_rejected_after_fairness():
    decision = _fallback_decide(_medium_after_fairness(0.59), iteration=2)
    assert decision["next_agent"] == "policy_join"
    assert decision["decision_proposal"] == "reject"


def test_medium_sufficient_confidence_can_be_approved_after_fairness():
    decision = _fallback_decide(_medium_after_fairness(0.60), iteration=2)
    assert decision["decision_proposal"] == "approve"


def test_fairness_bias_escalates_instead_of_approving():
    decision = _fallback_decide(_medium_after_fairness(0.90, bias_detected=True), iteration=3)
    assert decision["decision_proposal"] == "escalate"
