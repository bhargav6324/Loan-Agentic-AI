"""
Risk Adjustment Agent — True LLM-to-LLM Semantic Debate (V4)
─────────────────────────────────────────────────────────────
Two LLM personas with opposing objectives negotiate over a
financial decision:

  1. RISK OFFICER defends the score, citing financial evidence
  2. FAIRNESS OFFICER evaluates the defense for legitimacy

The outcome is determined by the quality of arguments, not
arithmetic subtraction.  This is genuinely novel: two LLMs
debating with the result influencing a financial decision.

If the defense is rejected → score adjusted, bias incident logged.
If the defense is accepted → delta is a legitimate risk signal.
"""

import time
import json
import httpx
from agents.state import AgentState
from agents.llm_client import call_groq
from config import settings


# ══════════════════════════════════════════════════════════
#  PROMPT 1: Risk Officer Defends the Score
# ══════════════════════════════════════════════════════════

RISK_DEFENSE_PROMPT = """
You are the Risk Officer defending a credit decision.

Applicant: {name}
Your Risk Score: {risk_score}/100
Your Key Reasoning: {risk_factors}
Credit Report: Credit Score={credit_score}, Defaults={defaults}, DTI={dti}

The Fairness Officer challenges you:
"{challenge}"

Defend your score OR concede that the variable ({bias_source}) is not
a legitimate risk proxy. Be specific. Cite financial evidence.

A legitimate defense must show that the demographic variable correlates
with a genuine financial risk factor (e.g., "rural applicants have
higher default rates due to agricultural income volatility").

You must NOT defend using circular logic (e.g., "the model scored them
higher because they are female" — that IS the bias, not a defense).

Respond ONLY with valid JSON:
{{
    "stance": "DEFEND" or "CONCEDE",
    "argument": "your specific argument citing financial evidence",
    "adjusted_score": null if DEFEND, or a number if CONCEDE
}}
"""


# ══════════════════════════════════════════════════════════
#  PROMPT 2: Fairness Officer Evaluates the Defense
# ══════════════════════════════════════════════════════════

FAIRNESS_EVALUATION_PROMPT = """
You are the Fairness Officer evaluating the Risk Officer's response.

Original Challenge: {challenge}
Risk Officer's Response: {risk_response}
Statistical Evidence: counterfactual delta = {delta} points when {bias_source} was changed

Is the Risk Officer's defense legitimate?

A legitimate defense must cite a genuine financial risk correlation
that is independent of demographic group membership.

Demographic group membership alone is NEVER a legitimate defense.
For example:
  - "Rural applicants default more" → NOT legitimate (statistical discrimination)
  - "This applicant's income is seasonal agricultural" → LEGITIMATE (specific financial risk)
  - "Female applicants have lower credit scores" → NOT legitimate (correlation ≠ causation)
  - "This applicant's DTI is 0.8" → LEGITIMATE (specific financial risk)

Respond ONLY with valid JSON:
{{
    "verdict": "ACCEPT_DEFENSE" or "REJECT_DEFENSE",
    "reasoning": "why you accept or reject the defense",
    "escalate_to_human": true or false
}}
"""


# ══════════════════════════════════════════════════════════
#  LLM CALL HELPER
# ══════════════════════════════════════════════════════════

def _call_llm(system_prompt: str, user_prompt: str, model: str = "llama-3.1-8b-instant") -> dict:
    """Make a single LLM call with JSON response format."""
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "gsk_placeholder":
        return None  # Signal to use fallback

    try:
        data = call_groq(
            payload={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            },
            api_key=settings.GROQ_API_KEY,
            timeout=12.0,
        )
        content = data["choices"][0]["message"]["content"].strip()
        return json.loads(content)
    except Exception as e:
        print(f"LLM debate call failed: {e}")
        return None


# ══════════════════════════════════════════════════════════
#  MAIN AGENT — True Semantic Debate
# ══════════════════════════════════════════════════════════

def risk_adjustment_agent(state: AgentState) -> AgentState:
    """
    True LLM-to-LLM semantic debate between Risk Officer and Fairness Officer.

    Two LLM personas exchange arguments:
      1. Risk Officer defends the score
      2. Fairness Officer evaluates the defense

    The outcome is determined by argument quality, not arithmetic.
    """
    logs = list(state.get("agent_logs", []))
    debate_log = list(state.get("fairness_debate_log", []))
    start = time.time()

    if not debate_log:
        return state  # Nothing to debate

    # Get the latest challenge from the Fairness Agent
    latest_challenge = debate_log[-1]
    name = state.get("applicant_data", {}).get("applicant_name", "Applicant")
    current_score = state.get("risk_score", 50.0)
    credit = state.get("credit_report", {}) or {}
    risk_factors = state.get("risk_factors", [])

    delta = latest_challenge.get("score_delta", 0)
    bias_source = latest_challenge.get("attribute_tested", "demographics")
    original_value = latest_challenge.get("original_value", "unknown")
    cf_value = latest_challenge.get("counterfactual_value", "unknown")

    challenge_text = (
        f"Changing {bias_source} from '{original_value}' to '{cf_value}' "
        f"shifted the risk score by {delta:+.1f} points. "
        f"This suggests the model is using {bias_source} as a risk factor. "
        f"Justify why this demographic variable is a legitimate risk proxy, "
        f"or concede that this constitutes algorithmic bias."
    )

    # ── Step 1: Risk Officer Defends ──────────────────────
    risk_defense_prompt = RISK_DEFENSE_PROMPT.format(
        name=name,
        risk_score=current_score,
        risk_factors=", ".join(risk_factors[:4]) or "None identified",
        credit_score=credit.get("credit_score", "N/A"),
        defaults=credit.get("total_defaults", "N/A"),
        dti=credit.get("current_outstanding", 0),
        challenge=challenge_text,
        bias_source=bias_source,
    )

    risk_response = _call_llm(
        "You are the Risk Officer at a bank. Defend your credit decision with financial evidence. Respond only with valid JSON.",
        risk_defense_prompt,
    )

    # Fallback if LLM unavailable
    if risk_response is None:
        risk_response = {
            "stance": "CONCEDE",
            "argument": f"Fallback: Unable to defend the use of {bias_source} as a risk proxy. Conceding to ensure fairness compliance.",
            "adjusted_score": round(current_score - abs(delta) * 0.5, 1),
        }

    logs.append({
        "agent": "risk_adjustment_agent",
        "step": len(logs) + 1,
        "action": f"SEMANTIC DEBATE: Risk Officer {risk_response.get('stance', 'N/A')}",
        "status": "defense",
        "details": f"Risk Officer argues: {risk_response.get('argument', 'No argument')}",
        "duration_ms": 0,
    })

    # ── Step 2: Fairness Officer Evaluates ────────────────
    fairness_verdict = None
    if risk_response.get("stance") == "DEFEND":
        fairness_eval_prompt = FAIRNESS_EVALUATION_PROMPT.format(
            challenge=challenge_text,
            risk_response=risk_response.get("argument", "No defense provided"),
            delta=abs(delta),
            bias_source=bias_source,
        )

        fairness_verdict = _call_llm(
            "You are the Fairness Officer at a bank. Evaluate risk defenses for legitimacy. Respond only with valid JSON.",
            fairness_eval_prompt,
        )

        # Fallback if LLM unavailable
        if fairness_verdict is None:
            fairness_verdict = {
                "verdict": "REJECT_DEFENSE",
                "reasoning": f"Fallback: Cannot verify defense for {bias_source}. Rejecting to ensure compliance.",
                "escalate_to_human": True,
            }

        logs.append({
            "agent": "risk_adjustment_agent",
            "step": len(logs) + 1,
            "action": f"SEMANTIC DEBATE: Fairness Officer {fairness_verdict.get('verdict', 'N/A')}",
            "status": "evaluation",
            "details": f"Fairness Officer: {fairness_verdict.get('reasoning', 'No reasoning')}",
            "duration_ms": 0,
        })

    # ── Step 3: Determine Outcome ─────────────────────────
    if risk_response.get("stance") == "CONCEDE":
        # Risk Officer conceded — adjust score
        new_score = risk_response.get("adjusted_score") or round(current_score - abs(delta) * 0.5, 1)
        new_score = max(0.0, min(100.0, float(new_score)))
        outcome = "CONCEDED"
        outcome_detail = (
            f"Risk Officer conceded: {risk_response.get('argument', 'N/A')}. "
            f"Score adjusted {current_score:.1f} → {new_score:.1f}."
        )
        state = {**state, "risk_score": new_score, "bias_detected": False}

    elif fairness_verdict and fairness_verdict.get("verdict") == "REJECT_DEFENSE":
        # Fairness Officer rejected the defense — bias confirmed
        adjustment = abs(delta) * 0.5  # Remove half the bias
        new_score = max(0.0, min(100.0, round(current_score - adjustment, 1)))
        outcome = "DEFENSE_REJECTED"
        outcome_detail = (
            f"Defense rejected: {fairness_verdict.get('reasoning', 'N/A')}. "
            f"Bias in {bias_source} confirmed. Score adjusted {current_score:.1f} → {new_score:.1f}."
        )
        escalate = fairness_verdict.get("escalate_to_human", False)
        state = {
            **state,
            "risk_score": new_score,
            "bias_detected": escalate,  # Keep bias flag if escalation needed
        }

    elif fairness_verdict and fairness_verdict.get("verdict") == "ACCEPT_DEFENSE":
        # Defense accepted — delta is a legitimate risk signal
        outcome = "DEFENSE_ACCEPTED"
        outcome_detail = (
            f"Defense accepted: {fairness_verdict.get('reasoning', 'N/A')}. "
            f"The {bias_source} delta of {delta:+.1f} is considered a legitimate risk signal. "
            f"Score unchanged at {current_score:.1f}."
        )
        state = {**state, "bias_detected": False}

    else:
        # Unknown state — conservative: adjust
        new_score = max(0.0, min(100.0, round(current_score - abs(delta) * 0.3, 1)))
        outcome = "INCONCLUSIVE"
        outcome_detail = f"Debate inconclusive. Conservative adjustment: {current_score:.1f} → {new_score:.1f}."
        state = {**state, "risk_score": new_score}

    # ── Record debate transcript ──────────────────────────
    debate_entry = {
        "agent": "risk_adjustment_agent",
        "turn": state.get("negotiation_turns", 0) + 1,
        "challenge": challenge_text,
        "risk_defense": {
            "stance": risk_response.get("stance"),
            "argument": risk_response.get("argument"),
        },
        "fairness_evaluation": {
            "verdict": fairness_verdict.get("verdict") if fairness_verdict else "N/A (Risk conceded)",
            "reasoning": fairness_verdict.get("reasoning") if fairness_verdict else "N/A",
        } if risk_response.get("stance") == "DEFEND" else None,
        "outcome": outcome,
        "resulting_score": state.get("risk_score", current_score),
    }
    debate_log.append(debate_entry)

    duration = int((time.time() - start) * 1000)

    logs.append({
        "agent": "risk_adjustment_agent",
        "step": len(logs) + 1,
        "action": f"Debate Turn {state.get('negotiation_turns', 0) + 1} — {outcome}",
        "status": "success",
        "details": outcome_detail,
        "duration_ms": duration,
    })

    return {
        **state,
        "fairness_debate_log": debate_log,
        "negotiation_turns": state.get("negotiation_turns", 0) + 1,
        "agent_logs": logs,
    }
