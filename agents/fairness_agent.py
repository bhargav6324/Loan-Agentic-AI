"""
Fairness & Bias Agent — Merged Counterfactual Math & LLM Debate (V6)
"""

from __future__ import annotations
import time
import json
import httpx
import os
from agents.state import AgentState
from config import settings
from agents.llm_client import call_groq

# --- FAIRNESS MATH LOGIC ---
"""
Fairness Agent — Counterfactual Bias Detection
───────────────────────────────────────────────
Runs counterfactual analysis on MEDIUM-risk (borderline) cases
to detect potential demographic bias.

Methodology:
  1. Takes the original risk score
  2. Recalculates with protected attributes flipped (gender, age group, location)
  3. Logs the SIGNED delta score (cf_score - original_score) for every test
  4. If the decision changes → bias detected → escalate to human review
  5. If no bias → approve the borderline case

Key patent claim: every bias determination is backed by a concrete, verifiable
numerical delta — not a black-box flag. The signed delta shows *direction*
(positive = applicant was scored more harshly with the original attribute).
"""



def _simulate_counterfactual(state: AgentState, flip_attribute: str, flip_value, logs: list) -> tuple[float, str]:
    """
    Recalculate risk score with one protected attribute changed.
    Uses the Risk Agent's core scoring logic to simulate inter-agent debate.
    """
    import copy
    from agents.risk_agent import _extract_features, evaluate_profile

    # Create a modified state to simulate the counterfactual
    modified_state = copy.deepcopy(state)
    if "applicant_data" not in modified_state:
        modified_state["applicant_data"] = {}
        
    original_value = None
    if flip_attribute == "gender":
        original_value = modified_state["applicant_data"].get("applicant_gender")
        modified_state["applicant_data"]["applicant_gender"] = flip_value
    elif flip_attribute == "age":
        original_value = modified_state["applicant_data"].get("applicant_age")
        modified_state["applicant_data"]["applicant_age"] = flip_value
    elif flip_attribute == "location":
        original_value = modified_state["applicant_data"].get("applicant_location")
        modified_state["applicant_data"]["applicant_location"] = flip_value

    # Extract features from the modified state
    modified_features = _extract_features(modified_state)
    
    # Send the challenge (log it)
    logs.append({
        "agent": "fairness_agent",
        "step": len(logs) + 1,
        "action": f"INTER-AGENT DEBATE: Challenge",
        "status": "challenge",
        "details": f"Re-evaluate profile with {flip_attribute}={flip_value} (was {original_value}). Justify any score difference.",
        "duration_ms": 0,
    })

    # Risk Agent responds by evaluating the modified features
    eval_result = evaluate_profile(modified_features, modified_state)
    cf_score = eval_result["risk_score"]
    agent_justification = eval_result.get("agent_justification", "No justification provided")

    # Log the Risk Agent's response
    logs.append({
        "agent": "risk_agent",
        "step": len(logs) + 1,
        "action": f"INTER-AGENT DEBATE: Response",
        "status": "response",
        "details": f"Re-evaluated with {flip_attribute}={flip_value}. New score is {cf_score}. Justification: {agent_justification}",
        "duration_ms": 0,
    })

    return cf_score, agent_justification


def _run_fairness_math(state: AgentState) -> AgentState:
    """
    Fairness Agent node for LangGraph.

    Only runs for MEDIUM-risk cases. Performs counterfactual analysis
    across three protected attributes: gender, age, and location.

    Every result records:
      - signed_delta = cf_score - original_score
          Positive → original was scored MORE harshly (possible adverse treatment)
          Negative → flip made score WORSE (possible adverse impact on that group)
      - score_difference = |signed_delta| (absolute magnitude)
      - bias_flag = True if magnitude exceeds per-attribute threshold

    A `counterfactual_delta_summary` is attached to the fairness_report,
    giving a single, auditor-readable table of worst deltas per attribute.
    This makes the bias determination quantifiable and technically defensible.
    """
    logs = list(state.get("agent_logs", []))
    errors = list(state.get("errors", []))
    start = time.time()

    original_score = state.get("risk_score", 50)
    applicant = state.get("applicant_data", {})
    original_gender = applicant.get("applicant_gender") or "male"
    original_age = applicant.get("applicant_age") or 30
    original_location = applicant.get("applicant_location") or "semi_urban"

    counterfactual_results = []
    bias_detected = False
    debate_log = []

    # Get the original justification from the state
    original_reflection = state.get("risk_agent_reflection", {})
    original_justification = state.get("risk_agent_summary", "Unknown")

    # ── Read Long Term Memory (Systemic Learning) ────────
    memory_file = os.path.join(os.path.dirname(__file__), "..", "data", "long_term_memory.json")
    systemic_bias_penalty = 0.0
    if os.path.exists(memory_file):
        try:
            with open(memory_file, "r") as f:
                memory = json.load(f)
            # Example learning: If >60% of past women were rejected, tighten the gender threshold
            women_records = [r for r in memory if r.get("gender") == "female"]
            if len(women_records) > 5:
                reject_rate = len([r for r in women_records if r.get("decision") == "rejected"]) / len(women_records)
                if reject_rate > 0.6:
                    systemic_bias_penalty = 2.0 # Tighten thresholds by 2 points
                    logs.append({
                        "agent": "fairness_agent",
                        "step": len(logs) + 1,
                        "action": "Systemic Memory Read",
                        "status": "warning",
                        "details": f"Historical rejection rate for females is high ({reject_rate*100:.0f}%). Tightening bias detection thresholds by {systemic_bias_penalty} pts.",
                        "duration_ms": 0
                    })
        except Exception:
            pass

    # Dynamic Thresholds
    gender_threshold = max(1.0, 5.0 - systemic_bias_penalty)
    age_threshold = max(1.0, 10.0 - systemic_bias_penalty)
    location_threshold = max(1.0, 8.0 - systemic_bias_penalty)

    # ── Test 1: Gender flip ──────────────────────────────
    # Gender MUST NOT change the score. Any delta here is direct evidence of bias.
    genders_to_test = [g for g in ["male", "female", "other"] if g != original_gender]
    for alt_gender in genders_to_test:
        cf_score, agent_justification = _simulate_counterfactual(state, "gender", alt_gender, logs)
        signed_delta = round(float(cf_score - original_score), 2)   # KEY: signed, not absolute
        score_diff = abs(signed_delta)
        
        debate_log.append({
            "attribute_tested": "gender",
            "original_value": original_gender,
            "counterfactual_value": alt_gender,
            "original_score": original_score,
            "counterfactual_score": cf_score,
            "score_delta": signed_delta,
            "risk_agent_justification": agent_justification
        })

        result = {
            "attribute": "gender",
            "original_value": original_gender,
            "counterfactual_value": alt_gender,
            "original_score": original_score,
            "counterfactual_score": round(float(cf_score), 2),
            "signed_delta": signed_delta,           # positive = original was harsher
            "score_difference": round(score_diff, 1),
            "threshold": gender_threshold,
            "bias_flag": bool(score_diff > gender_threshold),
            "interpretation": (
                "ORIGINAL scored harsher (possible adverse treatment)"
                if signed_delta > 5 else
                ("ORIGINAL scored more leniently (possible inverse bias)"
                 if signed_delta < -5 else
                 "No significant gender-driven score change")
            ),
        }
        counterfactual_results.append(result)
        if result["bias_flag"]:
            bias_detected = True

    gender_results = [r for r in counterfactual_results if r["attribute"] == "gender"]
    max_gender_delta = max((r["score_difference"] for r in gender_results), default=0)
    worst_gender = max(gender_results, key=lambda r: r["score_difference"], default=None)

    logs.append({
        "agent": "fairness_agent",
        "step": len(logs) + 1,
        "action": "Gender counterfactual analysis",
        "status": "bias_detected" if any(r["bias_flag"] for r in gender_results) else "no_bias",
        "details": (
            f"Tested {len(genders_to_test)} gender variants | "
            f"Max |Δ|={max_gender_delta:.1f} pts | "
            f"Worst: {original_gender}→{worst_gender['counterfactual_value'] if worst_gender else 'N/A'} "
            f"Δ={worst_gender['signed_delta']:+.2f} pts"
        ),
        "duration_ms": 0,
    })

    # ── Test 2: Age group flip ───────────────────────────
    # Age brackets above 10yr difference may legitimately affect score
    # but only via financial behaviour proxies — not directly.
    age_groups = [25, 35, 45, 55]
    for alt_age in age_groups:
        if abs(alt_age - original_age) < 5:
            continue
        cf_score, agent_justification = _simulate_counterfactual(state, "age", alt_age, logs)
        signed_delta = round(float(cf_score - original_score), 2)
        score_diff = abs(signed_delta)
        
        debate_log.append({
            "attribute_tested": "age",
            "original_value": original_age,
            "counterfactual_value": alt_age,
            "original_score": original_score,
            "counterfactual_score": cf_score,
            "score_delta": signed_delta,
            "risk_agent_justification": agent_justification
        })

        result = {
            "attribute": "age",
            "original_value": original_age,
            "counterfactual_value": alt_age,
            "original_score": original_score,
            "counterfactual_score": round(float(cf_score), 2),
            "signed_delta": signed_delta,
            "score_difference": round(score_diff, 1),
            "threshold": age_threshold,
            "bias_flag": bool(score_diff > age_threshold),
            "interpretation": (
                "ORIGINAL scored harsher (possible age-based adverse treatment)"
                if signed_delta > 10 else
                ("ORIGINAL scored more leniently (possible ageism against target bracket)"
                 if signed_delta < -10 else
                 "Within acceptable age-related variance")
            ),
        }
        counterfactual_results.append(result)
        if result["bias_flag"]:
            bias_detected = True

    age_results = [r for r in counterfactual_results if r["attribute"] == "age"]
    max_age_delta = max((r["score_difference"] for r in age_results), default=0)
    worst_age = max(age_results, key=lambda r: r["score_difference"], default=None)

    logs.append({
        "agent": "fairness_agent",
        "step": len(logs) + 1,
        "action": "Age counterfactual analysis",
        "status": "bias_detected" if any(r["bias_flag"] for r in age_results) else "no_bias",
        "details": (
            f"Tested {len(age_results)} age brackets | "
            f"Max |Δ|={max_age_delta:.1f} pts | "
            f"Worst: age {original_age}→{worst_age['counterfactual_value'] if worst_age else 'N/A'} "
            f"Δ={worst_age['signed_delta']:+.2f} pts"
        ),
        "duration_ms": 0,
    })

    # ── Test 3: Location flip ─────────────────────────────
    # Location (urban/semi_urban/rural) should NOT penalise applicants beyond
    # what objective credit behaviour warrants. Delta here flags systemic bias.
    locations_to_test = [loc for loc in ["urban", "semi_urban", "rural"] if loc != original_location]
    for alt_location in locations_to_test:
        cf_score, agent_justification = _simulate_counterfactual(state, "location", alt_location, logs)
        signed_delta = round(float(cf_score - original_score), 2)
        score_diff = abs(signed_delta)
        
        debate_log.append({
            "attribute_tested": "location",
            "original_value": original_location,
            "counterfactual_value": alt_location,
            "original_score": original_score,
            "counterfactual_score": cf_score,
            "score_delta": signed_delta,
            "risk_agent_justification": agent_justification
        })

        result = {
            "attribute": "location",
            "original_value": original_location,
            "counterfactual_value": alt_location,
            "original_score": original_score,
            "counterfactual_score": round(float(cf_score), 2),
            "signed_delta": signed_delta,
            "score_difference": round(score_diff, 1),
            "threshold": location_threshold,
            "bias_flag": bool(score_diff > location_threshold),
            "interpretation": (
                "ORIGINAL location scored harsher (possible geographic discrimination)"
                if signed_delta > 8 else
                ("ORIGINAL location scored more leniently (possible reverse location bias)"
                 if signed_delta < -8 else
                 "Within acceptable location-related variance")
            ),
        }
        counterfactual_results.append(result)
        if result["bias_flag"]:
            bias_detected = True

    location_results = [r for r in counterfactual_results if r["attribute"] == "location"]
    max_loc_delta = max((r["score_difference"] for r in location_results), default=0)
    worst_loc = max(location_results, key=lambda r: r["score_difference"], default=None)

    logs.append({
        "agent": "fairness_agent",
        "step": len(logs) + 1,
        "action": "Location counterfactual analysis",
        "status": "bias_detected" if any(r["bias_flag"] for r in location_results) else "no_bias",
        "details": (
            f"Tested {len(location_results)} location variants | "
            f"Max |Δ|={max_loc_delta:.1f} pts | "
            f"Worst: {original_location}→{worst_loc['counterfactual_value'] if worst_loc else 'N/A'} "
            f"Δ={worst_loc['signed_delta']:+.2f} pts"
        ),
        "duration_ms": 0,
    })

    duration = int((time.time() - start) * 1000)

    # ── Delta summary (patent-verifiable, one row per attribute) ──
    counterfactual_delta_summary = {
        "gender": {
            "max_absolute_delta": round(max_gender_delta, 2),
            "worst_signed_delta": round(worst_gender["signed_delta"], 2) if worst_gender else 0.0,
            "threshold": gender_threshold,
            "bias_flagged": any(r["bias_flag"] for r in gender_results),
        },
        "age": {
            "max_absolute_delta": round(max_age_delta, 2),
            "worst_signed_delta": round(worst_age["signed_delta"], 2) if worst_age else 0.0,
            "threshold": age_threshold,
            "bias_flagged": any(r["bias_flag"] for r in age_results),
        },
        "location": {
            "max_absolute_delta": round(max_loc_delta, 2),
            "worst_signed_delta": round(worst_loc["signed_delta"], 2) if worst_loc else 0.0,
            "threshold": location_threshold,
            "bias_flagged": any(r["bias_flag"] for r in location_results),
        },
    }

    # ── Build fairness report ────────────────────────────
    fairness_report = {
        "original_risk_score": original_score,
        "counterfactual_results": counterfactual_results,
        "counterfactual_delta_summary": counterfactual_delta_summary,
        "bias_detected": bias_detected,
        "recommendation": "escalate" if bias_detected else "approve",
        "summary": (
            "⚠️ Potential bias detected — decision varies with protected attributes. Escalating to human review."
            if bias_detected else
            "✅ No demographic bias detected — decision is consistent across all protected attributes."
        ),
    }

    if bias_detected:
        flagged_attrs = [a for a, v in counterfactual_delta_summary.items() if v["bias_flagged"]]
        summary = (
            f"⚠️ BIAS DETECTED in: {', '.join(flagged_attrs)} — "
            f"Worst delta: gender Δ={counterfactual_delta_summary['gender']['worst_signed_delta']:+.2f}, "
            f"age Δ={counterfactual_delta_summary['age']['worst_signed_delta']:+.2f}, "
            f"location Δ={counterfactual_delta_summary['location']['worst_signed_delta']:+.2f} — "
            f"Recommending escalation to human reviewer"
        )
    else:
        summary = (
            f"✅ No bias detected — gender Δ={counterfactual_delta_summary['gender']['worst_signed_delta']:+.2f}, "
            f"age Δ={counterfactual_delta_summary['age']['worst_signed_delta']:+.2f}, "
            f"location Δ={counterfactual_delta_summary['location']['worst_signed_delta']:+.2f} — Fairness check passed"
        )

    logs.append({
        "agent": "fairness_agent",
        "step": len(logs) + 1,
        "action": "Fairness assessment complete",
        "status": "escalate" if bias_detected else "passed",
        "details": summary,
        "duration_ms": duration,
    })

    return {
        **state,
        "fairness_report": fairness_report,
        "fairness_debate_log": debate_log,
        "bias_detected": bias_detected,
        "fairness_agent_summary": summary,
        "agent_logs": logs,
        "errors": errors,
    }

# --- FAIRNESS DEBATE LOGIC ---
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

def _run_fairness_debate(state: AgentState) -> AgentState:
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


def fairness_agent(state: AgentState) -> AgentState:
    """
    Consolidated Fairness Agent.
    1. Runs the counterfactual math.
    2. If any bias gaps > threshold are found, triggers multi-round LLM debate
       for EACH flagged attribute.
    """
    state = _run_fairness_math(state)

    # Trigger debate for every attribute where counterfactual math flagged bias
    debate_log = state.get("fairness_debate_log", [])
    flagged_entries = [
        entry for entry in debate_log
        if abs(entry.get("score_delta", 0)) > 5
        and entry.get("attribute_tested")          # must have an attribute
        and "outcome" not in entry                  # skip already-debated entries
    ]

    MAX_DEBATE_ROUNDS = 3
    rounds = 0
    for entry in flagged_entries:
        if rounds >= MAX_DEBATE_ROUNDS:
            break
        # Set the active challenge as the last entry so _run_fairness_debate picks it up
        state["fairness_debate_log"] = state.get("fairness_debate_log", [])
        state["fairness_debate_log"].append(entry)  # ensure it's the latest
        state = _run_fairness_debate(state)
        rounds += 1

    return state
