"""
Planner Agent — Autonomous LLM-Driven Execution Planner (V2)
─────────────────────────────────────────────────────────────
Generates genuinely open-ended execution plans using an LLM.

V1 was a router disguised as a planner: the prompt constrained the LLM
to exactly 4 hardcoded plan arrays.  V2 replaces that with a free-form
planning prompt that lets the LLM compose novel execution paths:

  • Call any agent 0 or more times
  • Repeat agents (e.g., risk_reeval after fairness mitigation)
  • Skip agents the LLM deems unnecessary for the profile
  • Re-order agents based on applicant demographics

Safety rails:
  • Maximum plan length: 8 steps (prevents infinite loops)
  • Plan MUST end with a terminal action (auto_approve / auto_reject / final_decision)
  • Every agent name is validated against a known registry
  • Deterministic fallback if LLM fails or returns invalid output
"""

from __future__ import annotations
import time
import json
import httpx
from agents.state import AgentState
from config import settings


# ══════════════════════════════════════════════════════════
#  AGENT REGISTRY — the single source of truth for what
#  agents the planner may invoke.  Each entry describes the
#  agent's purpose so the LLM can reason about when to use it.
# ══════════════════════════════════════════════════════════

AGENT_REGISTRY = {
    "policy_agent": (
        "Checks the applicant's data against regulatory lending policies "
        "(minimum age, loan-to-income cap, default limits, credit floor, "
        "high-value employment rules). Uses RAG over the RBI guidelines document."
    ),
    "fairness_agent": (
        "Runs counterfactual bias analysis by flipping protected attributes "
        "(gender, age, location) and re-scoring.  Detects demographic bias "
        "and can trigger a multi-turn debate with the Risk Adjustment Agent."
    ),
    "risk_reeval": (
        "Re-runs the full Risk Agent scoring engine.  Useful after the "
        "Fairness Agent adjusts the score, or after new data is gathered, "
        "to get an updated risk assessment on the modified state."
    ),
    "fraud_recheck": (
        "Re-runs the Fraud Agent's identity and anomaly checks.  Useful "
        "if the initial fraud check flagged a borderline case (REVIEW) and "
        "additional context is now available."
    ),
    "auto_approve": (
        "Terminates the pipeline with an APPROVED decision.  Use for LOW "
        "risk profiles with clean KYC and no policy concerns."
    ),
    "auto_reject": (
        "Terminates the pipeline with a REJECTED decision.  Use for HIGH "
        "risk profiles, KYC failures, fraud detection, or critical policy violations."
    ),
    "final_decision": (
        "Terminates the pipeline with a nuanced decision (approve / reject / "
        "escalate) for MEDIUM-risk or borderline cases.  Should only be used "
        "AFTER policy and/or fairness checks have been completed."
    ),
}

VALID_PLAN_AGENTS = set(AGENT_REGISTRY.keys())
TERMINAL_AGENTS = {"auto_approve", "auto_reject", "final_decision"}
MAX_PLAN_LENGTH = 8


# ══════════════════════════════════════════════════════════
#  STATE SUMMARY BUILDER
# ══════════════════════════════════════════════════════════

def _build_state_summary(state: AgentState, kyc_failed: bool) -> str:
    """Build a rich, structured summary of the current pipeline state for the LLM."""
    applicant = state.get("applicant_data", {})
    credit = state.get("credit_report", {}) or {}
    reflection = state.get("risk_agent_reflection", {}) or {}
    risk_factors = state.get("risk_factors", [])

    monthly_income = applicant.get("monthly_income", 0) or 0
    loan_amount = applicant.get("loan_amount", 0) or 0
    annual_income = monthly_income * 12
    lti_ratio = round(loan_amount / annual_income, 2) if annual_income > 0 else "N/A"

    return (
        f"- Name: {applicant.get('applicant_name', 'Unknown')}\n"
        f"- Age: {applicant.get('applicant_age', 'Unknown')} | "
        f"Gender: {applicant.get('applicant_gender', 'Unknown')} | "
        f"Location: {applicant.get('applicant_location', 'Unknown')}\n"
        f"- Employment: {applicant.get('employment_type', 'Unknown')} "
        f"({applicant.get('employment_years', 'Unknown')} years)\n"
        f"- Monthly Income: ₹{monthly_income:,.0f} | "
        f"Loan Amount: ₹{loan_amount:,.0f} | "
        f"Loan-to-Income: {lti_ratio}x\n"
        f"- KYC Status: {'FAILED ❌' if kyc_failed else 'VERIFIED ✅'}\n"
        f"- Fraud Detected: {state.get('fraud_detected', False)} | "
        f"Manual KYC Required: {state.get('requires_manual_kyc', False)}\n"
        f"- Risk Score: {state.get('risk_score', 'N/A')}/100 | "
        f"Category: {state.get('risk_category', 'N/A')}\n"
        f"- Risk Factors: {', '.join(risk_factors[:4]) or 'None identified'}\n"
        f"- Credit Score: {credit.get('credit_score', 'N/A')} | "
        f"Defaults: {credit.get('total_defaults', 'N/A')} | "
        f"Utilisation: {credit.get('credit_utilisation_pct', 'N/A')}%\n"
        f"- Data Agent Early Stop: {state.get('data_agent_early_stop', False)}\n"
        f"- Self-Reflection: {reflection.get('reflection_summary', 'No reflection data')}"
    )


# ══════════════════════════════════════════════════════════
#  LLM PROMPT
# ══════════════════════════════════════════════════════════

def _format_registry() -> str:
    """Format the agent registry into a human-readable block for the prompt."""
    lines = []
    for name, desc in AGENT_REGISTRY.items():
        tag = " [TERMINAL]" if name in TERMINAL_AGENTS else ""
        lines.append(f"  • {name}{tag}: {desc}")
    return "\n".join(lines)


PLANNER_PROMPT = """\
You are an autonomous loan evaluation orchestrator.  Your job is to generate \
a dynamic execution plan — an ordered sequence of agents to run — based on \
the applicant's current profile and pipeline state.

You are NOT a router.  You are a planner.  You may compose novel execution \
paths that were never hardcoded.

═══ AVAILABLE AGENTS ═══
{agent_registry}

═══ CURRENT PIPELINE STATE ═══
{state_summary}

═══ PLANNING RULES ═══
1. Your plan must contain between 1 and {max_steps} steps.
2. You may use any agent 0 or more times.  Repetition is allowed when justified \
   (e.g., risk_reeval after fairness mitigation).
3. Your plan MUST end with exactly one terminal action: \
   auto_approve, auto_reject, or final_decision.
4. If KYC failed or fraud was detected, the only valid plan is ["auto_reject"].
5. You MAY skip agents you deem unnecessary for this specific profile.
6. For MEDIUM-risk cases, you should usually include both policy_agent and \
   fairness_agent, but the ORDER and whether to add extra steps is your decision.
7. Consider whether re-evaluating risk (risk_reeval) after fairness adjustments \
   would produce a more accurate outcome.
8. For borderline HIGH/MEDIUM cases, consider running fraud_recheck to add confidence.

═══ EXAMPLES OF VALID PLANS (not exhaustive — you may invent new ones) ═══
- ["auto_reject"]
- ["auto_approve"]
- ["policy_agent", "fairness_agent", "final_decision"]
- ["fairness_agent", "policy_agent", "final_decision"]
- ["policy_agent", "fairness_agent", "risk_reeval", "final_decision"]
- ["fraud_recheck", "policy_agent", "fairness_agent", "final_decision"]
- ["fairness_agent", "risk_reeval", "policy_agent", "final_decision"]

Respond ONLY with valid JSON (no markdown, no explanation outside JSON):
{{
  "plan": ["agent1", "agent2", ...],
  "plan_reasoning": "Detailed 2-4 sentence explanation of why this specific \
sequence was chosen for this applicant, referencing their profile data."
}}
"""


# ══════════════════════════════════════════════════════════
#  PLAN VALIDATION
# ══════════════════════════════════════════════════════════

def _validate_plan(plan: list) -> tuple[list | None, str | None]:
    """
    Validate and sanitise the LLM-generated plan.

    Returns:
        (sanitised_plan, None) on success
        (None, error_reason) on failure
    """
    if not isinstance(plan, list) or len(plan) == 0:
        return None, "Plan is empty or not a list"

    # Enforce maximum length
    if len(plan) > MAX_PLAN_LENGTH:
        plan = plan[:MAX_PLAN_LENGTH]

    # Check all agent names are in the registry
    for agent in plan:
        if agent not in VALID_PLAN_AGENTS:
            return None, f"Unknown agent in plan: '{agent}'"

    # Ensure plan ends with a terminal action
    if plan[-1] not in TERMINAL_AGENTS:
        # Try to fix by appending final_decision (if under length cap)
        if len(plan) < MAX_PLAN_LENGTH:
            plan.append("final_decision")
        else:
            # Replace last step with final_decision
            plan[-1] = "final_decision"

    return plan, None


# ══════════════════════════════════════════════════════════
#  LLM CALL
# ══════════════════════════════════════════════════════════

def _call_llm_planner(state_summary: str, kyc_failed: bool, state: AgentState) -> dict:
    """
    Call the LLM to generate an execution plan.
    Falls back to deterministic logic on failure.
    """
    risk_category = state.get("risk_category", "MEDIUM")
    fraud_detected = state.get("fraud_detected", False)
    applicant = state.get("applicant_data", {})

    # ── Deterministic fallback ────────────────────────────
    def _fallback_logic(reason: str = "LLM unavailable") -> dict:
        if kyc_failed or fraud_detected:
            plan = ["auto_reject"]
            reasoning = f"Fallback ({reason}): KYC failed or fraud detected — auto-rejecting."
        elif risk_category == "HIGH":
            plan = ["auto_reject"]
            reasoning = f"Fallback ({reason}): HIGH risk — auto-rejecting."
        elif risk_category == "LOW":
            plan = ["auto_approve"]
            reasoning = f"Fallback ({reason}): LOW risk — auto-approving."
        else:
            # MEDIUM risk — use heuristics for ordering
            age = applicant.get("applicant_age", 30) or 30
            emp_type = applicant.get("employment_type", "salaried")
            gender = applicant.get("applicant_gender", "male")
            requires_manual_kyc = state.get("requires_manual_kyc", False)

            plan = []

            # Add fraud_recheck if manual KYC was flagged
            if requires_manual_kyc:
                plan.append("fraud_recheck")

            # High-variance demographics: fairness first
            high_variance = (
                emp_type == "self_employed"
                or age < 25
                or gender not in ("male", None)
            )

            if high_variance:
                plan.extend(["fairness_agent", "policy_agent"])
            else:
                plan.extend(["policy_agent", "fairness_agent"])

            # Consider risk re-evaluation for borderline scores
            risk_score = state.get("risk_score", 50)
            if risk_score and 55 <= risk_score <= 65:
                plan.append("risk_reeval")

            plan.append("final_decision")
            reasoning = (
                f"Fallback ({reason}): MEDIUM risk (score={risk_score}). "
                f"{'Fairness-first for high-variance demographic. ' if high_variance else ''}"
                f"{'Added fraud_recheck for manual KYC case. ' if requires_manual_kyc else ''}"
                f"{'Added risk_reeval for borderline score. ' if risk_score and 55 <= risk_score <= 65 else ''}"
                f"Deterministic routing applied."
            )

        return {"plan": plan, "plan_reasoning": reasoning}

    # If no API key, immediately fall back
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "gsk_placeholder":
        return _fallback_logic("no API key")

    # Format the prompt
    formatted_prompt = PLANNER_PROMPT.format(
        agent_registry=_format_registry(),
        state_summary=state_summary,
        max_steps=MAX_PLAN_LENGTH,
    )

    try:
        response = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an autonomous loan pipeline orchestrator. "
                            "Respond only with valid JSON.  No markdown."
                        ),
                    },
                    {"role": "user", "content": formatted_prompt},
                ],
                "temperature": 0.1,  # slight creativity for novel plans
                "response_format": {"type": "json_object"},
            },
            timeout=15.0,
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)

        raw_plan = parsed.get("plan")
        reasoning = parsed.get("plan_reasoning", "LLM provided no reasoning.")

        if not raw_plan:
            return _fallback_logic("LLM returned empty plan")

        # Validate
        validated_plan, error = _validate_plan(raw_plan)
        if error:
            fb = _fallback_logic(f"validation failed: {error}")
            fb["plan_reasoning"] = (
                f"LLM plan rejected ({error}). "
                f"Original LLM plan was: {raw_plan}. "
                f"{fb['plan_reasoning']}"
            )
            return fb

        return {"plan": validated_plan, "plan_reasoning": reasoning}

    except Exception as e:
        print(f"LLM planner failed: {e}")
        return _fallback_logic(str(e))


# ══════════════════════════════════════════════════════════
#  MAIN AGENT FUNCTION
# ══════════════════════════════════════════════════════════

def planner_agent(state: AgentState) -> AgentState:
    """
    Autonomous Planner Agent (V2) for the Loan System.

    Uses an open-ended LLM prompt to dynamically generate execution plans.
    The LLM can compose novel agent sequences — including repeating agents,
    skipping agents, and re-ordering steps — constrained only by the agent
    registry and a maximum plan length of 8 steps.
    """
    logs = list(state.get("agent_logs", []))
    start = time.time()

    # ── 1. Determine KYC status ───────────────────────────
    kyc_failed = False
    if state.get("data_agent_early_stop"):
        kyc_failed = True

    pan_result = state.get("pan_verification")
    aadhaar_result = state.get("aadhaar_verification")
    if not pan_result or not pan_result.get("verified") or not pan_result.get("name_match"):
        kyc_failed = True
    elif not aadhaar_result or not aadhaar_result.get("verified") or not aadhaar_result.get("name_match"):
        kyc_failed = True

    # ── 2. Build state summary for the LLM ────────────────
    state_summary = _build_state_summary(state, kyc_failed)

    # ── 3. Call the LLM planner ───────────────────────────
    llm_response = _call_llm_planner(state_summary, kyc_failed, state)

    plan = llm_response.get("plan", ["auto_reject"])
    reasoning = llm_response.get("plan_reasoning", "Fallback plan due to LLM error.")

    duration = int((time.time() - start) * 1000)
    summary = f"Plan generated ({len(plan)} steps): {' → '.join(plan)}"

    # Log whether the plan is LLM-generated or fallback
    is_fallback = "Fallback" in reasoning
    logs.append({
        "agent": "planner_agent",
        "step": len(logs) + 1,
        "action": "Generated autonomous execution plan",
        "status": "success" if not is_fallback else "fallback",
        "details": f"{summary} | Reasoning: {reasoning}",
        "duration_ms": duration,
    })

    return {
        **state,
        "plan": plan,
        "plan_reasoning": reasoning,
        "current_step_index": 0,
        "agent_logs": logs,
    }
