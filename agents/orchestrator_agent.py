"""
Orchestrator Agent — Continuous LLM-Driven Dynamic Orchestration (V5)
─────────────────────────────────────────────────────────────────────
Replaces the static plan[] + advance_step() architecture.

Instead of generating a plan array upfront and blindly iterating,
this orchestrator observes the CURRENT state after every agent
execution and makes a fresh decision about what to do next.

This is the single most impactful architectural change for achieving
genuine agentic behavior: the system can now REACT to new information
mid-pipeline.

Example:  policy_agent discovers applicant is 17 → orchestrator
          immediately routes to auto_reject (skipping fairness_agent).
          The old system would have continued executing the array.

Safety:
  • MAX_ITERATIONS = 12 (hard cap prevents infinite loops)
  • Mandatory terminal action (auto_approve / auto_reject / final_decision)
  • Deterministic fallback if LLM is unavailable
  • Full reasoning trace logged for auditability
"""

from __future__ import annotations
import time
import json
import httpx
from agents.state import AgentState
from agents.llm_client import call_groq
from config import settings


# ══════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════

MAX_ITERATIONS = 12  # Hard safety cap

AGENT_REGISTRY = {
    "policy_agent": (
        "Checks regulatory compliance (min age 21, LTI cap 5x, max 2 defaults, "
        "min credit 400, high-value employment rules). Uses RAG over RBI guidelines."
    ),
    "fairness_agent": (
        "Runs counterfactual bias testing: flips gender/age/location and re-scores. "
        "If bias is detected, triggers a multi-turn debate with the Risk Adjustment Agent."
    ),
    "risk_reeval": (
        "Re-runs the full Risk Agent scoring. Useful AFTER fairness adjustments "
        "or AFTER new data was gathered, to get an updated risk assessment."
    ),
    "fraud_recheck": (
        "Re-runs fraud checks. Useful if initial check was borderline (REVIEW) "
        "and you want higher confidence before approving."
    ),
    "decision_commit": "[TERMINAL] Propose a final decision (approve, reject, escalate) and route to commitment.",
}

TERMINAL_AGENTS = {"decision_commit", "policy_join"}
VALID_AGENTS = set(AGENT_REGISTRY.keys())


# ══════════════════════════════════════════════════════════
#  STATE SUMMARIZER
# ══════════════════════════════════════════════════════════

def _build_state_summary(state: AgentState) -> str:
    """Build a rich summary of the current pipeline state for the LLM."""
    applicant = state.get("applicant_data", {})
    credit = state.get("credit_report", {}) or {}
    reflection = state.get("risk_agent_reflection", {}) or {}

    monthly_income = applicant.get("monthly_income", 0) or 0
    loan_amount = applicant.get("loan_amount", 0) or 0
    annual_income = monthly_income * 12
    lti = round(loan_amount / annual_income, 2) if annual_income > 0 else "N/A"

    pan = state.get("pan_verification", {}) or {}
    aadhaar = state.get("aadhaar_verification", {}) or {}

    lines = [
        f"Applicant: {applicant.get('applicant_name', 'Unknown')}",
        f"Age: {applicant.get('applicant_age', 'N/A')} | Gender: {applicant.get('applicant_gender', 'N/A')} | Location: {applicant.get('applicant_location', 'N/A')}",
        f"Employment: {applicant.get('employment_type', 'N/A')}",
        f"Monthly Income: ₹{monthly_income:,.0f} | Loan: ₹{loan_amount:,.0f} | LTI: {lti}x",
        f"KYC: PAN={'✅' if pan.get('verified') else '❌'} Aadhaar={'✅' if aadhaar.get('verified') else '❌'}",
        f"Fraud: {state.get('fraud_detected', False)} | Manual KYC: {state.get('requires_manual_kyc', False)}",
        f"Risk Score: {state.get('risk_score', 'N/A')}/100 | Category: {state.get('risk_category', 'N/A')}",
        f"Risk Factors: {', '.join(state.get('risk_factors', [])[:4]) or 'None'}",
        f"Credit Score: {credit.get('credit_score', 'N/A')} | Defaults: {credit.get('total_defaults', 'N/A')}",
        f"Self-Reflection: {reflection.get('reflection_summary', 'N/A')}",
    ]

    # Add results of agents that have already run
    if state.get("policy_passed") is not None:
        violations = state.get("policy_violations", [])
        lines.append(f"Policy: {'PASSED ✅' if state['policy_passed'] else f'FAILED ❌ ({len(violations)} violations: {'; '.join(violations[:2])})'}")

    if state.get("fairness_report"):
        lines.append(f"Fairness: Bias={'DETECTED ⚠' if state.get('bias_detected') else 'None'} | Debate turns: {state.get('negotiation_turns', 0)}")

    # Add Episodic Memory Precedents for orchestrator routing context
    try:
        from memory.episodic_memory import retrieve_similar_cases
        similar = retrieve_similar_cases(applicant, credit, top_k=3)
        if similar:
            lines.append("\n═══ PAST ROUTING PRECEDENTS (EPISODIC MEMORY) ═══")
            for idx, case in enumerate(similar):
                outcome = case.get("outcome", {})
                lines.append(f"Case #{idx+1} (Sim: {case.get('similarity', 0):.2f}): {case.get('profile')}")
                lines.append(f"  → Final Decision: {outcome.get('decision')} | Bias: {outcome.get('bias_detected')}")
    except Exception as e:
        lines.append(f"\n[Episodic memory unavailable: {e}]")

    return "\n".join(lines)


def _build_execution_history(state: AgentState) -> str:
    """Build a history of orchestrator decisions so far."""
    history = state.get("orchestrator_history", [])
    if not history:
        return "No agents executed by orchestrator yet. This is the first decision."

    lines = []
    for entry in history:
        lines.append(f"Step {entry['step']}: {entry['agent']} — {entry['reasoning']}")
    return "\n".join(lines)


def _format_registry() -> str:
    lines = []
    for name, desc in AGENT_REGISTRY.items():
        lines.append(f"  • {name}: {desc}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════
#  LLM PROMPT
# ══════════════════════════════════════════════════════════

ORCHESTRATOR_PROMPT = """\
You are an autonomous loan evaluation orchestrator. After each agent \
completes, you observe the updated state and decide what to do NEXT.

You are NOT following a pre-generated plan. You reason in real-time.

═══ AVAILABLE AGENTS ═══
{agent_registry}

═══ CURRENT PIPELINE STATE ═══
{state_summary}

═══ ORCHESTRATOR HISTORY ═══
{execution_history}

═══ ITERATION ═══
Step {iteration} of {max_iterations} (you MUST terminate before step {max_iterations})

═══ DECISION RULES ═══
1. If KYC failed or fraud was detected → auto_reject immediately.
2. For LOW risk with clean KYC and no concerns → auto_approve.
3. For MEDIUM risk: usually run policy_agent then fairness_agent, but \
   the ORDER and whether to add extra steps is YOUR decision.
4. If policy_agent found critical violations → auto_reject (don't waste \
   compute on fairness for an applicant who fails basic eligibility).
5. After fairness_agent adjusts the score, consider risk_reeval to get \
   an updated risk assessment.
6. Do NOT re-run an agent that already executed UNLESS state has changed \
   since (e.g., risk_reeval after fairness adjustments).
7. For borderline cases where you are uncertain, propose escalate or reject.
8. You MUST terminate with decision_commit or policy_join.

Respond ONLY with valid JSON:
{{
    "next_agent": "agent_name",
    "decision_proposal": "approve, reject, or escalate (only required if next_agent is decision_commit or policy_join)",
    "reasoning": "2-3 sentences explaining your decision, referencing specific state data."
}}
"""


# ══════════════════════════════════════════════════════════
#  PLANNER LOGIC (V8)
# ══════════════════════════════════════════════════════════

ORCHESTRATOR_PLANNER_PROMPT = """\
You are the Loan Agentic AI Planner. Your job is to look at a new loan application \
and generate an INITIAL execution plan.

The pipeline always runs the Data Agent (KYC) and Fraud Agent first.
After Intake, the Orchestrator takes over.
You need to plan the sequence of agents to run AFTER intake.

═══ AVAILABLE AGENTS ═══
{agent_registry}

═══ APPLICATION INITIAL STATE ═══
{state_summary}

═══ PLANNING RULES ═══
1. The typical path for a normal application is:
   ['fan_out', 'policy_join', 'final_decision']
   (Note: fan_out automatically triggers risk_agent and policy_agent in parallel)
2. If you see immediate obvious reasons for rejection, you can plan to skip fairness and \
   go straight to ['fan_out', 'policy_join'].
3. If the profile has demographics known to historically suffer bias (e.g. rural, \
   certain employments), explicitly include 'fairness_agent' after the join.
4. Your plan is a STARTING POINT. The orchestrator will still dynamically route \
   after every step, and may deviate from this plan if KYC fails or ML confidence is low.

Respond ONLY with valid JSON:
{{
    "plan": ["agent_name_1", "agent_name_2", ...],
    "plan_reasoning": "Explain why you chose this initial sequence."
}}
"""

def generate_initial_plan(state: AgentState) -> AgentState:
    """Generate an upfront execution plan before agents run."""
    print("🧠 Planner: Generating initial execution plan...")
    
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "gsk_placeholder":
        return {
            **state,
            "plan": ["fan_out", "policy_join", "fairness_agent", "final_decision"],
            "plan_reasoning": "Fallback: Default deterministic plan."
        }

    try:
        prompt = ORCHESTRATOR_PLANNER_PROMPT.format(
            agent_registry=_format_registry(),
            state_summary=_build_state_summary(state)
        )

        data = call_groq(
            payload={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are the autonomous loan pipeline planner. Respond only with JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
            api_key=settings.GROQ_API_KEY,
            timeout=15.0,
        )

        parsed = json.loads(data["choices"][0]["message"]["content"].strip())
        plan = parsed.get("plan", ["fan_out", "policy_join", "final_decision"])
        reasoning = parsed.get("plan_reasoning", "No reasoning provided.")

        print(f"   Plan: {plan}")
        return {**state, "plan": plan, "plan_reasoning": reasoning}

    except Exception as e:
        print(f"   Planner failed: {e}")
        return {
            **state,
            "plan": ["fan_out", "policy_join", "fairness_agent", "final_decision"],
            "plan_reasoning": f"Fallback due to error: {e}"
        }


# ══════════════════════════════════════════════════════════
#  DETERMINISTIC FALLBACK
# ══════════════════════════════════════════════════════════

def _fallback_decide(state: AgentState, iteration: int) -> dict:
    """Deterministic fallback when LLM is unavailable."""
    risk_category = state.get("risk_category", "MEDIUM")
    fraud_detected = state.get("fraud_detected", False)
    kyc_failed = state.get("data_agent_early_stop", False)
    history = state.get("orchestrator_history", [])
    agents_run = {h["agent"] for h in history}

    # Immediate rejection cases
    if fraud_detected or kyc_failed:
        return {"next_agent": "policy_join", "decision_proposal": "reject", "reasoning": "Fallback: KYC/fraud failure → reject."}

    if risk_category == "HIGH":
        return {"next_agent": "policy_join", "decision_proposal": "reject", "reasoning": "Fallback: HIGH risk → reject."}

    if risk_category == "LOW" and "policy_agent" in agents_run:
        if state.get("policy_passed", True):
            return {"next_agent": "policy_join", "decision_proposal": "approve", "reasoning": "Fallback: LOW risk + policy passed → approve."}
        else:
            return {"next_agent": "policy_join", "decision_proposal": "reject", "reasoning": "Fallback: LOW risk but policy failed → reject."}

    # LOW risk, policy not yet run
    if risk_category == "LOW" and "policy_agent" not in agents_run:
        return {"next_agent": "policy_agent", "reasoning": "Fallback: LOW risk, running policy check first."}

    # MEDIUM risk — sequential execution
    if "policy_agent" not in agents_run:
        return {"next_agent": "policy_agent", "reasoning": "Fallback: MEDIUM risk, policy check first."}

    # Check if policy failed
    if not state.get("policy_passed", True):
        return {"next_agent": "policy_join", "decision_proposal": "reject", "reasoning": "Fallback: Policy violations found → reject."}

    if "fairness_agent" not in agents_run:
        return {"next_agent": "fairness_agent", "reasoning": "Fallback: MEDIUM risk, running fairness analysis."}

    # After fairness, check if risk_reeval is needed
    if "risk_reeval" not in agents_run and state.get("bias_detected"):
        return {"next_agent": "risk_reeval", "reasoning": "Fallback: Bias detected, re-evaluating risk."}

    decision_prop = "escalate" if state.get("bias_detected") else "approve"
    return {"next_agent": "policy_join", "decision_proposal": decision_prop, "reasoning": "Fallback: All checks done, making final decision."}


# ══════════════════════════════════════════════════════════
#  AMBIGUITY DETECTION — The Agentic Gate
# ══════════════════════════════════════════════════════════
#
#  The orchestrator is deterministic for the ~85% of applications
#  that match known patterns, and falls back to LLM-driven reasoning
#  only for the ambiguous ~15% — which is where it improves outcomes
#  over a pure rule engine.

def _is_ambiguous(state: AgentState) -> tuple[bool, str]:
    """
    Determine if the current pipeline state is genuinely ambiguous
    and requires LLM reasoning to resolve. Returns (is_ambiguous, reason).

    An ambiguous state is one where the deterministic fallback logic
    cannot cleanly map to a single confident next action — i.e., states
    we didn't anticipate at design time.
    """
    # 1. Low ML confidence — the model itself is uncertain
    confidence = state.get("risk_ml_confidence", 1.0)
    if confidence < 0.55:
        return True, f"Low ML model confidence ({confidence:.2f}) — model is uncertain about this profile"

    # 2. Conflicting verdicts — no rule covers this combination
    if state.get("risk_category") == "LOW" and state.get("requires_manual_kyc"):
        return True, "Low risk but fraud agent flagged manual KYC — conflicting signals"

    if state.get("policy_passed") and state.get("bias_detected"):
        return True, "Policy passed but fairness agent detected demographic bias — conflicting signals"

    if state.get("risk_category") == "LOW" and state.get("bias_detected"):
        return True, "Low risk but bias detected — need LLM to weigh trade-off"

    # 3. Fraud confidence in the ambiguous band (borderline identity match)
    fraud_conf = state.get("fraud_match_confidence", 1.0)
    if 0.3 < fraud_conf < 0.8:
        return True, f"Fraud match confidence in ambiguous band ({fraud_conf:.2f})"

    # 4. Deterministic path has stalled — fallback couldn't decide cleanly
    iterations = state.get("orchestrator_iterations", 0)
    if iterations >= 3 and not state.get("final_decision"):
        history = state.get("orchestrator_history", [])
        agents_run = {h["agent"] for h in history}
        # If we've run the key agents but still haven't terminated, something is unusual
        if "policy_agent" in agents_run and "fairness_agent" in agents_run:
            return True, f"Deterministic path stalled after {iterations} iterations with all checks done"

    return False, ""


# ══════════════════════════════════════════════════════════
#  LLM CALL (only invoked when state is ambiguous)
# ══════════════════════════════════════════════════════════

def _call_llm_orchestrator(state: AgentState, iteration: int) -> dict:
    """Call the LLM to decide the next action. Only reached for ambiguous states."""

    # No API key → fallback
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "gsk_placeholder":
        result = _fallback_decide(state, iteration)
        result["reasoning"] = f"[Fallback] {result['reasoning']}"
        return result

    state_summary = _build_state_summary(state)
    execution_history = _build_execution_history(state)

    prompt = ORCHESTRATOR_PROMPT.format(
        agent_registry=_format_registry(),
        state_summary=state_summary,
        execution_history=execution_history,
        iteration=iteration,
        max_iterations=MAX_ITERATIONS,
    )

    try:
        data = call_groq(
            payload={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an autonomous loan pipeline orchestrator. "
                            "You observe state and decide the next action in real-time. "
                            "Respond only with valid JSON. No markdown."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
            api_key=settings.GROQ_API_KEY,
            timeout=15.0,
        )

        content = data["choices"][0]["message"]["content"].strip()
        parsed = json.loads(content)

        next_agent = parsed.get("next_agent", "").strip()
        decision_proposal = parsed.get("decision_proposal", "approve") if next_agent in TERMINAL_AGENTS else None

        # Validate
        if next_agent not in VALID_AGENTS:
            fb = _fallback_decide(state, iteration)
            fb["reasoning"] = f"LLM returned invalid agent '{next_agent}'. {fb['reasoning']}"
            return fb

        return {"next_agent": next_agent, "decision_proposal": decision_proposal, "reasoning": f"[LLM-Agentic] {reasoning}"}

    except Exception as e:
        print(f"Orchestrator LLM failed: {e}")
        result = _fallback_decide(state, iteration)
        result["reasoning"] = f"[Fallback: {e}] {result['reasoning']}"
        return result


# ══════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR NODE
# ══════════════════════════════════════════════════════════

def orchestrator_decide(state: AgentState) -> AgentState:
    """
    Continuous Dynamic Orchestrator (V6 — Ambiguity-Gated).

    Called after every agent execution. Observes the updated state
    and decides what to do next.

    V6 upgrade: Uses deterministic fallback for routine cases (~85%),
    invokes LLM reasoning ONLY for genuinely ambiguous states (~15%).
    This is the true definition of agentic behavior — reasoning only
    where rules cannot safely decide.
    """
    logs = list(state.get("agent_logs", []))
    history = list(state.get("orchestrator_history", []))
    iteration = len(history) + 1
    start = time.time()

    # ── Safety: max iterations ────────────────────────────
    if iteration > MAX_ITERATIONS:
        decision = {
            "next_agent": "final_decision",
            "reasoning": f"Safety cap: reached {MAX_ITERATIONS} iterations. Forcing final_decision.",
        }
    else:
        # ── Ambiguity-Gated Decision ──────────────────────
        # Try deterministic fallback first (fast, free, reliable)
        fallback = _fallback_decide(state, iteration)

        # Check if state is genuinely ambiguous
        ambiguous, ambiguity_reason = _is_ambiguous(state)

        if not ambiguous:
            # Routine case — deterministic rules handle it
            decision = fallback
            decision["reasoning"] = f"[Deterministic] {decision['reasoning']}"
        else:
            # Ambiguous case — invoke LLM for agentic reasoning
            decision = _call_llm_orchestrator(state, iteration)
            decision["reasoning"] = f"(Ambiguity: {ambiguity_reason}) {decision['reasoning']}"

    next_agent = decision["next_agent"]
    reasoning = decision["reasoning"]
    decision_proposal = decision.get("decision_proposal")

    # Record this decision
    history.append({
        "step": iteration,
        "agent": next_agent,
        "reasoning": reasoning,
    })

    duration = int((time.time() - start) * 1000)
    is_terminal = next_agent in TERMINAL_AGENTS
    is_deterministic = "[Deterministic]" in reasoning
    is_llm = "[LLM-Agentic]" in reasoning

    logs.append({
        "agent": "orchestrator",
        "step": len(logs) + 1,
        "action": f"Orchestrator → {next_agent}" + (" [TERMINAL]" if is_terminal else ""),
        "status": "deterministic" if is_deterministic else ("llm_agentic" if is_llm else "fallback"),
        "details": f"Step {iteration}/{MAX_ITERATIONS}: {reasoning}",
        "duration_ms": duration,
    })

    # Build a running plan list for backward compatibility with frontend
    executed_agents = [h["agent"] for h in history]

    ret_state = {
        **state,
        "orchestrator_next_agent": next_agent,
        "orchestrator_history": history,
        "orchestrator_iterations": iteration,
        # Backward compatibility with frontend/test_pipeline
        "plan": executed_agents,
        "plan_reasoning": " → ".join(
            f"{h['agent']}({h['reasoning'][:60]}...)" for h in history[-3:]
        ),
        "agent_logs": logs,
    }
    
    if decision_proposal:
        ret_state["decision_proposal"] = decision_proposal
        
    return ret_state

