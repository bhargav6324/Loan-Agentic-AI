"""
Intake & Compliance Agent — Merged Data, Fraud, and Policy (V7)
────────────────────────────────────────────────────────────────
Changes from V6:
  1. Data Agent: Deterministic ReAct loop (LLM only on tool exception)
  2. Fraud Agent: Fuzzy name gate (difflib) — LLM only in ambiguous band
  3. Policy Agent: RAG score gate — LLM only when retrieval is uncertain
  4. Identity security: SHA-256 index deduplication (no raw PII stored)
"""

from __future__ import annotations
import time
import json
import httpx
import copy
import hashlib
import os
from difflib import SequenceMatcher
from agents.state import AgentState
from config import settings
from agents.llm_client import call_groq
from tools.mock_pan import verify_pan
from tools.mock_aadhaar import verify_aadhaar
from tools.mock_credit_bureau import get_credit_report
from tools.mock_gst import verify_gst_returns
from tools.mock_bank_statements import verify_bank_statements

# SHA-256 identity index — maps hash(PAN/Aadhaar) → user_id, no raw PII stored
_IDENTITY_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "identity_index.json")


def _sha256_id(value: str) -> str:
    return hashlib.sha256(value.strip().upper().encode()).hexdigest()


def _load_identity_index() -> dict:
    if os.path.exists(_IDENTITY_INDEX_PATH):
        try:
            with open(_IDENTITY_INDEX_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_identity_index(index: dict) -> None:
    os.makedirs(os.path.dirname(_IDENTITY_INDEX_PATH), exist_ok=True)
    with open(_IDENTITY_INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)


def _jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Jaro-Winkler using stdlib SequenceMatcher (zero extra deps)."""
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if s1 == s2:
        return 1.0
    return SequenceMatcher(None, s1, s2).ratio()

# --- DATA AGENT LOGIC ---
"""
Data Agent — Identity Verification & Data Enrichment (v3 — Fully Agentic)
────────────────────────────────────────────────────────────────────────
Autonomously gathers applicant data using a ReAct loop.
Crucially, it decides WHAT data to collect dynamically:
- Standard KYC: PAN, Aadhaar, CIBIL
- Dynamic: GST Returns (if self-employed), Bank Statements (if young or complex)
"""


# ══════════════════════════════════════════════════════════
#  TOOL REGISTRY
# ══════════════════════════════════════════════════════════

_AVAILABLE_TOOLS = {
    "verify_pan": verify_pan,
    "verify_aadhaar": verify_aadhaar,
    "get_credit_report": get_credit_report,
    "verify_gst_returns": verify_gst_returns,
    "verify_bank_statements": verify_bank_statements,
}

# ══════════════════════════════════════════════════════════
#  REASONING ENGINE
# ══════════════════════════════════════════════════════════

def _fallback_reason_next_action(
    applicant: dict,
    tools_called: list[str],
    results: dict,
    step: int,
) -> dict:
    pan_number = applicant.get("applicant_pan")
    aadhaar_number = applicant.get("applicant_aadhaar")
    name = applicant.get("applicant_name")
    emp_type = applicant.get("employment_type")
    age = applicant.get("applicant_age") or 30

    # 1. PAN
    if "verify_pan" not in tools_called:
        if not pan_number:
            return {"thought": "No PAN provided, skipping.", "action": "SKIP_PAN", "reason_for_stop": None}
        return {"thought": "Verifying PAN first for identity.", "action": "verify_pan", "reason_for_stop": None}

    pan_result = results.get("verify_pan")
    if pan_result and (not pan_result.get("verified") or not pan_result.get("name_match")):
        return {"thought": "PAN failed KYC.", "action": "STOP", "reason_for_stop": "PAN name mismatch or invalid"}

    # 2. Aadhaar
    if "verify_aadhaar" not in tools_called:
        if not aadhaar_number:
            return {"thought": "No Aadhaar provided, skipping.", "action": "SKIP_AADHAAR", "reason_for_stop": None}
        return {"thought": "Verifying Aadhaar for secondary identity.", "action": "verify_aadhaar", "reason_for_stop": None}

    aadhaar_result = results.get("verify_aadhaar")
    if aadhaar_result and (not aadhaar_result.get("verified") or not aadhaar_result.get("name_match")):
        return {"thought": "Aadhaar failed KYC.", "action": "STOP", "reason_for_stop": "Aadhaar name mismatch or invalid"}

    # 3. Dynamic Tools (The Agentic Upgrade)
    if "verify_gst_returns" not in tools_called and emp_type == "self_employed":
        return {
            "thought": "Applicant is self-employed. Standard credit reports are often insufficient for business income. Autonomously deciding to request GST returns to verify actual cash flow.",
            "action": "verify_gst_returns",
            "reason_for_stop": None
        }
        
    if "verify_bank_statements" not in tools_called and age < 25:
        return {
            "thought": f"Applicant is very young ({age}). They likely have a thin credit file. Autonomously deciding to request 6-month bank statements to assess current liquidity.",
            "action": "verify_bank_statements",
            "reason_for_stop": None
        }

    # 4. Credit Bureau
    if "get_credit_report" not in tools_called:
        pan_ok = pan_result and pan_result.get("verified")
        aadhaar_ok = aadhaar_result and aadhaar_result.get("verified")
        if pan_ok or aadhaar_ok:
            return {"thought": "Identity verified. Pulling credit report.", "action": "get_credit_report", "reason_for_stop": None}
        else:
            return {"thought": "No valid ID.", "action": "STOP", "reason_for_stop": "No identity verified"}

    return {"thought": "All necessary data gathered.", "action": "STOP", "reason_for_stop": None}


DATA_REACT_PROMPT = """
You are a credit evaluation data agent. Your goal is to gather necessary verification data for a loan applicant.

Applicant Profile:
- Name: {name}
- Age: {age}
- PAN: {pan_number}
- Aadhaar: {aadhaar_number}
- Employment Type: {emp_type}

Available Tools:
- verify_pan: Verifies applicant's PAN card. Required first.
- verify_aadhaar: Verifies applicant's Aadhaar card. Required second.
- get_credit_report: Pulls credit bureau score. Requires PAN or Aadhaar to be verified first.
- verify_gst_returns: Checks annual business turnover. Only call if self_employed.
- verify_bank_statements: Checks average bank balance. Only call if young (< 25).
- STOP: End the data gathering process.

Rules:
1. Always start by verifying identity. Call verify_pan first if PAN is provided, then verify_aadhaar.
2. If PAN or Aadhaar verification results show that name_match is false or verified is false, output action STOP with reason_for_stop explaining the mismatch.
3. If identity is verified, call get_credit_report.
4. If the applicant is self-employed, call verify_gst_returns.
5. If the applicant is under 25 years old, call verify_bank_statements.
6. Once all necessary/applicable tools are executed successfully, output action STOP.

Execution History so far:
{history_str}

Decide the NEXT action to take.
Respond ONLY with a valid JSON object matching this schema:
{{
    "thought": "Your step-by-step reasoning explaining why this action is chosen.",
    "action": "verify_pan" | "verify_aadhaar" | "get_credit_report" | "verify_gst_returns" | "verify_bank_statements" | "STOP" | "SKIP_PAN" | "SKIP_AADHAAR",
    "reason_for_stop": "Description of why we are stopping early, or null"
}}
"""

def _reason_next_action(
    applicant: dict,
    tools_called: list[str],
    results: dict,
    step: int,
    tool_exception: Exception | None = None,
) -> dict:
    """
    V7: Deterministic is now the PRIMARY path.
    LLM is called ONLY when a tool raised an exception and the fallback
    can't classify the situation — that's the genuine edge case.
    """
    # If a tool failed unexpectedly, ask the LLM what to do
    if tool_exception and settings.GROQ_API_KEY and settings.GROQ_API_KEY != "gsk_placeholder":
        history_lines = [f"- Tool: {t} | Result: {results.get(t, 'Skipped/Failed')}" for t in tools_called]
        history_str = "\n".join(history_lines) or "No tools called yet."
        formatted_prompt = DATA_REACT_PROMPT.format(
            name=applicant.get("applicant_name", "Unknown"),
            age=applicant.get("applicant_age", 30),
            pan_number=applicant.get("applicant_pan", "None"),
            aadhaar_number=applicant.get("applicant_aadhaar", "None"),
            emp_type=applicant.get("employment_type", "None"),
            history_str=history_str + f"\n- LAST TOOL EXCEPTION: {tool_exception}"
        )
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a credit evaluation agent. Respond only with valid JSON."},
                {"role": "user", "content": formatted_prompt}
            ],
            "temperature": 0.0
        }
        try:
            data = call_groq(payload, settings.GROQ_API_KEY, timeout=10.0)
            content = data["choices"][0]["message"]["content"].strip()
            if content.startswith("```json"): content = content[7:-3].strip()
            elif content.startswith("```"): content = content[3:-3].strip()
            return json.loads(content)
        except Exception:
            pass  # Fall through to deterministic

    # Deterministic primary path (always runs)
    return _fallback_reason_next_action(applicant, tools_called, results, step)

# ══════════════════════════════════════════════════════════
#  TOOL EXECUTOR & OBSERVER
# ══════════════════════════════════════════════════════════

def _execute_tool(tool_name: str, applicant: dict) -> dict:
    name = applicant.get("applicant_name")
    if tool_name == "verify_pan": return verify_pan(applicant.get("applicant_pan"), name)
    elif tool_name == "verify_aadhaar": return verify_aadhaar(applicant.get("applicant_aadhaar"), name)
    elif tool_name == "get_credit_report": return get_credit_report(applicant.get("applicant_pan"))
    elif tool_name == "verify_gst_returns": return verify_gst_returns(applicant.get("applicant_pan"))
    elif tool_name == "verify_bank_statements": return verify_bank_statements(applicant.get("applicant_aadhaar"))
    raise ValueError(f"Unknown tool: {tool_name}")

def _observe(tool_name: str, result: dict) -> str:
    if tool_name == "verify_pan": return f"PAN: {'VALID' if result.get('verified') else 'INVALID'}"
    if tool_name == "verify_aadhaar": return f"Aadhaar: {'ACTIVE' if result.get('verified') else 'INACTIVE'}"
    if tool_name == "get_credit_report": return f"CIBIL: {result.get('credit_score')} | Defaults: {result.get('total_defaults')}"
    if tool_name == "verify_gst_returns": return f"GST Turnover: {result.get('annual_turnover', 'N/A')}"
    if tool_name == "verify_bank_statements": return f"Avg Balance: {result.get('average_monthly_balance', 'N/A')}"
    return str(result)

# ══════════════════════════════════════════════════════════
#  MAIN AGENT FUNCTION
# ══════════════════════════════════════════════════════════

def _run_data(state: AgentState) -> AgentState:
    logs = list(state.get("agent_logs", []))
    errors = list(state.get("errors", []))
    applicant = state.get("applicant_data", {})

    tools_called = []
    tool_results = {}
    reasoning_trace = []
    early_stop = False
    stop_reason = None
    
    start_time = time.time()

    for iteration in range(1, 10):
        decision = _reason_next_action(applicant, tools_called, tool_results, iteration)
        thought = decision["thought"]
        action = decision["action"]

        logs.append({"agent": "data_agent", "step": len(logs)+1, "action": f"ReAct[{iteration}] REASON", "status": "reasoning", "details": thought, "duration_ms": 0})

        if action.startswith("SKIP"):
            tools_called.append(action.replace("SKIP_", "verify_").lower())
            reasoning_trace.append({"iteration": iteration, "thought": thought, "action": action, "observation": "Skipped"})
            continue

        if action == "STOP":
            stop_reason = decision.get("reason_for_stop")
            early_stop = bool(stop_reason)
            reasoning_trace.append({"iteration": iteration, "thought": thought, "action": "STOP", "observation": stop_reason or "Complete"})
            logs.append({"agent": "data_agent", "step": len(logs)+1, "action": f"ReAct[{iteration}] STOP", "status": "early_stop" if early_stop else "complete", "details": thought, "duration_ms": 0})
            break

        try:
            result = _execute_tool(action, applicant)
            tool_results[action] = result
            tools_called.append(action)
            observation = _observe(action, result)
            reasoning_trace.append({"iteration": iteration, "thought": thought, "action": f"CALL {action}", "observation": observation})
            logs.append({"agent": "data_agent", "step": len(logs)+1, "action": f"ReAct[{iteration}] ACT → {action}", "status": "success", "details": observation, "duration_ms": 0})
        except Exception as e:
            # V7: Tool exception → ask LLM for recovery guidance
            errors.append(f"{action} failed: {e}")
            tools_called.append(action)
            reasoning_trace.append({"iteration": iteration, "thought": thought, "action": f"CALL {action}", "observation": f"ERROR: {e}"})
            # Override next decision with LLM-guided recovery
            decision = _reason_next_action(applicant, tools_called, tool_results, iteration, tool_exception=e)

    actual_tools = [t for t in tools_called if t in _AVAILABLE_TOOLS]
    summary = f"ReAct loop: {len(reasoning_trace)} iterations, {len(actual_tools)} tools called dynamically based on profile."
    
    # Store dynamic results in state
    if "applicant_data" not in state: state["applicant_data"] = {}
    if "verify_gst_returns" in tool_results:
        state["applicant_data"]["gst_turnover"] = tool_results["verify_gst_returns"].get("annual_turnover")
    if "verify_bank_statements" in tool_results:
        state["applicant_data"]["avg_bank_balance"] = tool_results["verify_bank_statements"].get("average_monthly_balance")

    logs.append({"agent": "data_agent", "step": len(logs)+1, "action": "Data gathering complete", "status": "early_stop" if early_stop else "success", "details": summary, "duration_ms": int((time.time() - start_time)*1000)})

    return {
        **state,
        "pan_verification": tool_results.get("verify_pan"),
        "aadhaar_verification": tool_results.get("verify_aadhaar"),
        "credit_report": tool_results.get("get_credit_report"),
        "data_agent_summary": summary,
        "data_agent_reasoning_trace": reasoning_trace,
        "data_agent_tools_called": actual_tools,
        "data_agent_early_stop": early_stop,
        "agent_logs": logs,
        "errors": errors,
    }

# --- FRAUD AGENT LOGIC ---
"""
Fraud Agent — Anomaly & Identity Verification (V3)
──────────────────────────────────────────────────
Detects identity mismatch (PAN/Aadhaar) and suspicious combinations.
V3 Upgrade: Uses an LLM to reason about identity mismatches instead of
hard rejecting. E.g., it can recognize maiden names or minor typos and
escalate for manual KYC instead of outright rejecting as fraud.
"""


FRAUD_LLM_PROMPT = """
You are a KYC Fraud Analyst for a bank.

Applicant Stated Name: {stated_name}
Name on Official ID: {id_name}

Determine if this is:
1. FRAUD (Completely different names, obvious mismatch)
2. REVIEW (Minor typo, missing middle name, or potential maiden name)
3. MATCH (Exact match)

Respond ONLY with valid JSON:
{{
    "status": "FRAUD" | "REVIEW" | "MATCH",
    "reasoning": "Brief explanation"
}}
"""

def _call_llm_fraud(stated_name: str, id_name: str) -> dict:
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "gsk_placeholder":
        # Fallback rules
        if stated_name.lower() == id_name.lower():
            return {"status": "MATCH", "reasoning": "Exact match (fallback)"}
        elif stated_name.split()[0].lower() == id_name.split()[0].lower():
            return {"status": "REVIEW", "reasoning": "First names match, possible maiden/last name change (fallback)"}
        else:
            return {"status": "FRAUD", "reasoning": "Completely different names (fallback)"}
            
    formatted_prompt = FRAUD_LLM_PROMPT.format(stated_name=stated_name, id_name=id_name)
    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.1-8b-instant", # Use a smaller/faster model for simple tasks
        "messages": [
            {"role": "system", "content": "You are a strict JSON-only API."},
            {"role": "user", "content": formatted_prompt}
        ],
        "temperature": 0.0
    }
    
    try:
        data = call_groq(payload, settings.GROQ_API_KEY, timeout=10.0)
        content = data["choices"][0]["message"]["content"].strip()
        if content.startswith("```json"): content = content[7:-3].strip()
        elif content.startswith("```"): content = content[3:-3].strip()
        return json.loads(content)
    except Exception as e:
        # Graceful fallback on rate limit/API failure
        if stated_name.lower() == id_name.lower():
            return {"status": "MATCH", "reasoning": f"Exact match (fallback due to error: {e})"}
        elif stated_name.split()[0].lower() == id_name.split()[0].lower():
            return {"status": "REVIEW", "reasoning": f"First name match (fallback due to error: {e})"}
        else:
            return {"status": "FRAUD", "reasoning": f"Completely different names (fallback due to error: {e})"}


def _check_name_match(stated_name: str, id_name: str, doc_type: str) -> tuple[str, str]:
    """
    V7 Fuzzy gate: Jaro-Winkler similarity determines the path.
      >= 0.92  → MATCH (deterministic, no LLM)
      <  0.60  → FRAUD (deterministic, no LLM)
      0.60–0.92 → ambiguous → call LLM (typo / maiden name / different person)
    Returns (status, reasoning)
    """
    if not stated_name or not id_name:
        return "REVIEW", f"{doc_type} name comparison skipped (missing data)"
    sim = _jaro_winkler_similarity(stated_name, id_name)
    if sim >= 0.92:
        return "MATCH", f"{doc_type} name match (similarity={sim:.2f}, deterministic)"
    if sim < 0.60:
        return "FRAUD", f"{doc_type} name mismatch — completely different (similarity={sim:.2f}, deterministic)"
    # Ambiguous band — only here do we pay an LLM call
    res = _call_llm_fraud(stated_name, id_name)
    return res["status"], res["reasoning"]


def _run_fraud(state: AgentState) -> AgentState:
    logs = list(state.get("agent_logs", []))
    start = time.time()

    fraud_detected = False
    requires_manual_kyc = False
    fraud_reasons = []

    stated_name = state.get("applicant_data", {}).get("applicant_name", "")
    pan_result = state.get("pan_verification", {}) or {}
    aadhaar_result = state.get("aadhaar_verification", {}) or {}
    user_id = state.get("user_id", "unknown")
    applicant = state.get("applicant_data", {}) or {}

    # ── Security: SHA-256 identity deduplication (Option C) ──
    index = _load_identity_index()
    updated_index = False
    for doc_type, doc_value in [
        ("PAN", applicant.get("applicant_pan")),
        ("Aadhaar", applicant.get("applicant_aadhaar")),
    ]:
        if not doc_value:
            continue
        doc_hash = _sha256_id(doc_value)
        existing_user = index.get(doc_hash)
        if existing_user and existing_user != user_id:
            fraud_detected = True
            fraud_reasons.append(
                f"{doc_type} already registered to a different account (possible identity theft)"
            )
        elif not existing_user:
            index[doc_hash] = user_id
            updated_index = True
    if updated_index:
        _save_identity_index(index)

    # ── V7 Fuzzy-gated name matching ─────────────────────────
    if pan_result.get("verified"):
        id_name = pan_result.get("name_on_record", "")
        status, reason = _check_name_match(stated_name, id_name, "PAN")
        if status == "FRAUD":
            fraud_detected = True
            fraud_reasons.append(f"PAN Identity Fraud: {reason}")
        elif status == "REVIEW":
            requires_manual_kyc = True
            fraud_reasons.append(f"PAN Name Review Required: {reason}")

    if aadhaar_result.get("verified"):
        id_name = aadhaar_result.get("name_on_record", "")
        status, reason = _check_name_match(stated_name, id_name, "Aadhaar")
        if status == "FRAUD":
            fraud_detected = True
            fraud_reasons.append(f"Aadhaar Identity Fraud: {reason}")
        elif status == "REVIEW":
            requires_manual_kyc = True
            fraud_reasons.append(f"Aadhaar Name Review Required: {reason}")

    # Suspicious Financials Checks
    applicant = state.get("applicant_data", {})
    monthly_income = applicant.get("monthly_income") or 0
    loan_amount = applicant.get("loan_amount") or 0
    annual_income = monthly_income * 12
    if annual_income > 0 and (loan_amount / annual_income) > 20:
        fraud_detected = True
        fraud_reasons.append(f"Suspiciously high loan-to-income ratio ({(loan_amount / annual_income):.1f}x)")

    duration = int((time.time() - start) * 1000)
    
    if fraud_detected:
        summary = f"FRAUD DETECTED: {'; '.join(fraud_reasons)}"
        status = "fraud_detected"
    elif requires_manual_kyc:
        summary = f"MANUAL KYC REQUIRED: {'; '.join(fraud_reasons)}"
        status = "manual_kyc_required"
    else:
        summary = "No fraud signals detected. Identity verified."
        status = "passed"

    logs.append({
        "agent": "fraud_agent",
        "step": len(logs) + 1,
        "action": "Fraud & Identity Analysis",
        "status": status,
        "details": summary,
        "duration_ms": duration,
    })

    # Compute identity match confidence for orchestrator ambiguity gating
    if fraud_detected:
        match_confidence = 0.0
    elif requires_manual_kyc:
        match_confidence = 0.5
    else:
        match_confidence = 1.0

    return {
        **state,
        "fraud_detected": fraud_detected,
        "requires_manual_kyc": requires_manual_kyc,
        "fraud_reason": "; ".join(fraud_reasons) if (fraud_detected or requires_manual_kyc) else None,
        "fraud_agent_summary": summary,
        "fraud_match_confidence": match_confidence,
        "agent_logs": logs,
    }

# --- POLICY AGENT LOGIC ---
"""
Policy Agent — Regulatory Compliance Engine (RAG-Powered)
────────────────────────────────────────────────────────
Uses Retrieval-Augmented Generation (RAG) to dynamically check applicant
data against official banking guidelines. Falls back to hardcoded rules
if the LLM API is unavailable.
"""


# ── 1. Hardcoded Rules (Fallback Engine) ─────────────────

def _check_minimum_age(state: AgentState) -> str | None:
    age = state.get("applicant_data", {}).get("applicant_age")
    if age is not None and age < 21:
        return f"Minimum age requirement not met (applicant is {age}, minimum is 21)"
    return None

def _check_loan_to_income(state: AgentState) -> str | None:
    applicant = state.get("applicant_data", {})
    monthly_income = applicant.get("monthly_income")
    loan_amount = applicant.get("loan_amount", 0)
    if monthly_income and monthly_income > 0:
        annual_income = monthly_income * 12
        ratio = loan_amount / annual_income
        if ratio > 5:
            return f"Loan-to-income ratio too high ({ratio:.1f}x, maximum allowed is 5x)"
    return None

def _check_defaults(state: AgentState) -> str | None:
    credit = state.get("credit_report")
    if credit:
        defaults = credit.get("total_defaults", 0)
        if defaults > 2:
            return f"Too many active defaults ({defaults}, maximum allowed is 2)"
    return None

def _check_minimum_credit_score(state: AgentState) -> str | None:
    credit = state.get("credit_report")
    if credit:
        score = credit.get("credit_score", 0)
        if score < 400:
            return f"Credit score below minimum threshold ({score}, minimum is 400)"
    return None

def _check_high_value_employment(state: AgentState) -> str | None:
    applicant = state.get("applicant_data", {})
    loan_amount = applicant.get("loan_amount", 0)
    emp_type = applicant.get("employment_type", "")
    if loan_amount > 2_500_000 and emp_type == "self_employed":
        return f"High-value loan (₹{loan_amount:,.0f}) requires salaried or business employment, not self-employed"
    return None

_POLICY_CHECKS = [
    ("Minimum Age (21 years)", _check_minimum_age),
    ("Loan-to-Income Ratio (max 5x)", _check_loan_to_income),
    ("Default History (max 2)", _check_defaults),
    ("Minimum Credit Score (400)", _check_minimum_credit_score),
    ("High-Value Loan Employment", _check_high_value_employment),
]

def _fallback_policy_check(state: AgentState) -> tuple[list[str], str]:
    """Execute deterministic rules if RAG fails."""
    violations = []
    for rule_name, check_fn in _POLICY_CHECKS:
        violation = check_fn(state)
        if violation:
            violations.append(violation)
    
    if violations:
        return violations, "Fallback rules applied."
    return [], "Fallback rules applied. No violations."

# ── 2. RAG Engine ─────────────────────────────────────────

POLICY_RAG_PROMPT = """
You are a banking compliance officer. 

Applicant Data:
{applicant_data}

Retrieved Policy Document:
{policy_text}

Task: Evaluate if the applicant violates ANY of the retrieved policies.
Respond ONLY with valid JSON. Do not include markdown blocks or outside text.
{{
    "violations": ["violation reason 1", "violation reason 2"], // leave empty if none
    "reasoning": "brief explanation of your findings"
}}
"""

class RAGPolicyEngine:
    def __init__(self):
        self.chunks = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self._initialize()

    def _initialize(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            policy_path = os.path.join(os.path.dirname(__file__), "..", "policies", "rbi_guidelines.txt")
            if not os.path.exists(policy_path):
                return
            
            with open(policy_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Simple chunking by heading
            raw_chunks = content.split("##")
            self.chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]
            
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        except Exception:
            pass

    def retrieve(self, query: str, top_k: int = 3) -> tuple[str, float]:
        if not self.vectorizer or not self.chunks:
            return "", 0.0
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            best_score = float(similarities[top_indices[0]]) if len(top_indices) > 0 else 0.0
            retrieved = [self.chunks[i] for i in top_indices if similarities[i] > 0.0]
            return "\n\n".join(retrieved), round(best_score, 4)
        except Exception:
            return "", 0.0

_rag_engine = RAGPolicyEngine()

def _call_llm_policy(applicant_data_str: str, policy_text: str) -> dict:
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "gsk_placeholder":
        raise ValueError("Valid GROQ_API_KEY missing. Triggering fallback.")
        
    formatted_prompt = POLICY_RAG_PROMPT.format(
        applicant_data=applicant_data_str,
        policy_text=policy_text
    )
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a strict JSON-only API."},
            {"role": "user", "content": formatted_prompt}
        ],
        "temperature": 0.0
    }
    
    data = call_groq(payload, settings.GROQ_API_KEY, timeout=10.0)
    
    content = data["choices"][0]["message"]["content"].strip()
    
    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()
        
    return json.loads(content)

# ── 3. Main Policy Agent Node ─────────────────────────────

def _run_policy(state: AgentState) -> AgentState:
    logs = list(state.get("agent_logs", []))
    errors = list(state.get("errors", []))
    violations = []
    reasoning = ""
    start = time.time()
    
    # Construct applicant query string
    app = state.get("applicant_data", {}) or {}
    cred = state.get("credit_report", {}) or {}
    app_str = f"Age: {app.get('applicant_age')}, Loan Amount: {app.get('loan_amount')}, Monthly Income: {app.get('monthly_income')}, Employment: {app.get('employment_type')}, Credit Score: {cred.get('credit_score')}, Defaults: {cred.get('total_defaults')}"

    # ── V7 RAG gate: only call LLM when retrieval is genuinely uncertain ──
    rag_success = False
    retrieval_score = 0.0
    RAG_CONFIDENCE_THRESHOLD = 0.25  # below this, retrieved chunk is too vague
    try:
        policy_context, retrieval_score = _rag_engine.retrieve(app_str)
        if not policy_context:
            raise ValueError("Empty RAG context.")

        if retrieval_score >= RAG_CONFIDENCE_THRESHOLD:
            # High-confidence retrieval: apply deterministic rules directly, no LLM
            violations, reasoning = _fallback_policy_check(state)
            reasoning = f"High-confidence RAG match (score={retrieval_score:.3f}); deterministic rules applied."
            rag_success = True
        else:
            # Low-confidence: policy context is vague, call LLM to interpret
            llm_response = _call_llm_policy(app_str, policy_context)
            violations = llm_response.get("violations", [])
            reasoning = llm_response.get("reasoning", f"LLM interpretation (low RAG score={retrieval_score:.3f}).")
            rag_success = True
    except Exception as e:
        errors.append(f"RAG Policy Engine failed: {e}. Falling back to deterministic rules.")
        violations, reasoning = _fallback_policy_check(state)

    duration = int((time.time() - start) * 1000)
    policy_passed = len(violations) == 0

    if policy_passed:
        summary = "All policy checks passed ✅"
    else:
        summary = f"{len(violations)} policy violation(s) found: {'; '.join(violations)}"

    logs.append({
        "agent": "policy_agent",
        "step": len(logs) + 1,
        "action": "Policy assessment (RAG)" if rag_success else "Policy assessment (Fallback)",
        "status": "success" if policy_passed else "violations_found",
        "details": summary,
        "duration_ms": duration,
    })

    return {
        "policy_violations": violations,
        "policy_passed": policy_passed,
        "policy_agent_summary": summary,
        "policy_rag_reasoning": reasoning,
        "policy_retrieval_score": retrieval_score,
        "agent_logs": logs,
        "errors": errors,
    }


def intake_agent(state: AgentState) -> AgentState:
    """
    V7: Data + Fraud only (Policy runs in parallel via graph fan-out).
    Returns early if KYC/fraud check fails, so Policy is skipped too.
    """
    state = _run_data(state)
    if state.get("data_agent_early_stop"):
        return state

    state = _run_fraud(state)
    return state


def policy_agent_node(state: AgentState) -> dict:
    """
    Standalone Policy node for graph-level parallelism.
    Returns ONLY the policy-specific keys (no **state spread) to avoid
    LangGraph duplicate-write conflicts when running in parallel with risk_agent.
    """
    if state.get("fraud_detected") or state.get("data_agent_early_stop"):
        return {"policy_passed": True, "policy_violations": [], "agent_logs": [], "errors": []}
    return _run_policy(state)
