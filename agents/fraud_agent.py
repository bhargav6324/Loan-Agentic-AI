"""
Fraud Agent — Anomaly & Identity Verification (V3)
──────────────────────────────────────────────────
Detects identity mismatch (PAN/Aadhaar) and suspicious combinations.
V3 Upgrade: Uses an LLM to reason about identity mismatches instead of
hard rejecting. E.g., it can recognize maiden names or minor typos and
escalate for manual KYC instead of outright rejecting as fraud.
"""

import time
import json
import httpx
from agents.state import AgentState
from agents.llm_client import call_groq
from config import settings

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


def fraud_agent(state: AgentState) -> AgentState:
    logs = list(state.get("agent_logs", []))
    start = time.time()
    
    fraud_detected = False
    requires_manual_kyc = False
    fraud_reasons = []

    stated_name = state.get("applicant_data", {}).get("applicant_name", "")
    pan_result = state.get("pan_verification", {})
    aadhaar_result = state.get("aadhaar_verification", {})
    
    # Check PAN Match via LLM
    if pan_result and pan_result.get("verified"):
        id_name = pan_result.get("name_on_record", "")
        if stated_name and id_name:
            eval_res = _call_llm_fraud(stated_name, id_name)
            if eval_res["status"] == "FRAUD":
                fraud_detected = True
                fraud_reasons.append(f"PAN Identity Fraud: {eval_res['reasoning']}")
            elif eval_res["status"] == "REVIEW":
                requires_manual_kyc = True
                fraud_reasons.append(f"PAN Name Review Required: {eval_res['reasoning']}")

    # Check Aadhaar Match via LLM
    if aadhaar_result and aadhaar_result.get("verified"):
        id_name = aadhaar_result.get("name_on_record", "")
        if stated_name and id_name:
            eval_res = _call_llm_fraud(stated_name, id_name)
            if eval_res["status"] == "FRAUD":
                fraud_detected = True
                fraud_reasons.append(f"Aadhaar Identity Fraud: {eval_res['reasoning']}")
            elif eval_res["status"] == "REVIEW":
                requires_manual_kyc = True
                fraud_reasons.append(f"Aadhaar Name Review Required: {eval_res['reasoning']}")

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
