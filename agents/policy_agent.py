"""
Policy Agent — Regulatory Compliance Engine (RAG-Powered)
────────────────────────────────────────────────────────
Uses Retrieval-Augmented Generation (RAG) to dynamically check applicant
data against official banking guidelines. Falls back to hardcoded rules
if the LLM API is unavailable.
"""

from __future__ import annotations
import time
import os
import json
import httpx
from agents.state import AgentState
from config import settings

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

    def retrieve(self, query: str, top_k: int = 3) -> str:
        if not self.vectorizer or not self.chunks:
            return ""
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            retrieved = [self.chunks[i] for i in top_indices if similarities[i] > 0.0]
            return "\n\n".join(retrieved)
        except Exception:
            return ""

_rag_engine = RAGPolicyEngine()

def _call_llm_policy(applicant_data_str: str, policy_text: str) -> dict:
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "gsk_placeholder":
        raise ValueError("Valid GROQ_API_KEY missing. Triggering fallback.")
        
    formatted_prompt = POLICY_RAG_PROMPT.format(
        applicant_data=applicant_data_str,
        policy_text=policy_text
    )
    
    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a strict JSON-only API."},
            {"role": "user", "content": formatted_prompt}
        ],
        "temperature": 0.0
    }
    
    with httpx.Client(timeout=10.0) as client:
        response = client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        return json.loads(content)

# ── 3. Main Policy Agent Node ─────────────────────────────

def policy_agent(state: AgentState) -> AgentState:
    logs = list(state.get("agent_logs", []))
    errors = list(state.get("errors", []))
    violations = []
    reasoning = ""
    start = time.time()
    
    # Construct applicant query string
    app = state.get("applicant_data", {}) or {}
    cred = state.get("credit_report", {}) or {}
    app_str = f"Age: {app.get('applicant_age')}, Loan Amount: {app.get('loan_amount')}, Monthly Income: {app.get('monthly_income')}, Employment: {app.get('employment_type')}, Credit Score: {cred.get('credit_score')}, Defaults: {cred.get('total_defaults')}"

    # Try RAG
    rag_success = False
    try:
        policy_context = _rag_engine.retrieve(app_str)
        if not policy_context:
            raise ValueError("Empty RAG context.")
            
        llm_response = _call_llm_policy(app_str, policy_context)
        violations = llm_response.get("violations", [])
        reasoning = llm_response.get("reasoning", "RAG executed successfully.")
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
        **state,
        "policy_violations": violations,
        "policy_passed": policy_passed,
        "policy_agent_summary": summary,
        "policy_rag_reasoning": reasoning,
        "agent_logs": logs,
        "errors": errors,
    }
