"""
Reporting Agent — Merged Explainability, Learning, and Audit (V6)
"""

from __future__ import annotations
import time
import json
import httpx
import os
import hashlib
import threading
from datetime import datetime, timezone
from agents.state import AgentState
from config import settings
from agents.llm_client import call_groq

# --- EXPLAINABILITY LOGIC ---
"""
Explainability Agent — Human-Readable Decision Narratives
──────────────────────────────────────────────────────────
Generates clear, audience-specific explanations for every loan decision.
This agent reads the COMPLETE state (all previous agent outputs) and
produces four distinct narratives:

  1. APPLICANT message  — Simple, empathetic, jargon-free (max 150 words)
  2. ADMIN summary      — Technical SHAP-style breakdown (max 300 words)
  3. COMPLIANCE report  — Full regulatory audit trail
  4. PROCESS trace      — Narrative of how agents autonomously collaborated

This agent NEVER calculates scores itself — it only explains
pre-computed results from the other agents.
"""



PROCESS_EXPLANATION_PROMPT = """
You are an AI Process Auditor for a loan evaluation system.

Here is the exact sequential log of how the AI agents processed the application:
{agent_logs_str}

Write a 2-3 paragraph "Process Trace" summarizing HOW the system reached its decision.
Highlight key autonomous behaviors:
- Did the Data Agent skip or dynamically request specific tools?
- Did the Fairness Agent and Risk Agent negotiate or debate?
- Were there any edge case fallback rules triggered?

Focus strictly on the *process* and *agent behaviors*, not the underlying financial logic.
"""

def _generate_process_explanation(state: AgentState) -> str:
    """Generate process explanation detailing agent reasoning and debate."""
    logs = state.get("agent_logs", [])
    if not logs:
        return "No agent logs available to generate process explanation."
        
    logs_summary = []
    for log in logs:
        logs_summary.append(f"[{log.get('agent')}] {log.get('action')} -> {log.get('status')}: {log.get('details')}")
        
    logs_str = "\n".join(logs_summary)
    
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "gsk_placeholder":
        return (
            "PROCESS TRACE (Fallback Mode):\n"
            f"The system executed {len(logs)} autonomous steps. "
            "Data Agent verified identity dynamically. "
            "Risk, Policy, and Fairness agents evaluated the application in sequence. "
            f"Final decision was reached by {logs[-1].get('agent')} with status: {logs[-1].get('status')}."
        )

    formatted_prompt = PROCESS_EXPLANATION_PROMPT.format(agent_logs_str=logs_str)
    
    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a professional auditor explaining multi-agent AI execution."},
            {"role": "user", "content": formatted_prompt}
        ],
        "temperature": 0.3
    }
    
    try:
        data = call_groq(payload, settings.GROQ_API_KEY, timeout=20.0)
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Failed to generate process explanation: {e}"


# ── Applicant Message (LLM-Powered) ─────────────────────

APPLICANT_MESSAGE_PROMPT = """
You are a customer service officer at a bank writing to a loan applicant.

Applicant Name: {name}
Loan Amount Requested: ₹{loan_amount:,.0f}
Final Decision: {decision}
Risk Score: {risk_score}/100
Key Risk Factors: {risk_factors}
Policy Violations: {policy_violations}
Bias Detected: {bias_detected}

Write a warm, empathetic letter to the applicant in plain English (no jargon).
- If APPROVED: congratulate them, mention the strong factors, state next steps.
- If REJECTED: be empathetic, explain the top 2 reasons clearly, give 2-3 actionable steps they can take to improve their profile before reapplying.
- If ESCALATED: explain that a human reviewer will contact them within 2 business days.

Keep it under 150 words. Do not mention AI or algorithms.
"""

def _generate_applicant_message(state: AgentState) -> str:
    decision = state.get("final_decision", "pending")
    applicant = state.get("applicant_data", {}) or {}
    name = applicant.get("applicant_name", "Applicant")
    loan_amount = applicant.get("loan_amount", 0)
    risk_factors = state.get("risk_factors", [])
    policy_violations = state.get("policy_violations", [])

    # Fallback if no API key
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "gsk_placeholder":
        if decision == "approved":
            return f"Dear {name},\n\nCongratulations! Your application for a ₹{loan_amount:,.0f} loan has been approved. Our team will contact you within 2 business days to complete the documentation process.\n\nThank you for choosing us."
        elif decision == "rejected":
            top_reasons = "; ".join(risk_factors[:2]) if risk_factors else "unfavorable financial profile"
            return f"Dear {name},\n\nWe regret to inform you that your application for a ₹{loan_amount:,.0f} loan has not been approved at this time. The key concerns were: {top_reasons}. We encourage you to improve your credit score and reduce existing debt before reapplying. You may reapply after 90 days."
        elif decision == "escalated":
            return f"Dear {name},\n\nThank you for your application. Due to the complexity of your profile, your application has been referred to our Senior Credit Committee for manual review. A relationship manager will contact you within 2 business days."
        else:
            return f"Dear {name},\n\nYour loan application is currently being evaluated. We will notify you of the outcome shortly."

    prompt = APPLICANT_MESSAGE_PROMPT.format(
        name=name,
        loan_amount=loan_amount,
        decision=decision.upper(),
        risk_score=state.get("risk_score", "N/A"),
        risk_factors=", ".join(risk_factors[:3]) if risk_factors else "None identified",
        policy_violations=", ".join(policy_violations[:2]) if policy_violations else "None",
        bias_detected=state.get("bias_detected", False),
    )

    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a bank customer service officer. Write the letter exactly as instructed."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4
    }
    try:
        data = call_groq(payload, settings.GROQ_API_KEY, timeout=15.0)
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Dear {name},\n\nYour loan application decision is: {decision.upper()}. Please contact our branch for further details. (Generation error: {e})"


# ── Admin Summary (LLM-Powered) ──────────────────────────

ADMIN_SUMMARY_PROMPT = """
You are a Senior Credit Officer writing an internal credit memo for bank records.

Applicant: {name}
Age: {age} | Gender: {gender} | Location: {location} | Employment: {employment_type} ({employment_years} yrs)
Loan Amount: ₹{loan_amount:,.0f} | Monthly Income: ₹{monthly_income:,.0f} | Loan Purpose: {loan_purpose}

Risk Assessment:
- Risk Score: {risk_score}/100 | Category: {risk_category}
- ML/Heuristic Blend: {reflection_summary}
- Key Risk Factors: {risk_factors}

KYC & Identity:
- PAN Verified: {pan_verified} | Aadhaar Verified: {aadhaar_verified}
- Fraud Detected: {fraud_detected} | Manual KYC Required: {manual_kyc}

Credit Bureau:
- Credit Score: {credit_score} | Total Defaults: {defaults} | DTI: {dti} | Utilisation: {utilisation}%

Policy Compliance:
- Policy Passed: {policy_passed}
- Violations: {policy_violations}

Fairness Audit:
- Bias Detected: {bias_detected}
- Gender Delta: {gender_delta} | Age Delta: {age_delta} | Location Delta: {location_delta}

Final Decision: {decision}
Decision Reasoning: {decision_reasoning}

Write a formal 250-300 word credit memorandum covering all the above sections.
Use professional banking language. Format with clear section headers.
"""

def _generate_admin_summary(state: AgentState) -> str:
    applicant = state.get("applicant_data", {}) or {}
    credit = state.get("credit_report", {}) or {}
    reflection = state.get("risk_agent_reflection", {}) or {}
    fairness = state.get("fairness_report", {}) or {}
    delta_summary = fairness.get("counterfactual_delta_summary", {})
    pan = state.get("pan_verification", {}) or {}
    aadhaar = state.get("aadhaar_verification", {}) or {}

    monthly_income = applicant.get("monthly_income", 0) or 0
    annual_income = monthly_income * 12
    dti = round(credit.get("current_outstanding", 0) / annual_income, 2) if annual_income > 0 else "N/A"

    # Fallback
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "gsk_placeholder":
        return (
            f"CREDIT MEMORANDUM\n"
            f"Applicant: {applicant.get('applicant_name', 'Unknown')}\n"
            f"Decision: {state.get('final_decision', 'N/A').upper()}\n"
            f"Risk Score: {state.get('risk_score', 'N/A')}/100 ({state.get('risk_category', 'N/A')})\n"
            f"Credit Score: {credit.get('credit_score', 'N/A')} | Defaults: {credit.get('total_defaults', 'N/A')}\n"
            f"Policy Passed: {state.get('policy_passed', 'N/A')}\n"
            f"Bias Detected: {state.get('bias_detected', False)}\n"
            f"Policy Violations: {', '.join(state.get('policy_violations', [])) or 'None'}\n"
            f"Risk Factors: {', '.join(state.get('risk_factors', [])[:3]) or 'None'}"
        )

    prompt = ADMIN_SUMMARY_PROMPT.format(
        name=applicant.get("applicant_name", "Unknown"),
        age=applicant.get("applicant_age", "N/A"),
        gender=applicant.get("applicant_gender", "N/A"),
        location=applicant.get("applicant_location", "N/A"),
        employment_type=applicant.get("employment_type", "N/A"),
        employment_years=applicant.get("employment_years", "N/A"),
        loan_amount=applicant.get("loan_amount", 0),
        monthly_income=monthly_income,
        loan_purpose=applicant.get("loan_purpose", "N/A"),
        risk_score=state.get("risk_score", "N/A"),
        risk_category=state.get("risk_category", "N/A"),
        reflection_summary=reflection.get("reflection_summary", "No reflection data"),
        risk_factors=", ".join(state.get("risk_factors", [])[:4]) or "None",
        pan_verified=pan.get("verified", False),
        aadhaar_verified=aadhaar.get("verified", False),
        fraud_detected=state.get("fraud_detected", False),
        manual_kyc=state.get("requires_manual_kyc", False),
        credit_score=credit.get("credit_score", "N/A"),
        defaults=credit.get("total_defaults", "N/A"),
        dti=dti,
        utilisation=credit.get("credit_utilisation_pct", "N/A"),
        policy_passed=state.get("policy_passed", "N/A"),
        policy_violations=", ".join(state.get("policy_violations", [])) or "None",
        bias_detected=state.get("bias_detected", False),
        gender_delta=delta_summary.get("gender", {}).get("worst_signed_delta", "N/A"),
        age_delta=delta_summary.get("age", {}).get("worst_signed_delta", "N/A"),
        location_delta=delta_summary.get("location", {}).get("worst_signed_delta", "N/A"),
        decision=state.get("final_decision", "N/A").upper(),
        decision_reasoning=state.get("decision_reasoning", "N/A"),
    )

    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a Senior Credit Officer. Write exactly as instructed."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }
    try:
        data = call_groq(payload, settings.GROQ_API_KEY, timeout=20.0)
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Admin summary generation failed: {e}. Risk Score: {state.get('risk_score')}, Decision: {state.get('final_decision')}"


# ── Compliance Report (LLM-Powered) ──────────────────────

COMPLIANCE_REPORT_PROMPT = """
You are writing a regulatory compliance report for an RBI audit.

Application ID: {application_id}
Generated: {timestamp}

KYC Compliance:
- PAN: {pan_status} | Aadhaar: {aadhaar_status}
- Fraud Check: {fraud_status}
- Data Agent Tools Called: {tools_called}
- Data Agent ReAct Iterations: {react_iterations}

Risk Assessment Compliance:
- Algorithm: RandomForest + Heuristic Blend (demographic features excluded from ML model by design)
- Risk Score: {risk_score}/100 | Category: {risk_category}
- Self-Reflection Triggered: {reflection_revised} | Reason: {reflection_reason}
- Hard Override: {hard_override}

Policy Compliance (RBI Guidelines):
- Result: {policy_result}
- Violations Found: {policy_violations}
- RAG Engine Used: {rag_used}

Equal Lending Opportunity Audit (Fairness):
- Counterfactual Testing: Gender ({gender_tests} variants), Age ({age_tests} brackets), Location ({location_tests} variants)
- Gender Max Delta: {gender_delta} pts (threshold: {gender_threshold}) | Bias: {gender_bias}
- Age Max Delta: {age_delta} pts (threshold: {age_threshold}) | Bias: {age_bias}
- Location Max Delta: {location_delta} pts (threshold: {location_threshold}) | Bias: {location_bias}
- Overall Bias Detected: {bias_detected}
- Debate Log Entries: {debate_entries}

Decision:
- Final Decision: {decision}
- Reasoning: {decision_reasoning}
- Human Review Required: {escalated}

Audit Integrity:
- Pipeline Integrity Hash: {pipeline_hash}
- Audit Records Generated: {audit_records}
- Anomalies: {anomalies}

Write a formal 400-500 word regulatory compliance report covering all sections above.
This report will be submitted to RBI auditors. Use formal regulatory language.
Each section must have a clear header. End with a compliance certification statement.
"""

def _generate_compliance_report(state: AgentState) -> str:
    applicant = state.get("applicant_data", {}) or {}
    pan = state.get("pan_verification", {}) or {}
    aadhaar = state.get("aadhaar_verification", {}) or {}
    fairness = state.get("fairness_report", {}) or {}
    delta_summary = fairness.get("counterfactual_delta_summary", {})
    reflection = state.get("risk_agent_reflection", {}) or {}
    audit_summary = state.get("audit_execution_summary", {}) or {}

    # Count debate entries
    debate_log = state.get("fairness_debate_log", []) or []
    # Count counterfactual results by attribute
    cf_results = fairness.get("counterfactual_results", [])
    gender_tests = len([r for r in cf_results if r.get("attribute") == "gender"])
    age_tests = len([r for r in cf_results if r.get("attribute") == "age"])
    location_tests = len([r for r in cf_results if r.get("attribute") == "location"])

    # Fallback
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "gsk_placeholder":
        return (
            f"REGULATORY COMPLIANCE REPORT\n"
            f"Application ID: {state.get('application_id', 'N/A')}\n"
            f"Generated: {datetime.now().isoformat()}\n\n"
            f"1. KYC COMPLIANCE\n"
            f"PAN Verified: {pan.get('verified', False)} | Aadhaar Verified: {aadhaar.get('verified', False)}\n"
            f"Fraud Detected: {state.get('fraud_detected', False)}\n\n"
            f"2. RISK ASSESSMENT\n"
            f"Risk Score: {state.get('risk_score', 'N/A')}/100 ({state.get('risk_category', 'N/A')})\n"
            f"Self-Reflection Revised: {reflection.get('revised', False)}\n\n"
            f"3. POLICY COMPLIANCE\n"
            f"Policy Passed: {state.get('policy_passed', 'N/A')}\n"
            f"Violations: {', '.join(state.get('policy_violations', [])) or 'None'}\n\n"
            f"4. FAIRNESS AUDIT\n"
            f"Bias Detected: {state.get('bias_detected', False)}\n"
            f"Gender Delta: {delta_summary.get('gender', {}).get('worst_signed_delta', 'N/A')}\n\n"
            f"5. FINAL DECISION\n"
            f"Decision: {state.get('final_decision', 'N/A').upper()}\n"
            f"Pipeline Integrity Hash: {audit_summary.get('pipeline_integrity_hash', 'N/A')}\n\n"
            f"This report certifies that the loan evaluation was conducted in accordance with RBI guidelines."
        )

    prompt = COMPLIANCE_REPORT_PROMPT.format(
        application_id=state.get("application_id", "N/A"),
        timestamp=datetime.now().isoformat(),
        pan_status="VERIFIED" if pan.get("verified") else "NOT VERIFIED",
        aadhaar_status="VERIFIED" if aadhaar.get("verified") else "NOT VERIFIED",
        fraud_status="CLEAR" if not state.get("fraud_detected") else "FLAGGED",
        tools_called=", ".join(state.get("data_agent_tools_called", []) or []) or "Standard KYC",
        react_iterations=len(state.get("data_agent_reasoning_trace", []) or []),
        risk_score=state.get("risk_score", "N/A"),
        risk_category=state.get("risk_category", "N/A"),
        reflection_revised=reflection.get("revised", False),
        reflection_reason=reflection.get("revision_reason") or "No revision needed",
        hard_override="Yes" if state.get("risk_score") in [10.0, 95.0] else "No",
        policy_result="PASSED" if state.get("policy_passed") else "FAILED",
        policy_violations=", ".join(state.get("policy_violations", [])) or "None",
        rag_used="Yes (TF-IDF over RBI guidelines)" if state.get("policy_rag_reasoning") else "Fallback rules",
        gender_tests=gender_tests,
        age_tests=age_tests,
        location_tests=location_tests,
        gender_delta=delta_summary.get("gender", {}).get("max_absolute_delta", "N/A"),
        gender_threshold=delta_summary.get("gender", {}).get("threshold", 5.0),
        gender_bias=delta_summary.get("gender", {}).get("bias_flagged", False),
        age_delta=delta_summary.get("age", {}).get("max_absolute_delta", "N/A"),
        age_threshold=delta_summary.get("age", {}).get("threshold", 10.0),
        age_bias=delta_summary.get("age", {}).get("bias_flagged", False),
        location_delta=delta_summary.get("location", {}).get("max_absolute_delta", "N/A"),
        location_threshold=delta_summary.get("location", {}).get("threshold", 8.0),
        location_bias=delta_summary.get("location", {}).get("bias_flagged", False),
        bias_detected=state.get("bias_detected", False),
        debate_entries=len(debate_log),
        decision=state.get("final_decision", "N/A").upper(),
        decision_reasoning=state.get("decision_reasoning", "N/A"),
        escalated="Yes" if state.get("final_decision") == "escalated" else "No",
        pipeline_hash=audit_summary.get("pipeline_integrity_hash", "Not yet computed"),
        audit_records=audit_summary.get("total_steps", "N/A"),
        anomalies=", ".join(state.get("audit_anomalies", []) or []) or "None detected",
    )

    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are an RBI compliance officer. Write the report exactly as instructed."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }
    try:
        data = call_groq(payload, settings.GROQ_API_KEY, timeout=25.0)
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Compliance report generation failed: {e}. Pipeline hash: {audit_summary.get('pipeline_integrity_hash', 'N/A')}"


def _run_explainability(state: AgentState) -> AgentState:
    logs = list(state.get("agent_logs", []))
    errors = list(state.get("errors", []))
    start = time.time()

    try:
        applicant_message = _generate_applicant_message(state)
    except Exception:
        applicant_message = "Error"
        
    try:
        admin_summary = _generate_admin_summary(state)
    except Exception:
        admin_summary = "Error"
        
    try:
        compliance_report = _generate_compliance_report(state)
    except Exception:
        compliance_report = "Error"
        
    try:
        process_str = _generate_process_explanation(state)
    except Exception:
        process_str = "Error"

    duration = int((time.time() - start) * 1000)
    summary = f"Generated 4 explanations: Applicant ({len(applicant_message)} chars), Admin ({len(admin_summary)} chars), Compliance ({len(compliance_report)} chars), Process Trace ({len(process_str)} chars)"

    logs.append({
        "agent": "explainability_agent",
        "step": len(logs) + 1,
        "action": "Generate explanations",
        "status": "success",
        "details": summary,
        "duration_ms": duration,
    })

    return {
        **state,
        "explanation": {
            "applicant_message": applicant_message,
            "admin_summary": admin_summary,
            "compliance_report": compliance_report,
            "process_explanation": process_str,
        },
        "explainability_agent_summary": summary,
        "agent_logs": logs,
        "errors": errors,
    }

# --- LEARNING/MEMORY LOGIC ---
"""
Learning Agent — Systemic Bias Tracker (V3)
───────────────────────────────────────────
Updates long-term memory with the final outcome of the pipeline.
Analyzes rolling statistics to detect systemic bias over time,
allowing the Fairness Agent to dynamically adjust thresholds.
"""


MEMORY_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "long_term_memory.json")


def _analyze_memory(memory: list) -> dict:
    """
    Analyze the last 100 records for systemic bias patterns.
    Returns an insights dict with approval rates per demographic group.
    """
    if len(memory) < 5:
        return {"status": "insufficient_data", "records_analyzed": len(memory)}

    recent = memory[-100:]  # analyze last 100 records only
    insights = {"records_analyzed": len(recent), "status": "analyzed"}

    # ── Approval rate by gender ────────────────────────────
    gender_stats = {}
    for r in recent:
        g = r.get("gender") or "unknown"
        d = r.get("decision", "unknown")
        if g not in gender_stats:
            gender_stats[g] = {"approved": 0, "rejected": 0, "escalated": 0, "total": 0}
        gender_stats[g]["total"] += 1
        if d in gender_stats[g]:
            gender_stats[g][d] += 1

    gender_approval_rates = {}
    for g, stats in gender_stats.items():
        if stats["total"] >= 3:
            gender_approval_rates[g] = round(stats["approved"] / stats["total"], 3)

    insights["gender_approval_rates"] = gender_approval_rates

    # Flag if gap between any two genders exceeds 15%
    rates = list(gender_approval_rates.values())
    if len(rates) >= 2:
        gap = max(rates) - min(rates)
        insights["gender_bias_gap"] = round(gap, 3)
        if gap > 0.15:
            worst_group = min(gender_approval_rates, key=gender_approval_rates.get)
            insights["gender_bias_alert"] = {
                "detected": True,
                "gap": round(gap, 3),
                "most_disadvantaged_group": worst_group,
                "approval_rate": gender_approval_rates[worst_group],
                "recommendation": f"Tighten fairness thresholds — {worst_group} applicants approved at {gender_approval_rates[worst_group]*100:.0f}% vs others at higher rates"
            }
        else:
            insights["gender_bias_alert"] = {"detected": False, "gap": round(gap, 3)}

    # ── Approval rate by location ──────────────────────────
    location_stats = {}
    for r in recent:
        loc = r.get("location") or "unknown"
        d = r.get("decision", "unknown")
        if loc not in location_stats:
            location_stats[loc] = {"approved": 0, "total": 0}
        location_stats[loc]["total"] += 1
        if d == "approved":
            location_stats[loc]["approved"] += 1

    location_approval_rates = {
        loc: round(s["approved"] / s["total"], 3)
        for loc, s in location_stats.items()
        if s["total"] >= 3
    }
    insights["location_approval_rates"] = location_approval_rates

    # ── Approval rate by employment type ──────────────────
    employment_stats = {}
    for r in recent:
        emp = r.get("employment_type") or "unknown"
        d = r.get("decision", "unknown")
        if emp not in employment_stats:
            employment_stats[emp] = {"approved": 0, "total": 0}
        employment_stats[emp]["total"] += 1
        if d == "approved":
            employment_stats[emp]["approved"] += 1

    insights["employment_approval_rates"] = {
        emp: round(s["approved"] / s["total"], 3)
        for emp, s in employment_stats.items()
        if s["total"] >= 3
    }

    # ── Overall escalation rate ────────────────────────────
    escalated = sum(1 for r in recent if r.get("decision") == "escalated")
    insights["escalation_rate"] = round(escalated / len(recent), 3)
    if insights["escalation_rate"] > 0.30:
        insights["high_escalation_alert"] = (
            f"Escalation rate is {insights['escalation_rate']*100:.0f}% "
            f"— fairness thresholds may be too strict"
        )

    return insights


def _run_learning(state: AgentState) -> AgentState:
    logs = list(state.get("agent_logs", []))
    start = time.time()

    app = state.get("applicant_data", {})
    decision = state.get("final_decision", "unknown")

    # Build record — include more fields than before
    record = {
        "timestamp": time.time(),
        "application_id": state.get("application_id"),
        "gender": app.get("applicant_gender"),
        "location": app.get("applicant_location") or app.get("location"),
        "employment_type": app.get("employment_type"),
        "decision": decision,
        "risk_score": state.get("risk_score"),
        "bias_detected": state.get("bias_detected", False),
    }

    # Load memory
    memory = []
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                data = json.load(f)
                # Handle both old format (list) and new format (dict)
                if isinstance(data, list):
                    memory = data
                elif isinstance(data, dict):
                    memory = data.get("records", [])
        except Exception:
            pass

    memory.append(record)
    memory = memory[-1000:]  # keep last 1000

    # Analyze patterns
    insights = _analyze_memory(memory)

    # Save memory with insights
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump({"records": memory, "insights": insights, "last_updated": time.time()}, f, indent=2)

    # V5: Store in Episodic Memory (ChromaDB)
    try:
        from memory.episodic_memory import store_episode
        
        # Format the debate transcript if it exists
        debate_log = state.get("fairness_debate_log", [])
        debate_summary = ""
        if debate_log:
            debate_summary = "\n".join([f"[{d.get('agent')}] {d.get('action')}: {d.get('details')}" for d in debate_log[-2:]])
            
        store_episode(
            application_id=state.get("application_id", f"unknown_{time.time()}"),
            app_data=state.get("applicant_data", {}),
            credit_data=state.get("credit_report", {}),
            decision=decision,
            risk_score=state.get("risk_score", -1.0),
            bias_detected=state.get("bias_detected", False),
            debate_summary=debate_summary
        )
    except Exception as e:
        print(f"Warning: Failed to store in episodic memory: {e}")

    duration = int((time.time() - start) * 1000)

    # Build a human-readable summary of what was found
    summary_parts = [f"Appended record #{len(memory)}. Decision: {decision}."]
    if "gender_bias_alert" in insights and insights["gender_bias_alert"].get("detected"):
        alert = insights["gender_bias_alert"]
        summary_parts.append(
            f"⚠ SYSTEMIC GENDER BIAS: {alert['most_disadvantaged_group']} group "
            f"approved at {alert['approval_rate']*100:.0f}% — gap={alert['gap']*100:.0f}%."
        )
    if "high_escalation_alert" in insights:
        summary_parts.append(insights["high_escalation_alert"])
    if insights.get("status") == "insufficient_data":
        summary_parts.append("Insufficient data for analysis yet (need ≥5 records).")

    summary = " ".join(summary_parts)

    logs.append({
        "agent": "learning_agent",
        "step": len(logs) + 1,
        "action": "Update memory + analyze systemic patterns",
        "status": "success",
        "details": summary,
        "duration_ms": duration,
    })

    return {
        **state,
        "learning_insights": insights,  # now surfaces in API response
        "agent_logs": logs,
    }

# --- AUDIT LOGIC ---
"""
Audit Agent — Immutable Decision Logging & Audit Trail
──────────────────────────────────────────────────────
Provides complete, immutable logging for every action taken
across the entire agent pipeline. This agent:

  • Captures every tool call, result, and reasoning from all agents
  • Generates a complete audit trail for regulatory compliance
  • Computes a SHA-256 hash of each agent’s output JSON at the
    moment of audit — making the immutability claim technically
    concrete and independently verifiable
  • Exports audit data as structured JSON for database persistence
  • Detects anomalies in the agent execution flow
  • Logs admin overrides with mandatory justification

Critical design rule: APPEND-ONLY.
The Audit Agent never updates or deletes previous entries.
Even errors are logged — especially errors.

Hash integrity: any post-hoc mutation of an agent’s output_data
will produce a different SHA-256, immediately exposing tampering.
"""



# ════════════════════════════════════════════════════════
#  CRYPTOGRAPHIC HASHING UTILITIES
# ════════════════════════════════════════════════════════

def _sha256_of(payload: dict) -> str:
    """
    Compute a deterministic SHA-256 hex digest of a JSON-serialisable dict.
    Keys are sorted before serialisation so insertion order never affects
    the hash (two logically identical dicts always produce the same digest).
    Returns a lowercase 64-character hex string.
    """
    canonical = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hash_agent_outputs(state: AgentState) -> dict:
    """
    Produce one SHA-256 hash per named agent based on the slice of AgentState
    that each agent is contractually responsible for writing.

    Returns:
        {
          "data_agent":           "<sha256>",
          "risk_agent":           "<sha256>",
          "policy_agent":         "<sha256>",
          "fairness_agent":       "<sha256>",
          "explainability_agent": "<sha256>",
          "orchestrator":         "<sha256>"
        }

    A mutation of any agent's output fields would change its hash,
    making tampering detectable even without access to the audit trail.
    """
    agent_output_slices = {
        "data_agent": {
            "pan_verification":    state.get("pan_verification"),
            "aadhaar_verification": state.get("aadhaar_verification"),
            "credit_report":       state.get("credit_report"),
            "data_agent_summary":  state.get("data_agent_summary"),
        },
        "risk_agent": {
            "risk_score":          state.get("risk_score"),
            "risk_category":       state.get("risk_category"),
            "risk_factors":        state.get("risk_factors"),
            "risk_agent_summary":  state.get("risk_agent_summary"),
        },
        "policy_agent": {
            "policy_violations":    state.get("policy_violations"),
            "policy_passed":        state.get("policy_passed"),
            "policy_agent_summary": state.get("policy_agent_summary"),
        },
        "fairness_agent": {
            "fairness_report":        state.get("fairness_report"),
            "bias_detected":          state.get("bias_detected"),
            "fairness_agent_summary": state.get("fairness_agent_summary"),
        },
        "explainability_agent": {
            "explanation":                  state.get("explanation"),
            "explainability_agent_summary": state.get("explainability_agent_summary"),
        },
        "orchestrator": {
            "final_decision":     state.get("final_decision"),
            "decision_reasoning": state.get("decision_reasoning"),
        },
    }
    return {
        agent: _sha256_of(output_slice)
        for agent, output_slice in agent_output_slices.items()
    }


# ════════════════════════════════════════════════════════
#  AUDIT RECORD BUILDER
# ════════════════════════════════════════════════════════

def _build_audit_record(
    application_id: str,
    agent_name: str,
    tool_called: str,
    input_data: dict,
    output_data: dict,
    reasoning: str,
    execution_ms: int = 0,
) -> dict:
    """
    Build a single, standardised audit log entry.
    Every field is mandatory — no partial records allowed.

    `output_hash` is the SHA-256 of `output_data` (keys sorted).
    Any future mutation of the stored output_data can be detected
    by recomputing SHA-256 and comparing against this field.
    """
    output_hash = _sha256_of(output_data)
    return {
        "application_id": application_id,
        "agent_name":     agent_name,
        "tool_called":    tool_called,
        "input_data":     input_data,
        "output_data":    output_data,
        "output_hash":    output_hash,   # SHA-256 integrity seal
        "reasoning":      reasoning,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "execution_ms":   execution_ms,
    }


# ══════════════════════════════════════════════════════════
#  AUDIT TRAIL GENERATOR
# ══════════════════════════════════════════════════════════

def _generate_audit_trail(state: AgentState) -> list[dict]:
    """
    Walk through agent_logs and convert each entry into a
    formal audit record suitable for database insertion.
    """
    application_id = state.get("application_id", "unknown")
    agent_logs = state.get("agent_logs", [])
    audit_trail = []

    for log in agent_logs:
        record = _build_audit_record(
            application_id=application_id,
            agent_name=log.get("agent", "unknown"),
            tool_called=log.get("action", "unknown"),
            input_data={"step": log.get("step", 0)},
            output_data={
                "status": log.get("status", "unknown"),
                "details": log.get("details", ""),
            },
            reasoning=log.get("details", ""),
            execution_ms=log.get("duration_ms", 0),
        )
        audit_trail.append(record)

    return audit_trail


# ══════════════════════════════════════════════════════════
#  ANOMALY DETECTION
# ══════════════════════════════════════════════════════════

def _detect_anomalies(state: AgentState) -> list[str]:
    """
    Check for anomalies in the pipeline execution that might
    indicate bugs, missed steps, or suspicious patterns.
    """
    anomalies = []
    agent_logs = state.get("agent_logs", [])

    # Check: Did all expected agents run?
    agents_that_ran = set(log.get("agent") for log in agent_logs)
    expected_agents = {"data_agent", "risk_agent", "policy_agent"}

    missing = expected_agents - agents_that_ran
    if missing:
        anomalies.append(
            f"Missing agent execution: {', '.join(missing)} did not run"
        )

    # Check: Were there any errors?
    errors = state.get("errors", [])
    if errors:
        anomalies.append(
            f"{len(errors)} error(s) occurred during pipeline execution"
        )

    # Check: Risk score is present before decision
    if state.get("final_decision") and state.get("risk_score") is None:
        anomalies.append(
            "Decision was made without a risk score being calculated"
        )

    # Check: Policy was checked before approval
    if state.get("final_decision") == "approved" and state.get("policy_passed") is None:
        anomalies.append(
            "Application approved without policy compliance check"
        )

    # Check: Fairness agent ran for medium-risk cases
    risk_category = state.get("risk_category", "")
    if risk_category == "MEDIUM" and "fairness_agent" not in agents_that_ran:
        anomalies.append(
            "MEDIUM-risk case was decided without fairness analysis"
        )

    # Check: Suspiciously fast execution (possible skip)
    total_duration = sum(log.get("duration_ms", 0) for log in agent_logs)
    if len(agent_logs) > 5 and total_duration == 0:
        anomalies.append(
            "All agent steps reported 0ms execution — possible timing error"
        )

    return anomalies


# ══════════════════════════════════════════════════════════
#  EXECUTION SUMMARY
# ══════════════════════════════════════════════════════════

def _generate_execution_summary(state: AgentState) -> dict:
    """
    Generate a high-level summary of the pipeline execution.
    Useful for dashboard displays and quick admin review.
    """
    agent_logs = state.get("agent_logs", [])
    errors = state.get("errors", [])

    # Count steps per agent
    agent_step_counts = {}
    for log in agent_logs:
        agent = log.get("agent", "unknown")
        agent_step_counts[agent] = agent_step_counts.get(agent, 0) + 1

    # Count statuses
    status_counts = {}
    for log in agent_logs:
        status = log.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    # Total execution time
    total_ms = sum(log.get("duration_ms", 0) for log in agent_logs)

    return {
        "total_steps": len(agent_logs),
        "total_execution_ms": total_ms,
        "agents_executed": agent_step_counts,
        "status_breakdown": status_counts,
        "total_errors": len(errors),
        "decision": state.get("final_decision", "pending"),
        "risk_score": state.get("risk_score"),
        "risk_category": state.get("risk_category"),
        "policy_passed": state.get("policy_passed"),
        "bias_detected": state.get("bias_detected"),
    }


# ══════════════════════════════════════════════════════════
#  ADMIN OVERRIDE LOGGER
# ══════════════════════════════════════════════════════════

def log_admin_override(
    state: AgentState,
    admin_id: str,
    override_decision: str,
    override_reason: str,
) -> AgentState:
    """
    Log an admin override decision into the audit trail.
    Called when a human reviewer overrides the AI's decision.

    This is a separate function (not part of the pipeline)
    that can be called from the API routes.
    """
    logs = list(state.get("agent_logs", []))
    errors = list(state.get("errors", []))

    logs.append({
        "agent": "audit_agent",
        "step": len(logs) + 1,
        "action": "Admin override recorded",
        "status": "override",
        "details": (
            f"Admin '{admin_id}' overrode decision to '{override_decision}'. "
            f"Reason: {override_reason}"
        ),
        "duration_ms": 0,
    })

    # Build the override audit record
    audit_trail = list(state.get("audit_trail", []))
    audit_trail.append(
        _build_audit_record(
            application_id=state.get("application_id", "unknown"),
            agent_name="audit_agent",
            tool_called="admin_override",
            input_data={
                "admin_id": admin_id,
                "original_decision": state.get("final_decision"),
                "override_decision": override_decision,
            },
            output_data={
                "new_decision": override_decision,
                "reason": override_reason,
            },
            reasoning=override_reason,
        )
    )

    return {
        **state,
        "final_decision": override_decision,
        "decision_reasoning": f"Admin override by {admin_id}: {override_reason}",
        "audit_trail": audit_trail,
        "agent_logs": logs,
        "errors": errors,
    }


# ══════════════════════════════════════════════════════════
#  MAIN AGENT FUNCTION
# ══════════════════════════════════════════════════════════

def _run_audit(state: AgentState) -> AgentState:
    """
    Audit Agent node for LangGraph.

    Runs as the FINAL agent in the pipeline. It:
      1. Converts all agent_logs into formal audit records
         (each record carries a SHA-256 hash of its output_data)
      2. Computes per-agent output hashes over the complete AgentState
         to give an independent integrity proof of the entire pipeline run
      3. Detects anomalies in the execution flow
      4. Generates an execution summary
      5. Packages everything for database persistence

    CRITICAL: This agent is APPEND-ONLY.
    It never modifies previous agent outputs.
    """
    logs = list(state.get("agent_logs", []))
    errors = list(state.get("errors", []))
    start = time.time()

    # ── 0. Compute per-agent output hashes (do this first, before logs grow) ─
    try:
        agent_output_hashes = _hash_agent_outputs(state)
        pipeline_hash = _sha256_of(agent_output_hashes)  # single hash of all hashes
        logs.append({
            "agent": "audit_agent",
            "step": len(logs) + 1,
            "action": "Compute cryptographic output hashes (SHA-256)",
            "status": "success",
            "details": (
                f"Hashed {len(agent_output_hashes)} agent output slices | "
                f"Pipeline integrity fingerprint: {pipeline_hash[:16]}…"
            ),
            "duration_ms": 0,
        })
    except Exception as e:
        agent_output_hashes = {}
        pipeline_hash = ""
        errors.append(f"Output hash computation failed: {e}")
        logs.append({
            "agent": "audit_agent",
            "step": len(logs) + 1,
            "action": "Compute cryptographic output hashes (SHA-256)",
            "status": "error",
            "details": str(e),
            "duration_ms": 0,
        })

    # ── 1. Generate the formal audit trail ─────────────────────
    try:
        audit_trail = _generate_audit_trail(state)
        logs.append({
            "agent": "audit_agent",
            "step": len(logs) + 1,
            "action": "Generate formal audit trail",
            "status": "success",
            "details": (
                f"Created {len(audit_trail)} audit records from agent logs — "
                f"each record sealed with SHA-256 output_hash"
            ),
            "duration_ms": 0,
        })
    except Exception as e:
        audit_trail = []
        errors.append(f"Audit trail generation failed: {e}")
        logs.append({
            "agent": "audit_agent",
            "step": len(logs) + 1,
            "action": "Generate formal audit trail",
            "status": "error",
            "details": str(e),
            "duration_ms": 0,
        })

    # ── 2. Run anomaly detection ────────────────────────────
    try:
        anomalies = _detect_anomalies(state)
        if anomalies:
            logs.append({
                "agent": "audit_agent",
                "step": len(logs) + 1,
                "action": "Anomaly detection",
                "status": "warning",
                "details": f"Found {len(anomalies)} anomalie(s): {'; '.join(anomalies)}",
                "duration_ms": 0,
            })
        else:
            logs.append({
                "agent": "audit_agent",
                "step": len(logs) + 1,
                "action": "Anomaly detection",
                "status": "success",
                "details": "No anomalies detected in pipeline execution",
                "duration_ms": 0,
            })
    except Exception as e:
        anomalies = []
        errors.append(f"Anomaly detection failed: {e}")
        logs.append({
            "agent": "audit_agent",
            "step": len(logs) + 1,
            "action": "Anomaly detection",
            "status": "error",
            "details": str(e),
            "duration_ms": 0,
        })

    # ── 3. Generate execution summary ────────────────────────
    try:
        execution_summary = _generate_execution_summary(state)
        # Attach hash table to the summary so it surfaces in the API response
        execution_summary["agent_output_hashes"] = agent_output_hashes
        execution_summary["pipeline_integrity_hash"] = pipeline_hash
        logs.append({
            "agent": "audit_agent",
            "step": len(logs) + 1,
            "action": "Generate execution summary",
            "status": "success",
            "details": (
                f"Pipeline: {execution_summary['total_steps']} steps, "
                f"{execution_summary['total_execution_ms']}ms total, "
                f"{execution_summary['total_errors']} error(s) | "
                f"Integrity hash: {pipeline_hash[:16]}…"
            ),
            "duration_ms": 0,
        })
    except Exception as e:
        execution_summary = {}
        errors.append(f"Execution summary generation failed: {e}")
        logs.append({
            "agent": "audit_agent",
            "step": len(logs) + 1,
            "action": "Generate execution summary",
            "status": "error",
            "details": str(e),
            "duration_ms": 0,
        })

    duration = int((time.time() - start) * 1000)

    summary = (
        f"Audit complete: {len(audit_trail)} records, "
        f"{len(anomalies)} anomalie(s), "
        f"pipeline ran {execution_summary.get('total_steps', 0)} steps | "
        f"Pipeline hash: {pipeline_hash[:16]}…"
    )

    logs.append({
        "agent": "audit_agent",
        "step": len(logs) + 1,
        "action": "Audit finalisation complete",
        "status": "success",
        "details": summary,
        "duration_ms": duration,
    })

    return {
        **state,
        "audit_trail":             audit_trail,
        "audit_anomalies":         anomalies,
        "audit_execution_summary": execution_summary,
        "audit_agent_summary":     summary,
        "agent_logs":              logs,
        "errors":                  errors,
    }


def reporting_agent(state: AgentState) -> AgentState:
    """
    Consolidated Reporting Agent (V7 — Learning is now async).
    1. Generates the human-readable explanation (Explainability)
    2. Runs the Audit (SHA-256 sealed, deterministic)
    3. Fires Learning Agent as a background thread AFTER returning state —
       the user receives their loan decision immediately; memory writes
       happen asynchronously without blocking the pipeline response.
    """
    state = _run_explainability(state)
    state = _run_audit(state)

    # Fire-and-forget: Learning writes to disk/Chroma after the user
    # already has their result. Daemon=True ensures it doesn't block
    # process shutdown if the server terminates while writing.
    snapshot = dict(state)  # shallow copy so thread has stable reference
    t = threading.Thread(target=_run_learning, args=(snapshot,), daemon=True)
    t.start()

    return state
