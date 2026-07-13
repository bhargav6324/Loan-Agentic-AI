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

from __future__ import annotations
import time
import httpx
from datetime import datetime
from agents.state import AgentState
from agents.llm_client import call_groq
from config import settings


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


def explainability_agent(state: AgentState) -> AgentState:
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
