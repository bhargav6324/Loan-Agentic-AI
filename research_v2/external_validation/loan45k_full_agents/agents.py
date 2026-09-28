"""45K-compatible agents that replace only source-incompatible project agents."""
from __future__ import annotations

import time

from research_v2.external_validation.loan45k_governance import (
    FAIRNESS_EPSILON, deterministic_reassessment, native_risk_score,
    research_policy, risk_category, state_hash,
)


def _append_log(state, agent, action, details, started):
    logs = list(state.get("agent_logs", []))
    logs.append({"agent": agent, "step": len(logs) + 1, "action": action,
                 "status": "success", "details": details,
                 "duration_ms": int((time.perf_counter() - started) * 1000)})
    return logs


def intake_agent_45k(state):
    started = time.perf_counter()
    record = state["source_record"]
    digest = state_hash(record)
    applicant = {
        "applicant_name": f"Benchmark-{record['benchmark_id']}",
        "applicant_age": int(record["person_age"]),
        "applicant_gender": 1 if record["person_gender"] == "female" else 0,
        "loan_amount": float(record["loan_amnt"]),
        "monthly_income": float(record["monthly_income"]),
        "employment_years": float(record["person_emp_exp"]),
        "loan_purpose": str(record["loan_intent"]),
        "loan_to_income": float(record["loan_percent_income"]),
    }
    logs = _append_log(state, "intake_agent", "45K intake validated",
                       "Identity/document fields unavailable; marked not evaluated.", started)
    return {**state, "applicant_data": applicant, "state_version": 1,
            "state_hash": digest,
            "pan_verification": {"status": "not_evaluated", "verified": None},
            "aadhaar_verification": {"status": "not_evaluated", "verified": None},
            "data_agent_early_stop": False, "fraud_detected": False,
            "requires_manual_kyc": False, "agent_logs": logs}


def risk_agent_45k(state):
    started = time.perf_counter()
    record = state["source_record"]
    score = native_risk_score(record, 1 - float(state["approval_probability"]))
    original = score
    score, reassessed = deterministic_reassessment(score, record)
    category = risk_category(score)
    cert = {"type": "risk", "state_version": state["state_version"],
            "state_hash": state["state_hash"], "valid": True}
    logs = _append_log(state, "risk_agent", "45K risk evaluated",
                       f"score={score:.3f}; category={category}; reassessed={reassessed}", started)
    return {**state, "risk_score": score, "risk_category": category,
            "risk_ml_confidence": max(float(state["approval_probability"]),
                                      1 - float(state["approval_probability"])),
            "risk_agent_reflection": {"revised": reassessed,
                                      "original_score": original,
                                      "revised_score": score},
            "risk_factors": [], "risk_certificate": cert, "agent_logs": logs}


def policy_agent_45k(state):
    started = time.perf_counter()
    passed, violations = research_policy(state["source_record"])
    cert = {"type": "policy", "state_version": state["state_version"],
            "state_hash": state["state_hash"], "valid": True,
            "passed": passed, "violations": violations}
    logs = _append_log(state, "policy_agent", "45K research policy evaluated",
                       f"passed={passed}; violations={violations}", started)
    return {**state, "policy_passed": passed, "policy_violations": violations,
            "policy_certificate": cert, "agent_logs": logs}


def fairness_agent_45k(state):
    started = time.perf_counter()
    approval = float(state["approval_probability"])
    counterfactual = float(state["counterfactual_probability"])
    delta = counterfactual - approval
    flagged = abs(delta) > FAIRNESS_EPSILON
    report = {"method": "gender_counterfactual_probability_flip",
              "signed_delta": delta, "absolute_delta": abs(delta),
              "epsilon": FAIRNESS_EPSILON, "bias_flag": flagged}
    logs = _append_log(state, "fairness_agent", "Gender counterfactual audit",
                       f"signed_delta={delta:.6f}; flagged={flagged}", started)
    return {**state, "fairness_report": report, "bias_detected": flagged,
            "fairness_agent_summary": str(report), "agent_logs": logs}


def critic_agent_45k(state):
    started = time.perf_counter()
    findings = []
    if state.get("decision_proposal") == "approve" and state.get("policy_passed") is False:
        findings.append("approval_conflicts_with_policy")
    if state.get("risk_category") == "MEDIUM" and state.get("policy_passed") is True and not state.get("fairness_report"):
        findings.append("missing_required_fairness_audit")
    if not state.get("risk_certificate") or not state.get("policy_certificate"):
        findings.append("missing_required_certificate")
    logs = _append_log(state, "critic", "Structural consistency critique",
                       f"findings={findings}", started)
    return {**state, "critic_findings": findings, "agent_logs": logs}
