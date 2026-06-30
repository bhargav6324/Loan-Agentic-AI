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

from __future__ import annotations
import hashlib
import json
import time
from datetime import datetime, timezone
from agents.state import AgentState


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

def audit_agent(state: AgentState) -> AgentState:
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
