"""
Learning Agent — Systemic Bias Tracker (V3)
───────────────────────────────────────────
Updates long-term memory with the final outcome of the pipeline.
Analyzes rolling statistics to detect systemic bias over time,
allowing the Fairness Agent to dynamically adjust thresholds.
"""

import os
import json
import time
from agents.state import AgentState

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


def learning_agent(state: AgentState) -> AgentState:
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
