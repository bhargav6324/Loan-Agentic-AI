"""
Memory Seeder — Batch Run 200 Diverse Profiles
───────────────────────────────────────────────
Generates 200 synthetic Indian loan applicant profiles with
realistic diversity and runs them through the full pipeline
to populate long_term_memory.json with rich, complete records.

Also tallies orchestrator routing metrics (deterministic vs LLM vs fallback).

Run: python scripts/seed_memory.py
"""
import asyncio
import uuid
import sys
import os
import random
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Override API key to force deterministic fallbacks ─────
# This makes seeding fast (no LLM calls, no rate limits).
# The ML model, heuristics, and rule-based logic still run.
# The Learning Agent still records every decision.
os.environ["GROQ_API_KEY"] = "gsk_placeholder"

from orchestrator.graph import loan_pipeline

# ── Indian name pools ─────────────────────────────────────
MALE_NAMES = [
    "Ravi Kumar", "Suresh Patel", "Vikram Singh", "Arjun Sharma",
    "Rahul Gupta", "Amit Verma", "Sanjay Reddy", "Manoj Tiwari",
    "Deepak Joshi", "Karan Malhotra", "Nikhil Agarwal", "Ashish Dubey",
    "Rajesh Nair", "Pradeep Yadav", "Gaurav Mishra", "Sachin Pandey",
    "Anil Chauhan", "Rohit Saxena", "Vijay Menon", "Harish Iyer",
]
FEMALE_NAMES = [
    "Priya Sharma", "Anita Desai", "Neha Gupta", "Sunita Patel",
    "Kavita Reddy", "Meera Nair", "Deepika Singh", "Pooja Verma",
    "Swati Joshi", "Rani Malhotra", "Aditi Agarwal", "Shruti Tiwari",
    "Nisha Yadav", "Lakshmi Iyer", "Sonal Pandey", "Geeta Mishra",
    "Pallavi Chauhan", "Ritu Saxena", "Jaya Menon", "Divya Dubey",
]

EMPLOYMENT_TYPES = ["salaried", "self_employed", "business", "unemployed"]
LOCATIONS = ["urban", "semi_urban", "rural"]
LOAN_PURPOSES = ["home_improvement", "personal", "business", "car", "education", "medical", "wedding"]
PAN_PREFIXES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _random_pan():
    return "".join(random.choices(PAN_PREFIXES, k=5)) + "".join(str(random.randint(0, 9)) for _ in range(4)) + random.choice(PAN_PREFIXES)


def _random_aadhaar():
    return "".join(str(random.randint(0, 9)) for _ in range(12))


def generate_diverse_profiles(n: int = 200) -> list[dict]:
    """Generate N diverse Indian loan applicant profiles."""
    profiles = []

    for i in range(n):
        gender = random.choices(["male", "female"], weights=[0.65, 0.35])[0]
        name = random.choice(MALE_NAMES if gender == "male" else FEMALE_NAMES)
        location = random.choice(LOCATIONS)
        emp_type = random.choices(
            EMPLOYMENT_TYPES,
            weights=[0.45, 0.25, 0.20, 0.10]
        )[0]

        # Income varies by employment type
        if emp_type == "salaried":
            monthly_income = random.randint(25000, 300000)
        elif emp_type == "business":
            monthly_income = random.randint(40000, 500000)
        elif emp_type == "self_employed":
            monthly_income = random.randint(15000, 200000)
        else:  # unemployed
            monthly_income = random.randint(5000, 20000)

        # Loan amount varies realistically
        loan_amount = random.randint(
            max(50000, monthly_income),
            min(5000000, monthly_income * 36)
        )

        age = random.randint(21, 65)
        purpose = random.choice(LOAN_PURPOSES)

        profiles.append({
            "application_id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "applicant_data": {
                "applicant_name": name,
                "applicant_pan": _random_pan(),
                "applicant_aadhaar": _random_aadhaar(),
                "applicant_age": age,
                "applicant_gender": gender,
                "applicant_location": location,
                "employment_type": emp_type,
                "monthly_income": monthly_income,
                "loan_amount": loan_amount,
                "loan_purpose": purpose,
            }
        })

    return profiles


async def main():
    n = 200
    profiles = generate_diverse_profiles(n)

    counts = {"deterministic": 0, "llm_agentic": 0, "fallback": 0}
    decisions = {"approved": 0, "rejected": 0, "escalated": 0}
    errors = 0

    print(f"{'='*60}")
    print(f"MEMORY SEEDER — Running {n} diverse profiles through pipeline")
    print(f"{'='*60}")
    print(f"Start time: {time.strftime('%H:%M:%S')}")
    print()

    for i, p in enumerate(profiles):
        app_name = p["applicant_data"]["applicant_name"]
        app_income = p["applicant_data"]["monthly_income"]
        app_loan = p["applicant_data"]["loan_amount"]
        emp = p["applicant_data"]["employment_type"]

        print(f"[{i+1:3d}/{n}] {app_name:20s} | ₹{app_income:>8,}/mo | Loan: ₹{app_loan:>10,} | {emp:15s}", end="", flush=True)

        state = p.copy()
        state["agent_logs"] = []
        state["errors"] = []

        try:
            async for output in loan_pipeline.astream(state):
                node_name = list(output.keys())[0]
                state.update(output[node_name])

            decision = state.get("final_decision", "unknown")
            decisions[decision] = decisions.get(decision, 0) + 1

            # Count orchestrator routing
            for log in state.get("agent_logs", []):
                if log.get("agent") == "orchestrator":
                    status = log.get("status", "")
                    if status in counts:
                        counts[status] += 1

            print(f" → {decision.upper()}")

        except Exception as e:
            errors += 1
            print(f" → ERROR: {e}")

    # ── Final Report ──────────────────────────────────────
    total_routing = sum(counts.values())

    print(f"\n{'='*60}")
    print(f"MEMORY SEEDER — COMPLETE")
    print(f"{'='*60}")
    print(f"End time: {time.strftime('%H:%M:%S')}")
    print(f"\n📊 Decision Distribution ({n} applications):")
    for dec, count in sorted(decisions.items(), key=lambda x: -x[1]):
        print(f"   {dec.upper():12s}: {count:3d} ({count/n*100:.1f}%)")
    print(f"   {'ERRORS':12s}: {errors:3d}")

    print(f"\n🧠 Orchestrator Routing Metrics:")
    print(f"   Total routing decisions: {total_routing}")
    if total_routing > 0:
        print(f"   Deterministic / Fast-path: {counts['deterministic']:3d} ({counts['deterministic']/total_routing*100:.1f}%)")
        print(f"   LLM-Agentic / Reasoning:   {counts['llm_agentic']:3d} ({counts['llm_agentic']/total_routing*100:.1f}%)")
        print(f"   System Fallbacks:          {counts['fallback']:3d} ({counts['fallback']/total_routing*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
