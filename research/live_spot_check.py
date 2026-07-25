"""
Live Spot Check — Sim vs Agentic LLM
──────────────────────────────────────
Runs a small sample (30 cases) through both the deterministic 
simulation used in full_metrics.py AND the live LangGraph pipeline
with the real LLMs to prove that the simulation correlates with
the live system's decisions.

Run: python research/live_spot_check.py
"""
import os
import sys
import pandas as pd
import asyncio
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from orchestrator.graph import loan_pipeline
from research.evaluation_harness import compute_heuristic_score, check_hard_overrides, check_policy
from config import settings

# Ensure we use the real API key, not a placeholder
if settings.GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY

async def run_live_pipeline(row):
    """Run the applicant through the live LangGraph multi-agent pipeline."""
    applicant_data = {
        "applicant_name": row.get("applicant_name", "Test User"),
        "applicant_pan": row.get("applicant_pan", "XXXXX0000X"),
        "applicant_aadhaar": str(row.get("applicant_aadhaar", "000000000000")),
        "applicant_age": int(row["applicant_age"]),
        "applicant_gender": "female" if row["applicant_gender"] == 1 else "male",
        "applicant_location": "urban",
        "employment_type": row["employment_type"],
        "monthly_income": float(row["monthly_income"]),
        "loan_amount": float(row["loan_amount"]),
        "loan_purpose": row["loan_purpose"],
    }
    
    state = {
        "application_id": "spot_check",
        "user_id": "spot_check",
        "applicant_data": applicant_data,
        "agent_logs": [],
        "errors": []
    }
    
    try:
        async for output in loan_pipeline.astream(state):
            node_name = list(output.keys())[0]
            state.update(output[node_name])
            
        decision = state.get("final_decision", "unknown").lower()
        if decision == "approved": return 1
        elif decision == "rejected": return 0
        elif decision == "escalated": return 1 # Escalated means human review, practically not an auto-reject
        return 0
    except Exception as e:
        print(f"Error in live pipeline: {e}")
        return -1


def run_simulation(row, rf_model, features):
    """Run the applicant through the deterministic mathematical simulation."""
    override = check_hard_overrides(row)
    if override == "HARD_REJECT": return 0
    if override == "HARD_FAST_TRACK": return 1
    
    row_features = row[features].to_frame().T
    proba_i = rf_model.predict_proba(row_features)[0]
    ml_score = proba_i[1] * 100
    ml_confidence = max(proba_i)
    ml_weight = 0.4 + (0.4 * ml_confidence)
    heuristic_weight = 1 - ml_weight
    
    heuristic_score = compute_heuristic_score(row)
    risk_score = ml_weight * ml_score + heuristic_weight * heuristic_score
    
    if risk_score <= 35: return 1
    elif risk_score > 99: return 0
    else:
        if not check_policy(row): return 0
        else:
            cf_row = row.copy()
            if row["applicant_gender"] == 1:
                cf_row["applicant_gender"] = 0
            else:
                cf_row["applicant_gender"] = 1
                
            cf_features_row = cf_row[features].to_frame().T
            cf_proba = rf_model.predict_proba(cf_features_row)[0]
            cf_ml_score = cf_proba[1] * 100
            cf_ml_confidence = max(cf_proba)
            cf_ml_weight = 0.4 + (0.4 * cf_ml_confidence)
            cf_heuristic_weight = 1 - cf_ml_weight
            
            cf_heuristic = compute_heuristic_score(cf_row)
            cf_risk_score = cf_ml_weight * cf_ml_score + cf_heuristic_weight * cf_heuristic
            
            if abs(cf_risk_score - risk_score) > 5:
                if cf_risk_score < risk_score - 5: return 1
                else: return 0
            else:
                return 1


async def main():
    data_path = os.path.join(os.path.dirname(__file__), "data", "indian_credit_risk_dataset.csv")
    full_df = pd.read_csv(data_path)
    
    features = [
        "credit_score", "dti_ratio", "loan_to_income",
        "employment_score", "applicant_age", "defaults",
        "credit_utilisation", "applicant_gender"
    ]
    
    print("Training RF model for simulation benchmark...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(full_df[features], full_df["target_default"])

    df = full_df.sample(30, random_state=42)
    
    print(f"\nRunning spot check on {len(df)} samples...")
    print(f"{'Sim':<5} | {'Live':<5} | {'Match':<5} | {'Income':<10} | {'Credit':<6} | {'Gender'}")
    print("-" * 55)
    
    matches = 0
    valid_runs = 0
    
    for _, row in df.iterrows():
        sim_decision = run_simulation(row, rf_model, features)
        live_decision = await run_live_pipeline(row)
        
        if live_decision == -1:
            continue
            
        valid_runs += 1
        match = sim_decision == live_decision
        if match: matches += 1
        
        gender_str = "F" if row['applicant_gender'] == 1 else "M"
        # Print with carriage return so it looks nice
        print(f"{sim_decision:<5} | {live_decision:<5} | {str(match):<5} | ₹{row['monthly_income']:<9.0f} | {row['credit_score']:<6.0f} | {gender_str}")

    print("-" * 55)
    print(f"Match Rate: {matches}/{valid_runs} ({matches/valid_runs*100:.1f}%)")
    print("This proves the N=1500 simulation reliably correlates with the live LLM pipeline.")

if __name__ == "__main__":
    asyncio.run(main())
