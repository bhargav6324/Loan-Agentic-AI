import asyncio
import json
from sqlalchemy import select
from models.database import async_sessionmaker, engine, Application, AsyncSession
from orchestrator.graph import loan_pipeline
from datetime import datetime

async def main():
    async with async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)() as db:
        result = await db.execute(select(Application).where(Application.status == "submitted").limit(1))
        app = result.scalar_one_or_none()
        if not app:
            result = await db.execute(select(Application).where(Application.status == "processing").limit(1))
            app = result.scalar_one_or_none()
            if app:
                app.status = "submitted"
                await db.commit()
            else:
                return

        app_id = app.id
        initial_state = {
            "application_id": app_id,
            "user_id": app.user_id,
            "applicant_data": {
                "loan_amount": app.loan_amount,
                "loan_purpose": app.loan_purpose,
                "loan_tenure_months": app.loan_tenure_months,
                "applicant_name": app.applicant_name,
                "applicant_age": app.applicant_age,
                "applicant_gender": app.applicant_gender,
                "applicant_email": app.applicant_email,
                "applicant_phone": app.applicant_phone,
                "applicant_address": app.applicant_address,
                "applicant_pan": app.applicant_pan,
                "applicant_aadhaar": app.applicant_aadhaar,
                "monthly_income": app.monthly_income,
                "employment_type": app.employment_type,
                "employer_name": app.employer_name,
            },
            "agent_logs": [],
            "errors": [],
        }

        final_state = initial_state
        async for output in loan_pipeline.astream(initial_state):
            node_name = list(output.keys())[0]
            print(f"Finished {node_name}")
            final_state.update(output[node_name])
            
        try:
            app_result = await db.execute(select(Application).where(Application.id == app_id))
            app_to_update = app_result.scalar_one()

            app_to_update.shared_state = {
                "risk_score": final_state.get("risk_score"),
                "risk_category": final_state.get("risk_category"),
                "risk_factors": final_state.get("risk_factors"),
                "policy_violations": final_state.get("policy_violations"),
                "fairness_report": final_state.get("fairness_report"),
                "pan_verification": final_state.get("pan_verification"),
                "aadhaar_verification": final_state.get("aadhaar_verification"),
                "credit_report": final_state.get("credit_report"),
                "data_agent_summary": final_state.get("data_agent_summary"),
                "risk_agent_summary": final_state.get("risk_agent_summary"),
                "policy_agent_summary": final_state.get("policy_agent_summary"),
                "fairness_agent_summary": final_state.get("fairness_agent_summary"),
                "explanation": final_state.get("explanation"),
                "plan": final_state.get("plan"),
                "plan_reasoning": final_state.get("plan_reasoning"),
                "data_agent_reasoning_trace": final_state.get("data_agent_reasoning_trace"),
                "fairness_debate_log": final_state.get("fairness_debate_log"),
                # V5: Orchestrator reasoning trace
                "orchestrator_history": final_state.get("orchestrator_history"),
            }
            app_to_update.status = "approved"
            await db.commit()
            print("Successfully saved to DB!")
        except Exception as e:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
