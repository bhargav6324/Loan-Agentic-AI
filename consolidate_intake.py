import sys

def process():
    with open("agents/data_agent.py") as f:
        data_code = f.read()
    with open("agents/fraud_agent.py") as f:
        fraud_code = f.read()
    with open("agents/policy_agent.py") as f:
        policy_code = f.read()

    # We need to construct intake_agent.py
    # We will just append them but rename the main functions to _run_data, _run_fraud, _run_policy
    # and create a master intake_agent function
    
    data_code = data_code.replace("def data_agent(", "def _run_data(")
    fraud_code = fraud_code.replace("def fraud_agent(", "def _run_fraud(")
    policy_code = policy_code.replace("def policy_agent(", "def _run_policy(")
    
    out = '"""\nIntake & Compliance Agent — Merged Data, Fraud, and Policy (V6)\n"""\n\n'
    out += "from __future__ import annotations\nimport time\nimport json\nimport httpx\nimport copy\n"
    out += "from agents.state import AgentState\nfrom config import settings\nfrom agents.llm_client import call_groq\n"
    out += "from tools.mock_pan import verify_pan\nfrom tools.mock_aadhaar import verify_aadhaar\nfrom tools.mock_credit_bureau import get_credit_report\nfrom tools.mock_gst import verify_gst_returns\nfrom tools.mock_bank_statements import verify_bank_statements\n"
    
    # Remove imports from parts
    def clean_imports(code):
        lines = code.split("\n")
        return "\n".join([l for l in lines if not l.startswith("from ") and not l.startswith("import ")])

    out += "\n# --- DATA AGENT LOGIC ---\n" + clean_imports(data_code)
    out += "\n# --- FRAUD AGENT LOGIC ---\n" + clean_imports(fraud_code)
    out += "\n# --- POLICY AGENT LOGIC ---\n" + clean_imports(policy_code)
    
    out += """

def intake_agent(state: AgentState) -> AgentState:
    \"\"\"
    Consolidated agent that sequentially runs Data Gathering, Fraud Check, and Policy Check.
    If Fraud is detected, it returns early.
    \"\"\"
    state = _run_data(state)
    if state.get("data_agent_early_stop"):
        return state
        
    state = _run_fraud(state)
    if state.get("fraud_detected"):
        return state
        
    state = _run_policy(state)
    return state
"""
    with open("agents/intake_agent.py", "w") as f:
        f.write(out)

if __name__ == "__main__":
    process()
