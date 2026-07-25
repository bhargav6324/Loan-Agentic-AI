import sys

def process():
    with open("agents/explainability_agent.py") as f:
        explain_code = f.read()
    with open("agents/learning_agent.py") as f:
        learning_code = f.read()
    with open("agents/audit_agent.py") as f:
        audit_code = f.read()

    explain_code = explain_code.replace("def explainability_agent(", "def _run_explainability(")
    learning_code = learning_code.replace("def learning_agent(", "def _run_learning(")
    audit_code = audit_code.replace("def audit_agent(", "def _run_audit(")
    
    def clean_imports(code):
        lines = code.split("\n")
        return "\n".join([l for l in lines if not l.startswith("from ") and not l.startswith("import ")])

    out = '"""\nReporting Agent — Merged Explainability, Learning, and Audit (V6)\n"""\n\n'
    out += "from __future__ import annotations\nimport time\nimport json\nimport httpx\nimport os\nfrom datetime import datetime\n"
    out += "from agents.state import AgentState\nfrom config import settings\nfrom agents.llm_client import call_groq\n"

    out += "\n# --- EXPLAINABILITY LOGIC ---\n" + clean_imports(explain_code)
    out += "\n# --- LEARNING/MEMORY LOGIC ---\n" + clean_imports(learning_code)
    out += "\n# --- AUDIT LOGIC ---\n" + clean_imports(audit_code)
    
    out += """

def reporting_agent(state: AgentState) -> AgentState:
    \"\"\"
    Consolidated Reporting Agent.
    1. Generates the human-readable explanation (Explainability)
    2. Queries memory for similar cases to add to the report (Learning)
    3. Persists the final log and decision to disk (Audit)
    \"\"\"
    state = _run_explainability(state)
    state = _run_learning(state)
    state = _run_audit(state)
        
    return state
"""
    with open("agents/reporting_agent.py", "w") as f:
        f.write(out)

if __name__ == "__main__":
    process()
