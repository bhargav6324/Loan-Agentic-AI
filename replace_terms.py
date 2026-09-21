import os
import glob

replacements = {
    "Human Escalations (Workload)": "Override Rule Triggers",
    "Human Review Escalations (Workload)": "Override Rule Triggers",
    "Human Escalations": "Override Rule Triggers",
    "Escalation Rate": "Override Trigger Rate",
    "Bias Escalations Triggered": "Counterfactual Flags",
    "escalated for human review": "routed to the override rule",
    "human escalation": "override rule trigger",
    "human review": "counterfactual override",
    "human-in-the-loop": "automated override rule",
    "human reviewer": "the override rule",
    "deliberation": "counterfactual override",
    "semantic LLM debate": "",
    "Human-in-the-loop": "Automated override rule"
}

files = []
for root, _, filenames in os.walk("."):
    for filename in filenames:
        if filename.endswith(".py") or filename.endswith(".csv"):
            files.append(os.path.join(root, filename))

for filepath in files:
    with open(filepath, 'r') as f:
        content = f.read()
    
    new_content = content
    # Order matters: do longer phrases first
    # "escalated for human review" before "human review"
    # "Human Review Escalations (Workload)" before "Human Escalations"
    # The dictionary is ordered in python 3.7+
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)
        
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Updated {filepath}")
