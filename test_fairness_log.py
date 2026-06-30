from test_pipeline import *

print(f"Plan reasoning: {final_state.get('plan_reasoning')}")
debate_log = final_state.get("fairness_report", {}).get("debate_log", [])
import json
print(json.dumps(debate_log, indent=2))
