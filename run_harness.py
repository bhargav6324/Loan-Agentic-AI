import sys
import pytest
sys.path.append(".")

print("Running Pytest...")
pytest.main(['tests/test_research_v2/', '-v'])

print("Running Ablation...")
# Mock sys.argv
sys.argv = ["run_agentic_ablation.py", "--seeds", "42", "--configurations", "A0,A1,A1R,A2,A3"]
from research_v2.run_agentic_ablation import main
main()
