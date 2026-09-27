import sys
from pathlib import Path

# Allow this script to be run directly from the repository root:
# `.venv/bin/python scripts/smoke_test_ollama.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.ollama_client import generate_json, check_health, OllamaError

def main():
    print("Checking Ollama health...")
    try:
        check_health()
        print("Ollama health check passed.")
    except OllamaError as e:
        print(f"Health check failed: {e}")
        sys.exit(1)
        
    prompt = "Return only valid JSON containing monthly_income and employer_name extracted from: Salary ₹50,000, Employer Example Pvt Ltd."
    print("\nSending request to Ollama...")
    print(f"Prompt: {prompt}")
    
    try:
        result = generate_json(prompt)
        print("\nSuccess! Parsed JSON:")
        print(result)
    except Exception as e:
        print(f"\nFailed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
