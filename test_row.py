import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from research_v2.run_agentic_ablation import _run_a1r, _run_a2

# Load data and mock model
test_df = pd.read_csv("research_v2/synthetic_dataset/historical_applications.csv")
# row 4039 is id=4039
row = test_df[test_df['id'] == 4039].iloc[0]
print("ROW ID:", row['id'])

model = RandomForestClassifier(random_state=42)
# Need to mock the model's predict_proba? Actually, the tests do something similar.
# Let's just run it!

try:
    a1r_res, _ = _run_a1r(row, model, "hash", 6.0)
    print("A1R:", a1r_res)
except Exception as e:
    print("A1R failed:", e)

try:
    a2_res, _ = _run_a2(row, model, "hash", 6.0)
    print("A2:", a2_res)
except Exception as e:
    print("A2 failed:", e)
