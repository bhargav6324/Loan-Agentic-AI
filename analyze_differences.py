import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from research_v2.run_agentic_ablation import _run_a1, _run_a1r, _run_a2, _run_a3
import hashlib

FEATURES = [
    "credit_score", "dti_ratio", "loan_to_income", "employment_score",
    "applicant_age", "defaults", "credit_utilisation", "applicant_gender"
]

df = pd.read_csv("research/data/credit_risk_dataset.csv")
X = df[FEATURES]
y = df["target_default"]

# Same seed 42 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
model_hash = hashlib.sha256(b"RF_seed_42_n100").hexdigest()[:16]

test_indices = X_test.index
df_test = df.loc[test_indices].copy()

a1_a1r_diff = 0
a1r_a2_diff = 0
a2_a3_diff = 0

differences_a1r_a2 = []

for idx, row in df_test.iterrows():
    # Only run to count
    r1, _ = _run_a1(row, rf, model_hash, 6.0)
    r1r, _ = _run_a1r(row, rf, model_hash, 6.0)
    r2, _ = _run_a2(row, rf, model_hash, 6.0)
    r3, _ = _run_a3(row, rf, model_hash, 6.0)
    
    d1 = r1["final_decision"]
    d1r = r1r["final_decision"]
    d2 = r2["final_decision"]
    d3 = r3["final_decision"]
    
    if d1 != d1r:
        a1_a1r_diff += 1
    if d1r != d2:
        a1r_a2_diff += 1
        differences_a1r_a2.append({
            "id": row["id"],
            "A1R_decision": d1r,
            "A1R_nodes": r1r.get("nodes_executed"),
            "A1R_score": r1r.get("composite_score"),
            "A2_decision": d2,
            "A2_nodes": r2.get("nodes_executed", ""),
            "A2_score": r2.get("composite_score"),
            "A2_commit_status": r2.get("commit_status")
        })
    if d2 != d3:
        a2_a3_diff += 1

with open("differences.txt", "w") as f:
    f.write(f"A1 vs A1R diffs: {a1_a1r_diff}\n")
    f.write(f"A1R vs A2 diffs: {a1r_a2_diff}\n")
    f.write(f"A2 vs A3 diffs: {a2_a3_diff}\n")
    f.write("\nSample A1R vs A2 diffs:\n")
    for d in differences_a1r_a2[:20]:
        f.write(str(d) + "\n")
