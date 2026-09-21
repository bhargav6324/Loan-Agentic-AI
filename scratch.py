import pandas as pd
from research_v2.run_agentic_ablation import _run_a1r, _run_a2, _validate_gender
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from config import settings
import sys

settings.PAPER_MODE = True
df = pd.read_csv("research/data/credit_risk_dataset.csv")
features = ["credit_score","dti_ratio","loan_to_income","employment_score","applicant_age","defaults","credit_utilisation","applicant_gender"]
X = df[features]; y = df["target_default"]
X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

df_test = df.loc[X_test.index]

mismatches = []
for idx, row in df_test.iterrows():
    r1r, _ = _run_a1r(row, rf, "h", 6.0)
    r2, _ = _run_a2(row, rf, "h", 6.0)
    if r1r is None or r2 is None: continue
    if r1r["final_decision"] != r2["final_decision"]:
        mismatches.append((idx, r1r, r2))

print(f"Total mismatches: {len(mismatches)}")
for i, m, r2 in mismatches[:5]:
    print(f"ID: {i}")
    print(f"A1R: final={m['final_decision']}, commit={m['commit_status']}, risk={m['composite_score']}, proposal={m['decision']}, audit={m['audit_triggered']}, flag={m['audit_bias_flagged']}")
    print(f"A2 : final={r2['final_decision']}, commit={r2['commit_status']}, risk={r2['composite_score']}, proposal={r2['decision']}, audit={r2['audit_triggered']}, flag={r2['audit_bias_flagged']}")

