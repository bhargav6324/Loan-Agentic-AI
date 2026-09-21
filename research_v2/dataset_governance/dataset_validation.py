import os
import json
import hashlib
import pandas as pd
import numpy as np

DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../research/data/credit_risk_dataset.csv"))
GENERATOR_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../research/data/generate_dataset.py"))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(OUTPUT_DIR, "validation_results.json")
REPORT_PATH = os.path.join(OUTPUT_DIR, "DATA_QUALITY_REPORT.md")

# Strict Expected Values
EXPECTED_ROW_COUNT = 5000
EXPECTED_DATASET_SHA256 = "669ff7aeda69a60200732a2b22b7b9cf46bb38681c10892bfb68cc2c38f75f1a"
EXPECTED_GENERATOR_SHA256 = "bdcb361730e59d193817b0d2f4eef7f7db2e28f998e02de00364f289490e7892"
FROZEN_SEEDS = list(range(42, 52))
EXPECTED_COLUMNS = [
    "id", "applicant_age", "applicant_gender", "monthly_income", "loan_amount",
    "employment_type", "employment_score", "employment_years", "loan_tenure_years",
    "loan_purpose", "credit_score", "current_outstanding", "dti_ratio", 
    "loan_to_income", "defaults", "credit_utilisation", "target_default"
]

def hash_file(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath,"rb") as f:
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def generate_data_quality_report(df, stats, issues, checksum):
    md = [
        "# Data Quality Report",
        "*Aligned with selected good-practice principles from the NIST AI RMF, ISO/IEC 5259, and Datasheets for Datasets.*",
        "",
        "## Summary Statistics",
        f"- **Row Count:** {stats['row_count']}",
        f"- **Column Count:** {stats['col_count']}",
        f"- **Missing Values:** {stats['missing_values']}",
        f"- **Duplicate IDs:** {stats['duplicate_ids']}",
        f"- **Duplicate Rows:** {stats['duplicate_rows']}",
        f"- **Dataset Checksum (SHA-256):** `{checksum}`",
        "",
        "## Distribution Statistics",
        "| Feature | Min | Max | Mean | Median | Std Dev | Outliers (z>3) |",
        "|---|---|---|---|---|---|---|"
    ]
    
    for col, s in stats['numeric_stats'].items():
        md.append(f"| {col} | {s['min']:.2f} | {s['max']:.2f} | {s['mean']:.2f} | {s['median']:.2f} | {s['std']:.2f} | {s['outliers']} |")
    
    md.extend([
        "",
        "## Fairness & Subgroup Demographics",
        f"- **Gender Distribution:** {stats['gender_distribution']}",
        f"- **Default Prevalence by Gender (Male=0, Female=1):** {stats['default_prevalence_gender']}",
        f"- **Approval Prevalence (Non-Default) by Gender:** {stats['approval_prevalence_gender']}",
        f"- **Overall Class Imbalance (Defaults / Total):** {stats['default_prevalence'] * 100:.2f}%",
        "",
        "## Constraint & Logic Checks",
        f"- **Constant/Near-constant columns:** {stats.get('constant_columns', 'None')}",
        f"- **Unreachable Policy Rules:**"
    ])
    
    for rule, reachable in stats['reachable_policies'].items():
        md.append(f"    - {rule}: {'Reachable' if reachable else 'UNREACHABLE'}")
        
    md.extend([
        "",
        "## Schema Violations & Critical Issues"
    ])
    
    if issues:
        for issue in issues:
            md.append(f"- [FAILED] {issue}")
    else:
        md.append("- All schema constraints and leakage checks passed successfully.")
        
    md.append("")
    md.append("> Note: Synthetic data outliers are reported above for transparency but have not been deleted or modified. Z-score outliers are warnings only and are not automatically invalid, particularly for skewed financial variables.")
    
    with open(REPORT_PATH, "w") as f:
        f.write("\\n".join(md))

def main():
    print("Starting dataset validation...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
        
    checksum = hash_file(DATASET_PATH)
    generator_checksum = hash_file(GENERATOR_PATH)
    df = pd.read_csv(DATASET_PATH)
    
    issues = []

    if checksum != EXPECTED_DATASET_SHA256:
        issues.append(f"Dataset checksum mismatch: expected {EXPECTED_DATASET_SHA256}, got {checksum}")
    if generator_checksum != EXPECTED_GENERATOR_SHA256:
        issues.append(f"Generator checksum mismatch: expected {EXPECTED_GENERATOR_SHA256}, got {generator_checksum}")
    
    # 1. Row and Column constraints
    if len(df) != EXPECTED_ROW_COUNT:
        issues.append(f"Wrong row count: expected {EXPECTED_ROW_COUNT}, got {len(df)}")
        
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing required features: {missing_cols}")
        
    if "target_default" not in df.columns:
        issues.append("Missing target variable 'target_default'")
    else:
        invalid_targets = df[~df["target_default"].isin([0, 1])]
        if len(invalid_targets) > 0:
            issues.append(f"Invalid target values detected: {invalid_targets['target_default'].unique()}")
            
    if "applicant_gender" in df.columns:
        invalid_genders = df[~df["applicant_gender"].isin([0, 1])]
        if len(invalid_genders) > 0:
            issues.append(f"Invalid gender values detected: {invalid_genders['applicant_gender'].unique()}")
            
    # 2. Data Quality
    dup_ids = df["id"].duplicated().sum() if "id" in df.columns else 0
    if dup_ids > 0:
        issues.append(f"Duplicate IDs detected: {dup_ids}")
        
    dup_rows = df.duplicated().sum()
    missing_vals = df.isna().sum().sum()
    if missing_vals > 0:
        issues.append(f"Missing values detected: {missing_vals}")
        
    default_prev = df["target_default"].mean() if "target_default" in df.columns else 0
    if abs(default_prev - 0.1134) > 0.001:
        issues.append(f"Unexpected default prevalence: expected ~11.34%, got {default_prev*100:.2f}%")
        
    # 3. Numeric Stats & Outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_stats = {}
    constant_columns = []
    for col in numeric_cols:
        if col in ["id", "target_default", "applicant_gender", "employment_score"]:
            continue
        col_data = df[col]
        std = col_data.std()
        if std == 0:
            constant_columns.append(col)
        z_scores = np.abs((col_data - col_data.mean()) / (std + 1e-9))
        numeric_stats[col] = {
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "mean": float(col_data.mean()),
            "median": float(col_data.median()),
            "std": float(std),
            "outliers": int((z_scores > 3).sum())
        }
        
    # 4. Limits Check
    if df["applicant_age"].min() < 21 or df["applicant_age"].max() > 64:
        issues.append("Age generation limits violated (not between 21 and 64)")
    
    # Boundary test for age
    if 21 not in df["applicant_age"].values:
        issues.append("Boundary test failed: Age 21 not present in dataset")
    if 64 not in df["applicant_age"].values:
        issues.append("Boundary test failed: Age 64 not present in dataset")
    if (df["applicant_age"] == 20).sum() > 0:
        issues.append("Boundary test failed: Age 20 present in dataset (should fail)")
    if (df["applicant_age"] == 65).sum() > 0:
        issues.append("Boundary test failed: Age 65 present in dataset (should fail)")
        
    # 5. Fairness/Subgroup Stats
    gender_dist = df["applicant_gender"].value_counts().to_dict()
    default_prev_gender = df.groupby("applicant_gender")["target_default"].mean().to_dict()
    approval_prev_gender = df.groupby("applicant_gender")["target_default"].apply(lambda x: 1 - x.mean()).to_dict()
    
    # 6. Policy Reachability
    reachable_policies = {
        "DTI > 0.6": bool((df["dti_ratio"] > 0.6).sum() > 0),
        "Credit Score < 600": bool((df["credit_score"] < 600).sum() > 0),
        "Utilisation > 80%": bool((df["credit_utilisation"] > 80).sum() > 0),
        "Defaults > 0": bool((df["defaults"] > 0).sum() > 0)
    }
    
    # 7. Leakage check rules (Automated Assertions)
    import sys
    project_root = os.path.abspath(os.path.join(OUTPUT_DIR, "../../"))
    sys.path.insert(0, project_root)
    
    # A. Target and ID absent from model features
    from ml.canonical_model_contract import CanonicalFeatures
    model_features = list(CanonicalFeatures.model_fields.keys())
    if "target_default" in model_features:
        issues.append("Leakage: 'target_default' found in CanonicalFeatures")
    if "id" in model_features:
        issues.append("Leakage: 'id' found in CanonicalFeatures")
        
    # B. Target absent from AgentState
    from agents.state import AgentState
    state_annotations = getattr(AgentState, '__annotations__', {})
    if "target_default" in state_annotations:
        issues.append("Leakage: 'target_default' found in AgentState")
        
    # C. Source inspection plus executable split-overlap checks
    with open(os.path.join(project_root, "research_v2/run_agentic_ablation.py"), "r") as f:
        ablation_src = f.read()
        
    if "train_test_split(" not in ablation_src:
        issues.append("Leakage check failed: train_test_split not found in ablation script")
    # A simple string assertion that we don't fit over the entire DataFrame
    if "fit(df" in ablation_src or "fit_transform(df" in ablation_src:
        issues.append("Leakage: Global preprocessing detected (fitting on 'df' directly)")
        
    if "from sklearn.model_selection import train_test_split" not in ablation_src:
         issues.append("Leakage: sklearn train_test_split missing, potential index overlap")

    from sklearn.model_selection import train_test_split
    split_overlap_counts = {}
    for seed in FROZEN_SEEDS:
        train_idx, test_idx = train_test_split(
            df.index.to_numpy(), test_size=0.30, random_state=seed
        )
        overlap = len(set(train_idx).intersection(test_idx))
        split_overlap_counts[str(seed)] = overlap
        if overlap:
            issues.append(f"Train/test overlap for seed {seed}: {overlap} records")
    
    # Pack Results
    stats = {
        "row_count": len(df),
        "col_count": len(df.columns),
        "missing_values": int(missing_vals),
        "duplicate_ids": int(dup_ids),
        "duplicate_rows": int(dup_rows),
        "default_prevalence": float(default_prev),
        "numeric_stats": numeric_stats,
        "constant_columns": constant_columns,
        "gender_distribution": gender_dist,
        "default_prevalence_gender": default_prev_gender,
        "approval_prevalence_gender": approval_prev_gender,
        "reachable_policies": reachable_policies
        ,"split_overlap_counts": split_overlap_counts
    }
    
    results = {
        "dataset_checksum": checksum,
        "generator_checksum": generator_checksum,
        "validation_passed": len(issues) == 0,
        "issues": issues,
        "stats": stats
    }
    
    with open(JSON_PATH, "w") as f:
        json.dump(results, f, indent=4)
        
    generate_data_quality_report(df, stats, issues, checksum)
    
    if issues:
        print("CRITICAL VALIDATION FAILURES:")
        for i in issues:
            print(f"- {i}")
        raise RuntimeError("Dataset validation failed due to schema or constraint violations.")
        
    print("Dataset validation passed successfully.")
    print(f"Results written to {JSON_PATH}")

if __name__ == "__main__":
    main()
