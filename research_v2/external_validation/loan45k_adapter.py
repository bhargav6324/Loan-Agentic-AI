"""Validation and preparation for the 45K Loan Approval benchmark.

The source is explicitly synthetically enhanced with SMOTENC. This adapter keeps
the source columns intact, removes physically impossible records using a frozen
protocol, and adds only transparent derived fields.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "person_age", "person_gender", "person_education", "person_income",
    "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score", "previous_loan_defaults_on_file", "loan_status",
]


@dataclass(frozen=True)
class Loan45KPrepared:
    data: pd.DataFrame
    audit: dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_loan45k(path: str | Path) -> Loan45KPrepared:
    path = Path(path)
    raw = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    extra_columns = sorted(set(raw.columns) - set(REQUIRED_COLUMNS))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    invalid = {
        "duplicate_row": raw.duplicated(),
        "age_outside_20_100": ~raw["person_age"].between(20, 100),
        "employment_exceeds_age": raw["person_emp_exp"] > (raw["person_age"] - 14),
        "employment_outside_0_70": ~raw["person_emp_exp"].between(0, 70),
        "nonpositive_income": raw["person_income"] <= 0,
        "nonpositive_loan": raw["loan_amnt"] <= 0,
        "credit_score_outside_300_850": ~raw["credit_score"].between(300, 850),
        "invalid_gender": ~raw["person_gender"].isin(["male", "female"]),
        "invalid_previous_default": ~raw["previous_loan_defaults_on_file"].isin(["Yes", "No"]),
        "invalid_target": ~raw["loan_status"].isin([0, 1]),
    }
    invalid_any = pd.concat(invalid, axis=1).any(axis=1)
    clean = raw.loc[~invalid_any, REQUIRED_COLUMNS].copy().reset_index(drop=True)

    clean.insert(0, "benchmark_id", range(1, len(clean) + 1))
    clean["gender_encoded"] = clean["person_gender"].map({"male": 0, "female": 1}).astype(int)
    clean["monthly_income"] = clean["person_income"] / 12.0
    clean["loan_to_income"] = clean["loan_amnt"] / clean["person_income"]
    clean["previous_default_encoded"] = clean["previous_loan_defaults_on_file"].map({"No": 0, "Yes": 1}).astype(int)

    audit = {
        "source_path_name": path.name,
        "source_sha256": sha256_file(path),
        "source_rows": int(len(raw)),
        "source_columns": int(len(raw.columns)),
        "required_columns": REQUIRED_COLUMNS,
        "extra_columns": extra_columns,
        "missing_cells": int(raw.isna().sum().sum()),
        "invalid_counts": {name: int(mask.sum()) for name, mask in invalid.items()},
        "rows_excluded_total": int(invalid_any.sum()),
        "rows_retained": int(len(clean)),
        "approval_counts_retained": {
            str(k): int(v) for k, v in clean["loan_status"].value_counts().sort_index().items()
        },
        "provenance_classification": "synthetically_enhanced_public_benchmark",
        "known_generation_method": "SMOTENC augmentation reported by dataset publisher",
        "unsupported_current_canonical_fields": ["dti_ratio", "credit_utilisation"],
    }
    return Loan45KPrepared(clean, audit)
