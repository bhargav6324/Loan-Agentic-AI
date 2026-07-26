"""
Mock PAN Verification API
──────────────────────────
Simulates PAN card verification for development and testing.
In production: replace with real NSDL / UTIITSL API.
"""

from __future__ import annotations
from typing import Optional


# Configurable mock responses
_MOCK_PAN_DATABASE = {
    "ABCDE1234F": {"name": "Rahul Sharma", "status": "valid", "category": "individual"},
    "XYZAB5678G": {"name": "Priya Patel", "status": "valid", "category": "individual"},
    "BORDL5050B": {"name": "Amit Singh", "status": "valid", "category": "individual"},
    "POORQ9999Z": {"name": "Vijay Kumar", "status": "valid", "category": "individual"},
    "FRAUD1111X": {"name": "Fake Person", "status": "invalid", "category": "unknown"},
}


def verify_pan(pan_number: str, applicant_name: Optional[str] = None) -> dict:
    """
    Mock PAN verification.

    Returns:
        dict with: pan, status, name_on_record, name_match, category
    """
    pan_upper = pan_number.upper().strip()

    if pan_upper in _MOCK_PAN_DATABASE:
        record = _MOCK_PAN_DATABASE[pan_upper]
        name_match = False
        if applicant_name:
            name_match = applicant_name.lower().strip() == record["name"].lower()

        return {
            "pan": pan_upper,
            "status": record["status"],
            "name_on_record": record["name"],
            "name_match": name_match,
            "category": record["category"],
            "verified": record["status"] == "valid",
        }
    else:
        # Unknown PAN — not found in government database
        return {
            "pan": pan_upper,
            "status": "not_found",
            "name_on_record": None,
            "name_match": False,
            "category": "unknown",
            "verified": False,
        }


def seed_from_dataframe(df) -> int:
    """
    Evaluation helper: seeds the mock PAN database with records from a pandas DataFrame
    (e.g., indian_credit_risk_dataset.csv) so synthetic evaluation runs pass KYC verification.
    Does not affect production verification logic or existing database entries.
    """
    count = 0
    for _, row in df.iterrows():
        pan = str(row.get("applicant_pan", "")).upper().strip()
        if pan and pan != "NAN" and pan not in _MOCK_PAN_DATABASE:
            _MOCK_PAN_DATABASE[pan] = {
                "name": str(row.get("applicant_name", "Test User")),
                "status": "valid",
                "category": "individual",
            }
            count += 1
    return count

