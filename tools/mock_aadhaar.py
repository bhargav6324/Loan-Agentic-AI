"""
Mock Aadhaar Verification API
──────────────────────────────
Simulates Aadhaar identity verification for development and testing.
In production: replace with UIDAI e-KYC / Account Aggregator.
"""

from __future__ import annotations
from typing import Optional


_MOCK_AADHAAR_DATABASE = {
    "123456789012": {"name": "Rahul Sharma", "dob": "1990-05-15", "gender": "male", "address": "Mumbai, Maharashtra", "status": "active"},
    "234567890123": {"name": "Priya Patel", "dob": "1988-11-22", "gender": "female", "address": "Ahmedabad, Gujarat", "status": "active"},
    "345678901234": {"name": "Amit Singh", "dob": "1995-03-10", "gender": "male", "address": "Delhi, NCR", "status": "active"},
    "456789012345": {"name": "Vijay Kumar", "dob": "1985-07-08", "gender": "male", "address": "Pune, Maharashtra", "status": "active"},
    "999999999999": {"name": "Fake Person", "dob": "2000-01-01", "gender": "male", "address": "Unknown", "status": "inactive"},
}


def verify_aadhaar(
    aadhaar_number: str,
    applicant_name: Optional[str] = None,
) -> dict:
    """
    Mock Aadhaar verification.

    Returns:
        dict with: aadhaar, status, name_on_record, name_match,
                   dob, gender, address, verified
    """
    aadhaar = aadhaar_number.strip()

    if aadhaar in _MOCK_AADHAAR_DATABASE:
        record = _MOCK_AADHAAR_DATABASE[aadhaar]
        name_match = False
        if applicant_name:
            name_match = applicant_name.lower().strip() == record["name"].lower()

        return {
            "aadhaar": aadhaar,
            "status": record["status"],
            "name_on_record": record["name"],
            "name_match": name_match,
            "dob": record["dob"],
            "gender": record["gender"],
            "address": record["address"],
            "verified": record["status"] == "active",
        }
    else:
        # Unknown Aadhaar — not found in UIDAI database
        return {
            "aadhaar": aadhaar,
            "status": "not_found",
            "name_on_record": None,
            "name_match": False,
            "dob": None,
            "gender": None,
            "address": None,
            "verified": False,
        }


def seed_from_dataframe(df) -> int:
    """
    Evaluation helper: seeds the mock Aadhaar database with records from a pandas DataFrame
    (e.g., indian_credit_risk_dataset.csv) so synthetic evaluation runs pass KYC verification.
    Does not affect production verification logic or existing database entries.
    """
    count = 0
    for _, row in df.iterrows():
        aadhaar = str(row.get("applicant_aadhaar", "")).strip()
        if aadhaar and aadhaar != "nan" and aadhaar not in _MOCK_AADHAAR_DATABASE:
            _MOCK_AADHAAR_DATABASE[aadhaar] = {
                "name": str(row.get("applicant_name", "Test User")),
                "dob": "1990-01-01",
                "gender": "female" if row.get("applicant_gender") == 1 else "male",
                "address": "Urban, India",
                "status": "active",
            }
            count += 1
    return count

