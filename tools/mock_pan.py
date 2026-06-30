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
        # Unknown PAN — generate a plausible response
        return {
            "pan": pan_upper,
            "status": "valid",
            "name_on_record": applicant_name or "Unknown",
            "name_match": True,
            "category": "individual",
            "verified": True,
        }
