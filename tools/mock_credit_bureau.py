"""
Mock Credit Bureau API
──────────────────────
Simulates credit bureau data (CIBIL / Experian / Equifax) for development.
In production: replace with real credit bureau API integration.
"""

from __future__ import annotations
from typing import Optional


_MOCK_CREDIT_DATABASE = {
    "ABCDE1234F": {
        "credit_score": 780,
        "total_accounts": 5,
        "active_accounts": 3,
        "closed_accounts": 2,
        "total_defaults": 0,
        "current_outstanding": 250000,
        "oldest_account_years": 8,
        "recent_inquiries": 1,
        "credit_utilisation_pct": 25.0,
        "payment_history": "excellent",
    },
    "XYZAB5678G": {
        "credit_score": 710,
        "total_accounts": 4,
        "active_accounts": 3,
        "closed_accounts": 1,
        "total_defaults": 0,
        "current_outstanding": 500000,
        "oldest_account_years": 5,
        "recent_inquiries": 2,
        "credit_utilisation_pct": 45.0,
        "payment_history": "good",
    },
    "BORDL5050B": {
        "credit_score": 620,
        "total_accounts": 3,
        "active_accounts": 2,
        "closed_accounts": 1,
        "total_defaults": 1,
        "current_outstanding": 800000,
        "oldest_account_years": 3,
        "recent_inquiries": 4,
        "credit_utilisation_pct": 65.0,
        "payment_history": "fair",
    },
    "POORQ9999Z": {
        "credit_score": 420,
        "total_accounts": 6,
        "active_accounts": 4,
        "closed_accounts": 2,
        "total_defaults": 3,
        "current_outstanding": 1500000,
        "oldest_account_years": 2,
        "recent_inquiries": 8,
        "credit_utilisation_pct": 90.0,
        "payment_history": "poor",
    },
    "FRAUD1111X": {
        "credit_score": 350,
        "total_accounts": 10,
        "active_accounts": 8,
        "closed_accounts": 2,
        "total_defaults": 5,
        "current_outstanding": 3000000,
        "oldest_account_years": 1,
        "recent_inquiries": 15,
        "credit_utilisation_pct": 98.0,
        "payment_history": "very_poor",
    },
}


def get_credit_report(pan_number: str) -> dict:
    """
    Mock credit bureau lookup by PAN number.

    Returns:
        dict with: pan, credit_score, total_accounts, defaults,
                   outstanding, utilisation, payment_history, etc.
    """
    pan_upper = pan_number.upper().strip()

    if pan_upper in _MOCK_CREDIT_DATABASE:
        record = _MOCK_CREDIT_DATABASE[pan_upper].copy()
        record["pan"] = pan_upper
        record["bureau"] = "CIBIL"
        record["report_date"] = "2026-04-01"
        record["found"] = True
        return record
    else:
        # Default moderate credit for unknown PAN
        return {
            "pan": pan_upper,
            "bureau": "CIBIL",
            "report_date": "2026-04-01",
            "found": True,
            "credit_score": 680,
            "total_accounts": 2,
            "active_accounts": 2,
            "closed_accounts": 0,
            "total_defaults": 0,
            "current_outstanding": 100000,
            "oldest_account_years": 3,
            "recent_inquiries": 1,
            "credit_utilisation_pct": 30.0,
            "payment_history": "good",
        }
