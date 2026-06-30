"""
Mock Bank Statements API
────────────────────────
Simulates retrieving 6-month bank statements to verify cash flow stability.
"""
def verify_bank_statements(aadhaar: str) -> dict:
    if "1234" in aadhaar or "2345" in aadhaar:
        return {
            "status": "valid",
            "average_monthly_balance": 45000,
            "bounces_6_months": 0,
            "salary_credits": 6,
            "verified": True
        }
    return {
        "status": "warning",
        "average_monthly_balance": 12000,
        "bounces_6_months": 2,
        "salary_credits": 0,
        "verified": True
    }
