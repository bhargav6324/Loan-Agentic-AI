"""
Mock GST Returns API
────────────────────
Simulates retrieving GST returns for self-employed applicants.
"""
def verify_gst_returns(pan: str) -> dict:
    if "XYZ" in pan:
        return {
            "status": "valid",
            "annual_turnover": 1500000,
            "gst_status": "active",
            "verified": True
        }
    return {
        "status": "invalid",
        "verified": False,
        "reason": "No GST registration found for PAN."
    }
