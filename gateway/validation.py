"""
Zero Trust Gateway — Input Validation
──────────────────────────────────────
Sanitise and validate all incoming request data.
"""

import re
from typing import Optional
from fastapi import HTTPException, status


def sanitise_string(value: str, field_name: str = "field", max_length: int = 1000) -> str:
    """Remove potentially dangerous characters from string input."""
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail=f"{field_name} must be a string")

    # Strip whitespace
    value = value.strip()

    if len(value) > max_length:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} exceeds maximum length of {max_length} characters",
        )

    # Remove null bytes
    value = value.replace("\x00", "")

    # Basic XSS prevention — remove script tags
    value = re.sub(r"<script[^>]*>.*?</script>", "", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<[^>]+>", "", value)  # Strip all HTML tags

    return value


def validate_pan(pan: str) -> bool:
    """Validate Indian PAN number format: ABCDE1234F"""
    return bool(re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", pan))


def validate_aadhaar(aadhaar: str) -> bool:
    """Validate Aadhaar number format: 12 digits"""
    return bool(re.match(r"^[0-9]{12}$", aadhaar))


def validate_phone(phone: str) -> bool:
    """Validate Indian phone number."""
    return bool(re.match(r"^(\+91)?[6-9][0-9]{9}$", phone.replace(" ", "").replace("-", "")))


def validate_loan_amount(amount: float) -> bool:
    """Validate loan amount is within acceptable range."""
    return 10_000 <= amount <= 50_000_000  # ₹10K to ₹5Cr
