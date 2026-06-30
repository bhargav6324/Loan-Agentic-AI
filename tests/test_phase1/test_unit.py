"""
Phase 1 Tests — Auth, Database, Gateway, Mock APIs
───────────────────────────────────────────────────
Uses SQLite in-memory for testing (no PostgreSQL needed).
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════
# Auth Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPasswordHashing:
    """Test bcrypt password hashing."""

    def test_hash_password(self):
        from auth.password import hash_password, verify_password

        password = "MySecurePass123!"
        hashed = hash_password(password)

        assert hashed != password
        assert hashed.startswith("$2b$")

    def test_verify_correct_password(self):
        from auth.password import hash_password, verify_password

        password = "TestPassword456"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_wrong_password(self):
        from auth.password import hash_password, verify_password

        hashed = hash_password("CorrectPassword")

        assert verify_password("WrongPassword", hashed) is False


class TestJWTHandler:
    """Test JWT token creation and verification."""

    def test_create_access_token(self):
        from auth.jwt_handler import create_access_token, decode_token

        token = create_access_token("user-123", "user")
        payload = decode_token(token)

        assert payload is not None
        assert payload["sub"] == "user-123"
        assert payload["role"] == "user"
        assert payload["type"] == "access"

    def test_create_refresh_token(self):
        from auth.jwt_handler import create_refresh_token, decode_token

        token = create_refresh_token("user-456", "admin")
        payload = decode_token(token)

        assert payload is not None
        assert payload["sub"] == "user-456"
        assert payload["role"] == "admin"
        assert payload["type"] == "refresh"

    def test_create_token_pair(self):
        from auth.jwt_handler import create_token_pair

        tokens = create_token_pair("user-789", "user")

        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert tokens["token_type"] == "bearer"
        assert tokens["expires_in"] > 0

    def test_decode_invalid_token(self):
        from auth.jwt_handler import decode_token

        result = decode_token("invalid.token.here")
        assert result is None

    def test_decode_empty_token(self):
        from auth.jwt_handler import decode_token

        result = decode_token("")
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# Gateway Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestInputValidation:
    """Test input sanitisation and validation."""

    def test_sanitise_normal_string(self):
        from gateway.validation import sanitise_string

        result = sanitise_string("Hello World")
        assert result == "Hello World"

    def test_sanitise_strips_whitespace(self):
        from gateway.validation import sanitise_string

        result = sanitise_string("  hello  ")
        assert result == "hello"

    def test_sanitise_removes_html_tags(self):
        from gateway.validation import sanitise_string

        result = sanitise_string("<b>bold</b> text")
        assert "<b>" not in result
        assert "bold" in result

    def test_sanitise_removes_script_tags(self):
        from gateway.validation import sanitise_string

        result = sanitise_string('<script>alert("xss")</script>Hello')
        assert "<script>" not in result
        assert "Hello" in result

    def test_validate_pan_valid(self):
        from gateway.validation import validate_pan

        assert validate_pan("ABCDE1234F") is True
        assert validate_pan("XYZAB5678G") is True

    def test_validate_pan_invalid(self):
        from gateway.validation import validate_pan

        assert validate_pan("12345") is False
        assert validate_pan("abcde1234f") is False
        assert validate_pan("") is False

    def test_validate_aadhaar_valid(self):
        from gateway.validation import validate_aadhaar

        assert validate_aadhaar("123456789012") is True

    def test_validate_aadhaar_invalid(self):
        from gateway.validation import validate_aadhaar

        assert validate_aadhaar("12345") is False
        assert validate_aadhaar("abcdefghijkl") is False

    def test_validate_loan_amount(self):
        from gateway.validation import validate_loan_amount

        assert validate_loan_amount(500_000) is True
        assert validate_loan_amount(10_000) is True
        assert validate_loan_amount(5_000) is False  # Too low
        assert validate_loan_amount(100_000_000) is False  # Too high


# ═══════════════════════════════════════════════════════════════════════════
# Mock API Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMockPAN:
    """Test mock PAN verification API."""

    def test_known_pan(self):
        from tools.mock_pan import verify_pan

        result = verify_pan("ABCDE1234F", "Rahul Sharma")

        assert result["verified"] is True
        assert result["status"] == "valid"
        assert result["name_match"] is True

    def test_known_pan_name_mismatch(self):
        from tools.mock_pan import verify_pan

        result = verify_pan("ABCDE1234F", "Wrong Name")

        assert result["verified"] is True
        assert result["name_match"] is False

    def test_unknown_pan(self):
        from tools.mock_pan import verify_pan

        result = verify_pan("NEWPN0000A", "New Person")

        assert result["verified"] is True  # Default for unknown

    def test_fraud_pan(self):
        from tools.mock_pan import verify_pan

        result = verify_pan("FRAUD1111X")

        assert result["verified"] is False
        assert result["status"] == "invalid"


class TestMockAadhaar:
    """Test mock Aadhaar verification API."""

    def test_known_aadhaar(self):
        from tools.mock_aadhaar import verify_aadhaar

        result = verify_aadhaar("123456789012", "Rahul Sharma")

        assert result["verified"] is True
        assert result["name_match"] is True
        assert result["gender"] == "male"

    def test_inactive_aadhaar(self):
        from tools.mock_aadhaar import verify_aadhaar

        result = verify_aadhaar("999999999999")

        assert result["verified"] is False
        assert result["status"] == "inactive"


class TestMockCreditBureau:
    """Test mock credit bureau API."""

    def test_good_credit(self):
        from tools.mock_credit_bureau import get_credit_report

        result = get_credit_report("ABCDE1234F")

        assert result["credit_score"] == 780
        assert result["total_defaults"] == 0
        assert result["payment_history"] == "excellent"

    def test_poor_credit(self):
        from tools.mock_credit_bureau import get_credit_report

        result = get_credit_report("POORQ9999Z")

        assert result["credit_score"] == 420
        assert result["total_defaults"] == 3
        assert result["payment_history"] == "poor"

    def test_borderline_credit(self):
        from tools.mock_credit_bureau import get_credit_report

        result = get_credit_report("BORDL5050B")

        assert result["credit_score"] == 620
        assert result["total_defaults"] == 1

    def test_unknown_pan_default(self):
        from tools.mock_credit_bureau import get_credit_report

        result = get_credit_report("UNKNW0000X")

        assert result["found"] is True
        assert result["credit_score"] == 680  # Default moderate
