"""
Zero Trust Gateway — Rate Limiter
──────────────────────────────────
Uses slowapi for per-IP request throttling.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

# Create limiter instance — keyed by client IP
limiter = Limiter(key_func=get_remote_address)
