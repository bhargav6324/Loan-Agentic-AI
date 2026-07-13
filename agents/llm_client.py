"""
Shared Groq API Client
────────────────────────
Groq's free tier caps most models (including llama-3.1-8b-instant) at
30 requests/minute, tracked at the account level — not per model, per
agent, or per file. This project has 7 agents independently calling
Groq (fraud_agent x2, risk_agent, orchestrator_agent, policy_agent,
explainability_agent x4, learning_agent). Without a shared throttle,
a single batch of a handful of applications can burst 40-50+ calls in
a few seconds and get 429'd across the board.

This module gives every agent:
  1. A process-wide rate limiter (min ~2.2s between any Groq call,
     regardless of which agent makes it) — keeps us under 30 RPM.
  2. Automatic retry on 429, honoring the Retry-After header when
     present, before giving up.

On final failure it raises the same httpx exceptions callers already
expect, so each agent's existing `except Exception → fallback` logic
needs no changes beyond swapping the call site.
"""

from __future__ import annotations
import time
import threading
import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

_lock = threading.Lock()
_last_call_time = 0.0
_MIN_INTERVAL_SECONDS = 2.2  # ~27 RPM effective cap, under Groq's 30 RPM free-tier limit


def _throttle() -> None:
    """Block until at least _MIN_INTERVAL_SECONDS have passed since the
    last Groq call made by ANY agent in this process."""
    global _last_call_time
    with _lock:
        now = time.monotonic()
        wait = _MIN_INTERVAL_SECONDS - (now - _last_call_time)
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.monotonic()


def call_groq(payload: dict, api_key: str, timeout: float = 15.0, max_retries: int = 3) -> dict:
    """
    POST to Groq's chat completions endpoint with shared rate limiting
    and 429 retry/backoff.

    Returns the parsed JSON response body on success.
    Raises httpx.HTTPStatusError / httpx.RequestError on final failure —
    callers should keep their existing try/except fallback around this call.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        _throttle()
        try:
            response = httpx.post(GROQ_URL, headers=headers, json=payload, timeout=timeout)

            if response.status_code == 429:
                if attempt >= max_retries:
                    response.raise_for_status()  # exhausted retries -> raise
                retry_after = response.headers.get("retry-after")
                delay = float(retry_after) if retry_after else (2 ** attempt)
                print(f"[llm_client] 429 rate limited, retrying in {delay:.1f}s "
                      f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            last_exc = e
            if attempt >= max_retries or e.response.status_code != 429:
                raise
        except httpx.RequestError as e:
            last_exc = e
            if attempt >= max_retries:
                raise

    # Should be unreachable, but keeps type-checkers happy
    raise last_exc if last_exc else RuntimeError("call_groq: exhausted retries with no exception recorded")
