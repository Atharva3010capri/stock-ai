"""Thin wrapper around the Google Gemini API (free tier).

Every LLM call in the project goes through `generate_text()` so that
rate limits (429), busy or timed-out models (503/504), missing API keys and malformed JSON
are handled in one place with clear messages.

Retry policy (kept short so the UI never looks frozen):
  - wait 3s, 6s, 12s ... between attempts (or the delay Gemini asks for)
  - after two "busy" failures on the main model, switch to
    GEMINI_FALLBACK_MODEL for the remaining attempts
  - never sleep more than 60s in one go
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from config import (
    GEMINI_API_KEY,
    GEMINI_FALLBACK_MODEL,
    GEMINI_MODEL,
    LLM_MAX_RETRIES,
    LLM_TEMPERATURE,
)

BUSY_FAILURES_BEFORE_FALLBACK = 2


class LLMError(Exception):
    """Base class for LLM problems the UI can show to the user."""


class MissingAPIKeyError(LLMError):
    """Raised when GEMINI_API_KEY is not configured."""


class RateLimitError(LLMError):
    """Raised when Gemini keeps returning 429/503/504 after all retries."""


class LLMJSONError(LLMError):
    """Raised when the model did not return valid JSON."""


_client = None


def get_client():
    """Create (once) and return the Gemini client."""
    global _client
    if _client is not None:
        return _client

    if not GEMINI_API_KEY:
        raise MissingAPIKeyError(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and paste "
            "your free key from https://aistudio.google.com/apikey"
        )

    from google import genai  # imported lazily so tests can run without it
    from google.genai import types

    # 90 s per request, and let our own loop (not the SDK) handle retries so
    # a busy model never hangs the UI silently.
    _client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=90_000, retry_options=types.HttpRetryOptions(attempts=1)),
    )
    return _client


def _error_code(error: Exception) -> int | None:
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    return code if isinstance(code, int) else None


def _is_busy(error: Exception) -> bool:
    """True when Gemini is worth retrying: rate limit (429), overloaded (503)
    or the server gave up before finishing (504 DEADLINE_EXCEEDED)."""
    if _error_code(error) in (429, 503, 504):
        return True
    message = str(error)
    return any(
        token in message
        for token in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "high demand", "504", "DEADLINE_EXCEEDED")
    )


def _is_missing_model(error: Exception) -> bool:
    return _error_code(error) == 404 or "NOT_FOUND" in str(error) or " 404" in f" {error}"


def _retry_delay(error: Exception, attempt: int) -> float:
    """Use the API's suggested delay if present, else 3s, 6s, 12s... capped at 60s."""
    match = re.search(r"retry(?:Delay| in)\D*(\d+(?:\.\d+)?)", str(error))
    if match:
        return min(float(match.group(1)) + 1.0, 60.0)
    return min(3.0 * (2 ** attempt), 60.0)


def generate_text(
    prompt: str,
    system_instruction: str | None = None,
    json_mode: bool = False,
    temperature: float = LLM_TEMPERATURE,
    max_retries: int = LLM_MAX_RETRIES,
    status_callback=None,
) -> str:
    """Send a prompt to Gemini and return the text reply.

    Args:
        prompt: the user content.
        system_instruction: optional system prompt.
        json_mode: ask the model to answer with JSON only.
        temperature: 0 keeps answers deterministic and grounded.
        max_retries: how many times to retry after a 429/503.
        status_callback: optional function(str) used to tell the UI we are waiting.
    """
    from google.genai import types

    client = get_client()
    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction,
        response_mime_type="application/json" if json_mode else None,
    )

    def say(message: str) -> None:
        if status_callback:
            status_callback(message)

    model = GEMINI_MODEL
    fallback = GEMINI_FALLBACK_MODEL if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL != GEMINI_MODEL else ""
    busy_failures = 0
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            text = (response.text or "").strip()
            if not text:
                raise LLMError("Gemini returned an empty response.")
            return text
        except LLMError:
            raise
        except Exception as error:  # noqa: BLE001 - classified below
            last_error = error
            if _is_busy(error):
                busy_failures += 1
                if attempt >= max_retries:
                    break
                if fallback and model != fallback and busy_failures >= BUSY_FAILURES_BEFORE_FALLBACK:
                    model = fallback
                    say(f"{GEMINI_MODEL} is busy; switching to backup model {fallback}...")
                    continue
                delay = _retry_delay(error, attempt)
                say(f"Gemini is rate-limited or busy. Waiting {delay:.0f}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(delay)
                continue
            if "API key" in str(error) or "API_KEY_INVALID" in str(error) or _error_code(error) in (401, 403):
                raise MissingAPIKeyError("Gemini rejected the API key. Check GEMINI_API_KEY in .env.") from error
            if _is_missing_model(error):
                raise LLMError(
                    f"Model '{model}' was not found or is retired for your key. Run "
                    "`python check_setup.py` to list available models and set GEMINI_MODEL in .env."
                ) from error
            raise LLMError(f"Gemini request failed: {error}") from error

    raise RateLimitError(
        "Gemini is rate-limited or busy (free tier). Please wait a minute and try again, "
        "or set GEMINI_MODEL / GEMINI_FALLBACK_MODEL in .env to other Flash models."
    ) from last_error


def parse_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating ```json fences."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMJSONError("The model did not return valid JSON; the output was discarded.")


def generate_json(
    prompt: str,
    system_instruction: str | None = None,
    status_callback=None,
) -> Any:
    """Call Gemini in JSON mode and return the parsed object."""
    text = generate_text(
        prompt,
        system_instruction=system_instruction,
        json_mode=True,
        status_callback=status_callback,
    )
    return parse_json(text)
