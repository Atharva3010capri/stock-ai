"""TDD: the Gemini wrapper must survive a busy free-tier model without hanging.

The real SDK is replaced by a fake client; time.sleep is replaced so the
tests run instantly.
"""

import pytest

import gemini_client


class FakeError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class FakeModels:
    def __init__(self, script):
        # script: {model_name: [outcome, outcome, ...]} where outcome is str or Exception
        self.script = {name: list(outcomes) for name, outcomes in script.items()}
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        outcome = self.script[model].pop(0)
        if isinstance(outcome, Exception):
            raise outcome

        class Response:
            text = outcome

        return Response()


class FakeClient:
    def __init__(self, script):
        self.models = FakeModels(script)


@pytest.fixture
def fast(monkeypatch):
    sleeps = []
    monkeypatch.setattr(gemini_client.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(gemini_client, "GEMINI_MODEL", "primary")
    monkeypatch.setattr(gemini_client, "GEMINI_FALLBACK_MODEL", "backup")
    return sleeps


def use(monkeypatch, script):
    client = FakeClient(script)
    monkeypatch.setattr(gemini_client, "get_client", lambda: client)
    return client


def test_busy_model_falls_back_to_second_model(fast, monkeypatch):
    busy = FakeError(503, "UNAVAILABLE: high demand")
    client = use(monkeypatch, {"primary": [busy, busy], "backup": ["OK from backup"]})
    messages = []
    text = gemini_client.generate_text("hi", max_retries=4, status_callback=messages.append)
    assert text == "OK from backup"
    assert client.models.calls == ["primary", "primary", "backup"]
    assert any("backup" in m for m in messages)


def test_rate_limit_retries_then_gives_clear_error(fast, monkeypatch):
    limited = FakeError(429, "RESOURCE_EXHAUSTED")
    client = use(monkeypatch, {"primary": [limited] * 3, "backup": [limited] * 3})
    with pytest.raises(gemini_client.RateLimitError) as error:
        gemini_client.generate_text("hi", max_retries=2)
    assert "wait" in str(error.value).lower()
    assert client.models.calls == ["primary", "primary", "backup"]  # both models were tried
    assert len(fast) >= 1 and max(fast) <= 60  # backed off, but never a multi-minute sleep


def test_first_retry_is_short(fast, monkeypatch):
    busy = FakeError(503, "UNAVAILABLE")
    use(monkeypatch, {"primary": [busy, "OK"], "backup": []})
    assert gemini_client.generate_text("hi", max_retries=2) == "OK"
    assert fast[0] <= 5


def test_retired_model_message_names_check_setup(fast, monkeypatch):
    gone = FakeError(404, "NOT_FOUND: no longer available to new users")
    use(monkeypatch, {"primary": [gone], "backup": []})
    with pytest.raises(gemini_client.LLMError) as error:
        gemini_client.generate_text("hi")
    assert "check_setup" in str(error.value)


def test_no_fallback_configured_still_works(fast, monkeypatch):
    monkeypatch.setattr(gemini_client, "GEMINI_FALLBACK_MODEL", "")
    busy = FakeError(503, "UNAVAILABLE")
    client = use(monkeypatch, {"primary": [busy, busy, "OK"]})
    assert gemini_client.generate_text("hi", max_retries=3) == "OK"
    assert client.models.calls == ["primary"] * 3


def test_server_timeout_504_retries_and_falls_back(fast, monkeypatch):
    timeout = FakeError(504, "DEADLINE_EXCEEDED: Deadline expired before operation could complete.")
    client = use(monkeypatch, {"primary": [timeout, timeout], "backup": ["OK after timeout"]})
    assert gemini_client.generate_text("hi", max_retries=4) == "OK after timeout"
    assert client.models.calls == ["primary", "primary", "backup"]
    assert fast and fast[0] <= 5
