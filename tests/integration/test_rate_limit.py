"""Gateway rate-limit integration tests (Phase 6, Stage 1).

Drives the real FastAPI app through ``TestClient`` with a *small* fixed-window
limit and a fully injected clock, so the assertions are deterministic and never
touch the wall clock:

* the first ``N`` requests in a window succeed (200);
* request ``N+1`` in the same window -> 429 with a ``Retry-After`` header;
* after the injected clock advances past the window, the client is allowed
  again;
* ``/health`` is never rate-limited.

The clock is a tiny mutable holder injected via ``app.state.rate_limit_clock``
so a test can advance "now" by hand.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings


class FakeClock:
    """A monotonic clock whose value the test advances explicitly."""

    def __init__(self, start: float = 1000.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the offline ``fake`` provider and drop the settings cache per test."""
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_LLM_FALLBACK_PROVIDER", "none")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _rate_limited_app(
    monkeypatch: pytest.MonkeyPatch, *, limit: int, window: float
) -> tuple[TestClient, FakeClock]:
    """Build a TestClient with a tiny rate-limit window and an injected clock."""
    monkeypatch.setenv("FRIDAY_REQUIRE_AUTH", "false")
    monkeypatch.setenv("FRIDAY_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("FRIDAY_RATE_LIMIT_REQUESTS", str(limit))
    monkeypatch.setenv("FRIDAY_RATE_LIMIT_WINDOW_SECONDS", str(window))
    get_settings.cache_clear()
    app = create_app()
    clock = FakeClock()
    app.state.rate_limit_clock = clock
    return TestClient(app), clock


def _chat(client: TestClient) -> int:
    resp = client.post("/chat", json={"session_id": "rl", "text": "what's 2+2"})
    return resp.status_code


def test_in_window_allows_then_429(monkeypatch: pytest.MonkeyPatch) -> None:
    client, clock = _rate_limited_app(monkeypatch, limit=3, window=60.0)
    with client:
        # Re-inject the clock after lifespan startup so the rebuild doesn't drop it.
        client.app.state.rate_limit_clock = clock
        for _ in range(3):
            assert _chat(client) == 200
        resp = client.post(
            "/chat", json={"session_id": "rl", "text": "what's 2+2"}
        )
    assert resp.status_code == 429
    assert resp.json() == {"detail": "rate limit exceeded"}
    assert "Retry-After" in resp.headers
    # Retry-After is a non-negative integer number of seconds.
    assert int(resp.headers["Retry-After"]) >= 0


def test_window_reset_allows_again(monkeypatch: pytest.MonkeyPatch) -> None:
    client, clock = _rate_limited_app(monkeypatch, limit=2, window=60.0)
    with client:
        # The lifespan startup rebuilds runtime; re-pin our clock onto app.state.
        client.app.state.rate_limit_clock = clock
        assert _chat(client) == 200
        assert _chat(client) == 200
        # Third in-window -> blocked.
        assert _chat(client) == 429
        # Advance past the window; the fixed window resets and allows again.
        clock.advance(61.0)
        assert _chat(client) == 200


def test_health_never_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    client, clock = _rate_limited_app(monkeypatch, limit=1, window=60.0)
    with client:
        client.app.state.rate_limit_clock = clock
        # Exhaust the /chat budget.
        assert _chat(client) == 200
        assert _chat(client) == 429
        # /health stays open regardless of the exhausted budget.
        for _ in range(5):
            assert client.get("/health").status_code == 200


def test_rate_limit_disabled_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRIDAY_REQUIRE_AUTH", "false")
    monkeypatch.setenv("FRIDAY_RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("FRIDAY_RATE_LIMIT_REQUESTS", "1")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        for _ in range(5):
            assert _chat(client) == 200
