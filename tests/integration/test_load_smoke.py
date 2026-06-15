"""Load + error-budget smoke test (Phase 6, Stage 1).

Fires a burst of ``POST /chat`` turns at the real app (offline ``FakeLLM``,
sequentially and concurrently) and asserts:

* every request returns 200 (zero unexpected 5xx — the error budget is intact);
* the process-wide ``Metrics.requests`` counter counted every turn (the same
  counter the ``/admin/metrics`` view surfaces and the error budget watches via
  ``Metrics.errors``);
* ``Metrics.errors`` stays at 0 across the burst.

Auth and rate-limit are disabled for the smoke run so the burst is purely about
throughput and the error budget, not the gateway gates (those have their own
tests). The metrics instance is read straight off ``app.state.metrics`` — the
same one the orchestrator increments per turn.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings
from friday.observability.metrics import Metrics

_BURST = 50


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the offline ``fake`` provider and drop the settings cache per test."""
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_LLM_FALLBACK_PROVIDER", "none")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _open_app(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """An app with both gateway gates disabled (pure throughput smoke)."""
    monkeypatch.setenv("FRIDAY_REQUIRE_AUTH", "false")
    monkeypatch.setenv("FRIDAY_RATE_LIMIT_ENABLED", "false")
    get_settings.cache_clear()
    return TestClient(create_app())


def _metrics(client: TestClient) -> Metrics:
    metrics = client.app.state.metrics
    assert isinstance(metrics, Metrics)
    return metrics


def _chat(client: TestClient, i: int) -> int:
    resp = client.post(
        "/chat", json={"session_id": f"load-{i}", "text": "what's 2+2"}
    )
    return resp.status_code


def test_sequential_burst_all_200_and_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _open_app(monkeypatch) as client:
        before = _metrics(client).snapshot()["requests"]
        statuses = [_chat(client, i) for i in range(_BURST)]
        after = _metrics(client).snapshot()

    assert statuses == [200] * _BURST
    # Error budget: no turn surfaced an error.
    assert after["errors"] == 0
    # Every turn was counted by the process-wide requests counter.
    assert after["requests"] - before == _BURST


def test_concurrent_burst_all_200_and_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _open_app(monkeypatch) as client:
        before = _metrics(client).snapshot()["requests"]
        with ThreadPoolExecutor(max_workers=8) as pool:
            statuses = list(pool.map(lambda i: _chat(client, i), range(_BURST)))
        after = _metrics(client).snapshot()

    assert statuses == [200] * _BURST
    assert after["errors"] == 0
    assert after["requests"] - before == _BURST
