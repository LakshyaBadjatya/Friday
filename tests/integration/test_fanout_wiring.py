"""Integration tests for the Tier-3 feature fan-out wiring in :mod:`friday.app`.

The consolidation step wires the eleven disjoint feature modules into
``create_app`` / ``build_runtime``. These tests are the cross-feature wiring
contract:

* **Routers** — each flagged Tier-3 router (maps, presence, market, calendar,
  email, comms, hud, family) is *included unconditionally* (the router itself
  self-guards on its flag), so it is REACHABLE (not a missing-registration 404)
  when its flag is on and a clean self-guard ``404`` when its flag is off.
* **Offline mode** — ``build_runtime`` wraps the live LLM via
  :func:`friday.providers.offline.select_llm`, so ``enable_offline_mode`` swaps
  in the network-free :class:`~friday.providers.offline.OfflineLLM`.
* **Postgres** — ``build_runtime`` selects the Postgres long-term / vector stores
  when ``enable_postgres`` is on; absent ``psycopg``/DSN they raise a clear
  :class:`~friday.errors.FridayError` (only ever triggered by the flag).
* **Porcupine** — the wake-word backend is importable (no wiring required).
* **Orchestrator** — an "open maps" turn replies with a ``/maps`` deep link, and
  "distance to <X>" replies with a URL-encoded ``/maps?to=<X>`` link.

Everything is OFFLINE: ``llm_provider="fake"`` + ``memory_db_path=":memory:"``,
no network, no subprocess. Flags are flipped by setting the ``FRIDAY_*`` env vars
and clearing the cached :func:`~friday.config.get_settings`, so the *same* cached
accessor every router and ``create_app`` already imported returns settings that
reflect the flags.

A distinguishing detail: the self-guard ``404`` carries the feature's own
``{"detail": "<feature> disabled"}`` body, whereas a missing-registration ``404``
carries Starlette's ``{"detail": "Not Found"}``. The on-state assertions check
the response is NOT a missing-registration 404 (the router IS registered).
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from friday.app import build_runtime, create_app
from friday.config import Settings, get_settings
from friday.core.orchestrator import Orchestrator
from friday.providers.offline import OfflineLLM

# Each flagged Tier-3 router: its ``FRIDAY_*`` flag env var, a representative
# request (method, path, optional body) the self-guard answers, and the
# feature's own "disabled" detail substring (distinguishes the self-guard 404
# from a missing-registration "Not Found" 404). The chosen on-state request is
# offline-safe: GET surfaces that work with no credentials, or POST/GET surfaces
# whose flag check precedes any body parse / network client build.
_ROUTER_CASES: tuple[dict[str, object], ...] = (
    {
        "name": "maps",
        "flag": "FRIDAY_ENABLE_MAPS",
        "method": "GET",
        "path": "/maps",
        "disabled_detail": "maps disabled",
    },
    {
        "name": "presence",
        "flag": "FRIDAY_ENABLE_PRESENCE",
        "method": "GET",
        "path": "/presence",
        "disabled_detail": "presence disabled",
    },
    {
        "name": "market",
        "flag": "FRIDAY_ENABLE_MARKET_DATA",
        "method": "GET",
        "path": "/market/quote?symbol=NSE_EQ:1",
        "disabled_detail": "market data disabled",
    },
    {
        "name": "calendar",
        "flag": "FRIDAY_ENABLE_CALENDAR",
        "method": "POST",
        "path": "/calendar/events",
        "disabled_detail": "calendar disabled",
    },
    {
        "name": "email",
        "flag": "FRIDAY_ENABLE_EMAIL",
        "method": "POST",
        "path": "/email/draft",
        "disabled_detail": "email disabled",
    },
    {
        "name": "comms",
        "flag": "FRIDAY_ENABLE_COMMS",
        "method": "POST",
        "path": "/comms/sms",
        "disabled_detail": "comms disabled",
    },
    {
        "name": "hud",
        "flag": "FRIDAY_ENABLE_HUD",
        "method": "GET",
        "path": "/hud",
        "disabled_detail": "hud disabled",
    },
    {
        "name": "family",
        "flag": "FRIDAY_ENABLE_FAMILY_SHARING",
        "method": "POST",
        "path": "/family/optin",
        "disabled_detail": "family sharing disabled",
    },
)

# The path prefix every router must register in ``app.routes`` (so the fan-out
# included it at all, independent of the flag).
_ROUTER_PREFIXES: dict[str, str] = {
    "maps": "/maps",
    "presence": "/presence",
    "market": "/market",
    "calendar": "/calendar",
    "email": "/email",
    "comms": "/comms",
    "hud": "/hud",
    "family": "/family",
}


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """Clear the cached settings before and after each test (env isolation)."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _offline_settings(**flags: bool) -> Settings:
    """Offline settings with the given flags forced on (rest default off)."""
    return Settings(
        _env_file=None,
        llm_provider="fake",
        memory_db_path=":memory:",
        **flags,
    )


def _request(client: TestClient, method: str, path: str) -> object:
    """Issue ``method path`` with an empty JSON body for POST (forces validation)."""
    if method == "POST":
        return client.post(path, json={})
    return client.get(path)


# --------------------------------------------------------------------------- #
# Router fan-out: included unconditionally, self-guard on the flag.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", _ROUTER_CASES, ids=lambda c: str(c["name"]))
def test_router_404s_when_flag_off(
    case: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With every flag off, each Tier-3 router answers its self-guard 404.

    The 404 carries the feature's own ``"<feature> disabled"`` detail — proving
    it is the router's self-guard (the route IS registered), not a missing
    registration.
    """
    # Strip any ambient FRIDAY_* so the build is the pure offline default.
    for key in (str(c["flag"]) for c in _ROUTER_CASES):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_MEMORY_DB_PATH", ":memory:")
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as client:
        resp = _request(client, str(case["method"]), str(case["path"]))

    assert resp.status_code == 404, (case["name"], resp.status_code)
    body = resp.json()
    assert body.get("detail") == case["disabled_detail"], (case["name"], body)


@pytest.mark.parametrize("case", _ROUTER_CASES, ids=lambda c: str(c["name"]))
def test_router_reachable_when_flag_on(
    case: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With its flag on, each Tier-3 router is reachable (not a registration 404).

    The on-state request is offline-safe (no credentials needed, no network):
    a GET surface that answers locally, or a POST whose flag check precedes any
    body parse / network client build (so an empty body trips a 422). The only
    forbidden outcome is the missing-registration 404 — that would mean the
    fan-out never included the router.
    """
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_MEMORY_DB_PATH", ":memory:")
    monkeypatch.setenv(str(case["flag"]), "true")
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as client:
        resp = _request(client, str(case["method"]), str(case["path"]))

    # Reachable means: NOT a missing-registration 404. A 404 is only acceptable
    # if it is impossible here — when the flag is on the self-guard never fires,
    # so any 404 at all means the route was not registered.
    assert resp.status_code != 404, (case["name"], resp.status_code, resp.text)


def _registered_paths(app: object) -> set[str]:
    """Every registered route path, read from the OpenAPI schema.

    ``app.routes`` nests an ``include_router``'d router inside a mount wrapper in
    this FastAPI/Starlette version (so the included paths are not flat there); the
    OpenAPI ``paths`` map flattens every registered route regardless of how it was
    mounted, which is the reliable structural signal that the fan-out included a
    router at all.
    """
    schema = app.openapi()  # type: ignore[attr-defined]
    return set(schema.get("paths", {}).keys())


@pytest.mark.parametrize("case", _ROUTER_CASES, ids=lambda c: str(c["name"]))
def test_router_is_registered_on_app(case: dict[str, object]) -> None:
    """Each Tier-3 router's path prefix is present in the app's registered paths.

    A structural check independent of the flag: the fan-out includes every
    router unconditionally, so the path exists on the app regardless of config.
    """
    app = create_app()
    prefix = _ROUTER_PREFIXES[str(case["name"])]
    paths = _registered_paths(app)
    assert any(p == prefix or p.startswith(prefix + "/") for p in paths), (
        case["name"],
        prefix,
        sorted(paths),
    )


def test_all_flags_off_routers_still_registered() -> None:
    """The offline default registers all eight Tier-3 routers (none 404 as missing)."""
    app = create_app()
    paths = _registered_paths(app)
    for name, prefix in _ROUTER_PREFIXES.items():
        assert any(p == prefix or p.startswith(prefix + "/") for p in paths), (
            name,
            prefix,
        )


# --------------------------------------------------------------------------- #
# Offline mode: build_runtime swaps the live LLM for the network-free OfflineLLM.
# --------------------------------------------------------------------------- #
def test_offline_mode_selects_offline_llm() -> None:
    """``enable_offline_mode`` on -> the runtime LLM is an :class:`OfflineLLM`."""
    runtime = build_runtime(_offline_settings(enable_offline_mode=True))
    assert isinstance(runtime.llm, OfflineLLM)
    # The orchestrator shares the same offline provider (no live LLM reachable).
    assert isinstance(runtime.orchestrator._llm, OfflineLLM)  # noqa: SLF001


def test_offline_mode_off_keeps_live_llm() -> None:
    """Offline mode off -> the runtime keeps the (fake) live LLM, not OfflineLLM."""
    runtime = build_runtime(_offline_settings(enable_offline_mode=False))
    assert not isinstance(runtime.llm, OfflineLLM)


# --------------------------------------------------------------------------- #
# Postgres: build_runtime selects the PG stores only when the flag is on; absent
# psycopg/DSN they raise a clear FridayError (the flag is the only trigger).
# --------------------------------------------------------------------------- #
def test_postgres_off_uses_sqlite_stores() -> None:
    """Postgres off (the default) -> the SQLite long-term / vector stores are used."""
    from friday.memory.long_term import SQLiteLongTermStore
    from friday.memory.vector import SQLiteVectorStore

    runtime = build_runtime(_offline_settings(enable_postgres=False))
    assert isinstance(runtime.long_term, SQLiteLongTermStore)
    assert isinstance(runtime.vector, SQLiteVectorStore)


def test_postgres_on_without_dsn_raises_clear_error() -> None:
    """Postgres on but no DSN -> a clear FridayError (never a half-open store).

    The Postgres adapters validate the DSN before importing the driver, so an
    enabled-but-unconfigured build fails fast and loud — exactly the documented
    behaviour for the flag being on without ``psycopg``/``FRIDAY_POSTGRES_DSN``.
    """
    from friday.errors import FridayError

    with pytest.raises(FridayError):
        build_runtime(_offline_settings(enable_postgres=True))


# --------------------------------------------------------------------------- #
# Porcupine: no wiring required beyond it being importable.
# --------------------------------------------------------------------------- #
def test_porcupine_importable() -> None:
    """The Porcupine wake-word backend imports cleanly (no heavy dep at import)."""
    module = importlib.import_module("friday.voice.porcupine")
    assert hasattr(module, "PorcupineWakeWord")


# --------------------------------------------------------------------------- #
# Orchestrator: light "maps" intent emits a /maps deep link.
# --------------------------------------------------------------------------- #
def _orchestrator() -> Orchestrator:
    """A fully-wired orchestrator over the offline runtime (fake LLM, in-memory)."""
    return build_runtime(_offline_settings()).orchestrator


async def test_open_maps_emits_maps_link() -> None:
    """An "open maps" turn replies with a ``/maps`` deep link (no LLM needed)."""
    from friday.core.state import GraphState

    orch = _orchestrator()
    state = GraphState(session_id="m1", user_input="open maps")

    out = await orch.handle(state)

    assert out.response is not None
    assert "/maps" in out.response


async def test_distance_to_emits_encoded_maps_link() -> None:
    """A "distance to <X>" turn replies with a URL-encoded ``/maps?to=<X>`` link."""
    from friday.core.state import GraphState

    orch = _orchestrator()
    state = GraphState(
        session_id="m2", user_input="show me distance to New York City"
    )

    out = await orch.handle(state)

    assert out.response is not None
    assert "/maps?to=" in out.response
    # The destination in the LINK is URL-encoded (a space -> %20 or +), never a
    # raw space in the query. (The friendly prose may still name the place in
    # plain text; we assert on the deep link specifically.)
    link = out.response[out.response.index("/maps?to=") :].split()[0]
    assert " " not in link
    assert ("New%20York%20City" in link) or ("New+York+City" in link)


async def test_plain_distance_to_emits_encoded_maps_link() -> None:
    """The bare "distance to <X>" phrasing also emits the encoded ``/maps?to=`` link."""
    from friday.core.state import GraphState

    orch = _orchestrator()
    state = GraphState(session_id="m3", user_input="distance to Paris")

    out = await orch.handle(state)

    assert out.response is not None
    assert "/maps?to=Paris" in out.response
