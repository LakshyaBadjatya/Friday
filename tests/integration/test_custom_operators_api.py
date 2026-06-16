# © Lakshya Badjatya — Author
"""Integration tests for the Wave-0 custom-operators roster wiring in :mod:`friday.app`.

Fully offline (``llm_provider="fake"``, ``memory_db_path=":memory:"``): builds the
real runtime graph via :func:`friday.app.build_runtime` (and the FastAPI app via
:func:`friday.app.create_app`) and asserts:

* With a VALID ``FRIDAY_CUSTOM_OPERATORS`` entry, the parsed custom persona is
  merged into the always-on roster — ``GET /roster`` lists it (count 10) with the
  configured namespace + scope, alongside the nine built-ins.
* A MALFORMED entry does NOT crash app/runtime construction: boot falls back to the
  unmodified built-in roster (count 9), the warning is logged and the bad config
  skipped.
* The default (empty) leaves the roster exactly as the built-in nine.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from friday import app as app_mod
from friday.app import build_runtime, create_app
from friday.config import Settings

# The canonical built-in roster code-names.
_BUILTIN_NAMES = {
    "FRIDAY",
    "EDITH",
    "ORACLE",
    "GECKO",
    "KAREN",
    "VERONICA",
    "JOCASTA",
    "VISION",
    "FORGE",
}


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "llm_provider": "fake",
        "memory_db_path": ":memory:",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _app(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> FastAPI:
    monkeypatch.setattr(app_mod, "get_settings", lambda: settings)
    return create_app()


# --------------------------------------------------------------------------- #
# Default: empty custom_operators -> the roster is the built-in nine
# --------------------------------------------------------------------------- #
def test_default_no_custom_operators() -> None:
    runtime = build_runtime(_settings())
    assert set(runtime.roster.names()) == _BUILTIN_NAMES


# --------------------------------------------------------------------------- #
# Valid custom operator is merged + listed
# --------------------------------------------------------------------------- #
def test_runtime_merges_valid_custom_operator() -> None:
    runtime = build_runtime(
        _settings(
            custom_operators=[
                "SCOUT|Field Recon|web_search,notify|recon|be scout",
            ]
        )
    )
    names = set(runtime.roster.names())
    assert names == _BUILTIN_NAMES | {"SCOUT"}


def test_get_roster_includes_valid_custom_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        custom_operators=["SCOUT|Field Recon|web_search,notify|recon|be scout"]
    )
    app = _app(monkeypatch, settings)
    with TestClient(app) as client:
        resp = client.get("/roster")
        assert resp.status_code == 200
        body = resp.json()
        # Nine built-ins + one custom.
        assert body["count"] == 10
        by_name = {p["name"]: p for p in body["personas"]}
        assert "SCOUT" in by_name
        scout = by_name["SCOUT"]
        assert scout["title"] == "Field Recon"
        assert scout["namespace"] == "recon"
        assert "web_search" in scout["scope"]
        assert "notify" in scout["scope"]


# --------------------------------------------------------------------------- #
# Malformed config never crashes boot — falls back to the built-in roster
# --------------------------------------------------------------------------- #
def test_malformed_custom_operator_does_not_crash_runtime() -> None:
    # Only four fields (NAME|Title|tools|namespace) — the parser raises, the
    # wiring logs + skips, and the roster stays at the built-in nine.
    runtime = build_runtime(
        _settings(custom_operators=["BAD|only|three|fields"])
    )
    assert set(runtime.roster.names()) == _BUILTIN_NAMES


def test_malformed_custom_operator_does_not_crash_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(custom_operators=["BAD|only|three|fields"])
    app = _app(monkeypatch, settings)
    with TestClient(app) as client:
        resp = client.get("/roster")
        assert resp.status_code == 200
        body = resp.json()
        # The malformed entry was skipped; the default nine personas remain.
        assert body["count"] == 9
        assert {p["name"] for p in body["personas"]} == _BUILTIN_NAMES


# --------------------------------------------------------------------------- #
# A custom whose name collides with a built-in is dropped (built-ins win)
# --------------------------------------------------------------------------- #
def test_custom_operator_name_collision_is_dropped() -> None:
    runtime = build_runtime(
        _settings(
            custom_operators=["gecko|Impostor|web_search|gecko|shadow the builtin"]
        )
    )
    # The collision is dropped; the roster is unchanged (still nine, with the real
    # GECKO's built-in title intact).
    assert set(runtime.roster.names()) == _BUILTIN_NAMES
    gecko = runtime.roster.get("GECKO")
    assert gecko.title == "Finance & Markets"
