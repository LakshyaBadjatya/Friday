"""Integration tests for the Stage-1 security spine wiring in :mod:`friday.app`.

Fully offline (``llm_provider="fake"``, ``memory_db_path=":memory:"``): builds the
real runtime graph via :func:`friday.app.build_runtime` (and the FastAPI app via
:func:`friday.app.create_app`) and asserts the additive security spine the wiring
adds:

* **Hash-chained tool-call ledger.** Every tool the shared registry executes
  appends exactly one tamper-evident, hash-chained record to the on-disk
  ``audit_ledger_path`` (additive — the in-memory observability ``AuditLog`` the
  ``/admin/audit`` view reads is untouched), and ``GET /admin/audit/verify``
  returns ``{"ok": true, "broken_at": null}`` for an intact chain.
* **Tamper detection.** Corrupting a persisted ledger row makes
  ``GET /admin/audit/verify`` report ``ok=false`` with the offending index in
  ``broken_at``.
* **Startup secret self-check.** With ``enable_secret_self_check`` on and a
  planted fake secret under the scanned repo root, startup LOGS a WARNING per
  finding (captured via ``caplog``) and NEVER refuses to boot.
* **Broker is opt-in.** ``enable_broker`` defaults off, so the runtime carries a
  constructed :class:`~friday.broker.Broker` but the orchestrator's dispatch path
  is unchanged (the broker is not interposed) — existing tests stay green.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from friday import app as app_mod
from friday.app import build_runtime, create_app
from friday.broker import Broker, HashChainedAudit
from friday.config import Settings


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    """Offline settings with the ledger pinned under ``tmp_path``.

    ``":memory:"`` keeps every store ephemeral; the ledger path is a real file in
    ``tmp_path`` so a test can read/tamper it without touching ``data/``.
    """
    base: dict[str, object] = {
        "_env_file": None,
        "llm_provider": "fake",
        "memory_db_path": ":memory:",
        "audit_ledger_path": str(tmp_path / "audit.jsonl"),
        "enable_secret_self_check": False,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _app_with_settings(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> FastAPI:
    """Build the app with ``get_settings`` pinned to ``settings``.

    Patches both the app module's imported ``get_settings`` and the cached factory
    so ``create_app``'s eager runtime install reads the pinned (ledger-under-
    ``tmp_path``) settings — never the developer's real ``data/`` path.
    """
    monkeypatch.setattr(app_mod, "get_settings", lambda: settings)
    return create_app()


# --------------------------------------------------------------------------- #
# Config defaults
# --------------------------------------------------------------------------- #
def test_security_spine_settings_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.audit_ledger_path == "data/audit.jsonl"
    assert settings.secret_vault == "env"
    assert settings.enable_secret_self_check is True
    assert settings.enable_broker is False


# --------------------------------------------------------------------------- #
# Hash-chained ledger per tool execution
# --------------------------------------------------------------------------- #
async def test_tool_execution_appends_hash_chained_row(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    runtime = build_runtime(settings)
    ledger_path = Path(settings.audit_ledger_path)

    assert not ledger_path.exists()

    # Execute a registered, allow-listed, read-only tool through the shared
    # registry (the same path the orchestrator/agents use).
    result = await runtime.registry.execute(
        "web_search",
        {"query": "vector databases"},
        allowed_tools={"web_search"},
    )
    assert result is not None

    # One hash-chained record landed and the chain verifies.
    audit = HashChainedAudit(ledger_path)
    entries = audit.entries()
    assert len(entries) == 1
    assert entries[0].record["tool"] == "web_search"
    ok, broken_at = audit.verify()
    assert ok is True
    assert broken_at is None

    # The in-memory observability AuditLog (the /admin/audit system-of-record for
    # the dashboard) ALSO recorded the row — the ledger is purely additive.
    assert len(runtime.audit.recent(10)) == 1


async def test_broker_constructed_and_exposed_on_runtime(tmp_path: Path) -> None:
    runtime = build_runtime(_settings(tmp_path))
    assert isinstance(runtime.broker, Broker)
    assert isinstance(runtime.hash_audit, HashChainedAudit)


# --------------------------------------------------------------------------- #
# /admin/audit/verify
# --------------------------------------------------------------------------- #
def test_admin_audit_verify_ok_after_tool_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import asyncio

    settings = _settings(tmp_path)
    app = _app_with_settings(monkeypatch, settings)
    with TestClient(app) as client:
        # Before any tool call the (empty) chain still verifies.
        empty = client.get("/admin/audit/verify")
        assert empty.status_code == 200
        assert empty.json() == {"ok": True, "broken_at": None}

        # Run a tool through the shared registry, then re-verify.
        asyncio.run(
            app.state.registry.execute(
                "web_search",
                {"query": "q"},
                allowed_tools={"web_search"},
            )
        )
        verified = client.get("/admin/audit/verify")
        assert verified.status_code == 200
        assert verified.json() == {"ok": True, "broken_at": None}


def test_admin_audit_verify_detects_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import asyncio

    settings = _settings(tmp_path)
    app = _app_with_settings(monkeypatch, settings)
    ledger_path = Path(settings.audit_ledger_path)
    with TestClient(app) as client:
        asyncio.run(
            app.state.registry.execute(
                "web_search",
                {"query": "q"},
                allowed_tools={"web_search"},
            )
        )
        # Intact chain first.
        assert client.get("/admin/audit/verify").json()["ok"] is True

        # Tamper: rewrite the first row's record in place (its persisted
        # ``entry_hash`` no longer matches the recomputed hash of the mutated
        # record), which ``verify`` must flag.
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
        assert lines
        row = json.loads(lines[0])
        row["record"]["tool"] = "totally_different_tool"
        lines[0] = json.dumps(row)
        ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        broken = client.get("/admin/audit/verify").json()
        assert broken["ok"] is False
        assert broken["broken_at"] == 0


# --------------------------------------------------------------------------- #
# Startup secret self-check (warn-only, never blocks boot)
# --------------------------------------------------------------------------- #
def test_secret_self_check_warns_and_never_blocks_boot(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Plant a fake committed secret under a repo root the self-check scans.
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "leak.py").write_text(
        'NVIDIA_KEY = "nvapi-AAAAAAAAAAAAAAAAAAAAAAAAAAAA"\n',
        encoding="utf-8",
    )
    settings = _settings(tmp_path, enable_secret_self_check=True)

    with caplog.at_level("WARNING"):
        # Boot does not raise even with a planted secret.
        runtime = build_runtime(settings, repo_root=str(repo_root))
    assert runtime is not None
    # A WARNING was logged for the planted finding. The file path / kind ride on
    # the record's structured ``extra`` fields, not the format-string message.
    messages = " ".join(rec.getMessage() for rec in caplog.records)
    assert "secret" in messages.lower()
    assert any(
        getattr(rec, "file", "").endswith("leak.py")
        and getattr(rec, "kind", None) == "nvapi"
        for rec in caplog.records
    )


def test_secret_self_check_disabled_logs_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "leak.py").write_text(
        'KEY = "nvapi-BBBBBBBBBBBBBBBBBBBBBBBBBBBB"\n',
        encoding="utf-8",
    )
    settings = _settings(tmp_path, enable_secret_self_check=False)
    with caplog.at_level("WARNING"):
        build_runtime(settings, repo_root=str(repo_root))
    findings = [
        rec for rec in caplog.records if "plaintext secret" in rec.getMessage().lower()
    ]
    assert findings == []


# --------------------------------------------------------------------------- #
# enable_broker defaults off → dispatch unchanged
# --------------------------------------------------------------------------- #
def test_enable_broker_defaults_off_dispatch_unchanged(tmp_path: Path) -> None:
    runtime = build_runtime(_settings(tmp_path))
    # The broker is constructed and exposed, but with the flag off it is NOT
    # interposed into the orchestrator's dispatch path (which stays the plain
    # registry path the existing tests exercise).
    assert runtime.broker is not None
    assert getattr(runtime.orchestrator, "_broker", None) is None
