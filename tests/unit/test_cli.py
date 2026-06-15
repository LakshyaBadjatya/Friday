"""Unit tests for the ``friday`` command-line interface (:mod:`friday.cli`).

The CLI is the operator's front door to the running system: it serves the app,
reads/writes secrets through the configured vault, verifies the tamper-evident
audit ledger, lists the persona roster, and reports the version. These tests pin
that contract WITHOUT any network or process side effects:

* :func:`friday.cli.build_parser` parses every subcommand into a namespace whose
  fields drive the handlers, so argument wiring can be checked in isolation
  (no handler runs, ``uvicorn`` is never imported, the server never starts);
* ``audit verify`` exits ``0`` on an intact hash-chained ledger and non-zero on a
  tampered one (a real :class:`friday.broker.HashChainedAudit` over a tmp file);
* ``roster`` prints all nine canonical persona names;
* ``secrets set``/``get`` round-trip through an in-process vault;
* ``version`` prints the package version;
* ``serve`` is dispatched lazily — calling ``main(["serve", ...])`` with a stub
  in place of ``uvicorn.run`` proves the handler builds the right call and that
  importing the module never pulls ``uvicorn`` in.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from friday.broker import HashChainedAudit
from friday.cli import build_parser, main
from friday.roster import ROSTER


def test_build_parser_serve_defaults() -> None:
    """``serve`` defaults to 127.0.0.1:8000 and carries a callable handler."""
    parser = build_parser()
    args = parser.parse_args(["serve"])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert callable(args.func)


def test_build_parser_serve_host_port() -> None:
    """``serve --host --port`` are parsed onto the namespace."""
    parser = build_parser()
    args = parser.parse_args(["serve", "--host", "0.0.0.0", "--port", "9001"])
    assert args.host == "0.0.0.0"
    assert args.port == 9001


def test_build_parser_secrets_set_get() -> None:
    """``secrets set``/``get`` parse the name (and value for set)."""
    parser = build_parser()
    set_args = parser.parse_args(["secrets", "set", "nvidia_api_key", "nv-123"])
    assert set_args.secrets_command == "set"
    assert set_args.name == "nvidia_api_key"
    assert set_args.value == "nv-123"

    get_args = parser.parse_args(["secrets", "get", "nvidia_api_key"])
    assert get_args.secrets_command == "get"
    assert get_args.name == "nvidia_api_key"


def test_build_parser_audit_verify() -> None:
    """``audit verify`` parses to a callable handler."""
    parser = build_parser()
    args = parser.parse_args(["audit", "verify"])
    assert args.audit_command == "verify"
    assert callable(args.func)


def test_build_parser_roster_and_version() -> None:
    """``roster`` and ``version`` each parse to a callable handler."""
    parser = build_parser()
    for command in (["roster"], ["version"]):
        args = parser.parse_args(command)
        assert callable(args.func)


def test_no_subcommand_prints_help_and_errors(capsys: pytest.CaptureFixture[str]) -> None:
    """Invoking with no subcommand prints usage and returns a non-zero code."""
    code = main([])
    captured = capsys.readouterr()
    assert code != 0
    assert "usage" in (captured.out + captured.err).lower()


def test_version_prints_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    """``version`` prints the installed package version and exits 0."""
    from importlib.metadata import version as pkg_version

    code = main(["version"])
    captured = capsys.readouterr()
    assert code == 0
    assert pkg_version("friday") in captured.out


def test_roster_prints_all_nine_persona_names(capsys: pytest.CaptureFixture[str]) -> None:
    """``roster`` prints every canonical persona name (the prime + 8)."""
    code = main(["roster"])
    captured = capsys.readouterr()
    assert code == 0
    names = ROSTER.names()
    assert len(names) == 9
    for name in names:
        assert name in captured.out


def test_audit_verify_intact_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``audit verify`` on an intact ledger reports ok and exits 0."""
    ledger_path = tmp_path / "audit.jsonl"
    ledger = HashChainedAudit(ledger_path)
    ledger.append({"action": "first"})
    ledger.append({"action": "second"})

    monkeypatch.setenv("FRIDAY_AUDIT_LEDGER_PATH", str(ledger_path))
    _clear_settings_cache()

    code = main(["audit", "verify"])
    captured = capsys.readouterr()
    assert code == 0
    assert "ok" in captured.out.lower()


def test_audit_verify_tampered_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``audit verify`` on a tampered ledger reports the break and exits non-zero."""
    ledger_path = tmp_path / "audit.jsonl"
    ledger = HashChainedAudit(ledger_path)
    ledger.append({"action": "first"})
    ledger.append({"action": "second"})

    # Tamper: rewrite the first record's payload in place. The stored entry_hash
    # no longer recomputes, so verify() must flag the chain as broken.
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("first", "FORGED")
    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    monkeypatch.setenv("FRIDAY_AUDIT_LEDGER_PATH", str(ledger_path))
    _clear_settings_cache()

    code = main(["audit", "verify"])
    captured = capsys.readouterr()
    assert code != 0
    assert "broken" in captured.out.lower()


def test_secrets_set_then_get_round_trip(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``secrets set`` then ``get`` round-trips through the env vault backend."""
    monkeypatch.setenv("FRIDAY_SECRET_VAULT", "env")
    monkeypatch.delenv("FRIDAY_CLI_TEST_KEY", raising=False)
    _clear_settings_cache()

    set_code = main(["secrets", "set", "cli_test_key", "shhh"])
    assert set_code == 0
    # EnvVault writes FRIDAY_<NAME> uppercased into the process environment.
    assert sys.modules["os"].environ["FRIDAY_CLI_TEST_KEY"] == "shhh"

    get_code = main(["secrets", "get", "cli_test_key"])
    captured = capsys.readouterr()
    assert get_code == 0
    assert "shhh" in captured.out


def test_secrets_get_missing_exits_nonzero(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``secrets get`` for an unset secret reports it and exits non-zero."""
    monkeypatch.setenv("FRIDAY_SECRET_VAULT", "memory")
    _clear_settings_cache()

    code = main(["secrets", "get", "definitely_not_set"])
    captured = capsys.readouterr()
    assert code != 0
    assert "not set" in captured.out.lower()


def test_serve_dispatches_to_uvicorn_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    """``serve`` calls ``uvicorn.run`` with the factory + host/port; no real start.

    A fake ``uvicorn`` module is installed before dispatch so nothing real binds a
    socket. The handler must import ``uvicorn`` LAZILY (inside the handler, not at
    module import time), so installing the stub here is sufficient to intercept it.
    """
    calls: list[dict[str, Any]] = []

    class _FakeUvicorn:
        def run(self, app: str, **kwargs: Any) -> None:
            calls.append({"app": app, **kwargs})

    monkeypatch.setitem(sys.modules, "uvicorn", _FakeUvicorn())

    code = main(["serve", "--host", "127.0.0.1", "--port", "8123"])
    assert code == 0
    assert len(calls) == 1
    call = calls[0]
    assert call["app"] == "friday.app:create_app"
    assert call["factory"] is True
    assert call["host"] == "127.0.0.1"
    assert call["port"] == 8123


def _clear_settings_cache() -> None:
    """Drop the lru_cached settings so env overrides above take effect."""
    from friday.config import get_settings

    get_settings.cache_clear()
