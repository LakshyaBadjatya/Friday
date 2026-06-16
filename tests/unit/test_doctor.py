# © Lakshya Badjatya — Author
"""Unit tests for `friday doctor` (the health self-test + CLI wiring)."""

from __future__ import annotations

from friday.cli import _handle_doctor, build_parser
from friday.config import Settings
from friday.system.doctor import run_doctor


def _offline_settings() -> Settings:
    return Settings(_env_file=None, llm_provider="fake")


def test_offline_build_is_healthy() -> None:
    report = run_doctor(_offline_settings())
    assert report.ok is True
    names = {c.name for c in report.checks}
    assert {"providers", "llm_provider", "memory_store", "embeddings"} <= names
    providers = next(c for c in report.checks if c.name == "providers")
    assert "no keys needed" in providers.detail  # fake LLM needs none


def test_tampered_ledger_fails_the_report() -> None:
    report = run_doctor(_offline_settings(), audit_verify=lambda: (False, 3))
    assert report.ok is False
    audit = next(c for c in report.checks if c.name == "audit_ledger")
    assert audit.ok is False
    assert "broken at entry 3" in audit.detail
    assert "ISSUES FOUND" in report.render()


def test_intact_ledger_passes() -> None:
    report = run_doctor(_offline_settings(), audit_verify=lambda: (True, None))
    assert report.ok is True
    assert "all checks passed" in report.render()


def test_cli_registers_doctor_subcommand() -> None:
    args = build_parser().parse_args(["doctor"])
    assert args.func is _handle_doctor
