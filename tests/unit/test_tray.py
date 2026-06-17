# © Lakshya Badjatya — Author
"""Unit tests for the system-tray seam (flag-gated, lazy, with a fake fallback)."""

from __future__ import annotations

from friday.cli import _handle_tray, build_parser
from friday.config import Settings
from friday.desktop.tray import FakeTray, TrayController, build_tray


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, llm_provider="fake", **overrides)  # type: ignore[arg-type]


def test_fake_tray_records_notifications_and_run_is_noop() -> None:
    tray = FakeTray("FRIDAY", "http://x/hud")
    tray.notify("Heads up", "build done")
    assert tray.notifications == [("Heads up", "build done")]
    assert tray.ran is False
    tray.run()
    assert tray.ran is True


def test_fake_tray_satisfies_the_protocol() -> None:
    assert isinstance(FakeTray(), TrayController)


def test_build_tray_none_when_flag_off() -> None:
    assert build_tray(_settings()) is None


def test_build_tray_degrades_to_fake_when_backend_missing() -> None:
    # pystray/PIL are not installed here, so an enabled tray falls back to FakeTray
    # rather than crashing.
    tray = build_tray(_settings(enable_tray=True))
    assert isinstance(tray, FakeTray)


def test_cli_tray_disabled_returns_1() -> None:
    from friday.config import get_settings

    get_settings.cache_clear()
    args = build_parser().parse_args(["tray"])
    assert _handle_tray(args) == 1
    get_settings.cache_clear()


def test_cli_registers_tray_subcommand() -> None:
    args = build_parser().parse_args(["tray"])
    assert args.func is _handle_tray
