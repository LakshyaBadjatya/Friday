# © Lakshya Badjatya — Author
"""Unit tests for the encrypted, authenticated backup (and its CLI wiring)."""

from __future__ import annotations

from pathlib import Path

import pytest

from friday.cli import _handle_backup, build_parser
from friday.system.backup import BackupError, create_backup, restore_backup


def _make_files(root: Path) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    db = root / "memory.db"
    ledger = root / "audit.log"
    db.write_bytes(b"the durable memory database bytes")
    ledger.write_text("ledger-entry-1\nledger-entry-2\n", encoding="utf-8")
    return [db, ledger]


def test_round_trip_restores_identical_bytes(tmp_path: Path) -> None:
    files = _make_files(tmp_path / "src")
    blob = create_backup(files, "correct horse battery staple")

    dest = tmp_path / "restored"
    names = restore_backup(blob, "correct horse battery staple", dest)
    assert set(names) == {"memory.db", "audit.log"}
    assert (dest / "memory.db").read_bytes() == b"the durable memory database bytes"
    assert (dest / "audit.log").read_text(encoding="utf-8").startswith("ledger-entry-1")


def test_wrong_passphrase_fails_closed(tmp_path: Path) -> None:
    files = _make_files(tmp_path / "src")
    blob = create_backup(files, "right-key")
    with pytest.raises(BackupError, match="wrong passphrase or tampered"):
        restore_backup(blob, "WRONG-key", tmp_path / "out")
    assert not (tmp_path / "out" / "memory.db").exists()  # nothing written


def test_tampered_blob_is_detected(tmp_path: Path) -> None:
    files = _make_files(tmp_path / "src")
    blob = bytearray(create_backup(files, "k"))
    blob[40] ^= 0x01  # flip a ciphertext bit
    with pytest.raises(BackupError, match="wrong passphrase or tampered"):
        restore_backup(bytes(blob), "k", tmp_path / "out")


def test_empty_passphrase_rejected(tmp_path: Path) -> None:
    files = _make_files(tmp_path / "src")
    with pytest.raises(BackupError, match="non-empty passphrase"):
        create_backup(files, "")


def test_no_existing_files_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(BackupError, match="nothing to back up"):
        create_backup([tmp_path / "does-not-exist.db"], "k")


def test_short_blob_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(BackupError, match="too short"):
        restore_backup(b"tiny", "k", tmp_path / "out")


def test_cli_create_round_trips_via_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Point settings at real files so the create handler has something to bundle.
    db = tmp_path / "mem.db"
    ledger = tmp_path / "audit.log"
    db.write_bytes(b"db-bytes")
    ledger.write_text("entry\n", encoding="utf-8")
    monkeypatch.setenv("FRIDAY_MEMORY_DB_PATH", str(db))
    monkeypatch.setenv("FRIDAY_AUDIT_LEDGER_PATH", str(ledger))
    monkeypatch.setenv("FRIDAY_BACKUP_KEY", "cli-key")
    from friday.config import get_settings

    get_settings.cache_clear()

    out = tmp_path / "backup.fbk"
    create_args = build_parser().parse_args(["backup", "create", str(out)])
    assert _handle_backup(create_args) == 0
    assert out.exists()

    dest = tmp_path / "restored"
    restore_args = build_parser().parse_args(["backup", "restore", str(out), str(dest)])
    assert _handle_backup(restore_args) == 0
    assert (dest / "mem.db").read_bytes() == b"db-bytes"
    get_settings.cache_clear()


def test_cli_missing_passphrase_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FRIDAY_BACKUP_KEY", raising=False)
    args = build_parser().parse_args(["backup", "create", str(tmp_path / "x.fbk")])
    assert _handle_backup(args) == 2
