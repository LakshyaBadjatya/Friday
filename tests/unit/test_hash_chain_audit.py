"""Unit tests for :class:`friday.broker.HashChainedAudit`.

Covers the tamper-evident hash chain: each entry's hash binds the previous
hash and the canonical JSON of the record, so any in-place edit, deletion, or
insertion is detected by :meth:`HashChainedAudit.verify`. Also covers the
sensitive-key redaction applied to values before persistence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from friday.broker import HashChainedAudit
from friday.broker.audit import GENESIS_HASH, canonical_json


def _audit(tmp_path: Path) -> HashChainedAudit:
    return HashChainedAudit(tmp_path / "audit.jsonl")


def test_append_returns_entry_with_chained_hash(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    first = audit.append({"tool": "ping", "ok": True})
    assert first.index == 0
    assert first.prev_hash == GENESIS_HASH
    # The stored hash is sha256(prev_hash + canonical_json(record)).
    expected = hashlib.sha256(
        (GENESIS_HASH + canonical_json(first.record)).encode("utf-8")
    ).hexdigest()
    assert first.entry_hash == expected


def test_second_entry_links_to_first(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    first = audit.append({"a": 1})
    second = audit.append({"b": 2})
    assert second.index == 1
    assert second.prev_hash == first.entry_hash


def test_verify_ok_on_untampered_chain(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    audit.append({"a": 1})
    audit.append({"b": 2})
    audit.append({"c": 3})
    ok, broken_at = audit.verify()
    assert ok is True
    assert broken_at is None


def test_verify_ok_on_empty_chain(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    ok, broken_at = audit.verify()
    assert ok is True
    assert broken_at is None


def test_verify_detects_in_place_tampering(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    audit = HashChainedAudit(path)
    audit.append({"a": 1})
    audit.append({"b": 2})
    audit.append({"c": 3})

    # Mutate the record payload of the middle entry while leaving its stored
    # hash untouched -> the recomputed hash no longer matches.
    lines = path.read_text(encoding="utf-8").splitlines()
    middle = json.loads(lines[1])
    middle["record"]["b"] = 999
    lines[1] = json.dumps(middle)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, broken_at = HashChainedAudit(path).verify()
    assert ok is False
    assert broken_at == 1


def test_verify_detects_deleted_entry(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    audit = HashChainedAudit(path)
    audit.append({"a": 1})
    audit.append({"b": 2})
    audit.append({"c": 3})

    # Drop the middle entry: the survivor's prev_hash now points at a hash that
    # no longer precedes it.
    lines = path.read_text(encoding="utf-8").splitlines()
    del lines[1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, broken_at = HashChainedAudit(path).verify()
    assert ok is False
    assert broken_at == 1


def test_verify_detects_inserted_entry(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    audit = HashChainedAudit(path)
    audit.append({"a": 1})
    audit.append({"b": 2})

    # Forge an entry whose own hash is internally consistent but whose prev_hash
    # does not match the real predecessor -> insertion is detected at its index.
    lines = path.read_text(encoding="utf-8").splitlines()
    forged_record = {"forged": True}
    forged_prev = "0" * 64
    forged_hash = hashlib.sha256(
        (forged_prev + canonical_json(forged_record)).encode("utf-8")
    ).hexdigest()
    forged = {
        "index": 1,
        "prev_hash": forged_prev,
        "entry_hash": forged_hash,
        "record": forged_record,
    }
    lines.insert(1, json.dumps(forged))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, broken_at = HashChainedAudit(path).verify()
    assert ok is False
    assert broken_at == 1


def test_append_redacts_sensitive_values(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    entry = audit.append(
        {
            "tool": "send",
            "api_key": "sk-LIVE-123",
            "token": "abc",
            "secret": "s",
            "password": "p",
            "authorization": "Bearer x",
            "note": "kept",
        }
    )
    for key in ("api_key", "token", "secret", "password", "authorization"):
        assert entry.record[key] != "sk-LIVE-123"
        assert entry.record[key] != "abc"
        assert "REDACT" in str(entry.record[key]).upper()
    assert entry.record["note"] == "kept"
    assert entry.record["tool"] == "send"


def test_secret_value_never_on_disk(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    audit = HashChainedAudit(path)
    audit.append({"api_key": "TOP-SECRET-VALUE", "tool": "x"})
    raw = path.read_text(encoding="utf-8")
    assert "TOP-SECRET-VALUE" not in raw


def test_persisted_chain_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    a = HashChainedAudit(path)
    a.append({"a": 1})
    a.append({"b": 2})

    # A fresh instance over the same path continues the chain and still verifies.
    b = HashChainedAudit(path)
    third = b.append({"c": 3})
    assert third.index == 2
    ok, broken_at = b.verify()
    assert ok is True
    assert broken_at is None


def test_canonical_json_is_key_order_independent() -> None:
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


def test_entries_returns_all_records(tmp_path: Path) -> None:
    audit = _audit(tmp_path)
    audit.append({"a": 1})
    audit.append({"b": 2})
    entries = audit.entries()
    assert [e.index for e in entries] == [0, 1]
