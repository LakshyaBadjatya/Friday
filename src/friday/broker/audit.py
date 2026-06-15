"""Tamper-evident, append-only audit trail (hash-chained).

:class:`HashChainedAudit` persists records as an append-only JSONL ledger where
every entry's ``entry_hash`` is::

    sha256(prev_hash + canonical_json(record))

so each entry cryptographically binds the one before it (a blockchain-style
chain). The first entry links to :data:`GENESIS_HASH`. Because the hash covers
both the predecessor's hash *and* the canonical serialization of the record,
:meth:`HashChainedAudit.verify` walks the chain and detects any tampering —
an in-place edit, a deleted entry, or a forged/inserted entry — returning the
index of the first inconsistent link.

**Redaction.** Before a record is hashed or written, values whose key matches the
sensitive set (``api_key`` / ``token`` / ``secret`` / ``password`` /
``authorization`` — matched case-insensitively as a substring) are replaced with
:data:`REDACTED`. A credential therefore never reaches the ledger on disk, and
the hash is computed over the *redacted* form.

The module imports nothing from :mod:`friday.config` or :mod:`friday.app`; the
ledger location is the single constructor argument. Heavy/optional dependencies
are avoided entirely — only the stdlib is used.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# The hash a chain's first entry links to (64 zero hex chars: an empty SHA-256
# slot). Stable and well-known so an independent verifier can reproduce it.
GENESIS_HASH = "0" * 64

REDACTED = "***REDACTED***"

# Substrings marking a key as sensitive (case-insensitive). Matches the set the
# rest of FRIDAY redacts (api_key/token/secret/password/authorization); kept
# local so this module has no cross-package coupling.
_SENSITIVE_SUBSTRINGS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "authorization",
)


def is_sensitive(key: str) -> bool:
    """Return whether ``key`` names a sensitive value that must be redacted."""
    lowered = key.lower()
    return any(token in lowered for token in _SENSITIVE_SUBSTRINGS)


def redact(record: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of ``record`` with sensitive-keyed values masked.

    Nested mappings are redacted recursively so a secret nested one level down
    (e.g. ``{"headers": {"authorization": "..."}}``) is still masked.
    """
    out: dict[str, Any] = {}
    for key, value in record.items():
        if is_sensitive(key):
            out[key] = REDACTED
        elif isinstance(value, dict):
            out[key] = redact(value)
        else:
            out[key] = value
    return out


def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` to a stable, key-sorted, whitespace-free JSON string.

    Determinism is essential: the same logical record must always produce the
    same bytes so its hash is reproducible regardless of dict insertion order.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _entry_hash(prev_hash: str, record: dict[str, Any]) -> str:
    return hashlib.sha256((prev_hash + canonical_json(record)).encode("utf-8")).hexdigest()


class AuditEntry(BaseModel):
    """One link in the hash chain.

    Attributes:
        index: Zero-based position in the ledger.
        prev_hash: The preceding entry's ``entry_hash`` (or :data:`GENESIS_HASH`).
        entry_hash: ``sha256(prev_hash + canonical_json(record))``.
        record: The (already redacted) audited payload.
    """

    index: int
    prev_hash: str
    entry_hash: str
    record: dict[str, Any] = Field(default_factory=dict)


class HashChainedAudit:
    """An append-only, tamper-evident audit ledger backed by a JSONL file.

    Args:
        path: The ledger file. Parent directories are created on first append.
            An existing file is read on construction so the chain continues from
            wherever it left off (and ``verify`` runs against the persisted tail).
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)

    def entries(self) -> list[AuditEntry]:
        """Return every persisted entry, oldest-first (empty if no ledger yet)."""
        if not self._path.exists():
            return []
        out: list[AuditEntry] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                out.append(AuditEntry.model_validate_json(line))
        return out

    def append(self, record: dict[str, Any]) -> AuditEntry:
        """Redact, hash-chain, and append ``record``; return the written entry.

        The new entry's ``prev_hash`` is the current tail's ``entry_hash`` (or
        :data:`GENESIS_HASH` for the first entry). The redacted record — never
        the raw one — is what is hashed and persisted.
        """
        existing = self.entries()
        prev_hash = existing[-1].entry_hash if existing else GENESIS_HASH
        index = len(existing)

        clean = redact(record)
        entry = AuditEntry(
            index=index,
            prev_hash=prev_hash,
            entry_hash=_entry_hash(prev_hash, clean),
            record=clean,
        )

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")
        return entry

    def verify(self) -> tuple[bool, int | None]:
        """Walk the chain; return ``(ok, broken_at)``.

        ``ok`` is ``True`` iff every entry's ``entry_hash`` recomputes from its
        ``prev_hash`` and ``record`` *and* every ``prev_hash`` matches the actual
        predecessor's ``entry_hash`` (the first entry must link to
        :data:`GENESIS_HASH`). On the first inconsistency, returns
        ``(False, index)`` naming the offending entry; an empty or fully intact
        ledger returns ``(True, None)``.
        """
        prev_hash = GENESIS_HASH
        for expected_index, entry in enumerate(self.entries()):
            if entry.index != expected_index:
                return False, expected_index
            if entry.prev_hash != prev_hash:
                return False, expected_index
            if entry.entry_hash != _entry_hash(entry.prev_hash, entry.record):
                return False, expected_index
            prev_hash = entry.entry_hash
        return True, None
