"""The metadata index: the protocol, and a local-first SQLite implementation.

The document body is stored as one JSON blob so the model can grow without a
migration, while the columns that get filtered on (owner, space, subject, kind,
privacy, created_at) are promoted to real columns with indexes.

Design rules mirror :mod:`friday.study.store`: parametrized SQL only, idempotent
schema, connection-per-call for file paths and one shared connection for
``":memory:"``.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Protocol, runtime_checkable

from friday.vault.models import Item, Note, Privacy, Solve

_MEMORY_PATH = ":memory:"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id TEXT NOT NULL,
    owner_uid TEXT NOT NULL,
    space TEXT NOT NULL,
    privacy TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT '',
    subject TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    doc TEXT NOT NULL,
    PRIMARY KEY (owner_uid, id)
);
CREATE INDEX IF NOT EXISTS items_owner_created ON items (owner_uid, created_at DESC);
CREATE INDEX IF NOT EXISTS items_subject ON items (owner_uid, subject);
CREATE TABLE IF NOT EXISTS solves (
    id TEXT PRIMARY KEY,
    doc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notes (
    id TEXT NOT NULL,
    owner_uid TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    doc TEXT NOT NULL,
    PRIMARY KEY (owner_uid, id)
);
"""


@runtime_checkable
class VaultIndex(Protocol):
    """The index contract both the SQLite and Firestore backends satisfy."""

    def put_item(self, item: Item) -> None: ...

    def get_item(self, owner_uid: str, item_id: str) -> Item | None: ...

    def list_items(
        self,
        owner_uid: str,
        *,
        subject: str = "",
        kind: str = "",
        space: str = "",
        include_locked: bool = False,
        limit: int = 100,
    ) -> list[Item]: ...

    def delete_item(self, owner_uid: str, item_id: str) -> bool: ...

    def put_solve(self, solve: Solve) -> None: ...

    def get_solve(self, solve_id: str) -> Solve | None: ...

    def put_note(self, note: Note) -> None: ...

    def get_note(self, owner_uid: str, note_id: str) -> Note | None: ...

    def list_notes(self, owner_uid: str, *, limit: int = 100) -> list[Note]: ...

    def total_bytes(self, owner_uid: str) -> int: ...


class SQLiteVaultIndex:
    """Local-first, zero-server index. The test and offline backend."""

    def __init__(self, path: str = _MEMORY_PATH) -> None:
        self._path = path
        self._shared: sqlite3.Connection | None = None
        if path == _MEMORY_PATH:
            self._shared = sqlite3.connect(path, check_same_thread=False)
        conn = self._conn()
        try:
            conn.executescript(_SCHEMA)
        finally:
            self._close(conn)

    def _conn(self) -> sqlite3.Connection:
        if self._shared is not None:
            return self._shared
        return sqlite3.connect(self._path)

    def _close(self, conn: sqlite3.Connection) -> None:
        """Close a per-call connection; leave the shared (memory) one open."""
        if conn is not self._shared:
            conn.close()

    # ---------------------------------------------------------------- items
    def put_item(self, item: Item) -> None:
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO items (id, owner_uid, space, privacy, kind, subject, "
                "created_at, doc) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(owner_uid, id) DO UPDATE SET space=excluded.space, "
                "privacy=excluded.privacy, kind=excluded.kind, subject=excluded.subject, "
                "created_at=excluded.created_at, doc=excluded.doc",
                (
                    item.id,
                    item.owner_uid,
                    item.space,
                    str(item.privacy),
                    item.classification.kind,
                    item.classification.subject,
                    item.created_at,
                    item.model_dump_json(),
                ),
            )
            conn.commit()
        finally:
            self._close(conn)

    def get_item(self, owner_uid: str, item_id: str) -> Item | None:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT doc FROM items WHERE owner_uid = ? AND id = ?",
                (owner_uid, item_id),
            ).fetchone()
        finally:
            self._close(conn)
        return Item.model_validate_json(row[0]) if row else None

    def list_items(
        self,
        owner_uid: str,
        *,
        subject: str = "",
        kind: str = "",
        space: str = "",
        include_locked: bool = False,
        limit: int = 100,
    ) -> list[Item]:
        sql = "SELECT doc FROM items WHERE owner_uid = ?"
        args: list[object] = [owner_uid]
        if subject:
            sql += " AND subject = ?"
            args.append(subject)
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        if space:
            sql += " AND space = ?"
            args.append(space)
        if not include_locked:
            sql += " AND privacy != ?"
            args.append(str(Privacy.LOCKED))
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        conn = self._conn()
        try:
            rows = conn.execute(sql, args).fetchall()
        finally:
            self._close(conn)
        return [Item.model_validate_json(r[0]) for r in rows]

    def delete_item(self, owner_uid: str, item_id: str) -> bool:
        conn = self._conn()
        try:
            cur = conn.execute(
                "DELETE FROM items WHERE owner_uid = ? AND id = ?", (owner_uid, item_id)
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            self._close(conn)

    # --------------------------------------------------------------- solves
    def put_solve(self, solve: Solve) -> None:
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO solves (id, doc) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET doc=excluded.doc",
                (solve.id, solve.model_dump_json()),
            )
            conn.commit()
        finally:
            self._close(conn)

    def get_solve(self, solve_id: str) -> Solve | None:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT doc FROM solves WHERE id = ?", (solve_id,)
            ).fetchone()
        finally:
            self._close(conn)
        return Solve.model_validate_json(row[0]) if row else None

    # ---------------------------------------------------------------- notes
    def put_note(self, note: Note) -> None:
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO notes (id, owner_uid, subject, created_at, doc) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(owner_uid, id) DO UPDATE SET "
                "subject=excluded.subject, created_at=excluded.created_at, "
                "doc=excluded.doc",
                (
                    note.id,
                    note.owner_uid,
                    note.subject,
                    note.created_at,
                    note.model_dump_json(),
                ),
            )
            conn.commit()
        finally:
            self._close(conn)

    def get_note(self, owner_uid: str, note_id: str) -> Note | None:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT doc FROM notes WHERE owner_uid = ? AND id = ?",
                (owner_uid, note_id),
            ).fetchone()
        finally:
            self._close(conn)
        return Note.model_validate_json(row[0]) if row else None

    def list_notes(self, owner_uid: str, *, limit: int = 100) -> list[Note]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT doc FROM notes WHERE owner_uid = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (owner_uid, limit),
            ).fetchall()
        finally:
            self._close(conn)
        return [Note.model_validate_json(r[0]) for r in rows]

    # ---------------------------------------------------------------- quota
    def total_bytes(self, owner_uid: str) -> int:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT doc FROM items WHERE owner_uid = ?", (owner_uid,)
            ).fetchall()
        finally:
            self._close(conn)
        total = 0
        for (doc,) in rows:
            asset = json.loads(doc).get("cloudinary")
            if asset:
                total += int(asset.get("bytes", 0))
        return total
