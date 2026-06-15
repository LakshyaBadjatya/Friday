"""Local-first protocol store behind a SQLite backend (Tier 1 protocols).

FRIDAY's durable named-routine layer: protocols the assistant has been taught,
each an ordered list of registered tool calls fired by one trigger. The concrete,
gate-required backend is :class:`SQLiteProtocolStore`, a local-first, zero-server
adapter built on the stdlib :mod:`sqlite3` module (mirroring
:class:`friday.reminders.store.SQLiteReminderStore` and
:class:`friday.scheduler.store.SQLiteTriggerStore`).

Design rules (binding):

* **Local-first, zero-server.** SQLite is the concrete backend for both
  ``":memory:"`` (tests) and a file path (production). No daemon, no network.
* **Parametrized SQL only.** Every value reaches SQLite through a ``?``
  placeholder, so the store is injection-safe by construction.
* **Idempotent schema.** ``CREATE TABLE IF NOT EXISTS`` so constructing a store
  over an existing file never clobbers data.
* **Thread-safe by construction (file paths).** A *connection-per-call* is opened
  for a filesystem-backed database (shareable across threads); an in-memory
  database keeps a single shared connection (a new connection would otherwise see
  an empty database).
* **Steps as JSON.** A protocol's ``steps`` (each ``{tool, args}``) are serialized
  to a single JSON ``TEXT`` column, keeping the schema flat while preserving the
  ordered, typed step list on read.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from pydantic import BaseModel, Field

# In-memory SQLite database identifier — the ephemeral default for some callers.
_MEMORY_PATH = ":memory:"


# --------------------------------------------------------------------------- #
# Typed row models
# --------------------------------------------------------------------------- #
class ProtocolStep(BaseModel):
    """One step of a protocol: a registered tool name and its raw arguments.

    ``tool`` must name a tool registered in the shared
    :class:`~friday.tools.registry.ToolRegistry`; ``args`` is the raw argument
    mapping the registry validates against the tool's ``args_model`` before the
    tool runs. No arbitrary code is stored here — only a tool name + its inputs.
    """

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class Protocol(BaseModel):
    """A durable named protocol: an ordered list of :class:`ProtocolStep`.

    ``trigger_phrase`` is the spoken/typed phrase that fires the protocol (e.g.
    ``"goodnight"``); matching is the orchestrator's job. ``enabled`` gates
    whether the protocol participates in trigger matching.
    """

    id: int
    name: str
    trigger_phrase: str
    steps: list[ProtocolStep] = Field(default_factory=list)
    enabled: bool = True


# --------------------------------------------------------------------------- #
# SQLite implementation
# --------------------------------------------------------------------------- #
class SQLiteProtocolStore:
    """Local-first protocol store backed by stdlib :mod:`sqlite3`.

    Args:
        path: A filesystem path for a durable database, or ``":memory:"`` for an
            ephemeral in-process database. For a file path a fresh connection is
            opened per call (thread-safe); ``":memory:"`` keeps one shared
            connection for the life of the instance (a new connection would see
            an empty database).
    """

    def __init__(self, path: str = _MEMORY_PATH) -> None:
        self._path = path
        # An in-memory database cannot use connection-per-call: each new
        # connection would open a *separate* empty database. A file path can.
        self._shared: sqlite3.Connection | None = None
        if path == _MEMORY_PATH:
            self._shared = sqlite3.connect(path, check_same_thread=False)
            self._shared.row_factory = sqlite3.Row
        self.init_schema()

    # -- connection management --------------------------------------------- #
    def _connect(self) -> sqlite3.Connection:
        """Return the shared (memory) connection or a fresh per-call one (file)."""
        if self._shared is not None:
            return self._shared
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        """Create the ``protocols`` table if absent (idempotent).

        ``steps`` holds the JSON-serialized ordered step list; ``enabled`` is an
        integer flag. Safe to call repeatedly and against an existing file.
        """
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS protocols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    trigger_phrase TEXT NOT NULL,
                    steps TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            conn.commit()
        finally:
            self._close(conn)

    # -- writes ------------------------------------------------------------ #
    def add(
        self,
        *,
        name: str,
        trigger_phrase: str,
        steps: list[ProtocolStep],
        enabled: bool = True,
    ) -> Protocol:
        """Insert a protocol (parametrized; steps as JSON) and return the row."""
        steps_json = self._dump_steps(steps)
        conn = self._connect()
        try:
            cursor = conn.execute(
                "INSERT INTO protocols (name, trigger_phrase, steps, enabled) "
                "VALUES (?, ?, ?, ?)",
                (name, trigger_phrase, steps_json, 1 if enabled else 0),
            )
            conn.commit()
            protocol_id = int(cursor.lastrowid or 0)
        finally:
            self._close(conn)
        return Protocol(
            id=protocol_id,
            name=name,
            trigger_phrase=trigger_phrase,
            steps=list(steps),
            enabled=enabled,
        )

    def update(self, protocol: Protocol) -> bool:
        """Persist ``name``/``trigger_phrase``/``steps``/``enabled`` for a row.

        Returns ``True`` when a row with ``protocol.id`` existed and was updated,
        ``False`` otherwise.
        """
        steps_json = self._dump_steps(protocol.steps)
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE protocols SET name = ?, trigger_phrase = ?, steps = ?, "
                "enabled = ? WHERE id = ?",
                (
                    protocol.name,
                    protocol.trigger_phrase,
                    steps_json,
                    1 if protocol.enabled else 0,
                    protocol.id,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            self._close(conn)

    def set_enabled(self, protocol_id: int, enabled: bool) -> bool:
        """Toggle a protocol's ``enabled`` flag; ``False`` when no such row."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE protocols SET enabled = ? WHERE id = ?",
                (1 if enabled else 0, protocol_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            self._close(conn)

    def delete(self, protocol_id: int) -> int:
        """Delete a protocol by id; return the number of rows removed (0 or 1)."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "DELETE FROM protocols WHERE id = ?", (protocol_id,)
            )
            conn.commit()
            return cursor.rowcount
        finally:
            self._close(conn)

    # -- reads ------------------------------------------------------------- #
    def get(self, protocol_id: int) -> Protocol | None:
        """Return the protocol with ``protocol_id`` or ``None`` when absent."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, name, trigger_phrase, steps, enabled "
                "FROM protocols WHERE id = ?",
                (protocol_id,),
            ).fetchone()
            return None if row is None else self._row_to_protocol(row)
        finally:
            self._close(conn)

    def get_by_name(self, name: str) -> Protocol | None:
        """Return the protocol named ``name`` (case-insensitive) or ``None``.

        Uses SQLite's case-insensitive ``LOWER`` comparison so ``"goodNIGHT"``
        matches a stored ``"Goodnight"``; returns the first match (names are
        expected unique) or ``None`` when none exists.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, name, trigger_phrase, steps, enabled "
                "FROM protocols WHERE LOWER(name) = LOWER(?) ORDER BY id ASC",
                (name,),
            ).fetchone()
            return None if row is None else self._row_to_protocol(row)
        finally:
            self._close(conn)

    def list_protocols(self) -> list[Protocol]:
        """Return every protocol in insertion (ascending id) order."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, name, trigger_phrase, steps, enabled "
                "FROM protocols ORDER BY id ASC"
            ).fetchall()
            return [self._row_to_protocol(row) for row in rows]
        finally:
            self._close(conn)

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _dump_steps(steps: list[ProtocolStep]) -> str:
        """Serialize an ordered step list to a compact JSON array string."""
        return json.dumps([step.model_dump() for step in steps])

    @staticmethod
    def _load_steps(raw: str) -> list[ProtocolStep]:
        """Parse a JSON step array back into typed :class:`ProtocolStep` items."""
        data = json.loads(raw)
        return [ProtocolStep.model_validate(item) for item in data]

    def _row_to_protocol(self, row: sqlite3.Row) -> Protocol:
        return Protocol(
            id=int(row["id"]),
            name=str(row["name"]),
            trigger_phrase=str(row["trigger_phrase"]),
            steps=self._load_steps(str(row["steps"])),
            enabled=bool(row["enabled"]),
        )

    def _close(self, conn: sqlite3.Connection) -> None:
        """Close a per-call connection; leave the shared (memory) one open."""
        if conn is not self._shared:
            conn.close()
