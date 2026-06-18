# © Lakshya Badjatya — Author
"""Local-first flow store behind a SQLite backend (the Flow Engine checkpoint).

The durable home of a :class:`~friday.flows.models.Flow`: the engine calls
:meth:`SQLiteFlowStore.update` after every step, so a crash mid-run leaves a
fully-resumable row. Mirrors :class:`friday.scheduler.store.SQLiteTriggerStore`
exactly — same connection discipline (shared connection for ``":memory:"``,
connection-per-call for a file path), parametrized SQL only, idempotent schema.

The whole :class:`Flow` is persisted as one JSON blob in the ``data`` column, so
the schema never needs a migration as :class:`~friday.flows.models.FlowStep`
grows new fields; the ``status`` column is denormalized out of that JSON only so
``list``/``resumable`` can filter in SQL.
"""

from __future__ import annotations

import sqlite3

from friday.flows.models import Flow, FlowStatus

# In-memory SQLite database identifier — the ephemeral default (tests).
_MEMORY_PATH = ":memory:"

# Flows that a fresh process should pick back up after a restart.
_RESUMABLE = (FlowStatus.RUNNING.value, FlowStatus.PAUSED.value)


class SQLiteFlowStore:
    """Durable flow store backed by stdlib :mod:`sqlite3`.

    Args:
        path: A filesystem path for a durable database, or ``":memory:"`` for an
            ephemeral in-process database. For a file path a fresh connection is
            opened per call (thread-safe); ``":memory:"`` keeps one shared
            connection for the life of the instance.
    """

    def __init__(self, path: str = _MEMORY_PATH) -> None:
        self._path = path
        self._shared: sqlite3.Connection | None = None
        if path == _MEMORY_PATH:
            self._shared = sqlite3.connect(path, check_same_thread=False)
            self._shared.row_factory = sqlite3.Row
        self.init_schema()

    # -- connection management --------------------------------------------- #
    def _connect(self) -> sqlite3.Connection:
        if self._shared is not None:
            return self._shared
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _close(self, conn: sqlite3.Connection) -> None:
        if conn is not self._shared:
            conn.close()

    def init_schema(self) -> None:
        """Create the ``flows`` table if absent (idempotent)."""
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS flows (
                    id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.commit()
        finally:
            self._close(conn)

    # -- writes ------------------------------------------------------------ #
    def create(self, flow: Flow) -> Flow:
        """Insert ``flow`` (parametrized) and return it."""
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO flows (id, goal, status, data, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    flow.id,
                    flow.goal,
                    flow.status.value,
                    flow.model_dump_json(),
                    flow.created_at,
                    flow.updated_at,
                ),
            )
            conn.commit()
        finally:
            self._close(conn)
        return flow

    def update(self, flow: Flow) -> bool:
        """Persist the full ``flow`` JSON + denormalized status (the checkpoint).

        Returns ``True`` when a row with ``flow.id`` existed and was updated.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE flows SET goal = ?, status = ?, data = ?, updated_at = ? "
                "WHERE id = ?",
                (
                    flow.goal,
                    flow.status.value,
                    flow.model_dump_json(),
                    flow.updated_at,
                    flow.id,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            self._close(conn)

    def delete(self, flow_id: str) -> int:
        """Delete a flow by id; return the number of rows removed (0 or 1)."""
        conn = self._connect()
        try:
            cursor = conn.execute("DELETE FROM flows WHERE id = ?", (flow_id,))
            conn.commit()
            return cursor.rowcount
        finally:
            self._close(conn)

    # -- reads ------------------------------------------------------------- #
    def get(self, flow_id: str) -> Flow | None:
        """Return the flow with ``flow_id`` or ``None`` when absent."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT data FROM flows WHERE id = ?", (flow_id,)
            ).fetchone()
            return None if row is None else Flow.model_validate_json(row["data"])
        finally:
            self._close(conn)

    def list_flows(self, status: FlowStatus | None = None) -> list[Flow]:
        """Return every flow (newest-updated first), optionally filtered by status."""
        conn = self._connect()
        try:
            if status is None:
                rows = conn.execute(
                    "SELECT data FROM flows ORDER BY updated_at DESC, id ASC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT data FROM flows WHERE status = ? "
                    "ORDER BY updated_at DESC, id ASC",
                    (status.value,),
                ).fetchall()
            return [Flow.model_validate_json(row["data"]) for row in rows]
        finally:
            self._close(conn)

    def resumable(self) -> list[Flow]:
        """Return flows a fresh process should pick back up (``running``/``paused``)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT data FROM flows WHERE status IN (?, ?) ORDER BY id ASC",
                _RESUMABLE,
            ).fetchall()
            return [Flow.model_validate_json(row["data"]) for row in rows]
        finally:
            self._close(conn)
