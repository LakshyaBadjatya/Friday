"""Local-first family-sharing store behind a SQLite backend (build-spec §18).

FRIDAY's consent-enforced family layer: participants who have opted THEMSELVES in
to sharing, the directed share edges between them (each carrying a per-viewer
``raw_location`` grant), and an audit of every view (so the viewed participant
can always see who viewed them). The concrete, gate-required backend is
:class:`SQLiteFamilyStore`, a local-first, zero-server adapter built on the
stdlib :mod:`sqlite3` module (mirroring
:class:`friday.reminders.store.SQLiteReminderStore` and
:class:`friday.study.store.SQLiteStudyStore`).

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
* **Clock injectable.** The "now" used to stamp a recorded view comes from an
  injected ``clock() -> datetime``, never the wall clock, so timestamps are
  deterministic in tests.

**Privacy posture (binding, build-spec §18).** A participant's location is shared
as a coarse geofence *status* — one of ``home`` / ``work`` / ``away`` — and never
as raw coordinates by default. Raw latitude/longitude is gated by a per-viewer
``raw_location`` grant on the share edge; the store only records the grant, and
the :class:`~friday.family.service.FamilyService` enforces what a viewer sees.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

# In-memory SQLite database identifier — the ephemeral default for some callers.
_MEMORY_PATH = ":memory:"

#: The coarse geofence labels a participant's location may be shared as. This is
#: the ONLY location granularity shared by default — never raw coordinates.
GeofenceStatus = Literal["home", "work", "away"]
_GEOFENCE_LABELS: frozenset[str] = frozenset({"home", "work", "away"})


# --------------------------------------------------------------------------- #
# Typed row models
# --------------------------------------------------------------------------- #
class Participant(BaseModel):
    """A family-sharing participant who has opted THEMSELVES in.

    ``self_opted_in`` is always ``True`` for a persisted participant — the
    service rejects any attempt to add someone from another account (build-spec
    §18 guardrail 1). ``status`` is the coarse geofence label currently shared
    (``home`` / ``work`` / ``away``); raw coordinates are never stored on the
    participant. ``sharing_with`` is the list of viewer names this participant
    currently shares with (the directed share edges).
    """

    id: int
    name: str
    self_opted_in: bool = True
    status: GeofenceStatus = "away"
    sharing_with: list[str] = Field(default_factory=list)


class ViewRecord(BaseModel):
    """An audited view: ``viewer`` looked at ``viewed`` at ISO timestamp ``at``."""

    id: int
    viewer: str
    viewed: str
    at: str


# --------------------------------------------------------------------------- #
# SQLite implementation
# --------------------------------------------------------------------------- #
class SQLiteFamilyStore:
    """Local-first family-sharing store backed by stdlib :mod:`sqlite3`.

    Args:
        path: A filesystem path for a durable database, or ``":memory:"`` for an
            ephemeral in-process database. For a file path a fresh connection is
            opened per call (thread-safe); ``":memory:"`` keeps one shared
            connection for the life of the instance (a new connection would see
            an empty database).
        clock: A zero-arg callable returning the current time as a
            :class:`~datetime.datetime`. Used to stamp a recorded view; injected
            so timestamps are deterministic in tests. Defaults to the system wall
            clock (UTC).
    """

    def __init__(
        self,
        path: str = _MEMORY_PATH,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._path = path
        self._clock: Callable[[], datetime] = (
            clock if clock is not None else (lambda: datetime.now(UTC))
        )
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
        """Create the family tables if absent (idempotent).

        ``family_participants`` holds the opted-in members; ``family_shares`` the
        directed share edges (each with a ``raw_location`` grant flag); and
        ``family_views`` the audit of who viewed whom. Safe to call repeatedly
        and against an existing file.
        """
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS family_participants (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    self_opted_in INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'away'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS family_shares (
                    owner TEXT NOT NULL,
                    viewer TEXT NOT NULL,
                    raw_location INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (owner, viewer)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS family_views (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    viewer TEXT NOT NULL,
                    viewed TEXT NOT NULL,
                    at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            self._close(conn)

    # -- participant writes ------------------------------------------------ #
    def opt_in(self, name: str) -> Participant:
        """Opt ``name`` in (idempotent by name); return the stored participant.

        A persisted participant is always ``self_opted_in=True`` with a default
        geofence ``status`` of ``away``. Re-opting an existing name returns the
        existing row unchanged (no clobber of an already-shared participant).
        """
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO family_participants (name, self_opted_in, status) "
                "VALUES (?, 1, 'away')",
                (name,),
            )
            conn.commit()
        finally:
            self._close(conn)
        participant = self.get(name)
        assert participant is not None  # just inserted/ignored
        return participant

    def set_status(self, name: str, status: str) -> None:
        """Update a participant's coarse geofence status (``home``/``work``/``away``).

        Raises:
            ValueError: if ``status`` is not one of the known geofence labels —
                the store never accepts (and never stores) raw coordinates here.
        """
        if status not in _GEOFENCE_LABELS:
            raise ValueError(
                f"unknown geofence status {status!r}; "
                f"expected one of {sorted(_GEOFENCE_LABELS)}"
            )
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE family_participants SET status = ? WHERE name = ?",
                (status, name),
            )
            conn.commit()
        finally:
            self._close(conn)

    # -- share-edge writes ------------------------------------------------- #
    def add_share(self, owner: str, viewer: str, *, raw_location: bool = False) -> None:
        """Create/refresh the directed share edge ``owner -> viewer`` (parametrized).

        Idempotent on the ``(owner, viewer)`` key; ``raw_location`` records the
        per-viewer raw-coordinate grant (``False`` by default keeps the share at
        the coarse geofence-status level).
        """
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO family_shares (owner, viewer, raw_location) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(owner, viewer) DO UPDATE SET raw_location = excluded.raw_location",
                (owner, viewer, 1 if raw_location else 0),
            )
            conn.commit()
        finally:
            self._close(conn)

    def remove_share(self, owner: str, viewer: str) -> int:
        """Delete the share edge ``owner -> viewer``; return rows removed (0 or 1).

        Removing the edge also drops any per-viewer raw grant it carried, so a
        revoke fully stops sharing (build-spec §18 guardrail 3).
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "DELETE FROM family_shares WHERE owner = ? AND viewer = ?",
                (owner, viewer),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            self._close(conn)

    # -- view audit -------------------------------------------------------- #
    def record_view(self, viewer: str, viewed: str) -> ViewRecord:
        """Append an audited view (``at`` stamped from the injected clock)."""
        at = self._clock().isoformat()
        conn = self._connect()
        try:
            cursor = conn.execute(
                "INSERT INTO family_views (viewer, viewed, at) VALUES (?, ?, ?)",
                (viewer, viewed, at),
            )
            conn.commit()
            view_id = int(cursor.lastrowid or 0)
        finally:
            self._close(conn)
        return ViewRecord(id=view_id, viewer=viewer, viewed=viewed, at=at)

    # -- reads ------------------------------------------------------------- #
    def get(self, name: str) -> Participant | None:
        """Return the participant named ``name`` (with their share edges), or None."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, name, self_opted_in, status "
                "FROM family_participants WHERE name = ?",
                (name,),
            ).fetchone()
            if row is None:
                return None
            viewers = self._sharing_with(conn, name)
            return self._row_to_participant(row, viewers)
        finally:
            self._close(conn)

    def list_participants(self) -> list[Participant]:
        """Return all opted-in participants in insertion order (with their edges)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, name, self_opted_in, status "
                "FROM family_participants ORDER BY id ASC"
            ).fetchall()
            return [
                self._row_to_participant(
                    row, self._sharing_with(conn, str(row["name"]))
                )
                for row in rows
            ]
        finally:
            self._close(conn)

    def is_sharing(self, owner: str, viewer: str) -> bool:
        """Whether the directed share edge ``owner -> viewer`` currently exists."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM family_shares WHERE owner = ? AND viewer = ?",
                (owner, viewer),
            ).fetchone()
            return row is not None
        finally:
            self._close(conn)

    def shares_raw(self, owner: str, viewer: str) -> bool:
        """Whether ``owner`` has granted ``viewer`` the per-viewer RAW location."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT raw_location FROM family_shares "
                "WHERE owner = ? AND viewer = ?",
                (owner, viewer),
            ).fetchone()
            return row is not None and int(row["raw_location"]) == 1
        finally:
            self._close(conn)

    def views_of(self, name: str) -> list[ViewRecord]:
        """Return the audited views of ``name``, most-recent first.

        This is what lets the VIEWED participant see who viewed them (build-spec
        §18 guardrail 4).
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, viewer, viewed, at FROM family_views "
                "WHERE viewed = ? ORDER BY id DESC",
                (name,),
            ).fetchall()
            return [
                ViewRecord(
                    id=int(row["id"]),
                    viewer=str(row["viewer"]),
                    viewed=str(row["viewed"]),
                    at=str(row["at"]),
                )
                for row in rows
            ]
        finally:
            self._close(conn)

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _sharing_with(conn: sqlite3.Connection, owner: str) -> list[str]:
        """Return the viewer names ``owner`` currently shares with (sorted)."""
        rows = conn.execute(
            "SELECT viewer FROM family_shares WHERE owner = ? ORDER BY viewer ASC",
            (owner,),
        ).fetchall()
        return [str(row["viewer"]) for row in rows]

    @staticmethod
    def _row_to_participant(
        row: sqlite3.Row, sharing_with: list[str]
    ) -> Participant:
        raw_status = str(row["status"])
        # Defensive: coerce any unexpected stored value back to the safe default
        # (pydantic re-validates the literal on construction regardless).
        status: GeofenceStatus = (
            raw_status  # type: ignore[assignment]
            if raw_status in _GEOFENCE_LABELS
            else "away"
        )
        return Participant(
            id=int(row["id"]),
            name=str(row["name"]),
            self_opted_in=bool(row["self_opted_in"]),
            status=status,
            sharing_with=sharing_with,
        )

    def _close(self, conn: sqlite3.Connection) -> None:
        """Close a per-call connection; leave the shared (memory) one open."""
        if conn is not self._shared:
            conn.close()
