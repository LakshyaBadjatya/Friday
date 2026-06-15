"""Local-first knowledge-graph store behind a SQLite backend (Tier 2).

FRIDAY's durable entity/relation layer: a tiny knowledge graph of the people,
projects, and things the owner talks about, plus the relations between them. The
concrete backend is :class:`SQLiteGraphStore`, a local-first, zero-server adapter
built on the stdlib :mod:`sqlite3` module (mirroring
:class:`friday.protocols.store.SQLiteProtocolStore` and
:class:`friday.meetings.store.SQLiteMeetingStore`).

Design rules (binding):

* **Local-first, zero-server.** SQLite is the concrete backend for both
  ``":memory:"`` (tests) and a file path (production). No daemon, no network.
* **Parametrized SQL only.** Every value reaches SQLite through a ``?``
  placeholder, so the store is injection-safe by construction.
* **Idempotent schema + idempotent upsert.** ``CREATE TABLE IF NOT EXISTS`` so
  constructing a store over an existing file never clobbers data;
  :meth:`SQLiteGraphStore.upsert_entity` is keyed on ``(name, type)`` so the same
  entity seen twice merges (attrs are updated) rather than duplicating.
* **Thread-safe by construction (file paths).** A *connection-per-call* is opened
  for a filesystem-backed database (shareable across threads); an in-memory
  database keeps a single shared connection (a new connection would otherwise see
  an empty database).
* **Attrs as JSON.** An entity's free-form ``attrs`` map is serialized to a single
  JSON ``TEXT`` column, keeping the schema flat while preserving the typed dict on
  read.

The :meth:`SQLiteGraphStore.entity_card` view stitches an entity together with its
relations and — when a long-term store is supplied — the facts the owner has told
FRIDAY that mention the entity by name, so a single read answers "what do you know
about X?".
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# In-memory SQLite database identifier — the ephemeral default for some callers.
_MEMORY_PATH = ":memory:"

# Upper bound on facts pulled into an entity card from the long-term store, so a
# noisy fact history can never make a single card read unbounded.
_CARD_FACTS_LIMIT = 20


# --------------------------------------------------------------------------- #
# Read-side contract for the long-term store (only what the card view needs)
# --------------------------------------------------------------------------- #
@runtime_checkable
class _FactLookup(Protocol):
    """The slice of the long-term-store contract the entity card reads through.

    Structural so the concrete
    :class:`~friday.memory.long_term.SQLiteLongTermStore` satisfies it without an
    import-time coupling: the card only needs to *query* facts by substring. The
    returned rows need only expose a ``text`` attribute (a :class:`~friday.memory.
    long_term.Fact` does), which the card serializes back to plain strings.
    """

    def query_facts(self, query: str, limit: int = ...) -> list[Any]: ...


# --------------------------------------------------------------------------- #
# Typed row models
# --------------------------------------------------------------------------- #
class Entity(BaseModel):
    """A node in the knowledge graph: a named thing of some ``type``.

    Identity is the ``(name, type)`` pair (e.g. ``("Ada", "person")``);
    :meth:`SQLiteGraphStore.upsert_entity` is idempotent on it. ``attrs`` is a
    free-form, JSON-serializable map of extra structured facts about the entity
    (role, location, etc.) that the extractor or the owner may attach.
    """

    id: int
    name: str
    type: str
    attrs: dict[str, Any] = Field(default_factory=dict)


class Relation(BaseModel):
    """A directed edge between two entities, named by ``kind``.

    ``src`` and ``dst`` are entity *names* (e.g. ``("Ada", "Zephyr")`` with
    ``kind="works_on"``); the store does not require the endpoints to already be
    upserted, so a relation can be recorded before both nodes are fully known.
    """

    id: int
    src: str
    dst: str
    kind: str


# --------------------------------------------------------------------------- #
# SQLite implementation
# --------------------------------------------------------------------------- #
class SQLiteGraphStore:
    """Local-first knowledge-graph store backed by stdlib :mod:`sqlite3`.

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

    def _close(self, conn: sqlite3.Connection) -> None:
        """Close a per-call connection; leave the shared (memory) one open."""
        if conn is not self._shared:
            conn.close()

    def init_schema(self) -> None:
        """Create the ``entities`` and ``relations`` tables if absent (idempotent).

        ``entities`` is uniquely keyed on ``(name, type)`` so an upsert merges by
        identity; ``attrs`` holds the JSON-serialized attribute map. ``relations``
        records directed ``(src, dst, kind)`` edges by entity name. Safe to call
        repeatedly and against an existing file.
        """
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    attrs TEXT NOT NULL DEFAULT '{}',
                    UNIQUE (name, type)
                );
                CREATE TABLE IF NOT EXISTS relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    src TEXT NOT NULL,
                    dst TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    UNIQUE (src, dst, kind)
                );
                """
            )
            conn.commit()
        finally:
            self._close(conn)

    # -- writes ------------------------------------------------------------ #
    def upsert_entity(
        self, name: str, type: str, attrs: dict[str, Any] | None = None
    ) -> Entity:
        """Insert or merge an entity by ``(name, type)``; return the stored row.

        Idempotent on the ``(name, type)`` identity: a second call with the same
        pair updates the row's ``attrs`` in place (last write wins) rather than
        creating a duplicate. ``attrs`` is serialized to JSON; ``None`` is treated
        as an empty map. The returned :class:`Entity` carries the row's stable id.
        """
        merged = dict(attrs or {})
        attrs_json = json.dumps(merged)
        conn = self._connect()
        try:
            # ON CONFLICT keeps the existing id (idempotent identity) and refreshes
            # the attrs, so repeated extraction of the same entity never duplicates.
            conn.execute(
                "INSERT INTO entities (name, type, attrs) VALUES (?, ?, ?) "
                "ON CONFLICT(name, type) DO UPDATE SET attrs = excluded.attrs",
                (name, type, attrs_json),
            )
            conn.commit()
            row = conn.execute(
                "SELECT id, name, type, attrs FROM entities "
                "WHERE name = ? AND type = ?",
                (name, type),
            ).fetchone()
        finally:
            self._close(conn)
        return self._row_to_entity(row)

    def add_relation(self, src: str, dst: str, kind: str) -> Relation:
        """Insert a directed ``(src, dst, kind)`` edge; return the stored row.

        Idempotent on the full triple (``UNIQUE (src, dst, kind)``): re-adding the
        same edge returns the existing row rather than duplicating it. Endpoints
        are entity *names* and need not already be upserted as entities.
        """
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO relations (src, dst, kind) "
                "VALUES (?, ?, ?)",
                (src, dst, kind),
            )
            conn.commit()
            row = conn.execute(
                "SELECT id, src, dst, kind FROM relations "
                "WHERE src = ? AND dst = ? AND kind = ?",
                (src, dst, kind),
            ).fetchone()
        finally:
            self._close(conn)
        return self._row_to_relation(row)

    # -- reads ------------------------------------------------------------- #
    def get_entity(self, name: str) -> Entity | None:
        """Return the first entity named ``name`` (any type) or ``None``.

        Names are expected unique enough for a personal graph; when several types
        share a name the lowest-id (earliest) row wins, which is deterministic.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, name, type, attrs FROM entities "
                "WHERE name = ? ORDER BY id ASC",
                (name,),
            ).fetchone()
            return None if row is None else self._row_to_entity(row)
        finally:
            self._close(conn)

    def list_entities(self, type: str | None = None) -> list[Entity]:
        """List entities (optionally filtered by ``type``) in insertion order."""
        conn = self._connect()
        try:
            if type is None:
                rows = conn.execute(
                    "SELECT id, name, type, attrs FROM entities ORDER BY id ASC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, name, type, attrs FROM entities "
                    "WHERE type = ? ORDER BY id ASC",
                    (type,),
                ).fetchall()
            return [self._row_to_entity(row) for row in rows]
        finally:
            self._close(conn)

    def neighbors(self, name: str) -> list[Relation]:
        """Return every relation touching ``name`` (as ``src`` or ``dst``).

        Both outgoing (``src = name``) and incoming (``dst = name``) edges are
        returned so a card sees the entity's full local neighbourhood, ordered by
        id (insertion order) for a stable read.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, src, dst, kind FROM relations "
                "WHERE src = ? OR dst = ? ORDER BY id ASC",
                (name, name),
            ).fetchall()
            return [self._row_to_relation(row) for row in rows]
        finally:
            self._close(conn)

    def search(self, q: str) -> list[Entity]:
        """Return entities whose name contains ``q`` (case-insensitive substring).

        Parametrized SQL ``LIKE`` with the query wrapped in escaped ``%``
        wildcards, so ``q`` is always a literal substring — SQL metacharacters in
        it have no effect. Results are insertion-ordered. An empty query lists all
        entities (every name contains the empty string).
        """
        like = self._contains(q)
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, name, type, attrs FROM entities "
                "WHERE name LIKE ? ESCAPE '\\' ORDER BY id ASC",
                (like,),
            ).fetchall()
            return [self._row_to_entity(row) for row in rows]
        finally:
            self._close(conn)

    def entity_card(
        self, name: str, *, long_term: _FactLookup | None = None
    ) -> dict[str, Any]:
        """Assemble a single ``{entity, relations, facts}`` view of ``name``.

        ``entity`` is the matching :class:`Entity` serialized to a dict (or
        ``None`` when the name is not a known node — relations/facts may still be
        present). ``relations`` is every edge touching the name (see
        :meth:`neighbors`). ``facts`` is the list of long-term fact *texts* that
        mention the name, pulled from ``long_term`` when one is supplied (the same
        store the rest of FRIDAY records facts in); when no store is supplied
        ``facts`` is empty. Bounded by :data:`_CARD_FACTS_LIMIT`.
        """
        entity = self.get_entity(name)
        relations = self.neighbors(name)
        facts: list[str] = []
        if long_term is not None:
            facts = [
                str(fact.text)
                for fact in long_term.query_facts(name, limit=_CARD_FACTS_LIMIT)
            ]
        return {
            "entity": None if entity is None else entity.model_dump(),
            "relations": [relation.model_dump() for relation in relations],
            "facts": facts,
        }

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _contains(query: str) -> str:
        """Wrap ``query`` as a ``LIKE`` substring pattern, escaping wildcards.

        ``%``/``_`` (and the escape char ``\\``) in ``query`` are escaped so they
        match literally; the result is surrounded by ``%`` to mean "contains
        ``query``". Used with ``ESCAPE '\\'`` in the SQL (mirrors the long-term
        store's matcher).
        """
        escaped = (
            query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        return f"%{escaped}%"

    @staticmethod
    def _load_attrs(raw: str) -> dict[str, Any]:
        """Parse a JSON attrs string back to a dict; an empty/garbage value -> {}."""
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):  # pragma: no cover - defensive
            return {}
        return data if isinstance(data, dict) else {}

    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        return Entity(
            id=int(row["id"]),
            name=str(row["name"]),
            type=str(row["type"]),
            attrs=self._load_attrs(str(row["attrs"])),
        )

    @staticmethod
    def _row_to_relation(row: sqlite3.Row) -> Relation:
        return Relation(
            id=int(row["id"]),
            src=str(row["src"]),
            dst=str(row["dst"]),
            kind=str(row["kind"]),
        )
