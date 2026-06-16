"""Postgres / pgvector adapter swap for FRIDAY's durable memory (Tier 3).

This module is the optional ``postgres`` backend: drop-in replacements for the
local-first SQLite stores that persist into a Postgres database instead. They
keep the *exact same* structural contracts as the default adapters —
:class:`~friday.memory.long_term.LongTermStore` (via
:class:`PostgresLongTermStore`) and :class:`~friday.memory.vector.VectorStore`
(via :class:`PgVectorStore`, which stores embeddings in a ``pgvector`` column) —
so the integration layer can select them behind the ``enable_postgres`` flag with
no change to call sites.

Design rules (binding):

* **Optional, lazy driver.** ``psycopg`` is **not** a project dependency and is
  **not** in the uv lock. It is *lazy-imported inside* the adapters (never at
  module top), so ``import friday.memory.pg`` succeeds in the offline gate with
  no driver present. The local-first SQLite stores remain the default.
* **Fail loudly, never half-open.** Constructing an adapter validates the DSN and
  then imports ``psycopg``. If the DSN is missing or the driver is absent, the
  adapter raises a clear :class:`~friday.errors.FridayError` that tells the
  operator exactly what to do — *install ``psycopg`` and set
  ``FRIDAY_POSTGRES_DSN``* — rather than surfacing a bare ``ImportError`` or
  silently degrading.
* **Parametrized SQL only.** Every value reaches Postgres through a placeholder
  (``%s``); user/agent text is never interpolated into a statement, so the stores
  are injection-safe by construction — mirroring the SQLite adapters.
* **Typed rows.** Reads return the same pydantic models as the SQLite stores
  (:class:`~friday.memory.long_term.Fact`, :class:`~friday.memory.long_term.TaskRow`,
  :class:`~friday.memory.long_term.AuditRow`, and
  :class:`~friday.memory.vector.Chunk`), so callers get a stable schema.
* **Secrets stay secret.** The DSN may embed a password; it is held privately and
  never logged. Selection is config-driven (``postgres_dsn`` is a ``SecretStr``);
  this module receives the resolved string but treats it as sensitive.

No SDK, no ORM: the adapters talk to Postgres through ``psycopg`` (a thin driver)
exactly as the rest of FRIDAY talks to HTTP services through ``httpx``.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from friday.errors import FridayError
from friday.memory.long_term import (
    DEFAULT_LIMIT,
    AuditRow,
    Fact,
    TaskRow,
)
from friday.memory.vector import Chunk

if TYPE_CHECKING:
    from friday.providers.embeddings import EmbeddingProvider

# Operator-facing guidance raised whenever the Postgres backend cannot be used
# because the optional driver is missing or the DSN is unconfigured. It names
# BOTH remedies so a single message is actionable.
_POSTGRES_SETUP_NOTE = (
    "The Postgres backend is not available: install `psycopg` (it is an optional "
    "dependency, kept out of the default lock) and configure `FRIDAY_POSTGRES_DSN` "
    "with a valid Postgres connection string before enabling `enable_postgres`. "
    "The local-first SQLite stores remain the default and need neither."
)


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (second precision).

    Matches :func:`friday.memory.long_term._utc_now_iso` so rows written by the
    Postgres adapter are indistinguishable in shape from the SQLite ones.
    """
    return datetime.now(UTC).isoformat()


def _require_dsn(dsn: str | None) -> str:
    """Validate ``dsn`` is a non-empty string or raise the setup guidance.

    The DSN guard runs *before* any driver import so a misconfiguration is caught
    even in an environment without ``psycopg`` installed.
    """
    if not dsn or not dsn.strip():
        raise FridayError(_POSTGRES_SETUP_NOTE)
    return dsn


def _import_psycopg() -> Any:
    """Lazy-import and return the ``psycopg`` module, or raise the setup guidance.

    Imported here (not at module top) so this module imports cleanly with no
    driver present. A missing driver is reported as a
    :class:`~friday.errors.FridayError` carrying install/DSN guidance — never a
    bare ``ImportError`` leaking the optional dependency to the caller.
    """
    try:
        import psycopg  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - psycopg not installed in gate
        raise FridayError(_POSTGRES_SETUP_NOTE) from exc
    return psycopg


# --------------------------------------------------------------------------- #
# Long-term store over Postgres
# --------------------------------------------------------------------------- #
class PostgresLongTermStore:
    """:class:`~friday.memory.long_term.LongTermStore` backed by Postgres.

    A persistent, server-backed alternative to
    :class:`~friday.memory.long_term.SQLiteLongTermStore` with the identical
    method surface (``add_fact`` / ``query_facts`` / ``add_task`` /
    ``task_history`` / ``add_audit`` / ``forget``, plus ``audit_history``) and the
    same typed pydantic return rows. A single long-lived connection is opened in
    ``__init__`` via the lazy-imported ``psycopg`` driver; the schema is created
    idempotently (``CREATE TABLE IF NOT EXISTS``). All statements are parametrized
    (``%s``), so user/agent text is never interpolated — injection-safe by
    construction.

    Construction is the failure point: it validates the DSN and imports
    ``psycopg``, raising a clear :class:`~friday.errors.FridayError` (install
    ``psycopg`` + set ``FRIDAY_POSTGRES_DSN``) when either is absent, so a
    misconfigured deployment fails fast and loud rather than half-open.

    Args:
        dsn: A Postgres connection string (may embed a password; treated as
            sensitive and never logged).
    """

    def __init__(self, dsn: str | None) -> None:
        self._dsn = _require_dsn(dsn)
        psycopg = _import_psycopg()
        # Open eagerly so a bad DSN / unreachable server surfaces at construction
        # rather than on first use. ``connect`` lives on the lazily-imported
        # module, so this line only runs once the driver is present.
        self._conn = psycopg.connect(self._dsn)
        self.init_schema()

    def init_schema(self) -> None:
        """Create the ``facts``, ``tasks``, and ``audit`` tables if absent.

        Idempotent and safe to call repeatedly. ``sensitive``/``ok`` use a native
        ``BOOLEAN`` (Postgres has one, unlike SQLite), and ids are ``BIGSERIAL``.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id BIGSERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    sensitive BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id BIGSERIAL PRIMARY KEY,
                    intent TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    ok BOOLEAN NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS audit (
                    id BIGSERIAL PRIMARY KEY,
                    step TEXT NOT NULL,
                    ok BOOLEAN NOT NULL,
                    detail TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
        self._conn.commit()

    # -- facts -------------------------------------------------------------- #
    def add_fact(self, text: str, source_id: str, sensitive: bool = False) -> Fact:
        """Insert a fact (parametrized) and return the stored :class:`Fact`."""
        created_at = _utc_now_iso()
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO facts (text, source_id, sensitive, created_at) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (text, source_id, sensitive, created_at),
            )
            fact_id = int(self._scalar(cur))
        self._conn.commit()
        return Fact(
            id=fact_id,
            text=text,
            source_id=source_id,
            sensitive=sensitive,
            created_at=created_at,
        )

    def query_facts(self, query: str, limit: int = DEFAULT_LIMIT) -> list[Fact]:
        """Return up to ``limit`` facts whose text contains ``query`` (CI).

        Matching uses a parametrized ``ILIKE`` with the query wrapped in ``%``
        wildcards (escaped so SQL metacharacters in ``query`` match literally),
        so ``query`` is always a literal substring. Results are newest-first; a
        non-positive ``limit`` yields an empty list.
        """
        if limit <= 0:
            return []
        like = self._contains(query)
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, text, source_id, sensitive, created_at "
                "FROM facts WHERE text ILIKE %s "
                "ORDER BY id DESC LIMIT %s",
                (like, limit),
            )
            rows = cur.fetchall()
        return [self._row_to_fact(row) for row in rows]

    def all_facts(self, limit: int = DEFAULT_LIMIT) -> list[Fact]:
        """Return up to ``limit`` facts, newest first (a non-positive limit -> []).

        Unfiltered bulk read for second-brain export; newest-first to match
        :meth:`query_facts`.
        """
        if limit <= 0:
            return []
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, text, source_id, sensitive, created_at "
                "FROM facts ORDER BY id DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        return [self._row_to_fact(row) for row in rows]

    def forget(self, query: str) -> int:
        """Delete every fact whose text contains ``query`` (CI); return count.

        Parametrized like :meth:`query_facts`, so ``query`` is a literal
        substring. Returns the number of rows removed (0 when nothing matched).
        """
        like = self._contains(query)
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM facts WHERE text ILIKE %s", (like,))
            removed = cur.rowcount
        self._conn.commit()
        return int(removed)

    # -- tasks -------------------------------------------------------------- #
    def add_task(self, intent: str, summary: str, ok: bool) -> TaskRow:
        """Insert a task record (parametrized) and return the stored row."""
        created_at = _utc_now_iso()
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tasks (intent, summary, ok, created_at) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (intent, summary, ok, created_at),
            )
            task_id = int(self._scalar(cur))
        self._conn.commit()
        return TaskRow(
            id=task_id,
            intent=intent,
            summary=summary,
            ok=ok,
            created_at=created_at,
        )

    def task_history(self, limit: int = DEFAULT_LIMIT) -> list[TaskRow]:
        """Return up to ``limit`` task records, most recent first."""
        if limit <= 0:
            return []
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, intent, summary, ok, created_at "
                "FROM tasks ORDER BY id DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        return [self._row_to_task(row) for row in rows]

    # -- audit -------------------------------------------------------------- #
    def add_audit(self, step: str, ok: bool, detail: str) -> AuditRow:
        """Insert an audit record (parametrized) and return the stored row."""
        created_at = _utc_now_iso()
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO audit (step, ok, detail, created_at) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (step, ok, detail, created_at),
            )
            audit_id = int(self._scalar(cur))
        self._conn.commit()
        return AuditRow(
            id=audit_id,
            step=step,
            ok=ok,
            detail=detail,
            created_at=created_at,
        )

    def audit_history(self, limit: int = DEFAULT_LIMIT) -> list[AuditRow]:
        """Return up to ``limit`` audit records, most recent first."""
        if limit <= 0:
            return []
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, step, ok, detail, created_at "
                "FROM audit ORDER BY id DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        return [self._row_to_audit(row) for row in rows]

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _scalar(cursor: Any) -> Any:
        """Return the first column of the cursor's single fetched row.

        Used after a ``RETURNING id`` insert; raises if the driver returned no
        row (should never happen for a successful insert).
        """
        row = cursor.fetchone()
        if row is None:  # pragma: no cover - defensive: RETURNING always yields
            raise FridayError("Postgres insert returned no id")
        return row[0]

    @staticmethod
    def _contains(query: str) -> str:
        """Wrap ``query`` as an ``ILIKE`` substring pattern, escaping wildcards.

        ``%``, ``_`` and the escape char ``\\`` in ``query`` are escaped so they
        match literally; the result is surrounded by ``%`` so the pattern means
        "contains ``query``". Postgres ``LIKE``/``ILIKE`` use ``\\`` as the escape
        character by default, so no explicit ``ESCAPE`` clause is needed.
        """
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return f"%{escaped}%"

    @staticmethod
    def _row_to_fact(row: Any) -> Fact:
        return Fact(
            id=int(row[0]),
            text=str(row[1]),
            source_id=str(row[2]),
            sensitive=bool(row[3]),
            created_at=str(row[4]),
        )

    @staticmethod
    def _row_to_task(row: Any) -> TaskRow:
        return TaskRow(
            id=int(row[0]),
            intent=str(row[1]),
            summary=str(row[2]),
            ok=bool(row[3]),
            created_at=str(row[4]),
        )

    @staticmethod
    def _row_to_audit(row: Any) -> AuditRow:
        return AuditRow(
            id=int(row[0]),
            step=str(row[1]),
            ok=bool(row[2]),
            detail=str(row[3]),
            created_at=str(row[4]),
        )


# --------------------------------------------------------------------------- #
# Vector store over pgvector
# --------------------------------------------------------------------------- #
def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is zero).

    Mirrors :func:`friday.memory.vector._cosine`. Used as a Python-side fallback
    ranking that does not depend on the database's vector operators being present
    (and keeps scoring identical to the SQLite store's semantics).
    """
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for av, bv in zip(a, b, strict=False):
        dot += av * bv
        norm_a += av * av
        norm_b += bv * bv
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _format_vector(vector: list[float]) -> str:
    """Render a float vector in pgvector's text literal form ``[a,b,c]``."""
    return "[" + ",".join(repr(float(component)) for component in vector) + "]"


def _run_coro(coro: Any) -> Any:
    """Drive an async coroutine to completion from synchronous code.

    Mirrors :func:`friday.memory.vector._run_coro`: the :class:`VectorStore`
    contract is synchronous but the embedder is async. When no loop is running we
    use :func:`asyncio.run`; when one is, we offload to a short-lived worker
    thread that owns its own loop so we never reenter the caller's loop.
    """
    import asyncio
    import threading

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[Any] = []
    error: list[BaseException] = []

    def _worker() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001 - re-raised on caller thread
            error.append(exc)

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


class PgVectorStore:
    """:class:`~friday.memory.vector.VectorStore` backed by Postgres + pgvector.

    A server-backed alternative to
    :class:`~friday.memory.vector.SQLiteVectorStore` with the identical
    synchronous contract: :meth:`add` embeds each ``(text, source_id)`` pair with
    the injected :class:`~friday.providers.embeddings.EmbeddingProvider` and
    persists the text, source id, and a ``vector`` column; :meth:`query` embeds the
    query, ranks rows by cosine similarity, and returns the top ``k``
    :class:`~friday.memory.vector.Chunk` objects closest-first; :meth:`forget`
    deletes rows by exact ``source_id`` or text substring and returns the count.

    The embedder is async while the contract is sync, so embedding calls are
    driven through :func:`_run_coro` (loop-safe). Construction lazily imports
    ``psycopg``, enables the ``vector`` extension, and creates the table — failing
    loudly with the install/DSN guidance if the driver or DSN is absent.

    Args:
        dsn: A Postgres connection string (sensitive; never logged).
        embedder: The async embedding provider used for ``add``/``query``.
        dim: Vector dimensionality the ``vector`` column is sized to.
    """

    def __init__(
        self, dsn: str | None, embedder: EmbeddingProvider, dim: int
    ) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self._dsn = _require_dsn(dsn)
        self._embedder = embedder
        self._dim = dim
        psycopg = _import_psycopg()
        self._conn = psycopg.connect(self._dsn)
        self._init_schema()

    def _init_schema(self) -> None:
        """Enable pgvector and create the ``chunks`` table if absent (idempotent)."""
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                "CREATE TABLE IF NOT EXISTS chunks ("
                "  id BIGSERIAL PRIMARY KEY,"
                "  text TEXT NOT NULL,"
                "  source_id TEXT NOT NULL,"
                f"  embedding vector({self._dim}) NOT NULL"
                ")"
            )
        self._conn.commit()

    # -- VectorStore contract ----------------------------------------------- #
    def add(self, docs: list[tuple[str, str]]) -> None:
        """Embed and persist each ``(text, source_id)`` pair."""
        if not docs:
            return
        texts = [text for text, _ in docs]
        vectors = _run_coro(self._embedder.embed(texts))
        rows = [
            (text, source_id, _format_vector(vector))
            for (text, source_id), vector in zip(docs, vectors, strict=True)
        ]
        with self._conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (text, source_id, embedding) "
                "VALUES (%s, %s, %s)",
                rows,
            )
        self._conn.commit()

    def query(self, text: str, k: int = 4) -> list[Chunk]:
        """Return up to ``k`` chunks ranked by cosine similarity, closest first.

        An empty store (or ``k <= 0``) yields ``[]``. Chunks with non-positive
        similarity are dropped so unrelated material never surfaces. Ranking is
        computed in Python over the stored vectors, matching the SQLite store's
        semantics exactly.
        """
        if k <= 0:
            return []
        with self._conn.cursor() as cur:
            cur.execute("SELECT text, source_id, embedding FROM chunks")
            stored = cur.fetchall()
        if not stored:
            return []

        (query_vector,) = _run_coro(self._embedder.embed([text]))
        scored: list[Chunk] = []
        for row_text, source_id, embedding in stored:
            score = _cosine(query_vector, self._parse_vector(embedding))
            if score > 0.0:
                scored.append(
                    Chunk(text=str(row_text), source_id=str(source_id), score=score)
                )
        scored.sort(key=lambda chunk: chunk.score, reverse=True)
        return scored[:k]

    def forget(self, query_or_source_id: str) -> int:
        """Delete rows matching ``query_or_source_id``; return the count removed.

        A row matches when its ``source_id`` equals the argument *or* its text
        contains the argument as a case-insensitive substring. Parametrized
        (injection-safe). Returns the number of rows deleted (0 when none matched).
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE source_id = %s "
                "OR position(lower(%s) in lower(text)) > 0",
                (query_or_source_id, query_or_source_id),
            )
            removed = cur.rowcount
        self._conn.commit()
        return int(removed)

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _parse_vector(value: Any) -> list[float]:
        """Parse a stored embedding back into a list of floats.

        pgvector returns its ``vector`` column either as a Python ``list`` (when a
        type adapter is registered) or as its text literal ``"[a,b,c]"``; both are
        handled so the store works with or without the ``pgvector`` Python helper.
        """
        if isinstance(value, (list, tuple)):
            return [float(component) for component in value]
        text = str(value).strip().strip("[]")
        if not text:
            return []
        return [float(part) for part in text.split(",")]
