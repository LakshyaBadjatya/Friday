"""Unit tests for the Postgres/pgvector adapter swap (``friday.memory.pg``).

This is the Tier-3 ``postgres`` feature: drop-in replacements for the local-first
SQLite stores that talk to a Postgres database via ``psycopg`` (and pgvector for
the vector store). ``psycopg`` is an OPTIONAL dependency kept OUT of the uv lock,
so these tests pin three things WITHOUT a live database and WITHOUT ``psycopg``
installed:

* **Import is dependency-free.** ``import friday.memory.pg`` must succeed even
  when ``psycopg`` is absent — the driver is only ever lazy-imported inside the
  adapters, never at module top.
* **Constructing an adapter fails loudly.** With ``psycopg`` missing (or no DSN
  configured) building either adapter raises a clear
  :class:`~friday.errors.FridayError` telling the operator to install ``psycopg``
  and set ``FRIDAY_POSTGRES_DSN`` — never a bare ``ImportError`` or a half-open
  connection.
* **The classes satisfy the existing interfaces structurally.** ``Postgres
  LongTermStore`` carries the :class:`~friday.memory.long_term.LongTermStore`
  surface and ``PgVectorStore`` the :class:`~friday.memory.vector.VectorStore`
  surface, so the integration agent can swap them in behind the flag.

No network, no Postgres, no ``psycopg``: everything here is offline and
deterministic.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from friday.errors import FridayError
from friday.memory.long_term import LongTermStore
from friday.memory.vector import VectorStore

# A syntactically valid DSN; nothing here ever connects, so it is never dialled.
_DSN = "postgresql://user:pw@localhost:5432/friday"


@pytest.fixture
def pg_module() -> Any:
    """Import (fresh) and return ``friday.memory.pg``.

    Reimporting from a clean module-cache slot guarantees the import itself does
    not transitively require ``psycopg`` — if the module had a top-level
    ``import psycopg`` this fixture would raise here, since ``psycopg`` is not
    installed in the gate environment.
    """
    sys.modules.pop("friday.memory.pg", None)
    return importlib.import_module("friday.memory.pg")


@pytest.fixture
def block_psycopg(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Force ``import psycopg`` to fail, simulating the lib being absent.

    Even if a developer has ``psycopg`` installed locally, this makes the
    "driver missing" branch deterministic: any attempt to import ``psycopg``
    (or a submodule) raises :class:`ImportError`.
    """
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals_: Any = None,
        locals_: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        if name == "psycopg" or name.startswith("psycopg."):
            raise ImportError("No module named 'psycopg'")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "psycopg", raising=False)
    yield


# --------------------------------------------------------------------------- #
# Import is dependency-free
# --------------------------------------------------------------------------- #
def test_importing_pg_module_requires_no_psycopg(
    block_psycopg: None, pg_module: Any
) -> None:
    """``import friday.memory.pg`` succeeds with ``psycopg`` unimportable."""
    assert hasattr(pg_module, "PostgresLongTermStore")
    assert hasattr(pg_module, "PgVectorStore")
    # And the optional driver was never pulled into ``sys.modules`` by the import.
    assert "psycopg" not in sys.modules


def test_pg_module_has_no_top_level_psycopg_import(pg_module: Any) -> None:
    """No symbol named ``psycopg`` leaks into the module namespace at import."""
    assert not hasattr(pg_module, "psycopg")


# --------------------------------------------------------------------------- #
# Constructing without psycopg/dsn raises the clear error
# --------------------------------------------------------------------------- #
def test_long_term_store_without_psycopg_raises_clear_error(
    block_psycopg: None, pg_module: Any
) -> None:
    """Building the long-term adapter with no driver gives install guidance."""
    with pytest.raises(FridayError) as exc:
        pg_module.PostgresLongTermStore(_DSN)
    message = str(exc.value)
    assert "psycopg" in message
    assert "FRIDAY_POSTGRES_DSN" in message


def test_vector_store_without_psycopg_raises_clear_error(
    block_psycopg: None, pg_module: Any
) -> None:
    """Building the vector adapter with no driver gives install guidance."""
    with pytest.raises(FridayError) as exc:
        pg_module.PgVectorStore(_DSN, embedder=_FakeEmbedder(), dim=3)
    message = str(exc.value)
    assert "psycopg" in message
    assert "FRIDAY_POSTGRES_DSN" in message


def test_long_term_store_without_dsn_raises_clear_error(
    pg_module: Any,
) -> None:
    """An empty/None DSN is rejected before any driver import is attempted."""
    with pytest.raises(FridayError) as exc:
        pg_module.PostgresLongTermStore("")
    message = str(exc.value)
    assert "FRIDAY_POSTGRES_DSN" in message


def test_vector_store_without_dsn_raises_clear_error(pg_module: Any) -> None:
    """An empty/None DSN is rejected before any driver import is attempted."""
    with pytest.raises(FridayError) as exc:
        pg_module.PgVectorStore("", embedder=_FakeEmbedder(), dim=3)
    message = str(exc.value)
    assert "FRIDAY_POSTGRES_DSN" in message


def test_long_term_missing_dsn_does_not_need_psycopg(
    block_psycopg: None, pg_module: Any
) -> None:
    """The DSN guard fires even when psycopg is unimportable (no import needed)."""
    with pytest.raises(FridayError):
        pg_module.PostgresLongTermStore(None)


# --------------------------------------------------------------------------- #
# Structural interface conformance
# --------------------------------------------------------------------------- #
def test_postgres_long_term_store_satisfies_interface(pg_module: Any) -> None:
    """``PostgresLongTermStore`` carries the full ``LongTermStore`` surface."""
    cls = pg_module.PostgresLongTermStore
    for method in (
        "add_fact",
        "query_facts",
        "add_task",
        "task_history",
        "add_audit",
        "forget",
    ):
        assert callable(getattr(cls, method)), method
    # The protocol is structural; assert subclass conformance explicitly.
    assert issubclass(cls, LongTermStore)


def test_pg_vector_store_satisfies_interface(pg_module: Any) -> None:
    """``PgVectorStore`` carries the full ``VectorStore`` surface."""
    cls = pg_module.PgVectorStore
    for method in ("add", "query", "forget"):
        assert callable(getattr(cls, method)), method
    assert issubclass(cls, VectorStore)


def test_adapters_are_distinct_from_sqlite_stores(pg_module: Any) -> None:
    """The pg adapters are their own classes, not re-exported SQLite ones."""
    from friday.memory.long_term import SQLiteLongTermStore
    from friday.memory.vector import SQLiteVectorStore

    assert pg_module.PostgresLongTermStore is not SQLiteLongTermStore
    assert pg_module.PgVectorStore is not SQLiteVectorStore


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class _FakeEmbedder:
    """Minimal offline ``EmbeddingProvider`` stand-in (never actually called)."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


# Silence "unused import" for the typing-only ``Callable`` alias if linting runs
# before the symbol is referenced; it documents the fixture return shape.
_ = Callable
