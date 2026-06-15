"""Unit tests for ``friday.memory.long_term`` — the SQLite long-term store.

The Phase-4 long-term backend is a local-first, zero-server SQLite store
(stdlib ``sqlite3``) behind a ``LongTermStore`` protocol. Postgres is a
flagged, lazy-imported adapter swap that is never required for the gate.

These tests pin the spec (§10) round-trips and consent-relevant behaviour:

* ``add_fact`` then ``query_facts`` returns the fact carrying its ``source_id``;
* the ``sensitive`` flag is persisted and round-trips faithfully;
* ``forget(query)`` removes matching facts and a follow-up query returns ``[]``;
* tasks and audit records round-trip through their typed row models;
* tables are created idempotently and a file-backed store persists across
  reconnects;
* the ``PostgresLongTermStore`` stub raises a clear "Phase 6" error instead of
  silently pretending to work.

Everything runs offline against ``":memory:"`` or a ``tmp_path`` file — no
network, no key, no server.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from friday.errors import FridayError
from friday.memory.long_term import (
    AuditRow,
    Fact,
    LongTermStore,
    PostgresLongTermStore,
    SQLiteLongTermStore,
    TaskRow,
)


# --- protocol conformance --------------------------------------------------- #
def test_sqlite_store_satisfies_protocol() -> None:
    store: LongTermStore = SQLiteLongTermStore()
    assert isinstance(store, LongTermStore)


# --- facts round-trip ------------------------------------------------------- #
def test_add_fact_then_query_returns_it_with_source_id() -> None:
    store = SQLiteLongTermStore()
    fact = store.add_fact("the sky is blue today", source_id="weather-1")
    assert isinstance(fact, Fact)
    assert fact.id >= 1
    assert fact.text == "the sky is blue today"
    assert fact.source_id == "weather-1"
    assert fact.sensitive is False
    assert fact.created_at  # populated, non-empty timestamp

    results = store.query_facts("sky")
    assert len(results) == 1
    got = results[0]
    assert isinstance(got, Fact)
    assert got.id == fact.id
    assert got.text == "the sky is blue today"
    assert got.source_id == "weather-1"


def test_query_facts_substring_match_is_case_insensitive() -> None:
    store = SQLiteLongTermStore()
    store.add_fact("Boss prefers tea over coffee", source_id="pref-1")
    # Match on a differently-cased substring of the stored text.
    results = store.query_facts("TEA")
    assert [f.source_id for f in results] == ["pref-1"]


def test_query_facts_no_match_returns_empty() -> None:
    store = SQLiteLongTermStore()
    store.add_fact("apples and oranges", source_id="fruit")
    assert store.query_facts("quantum chromodynamics") == []


def test_query_facts_empty_store_returns_empty() -> None:
    store = SQLiteLongTermStore()
    assert store.query_facts("anything") == []


def test_query_facts_respects_limit() -> None:
    store = SQLiteLongTermStore()
    for i in range(5):
        store.add_fact(f"alpha fact number {i}", source_id=f"s{i}")
    results = store.query_facts("alpha", limit=2)
    assert len(results) == 2
    # All returned rows genuinely match the query token.
    assert all("alpha" in f.text for f in results)


# --- sensitive flag --------------------------------------------------------- #
def test_sensitive_flag_persisted() -> None:
    store = SQLiteLongTermStore()
    fact = store.add_fact(
        "home alarm code is 1234", source_id="secret-1", sensitive=True
    )
    assert fact.sensitive is True

    [got] = store.query_facts("alarm")
    assert got.sensitive is True


def test_non_sensitive_is_default_false() -> None:
    store = SQLiteLongTermStore()
    fact = store.add_fact("public note", source_id="n1")
    assert fact.sensitive is False
    [got] = store.query_facts("public")
    assert got.sensitive is False


# --- forget ----------------------------------------------------------------- #
def test_forget_removes_matching_and_followup_query_is_empty() -> None:
    store = SQLiteLongTermStore()
    store.add_fact("Boss lives in Mumbai", source_id="loc-1")
    store.add_fact("Boss likes filter coffee", source_id="pref-1")

    removed = store.forget("Mumbai")
    assert removed == 1

    # The forgotten fact is gone...
    assert store.query_facts("Mumbai") == []
    # ...but the unrelated fact survives.
    assert [f.source_id for f in store.query_facts("coffee")] == ["pref-1"]


def test_forget_removes_all_matching_rows_and_reports_count() -> None:
    store = SQLiteLongTermStore()
    store.add_fact("project apollo milestone one", source_id="a1")
    store.add_fact("project apollo milestone two", source_id="a2")
    store.add_fact("unrelated grocery list", source_id="g1")

    removed = store.forget("apollo")
    assert removed == 2
    assert store.query_facts("apollo") == []
    assert len(store.query_facts("grocery")) == 1


def test_forget_no_match_removes_nothing() -> None:
    store = SQLiteLongTermStore()
    store.add_fact("a single fact", source_id="s1")
    assert store.forget("nonexistent") == 0
    assert len(store.query_facts("single")) == 1


def test_forget_matches_sensitive_facts_too() -> None:
    store = SQLiteLongTermStore()
    store.add_fact("the safe combination is secret", source_id="x", sensitive=True)
    assert store.forget("combination") == 1
    assert store.query_facts("combination") == []


# --- tasks round-trip ------------------------------------------------------- #
def test_add_task_and_history_round_trip() -> None:
    store = SQLiteLongTermStore()
    row = store.add_task(intent="device", summary="turn on the lights", ok=True)
    assert isinstance(row, TaskRow)
    assert row.id >= 1
    assert row.intent == "device"
    assert row.summary == "turn on the lights"
    assert row.ok is True
    assert row.created_at

    history = store.task_history(limit=10)
    assert len(history) == 1
    assert history[0].id == row.id
    assert history[0].intent == "device"
    assert history[0].ok is True


def test_task_history_orders_most_recent_first_and_limits() -> None:
    store = SQLiteLongTermStore()
    for i in range(4):
        store.add_task(intent="analysis", summary=f"task {i}", ok=(i % 2 == 0))
    history = store.task_history(limit=2)
    assert len(history) == 2
    # Most recently inserted task has the highest id and comes first.
    assert history[0].id > history[1].id
    assert history[0].summary == "task 3"


def test_task_history_empty_returns_empty() -> None:
    store = SQLiteLongTermStore()
    assert store.task_history(limit=5) == []


# --- audit round-trip ------------------------------------------------------- #
def test_add_audit_and_round_trip() -> None:
    store = SQLiteLongTermStore()
    row = store.add_audit(step="route", ok=True, detail="intent=knowledge")
    assert isinstance(row, AuditRow)
    assert row.id >= 1
    assert row.step == "route"
    assert row.ok is True
    assert row.detail == "intent=knowledge"
    assert row.created_at

    history = store.audit_history(limit=10)
    assert len(history) == 1
    assert history[0].step == "route"
    assert history[0].detail == "intent=knowledge"


def test_audit_records_failures() -> None:
    store = SQLiteLongTermStore()
    store.add_audit(step="tool", ok=False, detail="boom")
    [got] = store.audit_history(limit=1)
    assert got.ok is False
    assert got.detail == "boom"


# --- persistence + idempotency ---------------------------------------------- #
def test_file_backed_store_persists_across_reconnect(tmp_path: Path) -> None:
    db = tmp_path / "friday.db"
    store = SQLiteLongTermStore(path=str(db))
    store.add_fact("durable knowledge", source_id="d1")

    # A fresh store over the same file sees the previously-written fact and does
    # not error re-creating the (already-existing) tables.
    reopened = SQLiteLongTermStore(path=str(db))
    results = reopened.query_facts("durable")
    assert [f.source_id for f in results] == ["d1"]


def test_init_is_idempotent_on_same_connection() -> None:
    store = SQLiteLongTermStore()
    # Re-initialising the schema must not raise (tables created IF NOT EXISTS).
    store.init_schema()
    store.init_schema()
    store.add_fact("still works", source_id="s")
    assert len(store.query_facts("works")) == 1


# --- SQL-injection safety (parametrized queries) ---------------------------- #
def test_query_with_sql_metacharacters_is_safe() -> None:
    store = SQLiteLongTermStore()
    store.add_fact("legitimate fact", source_id="ok")
    # A classic injection attempt must be treated as a literal substring, not
    # SQL, and simply match nothing — the table must still exist afterwards.
    assert store.query_facts("'; DROP TABLE facts; --") == []
    assert len(store.query_facts("legitimate")) == 1


def test_add_fact_with_quotes_round_trips_literally() -> None:
    store = SQLiteLongTermStore()
    weird = "O'Brien said \"hello\"; DROP TABLE facts;"
    store.add_fact(weird, source_id="q")
    [got] = store.query_facts("O'Brien")
    assert got.text == weird


# --- Postgres stub ---------------------------------------------------------- #
def test_postgres_store_raises_clear_phase6_error() -> None:
    store = PostgresLongTermStore(dsn="postgresql://localhost/friday")
    with pytest.raises(FridayError) as exc:
        store.add_fact("anything", source_id="x")
    message = str(exc.value)
    assert "Postgres" in message
    assert "Phase 6" in message


def test_postgres_store_query_also_raises() -> None:
    store = PostgresLongTermStore(dsn="postgresql://localhost/friday")
    with pytest.raises(FridayError):
        store.query_facts("anything")
