"""Unit tests for :class:`friday.graph.store.SQLiteGraphStore` (Tier 2).

The graph store is the local-first, SQLite-backed durable layer for FRIDAY's tiny
knowledge graph (entities + relations). Every test here is offline and uses a
*tmp-file* database (so the connection-per-call path is exercised and the store is
thread-safe by construction).

Pinned behaviours:

* ``upsert_entity`` is idempotent on ``(name, type)`` — a second upsert merges
  attrs into the *same* row (stable id, no duplicate); ``list_entities``/``search``
  see one entity.
* ``add_relation`` records a directed edge; ``neighbors`` returns every edge
  touching a name (outgoing *and* incoming).
* ``entity_card`` aggregates ``{entity, relations, facts}`` — facts are pulled
  from a supplied long-term store, matching the entity name; with no store the
  facts list is empty.
"""

from __future__ import annotations

from pathlib import Path

from friday.graph.store import Entity, Relation, SQLiteGraphStore
from friday.memory.long_term import SQLiteLongTermStore


def _store(tmp_path: Path) -> SQLiteGraphStore:
    """A tmp-file store (exercises the connection-per-call path)."""
    return SQLiteGraphStore(str(tmp_path / "graph.db"))


# --------------------------------------------------------------------------- #
# upsert idempotency
# --------------------------------------------------------------------------- #
def test_upsert_entity_returns_typed_entity_with_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    entity = store.upsert_entity("Ada", "person", {"role": "engineer"})
    assert isinstance(entity, Entity)
    assert entity.id >= 1
    assert entity.name == "Ada"
    assert entity.type == "person"
    assert entity.attrs == {"role": "engineer"}


def test_upsert_entity_is_idempotent_by_name_and_type(tmp_path: Path) -> None:
    """A second upsert of the same (name, type) merges into one row, same id."""
    store = _store(tmp_path)
    first = store.upsert_entity("Ada", "person", {"role": "engineer"})
    second = store.upsert_entity("Ada", "person", {"role": "lead", "team": "infra"})

    # Same identity -> same stable id, attrs updated (last write wins).
    assert second.id == first.id
    assert second.attrs == {"role": "lead", "team": "infra"}

    # And only one row exists for that identity.
    listed = store.list_entities()
    assert len(listed) == 1
    assert listed[0].id == first.id


def test_upsert_same_name_different_type_is_two_entities(tmp_path: Path) -> None:
    """Identity is the (name, type) pair, so a shared name across types stays distinct."""
    store = _store(tmp_path)
    store.upsert_entity("Mercury", "planet")
    store.upsert_entity("Mercury", "element")
    assert len(store.list_entities()) == 2
    assert {e.type for e in store.list_entities()} == {"planet", "element"}


def test_upsert_entity_defaults_attrs_to_empty(tmp_path: Path) -> None:
    store = _store(tmp_path)
    entity = store.upsert_entity("Zephyr", "project")
    assert entity.attrs == {}


def test_schema_init_is_idempotent(tmp_path: Path) -> None:
    db = str(tmp_path / "graph.db")
    first = SQLiteGraphStore(db)
    first.upsert_entity("Ada", "person")
    # Re-opening over the existing file must not clobber data.
    second = SQLiteGraphStore(db)
    assert second.get_entity("Ada") is not None


# --------------------------------------------------------------------------- #
# get / list / search
# --------------------------------------------------------------------------- #
def test_get_entity_returns_none_when_absent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.get_entity("Nobody") is None


def test_list_entities_filters_by_type(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert_entity("Ada", "person")
    store.upsert_entity("Grace", "person")
    store.upsert_entity("Zephyr", "project")

    people = store.list_entities(type="person")
    assert [e.name for e in people] == ["Ada", "Grace"]
    assert len(store.list_entities()) == 3


def test_search_matches_name_substring_case_insensitive(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert_entity("Zephyr", "project")
    store.upsert_entity("Apollo", "project")
    hits = store.search("zeph")
    assert [e.name for e in hits] == ["Zephyr"]


def test_search_treats_wildcards_literally(tmp_path: Path) -> None:
    """A ``%`` in the query matches a literal ``%``, not "anything"."""
    store = _store(tmp_path)
    store.upsert_entity("100% Done", "task")
    store.upsert_entity("Other", "task")
    hits = store.search("100%")
    assert [e.name for e in hits] == ["100% Done"]


# --------------------------------------------------------------------------- #
# relations + neighbors
# --------------------------------------------------------------------------- #
def test_add_relation_returns_typed_relation_with_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    relation = store.add_relation("Ada", "Zephyr", "works_on")
    assert isinstance(relation, Relation)
    assert relation.id >= 1
    assert relation.src == "Ada"
    assert relation.dst == "Zephyr"
    assert relation.kind == "works_on"


def test_add_relation_is_idempotent_by_triple(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.add_relation("Ada", "Zephyr", "works_on")
    second = store.add_relation("Ada", "Zephyr", "works_on")
    assert second.id == first.id
    assert len(store.neighbors("Ada")) == 1


def test_neighbors_returns_outgoing_and_incoming_edges(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.add_relation("Ada", "Zephyr", "works_on")  # outgoing for Ada
    store.add_relation("Grace", "Ada", "mentors")  # incoming for Ada
    store.add_relation("Grace", "Zephyr", "works_on")  # unrelated to Ada

    ada = store.neighbors("Ada")
    kinds = {(r.src, r.dst, r.kind) for r in ada}
    assert kinds == {
        ("Ada", "Zephyr", "works_on"),
        ("Grace", "Ada", "mentors"),
    }


def test_neighbors_empty_for_unknown_name(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.neighbors("Nobody") == []


# --------------------------------------------------------------------------- #
# entity_card aggregation
# --------------------------------------------------------------------------- #
def test_entity_card_aggregates_entity_relations_and_facts(tmp_path: Path) -> None:
    """The card stitches the entity + its relations + matching long-term facts."""
    store = _store(tmp_path)
    store.upsert_entity("Ada", "person", {"role": "lead"})
    store.add_relation("Ada", "Zephyr", "works_on")
    store.add_relation("Grace", "Ada", "mentors")

    long_term = SQLiteLongTermStore(":memory:")
    # Seed a fact that mentions the entity by name, plus one that does not.
    long_term.add_fact("Ada prefers async standups.", source_id="chat:1")
    long_term.add_fact("The kitchen light is in the allow-list.", source_id="chat:2")

    card = store.entity_card("Ada", long_term=long_term)

    assert card["entity"] is not None
    assert card["entity"]["name"] == "Ada"
    assert card["entity"]["attrs"] == {"role": "lead"}

    rel_triples = {(r["src"], r["dst"], r["kind"]) for r in card["relations"]}
    assert rel_triples == {
        ("Ada", "Zephyr", "works_on"),
        ("Grace", "Ada", "mentors"),
    }

    # Only the fact mentioning "Ada" is included.
    assert card["facts"] == ["Ada prefers async standups."]


def test_entity_card_without_long_term_has_empty_facts(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert_entity("Ada", "person")
    store.add_relation("Ada", "Zephyr", "works_on")

    card = store.entity_card("Ada")
    assert card["facts"] == []
    assert card["entity"]["name"] == "Ada"
    assert len(card["relations"]) == 1


def test_entity_card_unknown_name_returns_null_entity(tmp_path: Path) -> None:
    """An unknown node is not an error: entity is None, relations/facts empty."""
    store = _store(tmp_path)
    card = store.entity_card("Nobody")
    assert card["entity"] is None
    assert card["relations"] == []
    assert card["facts"] == []
