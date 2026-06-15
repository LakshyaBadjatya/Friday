"""Integration tests for the ``/graph`` knowledge-graph API (Tier 2).

All offline against a :class:`~friday.providers.llm.FakeLLM` and ``":memory:"``
stores, with a ``TestClient`` whose ``FRIDAY_ENABLE_KNOWLEDGE_GRAPH`` flag is
forced on/off via a monkeypatched ``get_settings`` (mirroring the RAG/reminders
API tests). No network, no key.

Covered:
* Every ``/graph`` route is ``404`` when the flag is off (the feature does not
  exist for callers).
* ``POST /graph/extract`` with a scripted FakeLLM JSON upserts entities/relations
  and ``GET /graph/entities`` lists them (and filters by ``type``).
* ``GET /graph/entity/{name}`` returns the entity card aggregating the entity,
  its relations, and the long-term facts that mention the name (seeded on the
  shared long-term store).
* An extraction whose LLM pass fails is NON-FATAL: ``POST /graph/extract`` still
  returns ``200`` with an empty ``{entities, relations}``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import Settings
from friday.providers.llm import FakeLLM, LLMResponse


def _enable_graph_settings() -> Settings:
    # ``":memory:"`` keeps every app instance's stores ephemeral and isolated so
    # tests never touch the developer's real ``data/`` files or each other.
    return Settings(
        _env_file=None,
        enable_knowledge_graph=True,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _disable_graph_settings() -> Settings:
    return Settings(
        _env_file=None,
        enable_knowledge_graph=False,
        llm_provider="fake",
        memory_db_path=":memory:",
    )


def _client(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> TestClient:
    """A ``TestClient`` whose graph flag is forced via a patched ``get_settings``."""
    import friday.app as app_module

    factory = _enable_graph_settings if enabled else _disable_graph_settings
    monkeypatch.setattr(app_module, "get_settings", factory)
    app = create_app()
    return TestClient(app)


def _script_extractor(client: TestClient, *responses: LLMResponse) -> None:
    """Replace the app's entity extractor with one over a scripted FakeLLM.

    The default offline LLM is an empty-script ``FakeLLM`` (every completion
    raises), so to exercise the happy extract path the test installs an extractor
    whose LLM returns the given scripted JSON. The store is left as the wired one,
    so the upsert lands in the same graph the card/list routes read.
    """
    from friday.graph.extractor import EntityExtractor

    client.app.state.graph_extractor = EntityExtractor(FakeLLM(responses=list(responses)))


# --------------------------------------------------------------------------- #
# Disabled -> 404 on every surface
# --------------------------------------------------------------------------- #
def test_graph_disabled_entities_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_graph_settings()
        resp = client.get("/graph/entities")
    assert resp.status_code == 404


def test_graph_disabled_entity_card_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_graph_settings()
        resp = client.get("/graph/entity/Ada")
    assert resp.status_code == 404


def test_graph_disabled_extract_is_404(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=False) as client:
        client.app.state.settings = _disable_graph_settings()
        resp = client.post("/graph/extract", json={"text": "Ada leads Zephyr."})
    assert resp.status_code == 404


def test_graph_default_off_is_404() -> None:
    """With pristine env-default settings (flag off), the entities list is 404."""
    app = create_app()
    with TestClient(app) as client:
        client.app.state.settings = _disable_graph_settings()
        resp = client.get("/graph/entities")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Enabled -> extract, list (+ filter), entity card
# --------------------------------------------------------------------------- #
def test_graph_extract_then_list_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scripted extract upserts entities/relations; the list reflects them."""
    payload = (
        '{"entities": [{"name": "Ada", "type": "person"},'
        ' {"name": "Zephyr", "type": "project"}],'
        ' "relations": [{"src": "Ada", "dst": "Zephyr", "kind": "works_on"}]}'
    )
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_graph_settings()
        _script_extractor(client, LLMResponse(text=payload))

        resp = client.post("/graph/extract", json={"text": "Ada leads project Zephyr."})
        assert resp.status_code == 200
        applied = resp.json()
        assert {e["name"] for e in applied["entities"]} == {"Ada", "Zephyr"}
        assert applied["relations"][0]["kind"] == "works_on"

        listed = client.get("/graph/entities")
        assert listed.status_code == 200
        body = listed.json()
        assert body["count"] == 2
        assert {e["name"] for e in body["entities"]} == {"Ada", "Zephyr"}

        # ?type= filters to one type.
        people = client.get("/graph/entities", params={"type": "person"})
        assert [e["name"] for e in people.json()["entities"]] == ["Ada"]


def test_graph_entity_card_aggregates_relations_and_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The entity card stitches the entity + relations + matching long-term facts."""
    payload = (
        '{"entities": [{"name": "Ada", "type": "person", "attrs": {"role": "lead"}},'
        ' {"name": "Zephyr", "type": "project"}],'
        ' "relations": [{"src": "Ada", "dst": "Zephyr", "kind": "works_on"}]}'
    )
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_graph_settings()
        _script_extractor(client, LLMResponse(text=payload))

        # Seed a long-term fact that mentions the entity (the card pulls it in).
        client.app.state.long_term.add_fact(
            "Ada prefers async standups.", source_id="chat:1"
        )

        client.post("/graph/extract", json={"text": "Ada leads project Zephyr."})

        card = client.get("/graph/entity/Ada")
        assert card.status_code == 200
        body = card.json()
        assert body["entity"]["name"] == "Ada"
        assert body["entity"]["attrs"] == {"role": "lead"}
        rel_triples = {
            (r["src"], r["dst"], r["kind"]) for r in body["relations"]
        }
        assert ("Ada", "Zephyr", "works_on") in rel_triples
        assert body["facts"] == ["Ada prefers async standups."]


def test_graph_entity_card_unknown_name_is_200_with_null_entity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown node is not a 404 — the card returns null entity + empty lists."""
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_graph_settings()
        card = client.get("/graph/entity/Nobody")
        assert card.status_code == 200
        body = card.json()
        assert body["entity"] is None
        assert body["relations"] == []
        assert body["facts"] == []


def test_graph_extract_failure_is_non_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An extraction whose LLM pass fails returns 200 with an empty result."""
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_graph_settings()
        # An empty-script FakeLLM raises on completion -> non-fatal empty result.
        _script_extractor(client)

        resp = client.post("/graph/extract", json={"text": "Ada leads Zephyr."})
        assert resp.status_code == 200
        assert resp.json() == {"entities": [], "relations": []}

        # Nothing was written.
        listed = client.get("/graph/entities")
        assert listed.json()["count"] == 0


def test_graph_extract_bad_body_is_422(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch, enabled=True) as client:
        client.app.state.settings = _enable_graph_settings()
        resp = client.post("/graph/extract", json={"text": ""})
    assert resp.status_code == 422
