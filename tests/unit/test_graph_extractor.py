"""Unit tests for :class:`friday.graph.extractor.EntityExtractor` (Tier 2).

The extractor turns a free-text note into a small ``{entities, relations}`` set
via one LLM pass and applies it to a :class:`SQLiteGraphStore`. Every test here is
offline against a scripted :class:`~friday.providers.llm.FakeLLM` (zero network)
and a ``":memory:"`` graph store.

Pinned behaviours:

* A scripted strict-JSON ``{entities, relations}`` response is parsed and
  ``upsert_into`` applies it: entities are upserted (idempotent by ``(name, type)``)
  and relations added.
* Extraction is NON-FATAL: an LLM error (exhausted script / provider error), an
  empty completion, and an unparseable payload each yield the *empty* result and
  write nothing — ``extract`` never raises.
* JSON wrapped in prose / a code fence still parses (bounded ``{...}`` slice).
"""

from __future__ import annotations

from friday.graph.extractor import EntityExtractor
from friday.graph.store import SQLiteGraphStore
from friday.providers.llm import FakeLLM, LLMProvider, LLMResponse, Message, ToolSpec


def _llm(text: str | None) -> FakeLLM:
    """A FakeLLM scripted with a single completion whose ``text`` is ``text``."""
    return FakeLLM(responses=[LLMResponse(text=text)])


class _BoomLLM(LLMProvider):
    """An LLM that always raises, to exercise the non-fatal provider-error path."""

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        raise RuntimeError("provider exploded")


# --------------------------------------------------------------------------- #
# happy path: scripted JSON -> parsed + applied
# --------------------------------------------------------------------------- #
async def test_extract_parses_scripted_json() -> None:
    payload = (
        '{"entities": [{"name": "Ada", "type": "person", "attrs": {"role": "lead"}},'
        ' {"name": "Zephyr", "type": "project", "attrs": {}}],'
        ' "relations": [{"src": "Ada", "dst": "Zephyr", "kind": "works_on"}]}'
    )
    extractor = EntityExtractor(_llm(payload))
    result = await extractor.extract("Ada leads project Zephyr.")

    assert [e["name"] for e in result["entities"]] == ["Ada", "Zephyr"]
    assert result["entities"][0]["attrs"] == {"role": "lead"}
    assert result["relations"] == [
        {"src": "Ada", "dst": "Zephyr", "kind": "works_on"}
    ]


async def test_upsert_into_applies_entities_and_relations() -> None:
    payload = (
        '{"entities": [{"name": "Ada", "type": "person"},'
        ' {"name": "Zephyr", "type": "project"}],'
        ' "relations": [{"src": "Ada", "dst": "Zephyr", "kind": "works_on"}]}'
    )
    store = SQLiteGraphStore(":memory:")
    extractor = EntityExtractor(_llm(payload))

    result = await extractor.upsert_into(store, "Ada leads project Zephyr.")

    # The graph now has both entities and the relation.
    names = {e.name for e in store.list_entities()}
    assert names == {"Ada", "Zephyr"}
    assert store.get_entity("Ada").type == "person"  # type: ignore[union-attr]
    neighbors = store.neighbors("Ada")
    assert len(neighbors) == 1
    assert neighbors[0].kind == "works_on"
    # The applied result is returned for reporting.
    assert len(result["entities"]) == 2


async def test_extract_tolerates_prose_wrapped_json() -> None:
    """A model that wraps the JSON in prose / a fence still parses (bounded slice)."""
    payload = (
        "Sure! Here is the graph:\n```json\n"
        '{"entities": [{"name": "Ada", "type": "person"}], "relations": []}\n'
        "```\nLet me know if you need more."
    )
    extractor = EntityExtractor(_llm(payload))
    result = await extractor.extract("Ada is here.")
    assert [e["name"] for e in result["entities"]] == ["Ada"]


# --------------------------------------------------------------------------- #
# non-fatal: every failure mode -> empty result, never raises, never writes
# --------------------------------------------------------------------------- #
async def test_extract_provider_error_yields_empty() -> None:
    extractor = EntityExtractor(_BoomLLM())
    result = await extractor.extract("anything")
    assert result == {"entities": [], "relations": []}


async def test_extract_exhausted_script_yields_empty() -> None:
    """An exhausted FakeLLM raises ProviderError -> caught -> empty result."""
    extractor = EntityExtractor(FakeLLM(responses=[]))
    result = await extractor.extract("anything")
    assert result == {"entities": [], "relations": []}


async def test_extract_empty_completion_yields_empty() -> None:
    extractor = EntityExtractor(_llm(""))
    result = await extractor.extract("anything")
    assert result == {"entities": [], "relations": []}


async def test_extract_unparseable_payload_yields_empty() -> None:
    extractor = EntityExtractor(_llm("not json at all, just chatter"))
    result = await extractor.extract("anything")
    assert result == {"entities": [], "relations": []}


async def test_extract_malformed_shape_yields_empty() -> None:
    """Valid JSON but the wrong shape (entity missing ``type``) trips the fallback."""
    extractor = EntityExtractor(_llm('{"entities": [{"name": "Ada"}], "relations": []}'))
    result = await extractor.extract("anything")
    assert result == {"entities": [], "relations": []}


async def test_upsert_into_on_failure_writes_nothing() -> None:
    store = SQLiteGraphStore(":memory:")
    extractor = EntityExtractor(_BoomLLM())
    result = await extractor.upsert_into(store, "anything")
    assert result == {"entities": [], "relations": []}
    assert store.list_entities() == []
