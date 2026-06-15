"""LLM-driven entity/relation extraction for the knowledge graph (Tier 2).

:class:`EntityExtractor` is the single seam that turns a free-text note into a
small set of typed entities + relations and applies them to a
:class:`~friday.graph.store.SQLiteGraphStore`. It depends only on the typed
:class:`~friday.providers.llm.LLMProvider` boundary, so it imports no SDK and runs
fully offline against a scripted :class:`~friday.providers.llm.FakeLLM` in tests.

One binding rule:

* **Extraction is NON-FATAL.** Exactly one LLM pass asks for a single strict-JSON
  ``{entities, relations}`` object derived from the text. *Any* failure — a
  provider error/timeout, empty text, or a payload that does not parse into the
  expected shape — degrades to the **empty result** (no entities, no relations).
  :meth:`EntityExtractor.extract` never raises because the LLM (or its output)
  misbehaved, so a graph write can be attempted on every turn without risk.

The parse reuses the bounded ``{...}`` slice helper (a model may wrap the JSON in
prose or a code fence) and a strict pydantic shape so any deviation simply trips
the non-fatal fallback rather than corrupting the graph.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from friday.graph.store import SQLiteGraphStore
from friday.providers.llm import LLMProvider, Message

logger = logging.getLogger("friday.graph.extractor")

# Instruction handed to the LLM. It asks for a single strict-JSON object so the
# parse is deterministic; any deviation simply trips the non-fatal empty fallback.
_EXTRACT_INSTRUCTION = (
    "You are building a small knowledge graph from a note. Reply with a single "
    "JSON object and nothing else, of the exact shape "
    '{"entities": [{"name": str, "type": str, "attrs": object}], '
    '"relations": [{"src": str, "dst": str, "kind": str}]}. '
    "An entity is a concrete person, project, organization, place, or thing named "
    "in the note; type is a short lowercase label (e.g. person, project, org). "
    "attrs is an optional object of extra structured facts (may be empty). A "
    "relation connects two entity names by a short lowercase kind (e.g. "
    "works_on, located_in, member_of). Use entity names exactly as they appear "
    "in the relations. Do not invent entities or relations not supported by the "
    "note; return empty lists if there are none.\n\nNote:\n"
)


class ExtractedEntity(BaseModel):
    """One extracted entity: a name, a short type, and optional attrs.

    ``extra`` keys are ignored so a chatty model that adds fields still parses;
    ``name``/``type`` are required so a malformed item raises and trips the
    non-fatal fallback for the whole pass.
    """

    model_config = {"extra": "ignore"}

    name: str
    type: str
    attrs: dict[str, Any] = Field(default_factory=dict)


class ExtractedRelation(BaseModel):
    """One extracted relation: a directed ``(src, dst)`` edge named by ``kind``."""

    model_config = {"extra": "ignore"}

    src: str
    dst: str
    kind: str


class ExtractionResult(BaseModel):
    """The strict shape the LLM extraction pass is parsed into.

    Both lists default to empty so a payload that omits one still parses; the
    items are strictly typed, so a malformed entry raises
    :class:`~pydantic.ValidationError` and trips the non-fatal empty fallback.
    """

    model_config = {"extra": "ignore"}

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


# The empty (non-fatal) result returned on any extraction failure.
_EMPTY_RESULT: dict[str, list[Any]] = {"entities": [], "relations": []}


class EntityExtractor:
    """Extract entities + relations from text and apply them to the graph.

    Args:
        llm: The live LLM provider used for the single, non-fatal extraction pass.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def extract(self, text: str) -> dict[str, list[dict[str, Any]]]:
        """One LLM pass for ``{entities, relations}``; never raise.

        The completion and its JSON parse are wrapped in a broad ``except``: any
        provider error/timeout, empty text, or a payload that does not parse into
        the expected ``{entities, relations}`` shape degrades to the empty result
        (no entities, no relations). The return value is plain dicts so callers
        (the route, :meth:`upsert_into`) need no model coupling.
        """
        prompt = _EXTRACT_INSTRUCTION + text
        try:
            response = await self._llm.complete(
                [Message(role="user", content=prompt)]
            )
            raw = (response.text or "").strip()
            if not raw:
                raise ValueError("empty LLM extraction")
            parsed = ExtractionResult.model_validate_json(_extract_json(raw))
        except Exception:  # noqa: BLE001 - extraction is optional + non-fatal
            logger.warning(
                "graph entity extraction failed; returning empty result"
            )
            return {"entities": [], "relations": []}
        return {
            "entities": [entity.model_dump() for entity in parsed.entities],
            "relations": [relation.model_dump() for relation in parsed.relations],
        }

    async def upsert_into(
        self, store: SQLiteGraphStore, text: str
    ) -> dict[str, list[dict[str, Any]]]:
        """Extract from ``text`` and apply the result to ``store``; never raise.

        Runs :meth:`extract` (already non-fatal) and upserts each entity (idempotent
        by ``(name, type)``) and adds each relation (idempotent by triple). The same
        extraction dict is returned so the caller can report what was applied; a
        failed extraction yields the empty result and writes nothing.
        """
        result = await self.extract(text)
        for entity in result["entities"]:
            store.upsert_entity(
                name=str(entity["name"]),
                type=str(entity["type"]),
                attrs=dict(entity.get("attrs") or {}),
            )
        for relation in result["relations"]:
            store.add_relation(
                src=str(relation["src"]),
                dst=str(relation["dst"]),
                kind=str(relation["kind"]),
            )
        return result


def _extract_json(text: str) -> str:
    """Return the first ``{...}`` JSON object substring of ``text``.

    Tolerates a model that wraps the JSON in prose or a ```` ```json ```` fence by
    slicing from the first ``{`` to the last ``}``. When no braces are present the
    original text is returned so the downstream parse fails loudly into the
    non-fatal fallback rather than silently succeeding on garbage. Mirrors the
    meeting-capture helper.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start : end + 1]
