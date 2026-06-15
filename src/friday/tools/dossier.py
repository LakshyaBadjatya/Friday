"""Entity dossier tool: compile what FRIDAY knows about a named entity.

:class:`DossierTool` answers "what do you know about X?" by stitching together
three injected, read-only sources into a single structured dossier:

* the **knowledge graph** (``graph_store``) — the entity node and the relations
  touching it (people/projects/things and how they connect);
* the **long-term store** (``long_term``) — durable facts the owner has told
  FRIDAY that mention the entity by name; and
* an optional **searcher** (``searcher``) — a read-only tool (e.g. a web search)
  consulted only when the local graph/memory turns up nothing, so a brand-new
  name still yields *some* grounding rather than an empty dossier.

The dossier is ``{entity, facts, relations, summary}``: ``entity`` is the graph
node (or ``None`` when the name is unknown locally), ``facts`` is the list of
matching long-term fact texts, ``relations`` is the entity's local
neighbourhood, and ``summary`` is a short deterministic, human-readable line
assembled from those parts (no LLM, no fabrication).

Dependency injection only: every collaborator arrives through the constructor,
so this module imports neither :mod:`friday.config` nor :mod:`friday.app` and is
trivially unit-testable with fakes. The tool is read-only
(``side_effecting=False``, ``idempotent=True``).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from friday.tools.base import ToolResult

logger = logging.getLogger("friday.tools.dossier")

# Upper bound on facts pulled into a dossier from the long-term store, so a noisy
# fact history can never make a single dossier read unbounded.
_DOSSIER_FACTS_LIMIT = 20

# Upper bound on optional-searcher results folded into the summary/dossier.
_SEARCH_RESULTS_LIMIT = 5


# --------------------------------------------------------------------------- #
# Structural contracts for the injected collaborators (read-only slices)
# --------------------------------------------------------------------------- #
@runtime_checkable
class _GraphReader(Protocol):
    """The read slice of a knowledge-graph store the dossier needs.

    Structural so the concrete
    :class:`~friday.graph.store.SQLiteGraphStore` satisfies it without an
    import-time coupling. The dossier only ever *reads* — it never mutates the
    graph. ``get_entity`` returns a node (anything exposing ``model_dump``) or
    ``None``; ``neighbors`` returns the relations (each exposing ``model_dump``)
    touching the name.
    """

    def get_entity(self, name: str) -> Any | None: ...

    def neighbors(self, name: str) -> list[Any]: ...


@runtime_checkable
class _FactReader(Protocol):
    """The read slice of the long-term store the dossier reads facts through.

    Structural so the concrete
    :class:`~friday.memory.long_term.SQLiteLongTermStore` satisfies it. Each
    returned row need only expose a ``text`` attribute (a
    :class:`~friday.memory.long_term.Fact` does), which the dossier serializes to
    a plain string.
    """

    def query_facts(self, query: str, limit: int = ...) -> list[Any]: ...


@runtime_checkable
class _Searcher(Protocol):
    """The optional read-only searcher fallback (e.g. a web-search tool).

    Anything implementing the :class:`~friday.tools.base.Tool` ``__call__``
    contract qualifies; the dossier passes an object exposing ``query`` and reads
    ``ToolResult.data["results"]`` back out. It is only consulted when the local
    graph + memory yield nothing, and any failure is swallowed (the dossier is
    best-effort and never raises because of the fallback).
    """

    args_model: type[BaseModel]

    async def __call__(self, args: Any) -> ToolResult: ...


class DossierArgs(BaseModel):
    """Arguments for :class:`DossierTool`: the entity name to compile a dossier on."""

    name: str = Field(min_length=1)


class DossierTool:
    """Compile a structured dossier on an entity from graph + memory (+ search).

    Args:
        graph_store: The knowledge-graph read source (``get_entity`` /
            ``neighbors``). Read-only; never mutated.
        long_term: The durable long-term store (``query_facts``); facts that
            mention the entity by name are folded into the dossier.
        searcher: Optional read-only fallback tool (keyword-only). Consulted only
            when the graph and long-term store return nothing for the name, so a
            brand-new entity still produces some grounding. Any failure is
            swallowed; the dossier remains best-effort.
    """

    name = "entity_dossier"
    description = (
        "Compile a dossier on a person, project, or thing: the known entity, "
        "facts, relations, and a short summary, from the local knowledge graph "
        "and long-term memory (with an optional search fallback)."
    )
    args_model = DossierArgs
    required_permission = "memory"
    idempotent = True
    side_effecting = False

    def __init__(
        self,
        graph_store: _GraphReader,
        long_term: _FactReader,
        *,
        searcher: _Searcher | None = None,
    ) -> None:
        self._graph = graph_store
        self._long_term = long_term
        self._searcher = searcher

    async def __call__(self, args: Any) -> ToolResult:
        """Assemble ``{entity, facts, relations, summary}`` for ``args.name``."""
        if not isinstance(args, DossierArgs):
            args = DossierArgs.model_validate(args)
        name = args.name

        entity = self._read_entity(name)
        relations = self._read_relations(name)
        facts = self._read_facts(name)

        search_results: list[dict[str, Any]] = []
        # Only reach out to the (optional) searcher when local knowledge is empty,
        # so we never pay for a network call when the graph/memory already answers.
        if (
            self._searcher is not None
            and entity is None
            and not relations
            and not facts
        ):
            search_results = await self._search(name)

        summary = self._summarize(name, entity, relations, facts, search_results)
        data: dict[str, Any] = {
            "entity": entity,
            "facts": facts,
            "relations": relations,
            "summary": summary,
        }
        if search_results:
            data["search_results"] = search_results
        logger.info(
            "dossier compiled name=%r facts=%d relations=%d entity=%s search=%d",
            name,
            len(facts),
            len(relations),
            entity is not None,
            len(search_results),
        )
        return ToolResult(ok=True, data=data, error=None)

    # -- read helpers ------------------------------------------------------- #
    def _read_entity(self, name: str) -> dict[str, Any] | None:
        """Return the graph node for ``name`` serialized to a dict, or ``None``."""
        entity = self._graph.get_entity(name)
        if entity is None:
            return None
        return self._to_dict(entity)

    def _read_relations(self, name: str) -> list[dict[str, Any]]:
        """Return every relation touching ``name`` serialized to dicts."""
        return [self._to_dict(rel) for rel in self._graph.neighbors(name)]

    def _read_facts(self, name: str) -> list[str]:
        """Return long-term fact texts mentioning ``name`` (bounded)."""
        rows = self._long_term.query_facts(name, limit=_DOSSIER_FACTS_LIMIT)
        return [str(row.text) for row in rows]

    async def _search(self, name: str) -> list[dict[str, Any]]:
        """Best-effort fallback search; swallow any failure (never raises)."""
        searcher = self._searcher
        if searcher is None:  # pragma: no cover - guarded by caller
            return []
        try:
            search_args = searcher.args_model.model_validate({"query": name})
            result = await searcher(search_args)
        except Exception:  # noqa: BLE001 - fallback is strictly best-effort
            logger.warning("dossier search fallback failed for %r", name, exc_info=True)
            return []
        if not result.ok:
            return []
        raw = result.data.get("results", [])
        if not isinstance(raw, list):
            return []
        results: list[dict[str, Any]] = [
            item for item in raw if isinstance(item, dict)
        ]
        return results[:_SEARCH_RESULTS_LIMIT]

    # -- summary ------------------------------------------------------------ #
    @staticmethod
    def _summarize(
        name: str,
        entity: dict[str, Any] | None,
        relations: list[dict[str, Any]],
        facts: list[str],
        search_results: list[dict[str, Any]],
    ) -> str:
        """Build a short deterministic, human-readable summary line.

        Purely mechanical (no LLM): describes what was found so the caller has a
        one-line gist without re-deriving it from the structured parts.
        """
        parts: list[str] = []
        if entity is not None:
            etype = entity.get("type")
            parts.append(
                f"{name} is a known {etype}." if etype else f"{name} is known."
            )
        else:
            parts.append(f"No knowledge-graph entry for {name}.")
        if relations:
            parts.append(
                f"{len(relations)} relation"
                f"{'' if len(relations) == 1 else 's'} recorded."
            )
        if facts:
            parts.append(
                f"{len(facts)} fact{'' if len(facts) == 1 else 's'} on file."
            )
        if not relations and not facts and entity is None:
            if search_results:
                parts.append(
                    f"No local records; {len(search_results)} web result"
                    f"{'' if len(search_results) == 1 else 's'} found."
                )
            else:
                parts.append("Nothing on file.")
        return " ".join(parts)

    @staticmethod
    def _to_dict(obj: Any) -> dict[str, Any]:
        """Serialize a pydantic-like row to a dict, tolerating plain mappings."""
        dump = getattr(obj, "model_dump", None)
        if callable(dump):
            result = dump()
            if isinstance(result, dict):
                return result
        if isinstance(obj, dict):
            return dict(obj)
        return {"value": obj}
