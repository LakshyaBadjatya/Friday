"""Unit tests for :class:`friday.tools.dossier.DossierTool`.

The tool is exercised with **fakes** for its injected collaborators — a fake
graph store, a fake long-term store, and (where relevant) a fake searcher — so
the tests are fully isolated, offline, and deterministic. They pin the protocol
attributes, the dossier assembly (entity + facts + relations + summary), the
search fallback (used only when local knowledge is empty, and best-effort on
failure), and a round-trip through a real :class:`ToolRegistry`.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel

from friday.tools.base import Tool, ToolError, ToolResult
from friday.tools.dossier import DossierArgs, DossierTool
from friday.tools.registry import ToolRegistry


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _Row(BaseModel):
    """A minimal pydantic row exposing ``model_dump`` like the real models."""

    id: int
    name: str = ""
    type: str = ""


class _Relation(BaseModel):
    id: int
    src: str
    dst: str
    kind: str


class _Fact(BaseModel):
    text: str


class FakeGraph:
    """A fake knowledge-graph reader (``get_entity`` / ``neighbors``)."""

    def __init__(
        self,
        entities: dict[str, _Row] | None = None,
        relations: dict[str, list[_Relation]] | None = None,
    ) -> None:
        self._entities = entities or {}
        self._relations = relations or {}

    def get_entity(self, name: str) -> _Row | None:
        return self._entities.get(name)

    def neighbors(self, name: str) -> list[_Relation]:
        return self._relations.get(name, [])


class FakeLongTerm:
    """A fake long-term store returning facts whose text contains the query."""

    def __init__(self, facts: list[str] | None = None) -> None:
        self._facts = facts or []

    def query_facts(self, query: str, limit: int = 10) -> list[_Fact]:
        matches = [f for f in self._facts if query.lower() in f.lower()]
        return [_Fact(text=t) for t in matches[:limit]]


class _SearchArgs(BaseModel):
    query: str


class FakeSearcher:
    """A fake read-only searcher matching the Tool ``__call__`` contract."""

    name = "web_search"
    description = "fake search"
    args_model = _SearchArgs
    required_permission = "web_search"
    idempotent = True
    side_effecting = False

    def __init__(
        self, results: list[dict[str, str]] | None = None, *, ok: bool = True
    ) -> None:
        self._results = results or []
        self._ok = ok
        self.calls: list[str] = []

    async def __call__(self, args: Any) -> ToolResult:
        self.calls.append(args.query)
        if not self._ok:
            return ToolResult(
                ok=False,
                data={"results": []},
                error=ToolError(code="search_failed", message="boom"),
            )
        return ToolResult(ok=True, data={"results": self._results})


class ExplodingSearcher:
    """A searcher that raises — the dossier must swallow it (best-effort)."""

    name = "web_search"
    description = "exploding search"
    args_model = _SearchArgs
    required_permission = "web_search"
    idempotent = True
    side_effecting = False

    async def __call__(self, args: Any) -> ToolResult:
        raise RuntimeError("network on fire")


# --------------------------------------------------------------------------- #
# Attributes / protocol conformance
# --------------------------------------------------------------------------- #
def test_dossier_tool_attrs() -> None:
    tool = DossierTool(FakeGraph(), FakeLongTerm())
    assert isinstance(tool, Tool)
    assert tool.name == "entity_dossier"
    assert tool.args_model is DossierArgs
    assert tool.side_effecting is False
    assert tool.idempotent is True


def test_dossier_args_requires_non_empty_name() -> None:
    args = DossierArgs(name="Ada")
    assert args.name == "Ada"


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
async def test_dossier_compiles_entity_facts_and_relations() -> None:
    graph = FakeGraph(
        entities={"Ada": _Row(id=1, name="Ada", type="person")},
        relations={
            "Ada": [_Relation(id=1, src="Ada", dst="Zephyr", kind="works_on")]
        },
    )
    long_term = FakeLongTerm(facts=["Ada likes climbing", "unrelated note"])
    tool = DossierTool(graph, long_term)

    result = await tool(DossierArgs(name="Ada"))
    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.error is None

    data = result.data
    assert data["entity"]["name"] == "Ada"
    assert data["entity"]["type"] == "person"
    assert data["facts"] == ["Ada likes climbing"]
    assert len(data["relations"]) == 1
    assert data["relations"][0]["dst"] == "Zephyr"
    # Summary mechanically reflects what was found.
    assert "person" in data["summary"]
    assert "1 relation" in data["summary"]
    assert "1 fact" in data["summary"]
    # No searcher injected -> no search_results key.
    assert "search_results" not in data


async def test_dossier_unknown_entity_returns_none_entity() -> None:
    tool = DossierTool(FakeGraph(), FakeLongTerm())
    result = await tool(DossierArgs(name="Nobody"))
    assert result.ok is True
    assert result.data["entity"] is None
    assert result.data["facts"] == []
    assert result.data["relations"] == []
    assert "Nothing on file" in result.data["summary"]


# --------------------------------------------------------------------------- #
# Search fallback
# --------------------------------------------------------------------------- #
async def test_dossier_uses_searcher_only_when_local_is_empty() -> None:
    searcher = FakeSearcher(
        results=[{"title": "Ada Lovelace", "url": "https://x", "snippet": "..."}]
    )
    tool = DossierTool(FakeGraph(), FakeLongTerm(), searcher=searcher)

    result = await tool(DossierArgs(name="Ada Lovelace"))
    assert result.ok is True
    assert searcher.calls == ["Ada Lovelace"]
    assert result.data["search_results"][0]["title"] == "Ada Lovelace"
    assert "web result" in result.data["summary"]


async def test_dossier_skips_searcher_when_local_has_data() -> None:
    graph = FakeGraph(entities={"Ada": _Row(id=1, name="Ada", type="person")})
    searcher = FakeSearcher(results=[{"title": "should not be used"}])
    tool = DossierTool(graph, FakeLongTerm(), searcher=searcher)

    result = await tool(DossierArgs(name="Ada"))
    assert result.ok is True
    # Local knowledge answered, so the fallback was never invoked.
    assert searcher.calls == []
    assert "search_results" not in result.data


async def test_dossier_search_failure_is_best_effort() -> None:
    # A failing search result must not break the dossier.
    searcher = FakeSearcher(ok=False)
    tool = DossierTool(FakeGraph(), FakeLongTerm(), searcher=searcher)
    result = await tool(DossierArgs(name="Ghost"))
    assert result.ok is True
    assert "search_results" not in result.data
    assert "Nothing on file" in result.data["summary"]


async def test_dossier_search_exception_is_swallowed() -> None:
    tool = DossierTool(
        FakeGraph(), FakeLongTerm(), searcher=cast(Any, ExplodingSearcher())
    )
    result = await tool(DossierArgs(name="Ghost"))
    # The exploding searcher must not propagate; dossier stays ok.
    assert result.ok is True
    assert "search_results" not in result.data


# --------------------------------------------------------------------------- #
# Round-trip through the real registry
# --------------------------------------------------------------------------- #
async def test_dossier_round_trip_via_registry() -> None:
    graph = FakeGraph(entities={"Zephyr": _Row(id=2, name="Zephyr", type="project")})
    registry = ToolRegistry()
    registry.register(cast(Tool, DossierTool(graph, FakeLongTerm())))

    result = await registry.execute(
        "entity_dossier", {"name": "Zephyr"}, allowed_tools={"entity_dossier"}
    )
    assert result.ok is True
    assert result.data["entity"]["type"] == "project"


async def test_dossier_bad_args_rejected_by_registry() -> None:
    registry = ToolRegistry()
    registry.register(cast(Tool, DossierTool(FakeGraph(), FakeLongTerm())))
    # Empty name violates ``min_length=1`` -> rejected pre-execution.
    result = await registry.execute(
        "entity_dossier", {"name": ""}, allowed_tools={"entity_dossier"}
    )
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "bad_args"
