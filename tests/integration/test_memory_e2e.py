"""End-to-end memory test: seed -> grounded answer -> forget -> empty.

Exercises the full Phase-4 Stage-2 loop with offline fakes (FakeEmbeddings,
FakeLLM) and in-memory stores:

1. An agent proposes a non-sensitive :class:`MemoryWrite`; the orchestrator
   auto-commits it (``memory_autowrite`` on) to BOTH the long-term store and the
   persistent vector store.
2. A Knowledge turn retrieves the seeded chunk and answers grounded in it,
   citing its ``source_id``.
3. A "forget X" turn removes the fact from both stores.
4. A follow-up Knowledge turn now retrieves nothing and says it has nothing.

Every dependency is deterministic and offline: no key, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from friday.agents.base import AgentRegistry, AgentResult
from friday.agents.knowledge import KnowledgeAgent
from friday.core.orchestrator import MemoryWrite, Orchestrator
from friday.core.state import GraphState
from friday.memory.long_term import SQLiteLongTermStore
from friday.memory.short_term import ShortTermMemory
from friday.memory.vector import SQLiteVectorStore
from friday.providers.embeddings import FakeEmbeddings
from friday.providers.llm import FakeLLM, LLMResponse, Message, ToolSpec, Usage
from friday.tools.registry import ToolRegistry

PERSONA_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "friday" / "persona" / "friday.md"
)

_SOURCE_ID = "doc-vault-location"
_SECRET_TEXT = "The spare key is hidden under the third flowerpot on the patio."


class _SeedingAgent:
    """An automation agent that proposes one non-sensitive memory write."""

    name = "automation"
    allowed_tools: frozenset[str] = frozenset()

    async def run(self, state: GraphState) -> AgentResult:
        return AgentResult(
            output="Stored that for you.",
            memory_writes=[
                MemoryWrite(text=_SECRET_TEXT, source_id=_SOURCE_ID, sensitive=False)
            ],
        )


class _RelayLLM(FakeLLM):
    """A FakeLLM that relays the draft it is asked to re-voice, verbatim-ish.

    It returns the last user message's content so persona-wrap conveys the
    agent/knowledge draft (which contains the cited source_id) rather than
    starving on an empty script.
    """

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user" and m.content),
            "ok",
        )
        return LLMResponse(text=last_user, tool_calls=[], usage=Usage())


def _state(text: str) -> GraphState:
    return GraphState(session_id="e2e", user_input=text)


def _build(monkeypatch: pytest.MonkeyPatch) -> tuple[
    Orchestrator, SQLiteVectorStore, SQLiteLongTermStore
]:
    from friday import config

    settings = config.Settings(memory_autowrite=True)
    import friday.core.orchestrator as orch_mod
    import friday.core.router as router_mod

    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(orch_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(router_mod, "get_settings", lambda: settings)

    embedder = FakeEmbeddings(dim=64)
    vector = SQLiteVectorStore(":memory:", embedder=embedder, dim=64)
    long_term = SQLiteLongTermStore(":memory:")

    agents = AgentRegistry()
    agents.register(_SeedingAgent())
    agents.register(
        KnowledgeAgent(store=vector, memory=ShortTermMemory(), long_term=long_term)
    )

    orch = Orchestrator(
        llm=_RelayLLM(responses=[]),
        registry=ToolRegistry(),
        memory=ShortTermMemory(),
        persona_path=PERSONA_PATH,
        agents=agents,
        long_term=long_term,
        vector=vector,
    )
    return orch, vector, long_term


async def test_seed_then_ground_then_forget_then_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orch, vector, long_term = _build(monkeypatch)

    # 1. Seed a fact via an agent write (AUTOMATION mode, auto-committed).
    await orch.handle(_state("automate stashing where the spare key is"))

    # The auto-commit reached both stores. The vector store is checked by
    # retrieving the seeded text itself (FakeEmbeddings is deterministic; a text
    # matches itself with cosine 1.0), which proves the chunk was indexed.
    assert long_term.query_facts("spare key") != []
    seeded = vector.query(_SECRET_TEXT, k=4)
    assert seeded and seeded[0].source_id == _SOURCE_ID

    # 2. A Knowledge turn answers grounded in the seeded chunk, citing source_id.
    #    "look up" routes to RESEARCH; force the knowledge path by dispatching
    #    the knowledge agent directly through the registry-backed mode.
    grounded = await _ask_knowledge(orch, "where is the spare key hidden")
    assert _SOURCE_ID in grounded

    # 3. "forget" the fact -> removed from both stores.
    forget_state = await orch.handle(_state("forget what you know about spare key"))
    assert forget_state.response is not None
    assert long_term.query_facts("spare key") == []
    # The store is now empty, so even the seeded text retrieves nothing.
    assert vector.query(_SECRET_TEXT, k=4) == []

    # 4. Knowledge now retrieves nothing and says so.
    empty = await _ask_knowledge(orch, "where is the spare key hidden")
    assert _SOURCE_ID not in empty
    assert "nothing" in empty.lower() or "no " in empty.lower()


async def _ask_knowledge(orch: Orchestrator, text: str) -> str:
    """Drive a Knowledge-agent turn and return the response text.

    The deterministic router has no KNOWLEDGE mode, so we invoke the knowledge
    agent through the orchestrator's public knowledge entrypoint.
    """
    state = _state(text)
    out = await orch.knowledge_turn(state)
    assert out.response is not None
    return out.response
