"""Unit tests for the knowledge agent (Stage 3, build-spec section 9.7).

The :class:`~friday.agents.knowledge.KnowledgeAgent` is a *grounded* retrieval
agent: it answers ONLY from chunks pulled out of an injected ``VectorStore``,
citing each chunk's ``source_id``. It declares no tools
(``allowed_tools == frozenset()``) and never reaches the network.

The two pinned behaviours:

* **Empty store** -> the agent explicitly states it has nothing on the topic and
  emits NO parametric/from-memory answer (honest "I don't know", not a guess).
* **Seeded store** -> the answer is grounded in the retrieved chunk and cites
  that chunk's ``source_id``.

All retrieval is deterministic and offline (the in-memory token-overlap store),
so these tests need no network, no model, and no wall-clock.
"""

from __future__ import annotations

import pytest

from friday.agents.base import Agent, AgentResult
from friday.agents.knowledge import KnowledgeAgent
from friday.core.state import GraphState, Mode
from friday.memory.short_term import ShortTermMemory
from friday.memory.vector import InMemoryVectorStore
from friday.providers.llm import Message


def _state(user_input: str) -> GraphState:
    return GraphState(
        session_id="know-test",
        mode=Mode.CONVERSATION,
        user_input=user_input,
    )


def test_knowledge_agent_satisfies_agent_protocol() -> None:
    agent = KnowledgeAgent(store=InMemoryVectorStore())
    assert isinstance(agent, Agent)
    assert agent.name == "knowledge"
    # No tools: this agent only reads its vector store + short-term memory.
    assert agent.allowed_tools == frozenset()


async def test_empty_store_says_it_has_nothing_no_parametric_answer() -> None:
    # An empty store retrieves nothing; the agent must decline honestly rather
    # than answer the (famous, easily-parametric) question from model memory.
    store = InMemoryVectorStore()
    agent = KnowledgeAgent(store=store)

    result = await agent.run(_state("What is the capital of France?"))

    assert isinstance(result, AgentResult)
    # No fabricated parametric answer: it must NOT just say "Paris".
    assert "paris" not in result.output.lower()
    # It explicitly states it has nothing grounded on the topic.
    lowered = result.output.lower()
    assert "nothing" in lowered or "no " in lowered or "don't have" in lowered
    # No chunks -> no citations and (calibrated) low confidence.
    assert result.confidence < 1.0
    assert result.tool_calls_made == []


async def test_seeded_store_answer_references_seeded_source_id() -> None:
    store = InMemoryVectorStore()
    store.add(
        [
            (
                "FRIDAY is a defensive-only local assistant built in Python.",
                "doc-friday-overview",
            ),
            ("Unrelated note about gardening and tomatoes.", "doc-garden"),
        ]
    )
    agent = KnowledgeAgent(store=store)

    result = await agent.run(_state("What is FRIDAY, the local assistant?"))

    assert isinstance(result, AgentResult)
    # Grounded: the answer must cite the seeded source id it retrieved from.
    assert "doc-friday-overview" in result.output
    # And it should not have hallucinated an unrelated citation as the basis.
    assert "doc-friday-overview" in {w.source_id for w in result.memory_writes}
    assert result.confidence > 0.0
    assert result.tool_calls_made == []


async def test_short_term_memory_is_consulted_but_does_not_fabricate() -> None:
    # Even with conversation history present, an empty store yields no grounding.
    store = InMemoryVectorStore()
    memory = ShortTermMemory()
    memory.append(
        "know-test", Message(role="user", content="Tell me about quantum gravity.")
    )
    agent = KnowledgeAgent(store=store, memory=memory)

    result = await agent.run(_state("Continue that explanation."))

    # Still honest: nothing retrieved -> no invented physics lecture.
    lowered = result.output.lower()
    assert "nothing" in lowered or "no " in lowered or "don't have" in lowered
    assert result.confidence < 1.0


@pytest.mark.parametrize("k", [1, 4])
async def test_query_respects_top_k_and_only_cites_retrieved_sources(k: int) -> None:
    store = InMemoryVectorStore()
    store.add(
        [
            ("alpha beta gamma about widgets", "doc-a"),
            ("alpha beta delta about widgets", "doc-b"),
            ("completely orthogonal subject matter", "doc-c"),
        ]
    )
    agent = KnowledgeAgent(store=store, top_k=k)

    result = await agent.run(_state("alpha beta widgets"))

    cited = {w.source_id for w in result.memory_writes}
    # Only overlapping docs can be cited; the orthogonal one never surfaces.
    assert "doc-c" not in cited
    assert cited.issubset({"doc-a", "doc-b"})
    assert len(cited) <= k
