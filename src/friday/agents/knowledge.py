"""The knowledge agent: grounded retrieval over a :class:`VectorStore`.

:class:`KnowledgeAgent` answers questions *only* from passages it retrieves out
of an injected :class:`~friday.memory.vector.VectorStore` (Phase 2 ships the
deterministic in-memory adapter; Chroma/pgvector is Phase 4). It declares no
tools — ``allowed_tools == frozenset()`` — because retrieval is a direct,
in-process read of the store, not a tool call routed through the registry. It
may also peek at per-session short-term memory to enrich the retrieval query,
but never to invent an answer.

Honesty is structural, exactly as the build-spec (section 9.7) requires:

* **Nothing retrieved -> say so.** When the store returns no relevant chunk the
  agent emits an explicit "I have nothing on that" and refuses to answer from
  parametric/model memory. There is no LLM in this path, so a fabricated
  "from-memory" answer is impossible by construction.
* **Something retrieved -> cite it.** The answer is assembled from the retrieved
  chunks and cites each chunk's ``source_id`` inline, so every claim is
  traceable to the document it came from. The retrieved chunks are also returned
  as ``memory_writes`` for the orchestrator/audit trail.

Determinism: retrieval is the token-overlap store and answer assembly is pure
string formatting — no network, no model, no wall-clock — so a given
``(store, query)`` always yields the same :class:`AgentResult`.

This module imports no LLM SDK; it depends only on in-repo abstractions, keeping
the ``agents`` package clean for the SDK-isolation guard.
"""

from __future__ import annotations

from friday.agents.base import AgentResult
from friday.core.state import GraphState
from friday.memory.short_term import ShortTermMemory
from friday.memory.vector import Chunk, VectorStore

# Default number of chunks to retrieve and ground the answer in.
DEFAULT_TOP_K = 4

# Confidence the agent reports when it retrieved nothing and is honestly
# declining — deliberately below 1.0 so callers can see it is not a confident
# grounded answer.
_NO_GROUNDING_CONFIDENCE = 0.0


class KnowledgeAgent:
    """Answers strictly from retrieved vector-store chunks, citing their sources.

    Args:
        store: The vector store queried for grounding passages. Only the
            structural :class:`~friday.memory.vector.VectorStore` contract is
            depended upon, so any adapter (in-memory now, Chroma later) works.
        memory: Optional per-session short-term buffer. When present, recent
            user turns are folded into the retrieval query to sharpen recall;
            it is never used to fabricate an answer.
        top_k: Maximum number of chunks to retrieve per query.
    """

    name: str = "knowledge"
    allowed_tools: frozenset[str] = frozenset()

    def __init__(
        self,
        store: VectorStore,
        memory: ShortTermMemory | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self._store = store
        self._memory = memory
        self._top_k = top_k

    # -- query construction ------------------------------------------------- #
    def _retrieval_query(self, state: GraphState) -> str:
        """Build the retrieval query from the turn input + recent user history.

        Short-term memory only *enriches* the query string; it never becomes an
        answer. With no memory configured (or none for the session) this is just
        the current ``user_input``.
        """
        query = state.user_input
        if self._memory is None:
            return query
        recent_user_turns = [
            msg.content
            for msg in self._memory.history(state.session_id)
            if msg.role == "user" and msg.content
        ]
        if not recent_user_turns:
            return query
        # Most-recent history first, then the current input, so overlap scoring
        # is dominated by what the owner is asking right now.
        return " ".join([*recent_user_turns, query])

    # -- answer assembly ---------------------------------------------------- #
    @staticmethod
    def _no_grounding_answer() -> str:
        """An explicit, honest decline used when nothing was retrieved.

        Deliberately contains no topic-specific content so it can never read as
        a parametric guess — it states the absence of grounding and stops.
        """
        return (
            "I have nothing in my knowledge base on that. I answer only from "
            "retrieved sources, and the store returned no relevant material, so "
            "I won't guess from memory. Point me at a document and I'll ground "
            "an answer in it."
        )

    @staticmethod
    def _grounded_answer(chunks: list[Chunk]) -> str:
        """Assemble a grounded answer that cites each chunk's ``source_id``."""
        lines = ["Grounded in the retrieved sources:"]
        for chunk in chunks:
            lines.append(f"- [{chunk.source_id}] {chunk.text}")
        cited = ", ".join(chunk.source_id for chunk in chunks)
        lines.append(f"(Sources: {cited})")
        return "\n".join(lines)

    # -- agent entrypoint --------------------------------------------------- #
    async def run(self, state: GraphState) -> AgentResult:
        """Retrieve grounding chunks and answer from them, or decline honestly.

        Returns an :class:`AgentResult` whose ``output`` is either a grounded,
        cited answer or an explicit "nothing retrieved" decline. Retrieved
        chunks (if any) are returned in ``memory_writes`` for audit, and
        ``confidence`` reflects the best retrieved chunk's score (0.0 when
        nothing was retrieved).
        """
        query = self._retrieval_query(state)
        chunks = self._store.query(query, k=self._top_k)

        if not chunks:
            return AgentResult(
                output=self._no_grounding_answer(),
                tool_calls_made=[],
                memory_writes=[],
                confidence=_NO_GROUNDING_CONFIDENCE,
            )

        # The store returns chunks closest-first; the top score is the agent's
        # calibrated confidence in the grounding (already in [0.0, 1.0]).
        confidence = chunks[0].score
        return AgentResult(
            output=self._grounded_answer(chunks),
            tool_calls_made=[],
            memory_writes=list(chunks),
            confidence=confidence,
        )
