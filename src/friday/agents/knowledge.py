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

import re

from friday.agents.base import AgentResult
from friday.core.state import GraphState
from friday.memory.long_term import LongTermStore
from friday.memory.short_term import ShortTermMemory
from friday.memory.vector import Chunk, VectorStore

# Default number of chunks to retrieve and ground the answer in.
DEFAULT_TOP_K = 4

# Confidence the agent reports when it retrieved nothing and is honestly
# declining — deliberately below 1.0 so callers can see it is not a confident
# grounded answer.
_NO_GROUNDING_CONFIDENCE = 0.0

# Score assigned to a long-term fact match. The long-term store ranks by a
# binary case-insensitive substring (LIKE) match rather than a graded
# similarity, so a matched fact gets a fixed, deliberately-modest score: it is
# real grounding (> 0.0) but should not outrank a strong semantic vector hit.
_LONG_TERM_MATCH_SCORE = 0.5

# Tokenizer for long-term recall: lowercase alphanumeric words of length >= 3 so
# the substring query is built from salient terms, not stopwords/punctuation.
_LT_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

# Common short words that carry no retrieval signal — dropped before matching so
# a long-term LIKE query is anchored on content words.
_LT_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "are", "was", "were", "you", "your", "what",
        "where", "when", "who", "why", "how", "does", "did", "can", "could",
        "would", "should", "about", "tell", "know", "with", "that", "this",
        "from", "have", "has", "had", "into", "out", "any", "all",
    }
)


class KnowledgeAgent:
    """Answers strictly from retrieved sources (vector + long-term), citing them.

    Retrieval is *hybrid* (build-spec §10, §9.7): the agent queries the injected
    vector store for semantically-near chunks AND folds in recent long-term facts
    matching the query. Both arrive as :class:`~friday.memory.vector.Chunk`
    objects carrying a ``source_id``, are merged (deduplicated by source), and the
    answer cites each. When nothing is retrieved from either source the agent says
    so honestly — there is no LLM in this path, so a parametric guess is
    impossible by construction.

    Args:
        store: The vector store queried for grounding passages. Only the
            structural :class:`~friday.memory.vector.VectorStore` contract is
            depended upon, so any adapter (in-memory now, SQLite/Chroma later)
            works.
        memory: Optional per-session short-term buffer. When present, recent
            user turns are folded into the retrieval query to sharpen recall;
            it is never used to fabricate an answer.
        long_term: Optional durable store. When wired, recent facts whose text
            matches the query are retrieved alongside the vector chunks so the
            agent grounds in persisted memory as well as indexed documents.
        top_k: Maximum number of chunks to retrieve per source.
    """

    name: str = "knowledge"
    allowed_tools: frozenset[str] = frozenset()

    def __init__(
        self,
        store: VectorStore,
        memory: ShortTermMemory | None = None,
        top_k: int = DEFAULT_TOP_K,
        long_term: LongTermStore | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self._store = store
        self._memory = memory
        self._long_term = long_term
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

    # -- long-term grounding ------------------------------------------------ #
    @staticmethod
    def _salient_terms(text: str) -> list[str]:
        """Content words (>=3 chars, non-stopword) of ``text``, order-preserving."""
        seen: set[str] = set()
        terms: list[str] = []
        for tok in _LT_TOKEN_RE.findall(text.lower()):
            if tok in _LT_STOPWORDS or tok in seen:
                continue
            seen.add(tok)
            terms.append(tok)
        return terms

    def _long_term_chunks(self, state: GraphState) -> list[Chunk]:
        """Retrieve recent long-term facts matching the turn as :class:`Chunk`s.

        Long-term recall is a case-insensitive substring (LIKE) match, so a
        whole-sentence query would rarely hit a stored fact. Instead each salient
        content word of the user input is matched independently and the results
        are merged (deduplicated by ``source_id``). Returns ``[]`` when no
        long-term store is wired or nothing matches. Each matched fact becomes a
        :class:`Chunk` carrying its ``source_id`` and :data:`_LONG_TERM_MATCH_SCORE`.
        """
        if self._long_term is None:
            return []
        seen: set[str] = set()
        chunks: list[Chunk] = []
        for term in self._salient_terms(state.user_input):
            for fact in self._long_term.query_facts(term, limit=self._top_k):
                if fact.source_id in seen:
                    continue
                seen.add(fact.source_id)
                chunks.append(
                    Chunk(
                        text=fact.text,
                        source_id=fact.source_id,
                        score=_LONG_TERM_MATCH_SCORE,
                    )
                )
        return chunks

    @staticmethod
    def _merge(vector_chunks: list[Chunk], long_term_chunks: list[Chunk]) -> list[Chunk]:
        """Merge two chunk lists, deduplicating by ``source_id`` (highest score).

        Vector hits come first (they carry graded similarity); a long-term fact
        with the same ``source_id`` is folded in only if not already present, so
        a source is cited once. The merged list is sorted by descending score so
        the strongest grounding leads.
        """
        by_source: dict[str, Chunk] = {}
        for chunk in [*vector_chunks, *long_term_chunks]:
            existing = by_source.get(chunk.source_id)
            if existing is None or chunk.score > existing.score:
                by_source[chunk.source_id] = chunk
        merged = list(by_source.values())
        merged.sort(key=lambda chunk: chunk.score, reverse=True)
        return merged

    # -- agent entrypoint --------------------------------------------------- #
    async def run(self, state: GraphState) -> AgentResult:
        """Retrieve grounding chunks and answer from them, or decline honestly.

        Hybrid retrieval: vector-store chunks plus matching long-term facts,
        merged and deduplicated by ``source_id``. Returns an :class:`AgentResult`
        whose ``output`` is either a grounded, cited answer or an explicit
        "nothing retrieved" decline. Retrieved chunks (if any) are returned in
        ``memory_writes`` for audit, and ``confidence`` reflects the best
        retrieved chunk's score (0.0 when nothing was retrieved).
        """
        query = self._retrieval_query(state)
        vector_chunks = self._store.query(query, k=self._top_k)
        long_term_chunks = self._long_term_chunks(state)
        chunks = self._merge(vector_chunks, long_term_chunks)

        if not chunks:
            return AgentResult(
                output=self._no_grounding_answer(),
                tool_calls_made=[],
                memory_writes=[],
                confidence=_NO_GROUNDING_CONFIDENCE,
            )

        # Chunks are merged closest-first; the top score is the agent's
        # calibrated confidence in the grounding (already in [0.0, 1.0]).
        confidence = chunks[0].score
        return AgentResult(
            output=self._grounded_answer(chunks),
            tool_calls_made=[],
            memory_writes=list(chunks),
            confidence=confidence,
        )
