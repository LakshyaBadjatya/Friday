"""The analysis agent: evidence-grounded synthesis with honest confidence.

:class:`AnalysisAgent` is the FRIDAY specialist for "analyze / assess / what's
the outlook" turns. It implements the :class:`~friday.agents.base.Agent`
protocol (``name="analysis"``, ``allowed_tools={"web_search"}``) and runs in two
phases:

1. **Retrieve.** It calls ``web_search`` through the injected
   :class:`~friday.tools.registry.ToolRegistry` (respecting ``allowed_tools``),
   turning each hit into a numbered, citable source (``[S1]``, ``[S2]`` …). It
   records the issued :class:`~friday.providers.llm.ToolCall` for audit.
2. **Synthesize.** It asks the LLM to write an answer-first analysis that cites
   those sources and ends with an explicit ``[confidence: low|medium|high]``
   tag — and, critically, that NEVER states a numeric probability/percentage
   without an accompanying source.

**The guard is structural, not advisory.** Prompting the model to behave is not
enough — a model asked "what's the *exact* probability?" will happily invent a
number. So after synthesis a deterministic guard (:func:`_scrub_unsourced`)
strips any bare percentage that is not immediately backed by a ``[S#]``
citation, drops over-claiming certainty words, and guarantees a confidence tag
is present. If retrieval produced no evidence, the agent does not even surface
the model's draft: it returns an honest, low-confidence statement that it has
nothing to go on. Honesty is enforced by code, not hope.

This module imports no LLM SDK — only the :class:`LLMProvider` abstraction — so
it keeps ``friday.agents`` clean of provider SDKs (grep-enforced by
``tests/unit/test_architecture.py``).
"""

from __future__ import annotations

import logging
import re
import uuid

from friday.agents.base import AgentResult
from friday.core.state import GraphState
from friday.errors import PermissionError, ProviderError
from friday.providers.llm import FakeLLM, LLMProvider, Message, ToolCall
from friday.tools.registry import ToolRegistry

logger = logging.getLogger("friday.agents.analysis")

# The single tool this agent is permitted to reach.
_ALLOWED_TOOLS: frozenset[str] = frozenset({"web_search"})
# How many search hits to fold into the evidence block.
_MAX_RESULTS = 5

# A numeric probability/percentage figure, e.g. "73%", "12.5 %", "40 percent".
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|percent\b)", re.IGNORECASE)
# A bracketed source citation, e.g. "[S1]" or "[s12]". A percentage is only
# considered *sourced* when such a marker sits within ``_CITATION_WINDOW``
# characters after it.
_CITATION_RE = re.compile(r"\[s\d+\]", re.IGNORECASE)
_CITATION_WINDOW = 40
# An accepted confidence tag at the level of the whole answer.
_CONFIDENCE_TAG_RE = re.compile(r"\[confidence:\s*(low|medium|high)\]", re.IGNORECASE)
# Over-claiming certainty words that have no place in a probabilistic forecast;
# scrubbed so the agent never promises an outcome it cannot know.
_OVERCLAIM_RE = re.compile(
    r"\b(?:definitely|certainly|guaranteed|for\s+sure|without\s+a\s+doubt|"
    r"will\s+absolutely)\b",
    re.IGNORECASE,
)

# Map a qualitative confidence label to a numeric confidence for AgentResult.
_CONFIDENCE_SCORE: dict[str, float] = {"low": 0.3, "medium": 0.6, "high": 0.85}

_SYSTEM_PROMPT = (
    "You are FRIDAY's analysis specialist. Write an answer-first, evidence-"
    "grounded analysis using ONLY the numbered sources provided. Cite each "
    "claim with its source marker (e.g. [S1]). Do NOT state any numeric "
    "probability or percentage unless it comes directly from a cited source — "
    "for forecasts, give a qualitative read instead of an invented number. "
    "Never promise a certain outcome. End your answer with exactly one tag: "
    "[confidence: low|medium|high], reflecting how strongly the sources support "
    "your read."
)


class AnalysisAgent:
    """Evidence-grounded analysis agent with a structural anti-fabrication guard.

    Args:
        registry: The tool registry the agent dispatches ``web_search`` through.
        llm: The provider used to synthesize the analysis. Defaults to an empty
            :class:`~friday.providers.llm.FakeLLM` so the agent is trivially
            constructible in tests; production wiring injects the real provider.
    """

    name = "analysis"
    allowed_tools = _ALLOWED_TOOLS

    def __init__(
        self,
        registry: ToolRegistry,
        llm: LLMProvider | None = None,
    ) -> None:
        self._registry = registry
        self._llm: LLMProvider = llm if llm is not None else FakeLLM(responses=[])

    # -- retrieval --------------------------------------------------------- #
    async def _search(
        self, query: str
    ) -> tuple[list[dict[str, str]], ToolCall, bool]:
        """Run ``web_search``; return (results, the issued call, ok flag).

        Any permission denial or handled tool failure yields an empty result
        list with ``ok=False`` — the caller then refuses to fabricate.
        """
        raw_args: dict[str, object] = {"query": query, "max_results": _MAX_RESULTS}
        call = ToolCall(id=f"call_{uuid.uuid4().hex}", name="web_search", arguments=raw_args)
        try:
            result = await self._registry.execute(
                "web_search", raw_args, allowed_tools=self.allowed_tools
            )
        except PermissionError as exc:  # pragma: no cover - defensive
            logger.warning("analysis denied web_search: %s", exc)
            return [], call, False

        if not result.ok:
            detail = result.error.message if result.error is not None else "unknown"
            logger.warning("analysis web_search failed: %s", detail)
            return [], call, False

        rows = result.data.get("results", [])
        # Normalize to a list of str->str dicts.
        results: list[dict[str, str]] = [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "snippet": str(r.get("snippet", "")),
            }
            for r in rows
        ]
        return results, call, True

    @staticmethod
    def _evidence_block(results: list[dict[str, str]]) -> tuple[str, set[str]]:
        """Render numbered sources and return (block, the set of source ids)."""
        lines: list[str] = []
        ids: set[str] = set()
        for i, row in enumerate(results, start=1):
            sid = f"S{i}"
            ids.add(sid)
            lines.append(
                f"[{sid}] {row['title']} ({row['url']}): {row['snippet']}".strip()
            )
        return "\n".join(lines), ids

    # -- synthesis --------------------------------------------------------- #
    async def _synthesize(self, user_input: str, evidence: str) -> str | None:
        """Ask the LLM for an analysis; return its text or ``None`` on failure."""
        task = Message(
            role="user",
            content=(
                f"The owner asked: {user_input!r}\n\n"
                f"NUMBERED SOURCES:\n{evidence}\n\n"
                "Write the analysis now, following every rule."
            ),
        )
        messages = [Message(role="system", content=_SYSTEM_PROMPT), task]
        try:
            response = await self._llm.complete(messages, tools=None)
        except ProviderError as exc:
            logger.warning("analysis synthesis failed: %s", exc)
            return None
        return response.text

    # -- guard ------------------------------------------------------------- #
    @staticmethod
    def _scrub_unsourced(text: str) -> str:
        """Remove every percentage figure not backed by a nearby ``[S#]`` cite.

        A percentage is kept only when a citation marker appears within
        ``_CITATION_WINDOW`` characters *after* it (i.e. the figure is sourced).
        Any unsourced figure is replaced with the qualitative placeholder
        ``"an uncertain amount"`` so the sentence stays readable but carries no
        fabricated number. Over-claiming certainty words are also dropped.
        """

        def _replace(match: re.Match[str]) -> str:
            tail = text[match.end() : match.end() + _CITATION_WINDOW]
            if _CITATION_RE.search(tail):
                return match.group(0)  # sourced -> keep verbatim
            return "an uncertain amount"

        scrubbed = _PERCENT_RE.sub(_replace, text)
        scrubbed = _OVERCLAIM_RE.sub("likely", scrubbed)
        # Collapse any double spaces the substitutions may have introduced.
        return re.sub(r"[ \t]{2,}", " ", scrubbed).strip()

    @staticmethod
    def _ensure_confidence_tag(text: str, label: str) -> str:
        """Guarantee the answer ends with exactly one confidence tag."""
        if _CONFIDENCE_TAG_RE.search(text):
            return text
        suffix = "" if not text or text.endswith((".", "!", "?")) else "."
        return f"{text}{suffix} [confidence: {label}]".strip()

    @staticmethod
    def _confidence_from_tag(text: str, default: float) -> float:
        match = _CONFIDENCE_TAG_RE.search(text)
        if match is None:
            return default
        return _CONFIDENCE_SCORE.get(match.group(1).lower(), default)

    def _no_evidence_answer(self, user_input: str) -> str:
        """Honest, low-confidence answer when retrieval yielded nothing."""
        return (
            "I couldn't retrieve any sources to ground an analysis of "
            f"{user_input!r}, so I won't put a number on it. I have nothing "
            "solid to go on right now — say the word and I'll dig deeper. "
            "[confidence: low]"
        )

    # -- public entrypoint ------------------------------------------------- #
    async def run(self, state: GraphState) -> AgentResult:
        """Search, synthesize, and return a guarded, confidence-tagged result.

        The returned :class:`AgentResult` records the ``web_search`` call it
        issued; ``output`` is guaranteed to (a) carry a single
        ``[confidence: …]`` tag and (b) contain no bare, unsourced percentage.
        """
        results, call, ok = await self._search(state.user_input)
        tool_calls_made: list[ToolCall] = [call]

        # No evidence (failed search or empty page): never fabricate.
        if not ok or not results:
            output = self._no_evidence_answer(state.user_input)
            return AgentResult(
                output=output,
                tool_calls_made=tool_calls_made,
                confidence=_CONFIDENCE_SCORE["low"],
            )

        evidence, _source_ids = self._evidence_block(results)
        draft = await self._synthesize(state.user_input, evidence)

        if draft is None or not draft.strip():
            # Synthesis failed/empty -> fall back to honest low-confidence.
            output = self._no_evidence_answer(state.user_input)
            return AgentResult(
                output=output,
                tool_calls_made=tool_calls_made,
                confidence=_CONFIDENCE_SCORE["low"],
            )

        # Structural guard: strip any unsourced percentage + over-claims, then
        # guarantee a confidence tag. Default the inferred label to "medium"
        # when the model omitted a tag.
        guarded = self._scrub_unsourced(draft)
        guarded = self._ensure_confidence_tag(guarded, "medium")
        confidence = self._confidence_from_tag(guarded, _CONFIDENCE_SCORE["medium"])

        return AgentResult(
            output=guarded,
            tool_calls_made=tool_calls_made,
            confidence=confidence,
        )
