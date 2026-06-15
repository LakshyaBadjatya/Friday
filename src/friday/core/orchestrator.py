"""The orchestrator: persona, memory, routing dispatch, synthesis, refusal.

The :class:`Orchestrator` is FRIDAY's brain stem for one turn. Per build-spec
section 5 it:

1. **Loads short-term history** for the session.
2. **Refuses out-of-scope asks first.** Requests for cut capabilities — facial
   recognition, people-tracking, offensive cyber (build-spec sections 2.1-2.2)
   — get a short, honest, in-character decline that names the reason and never
   fabricates the capability. This check precedes routing/LLM so we never spend
   a model call dressing up a "no".
3. **Routes** the turn via :func:`friday.core.router.route`.
4. **Dispatches** on the resulting :class:`Mode`:
   * ``CLARIFY`` -> return a clarifying *question* (never a guess).
   * ``CONVERSATION`` -> answer inline, in persona, via the LLM.
   * ``RESEARCH`` -> a minimal research path that *may* call ``web_search``
     through the registry (respecting the agent's ``allowed_tools``) and then
     synthesizes from what was actually retrieved.
5. **Synthesizes** the final reply in the FRIDAY persona — the persona spec
   (``persona/friday.md``) is injected as the system prompt and the owner is
   addressed as ``settings.owner_address``.

Honesty is structural: any :class:`~friday.errors.FridayError` is surfaced
plainly rather than masked as success, and tool failures are reported as
"couldn't retrieve" rather than fabricated.

This module never imports an LLM SDK — it depends only on the
:class:`~friday.providers.llm.LLMProvider` abstraction (grep-enforced by
``tests/unit/test_architecture.py``).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from friday.config import get_settings
from friday.core.router import route
from friday.core.state import GraphState, Mode
from friday.errors import FridayError, PermissionError, ProviderError
from friday.memory.short_term import ShortTermMemory
from friday.providers.llm import LLMProvider, Message
from friday.tools.registry import ToolRegistry

logger = logging.getLogger("friday.core.orchestrator")

# Tools the minimal Research path is allowed to reach this phase.
_RESEARCH_ALLOWED_TOOLS: frozenset[str] = frozenset({"web_search"})

# Out-of-scope / cut capabilities that earn an in-character refusal (build-spec
# sections 2.1-2.2). Each entry is (regex, human-readable reason). The regexes
# are intentionally specific so ordinary words ("track the build", "face the
# problem") do not trip a false refusal.
_REFUSAL_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"facial\s+recognition|recognize\s+(?:the\s+)?(?:person|people|face)"
            r"|identify\s+(?:the\s+|this\s+|that\s+)?(?:person|people|face|individual)"
            r"|who\s+is\s+(?:the\s+)?(?:person|individual)\s+in\s+(?:this|the)\s+"
            r"(?:photo|image|picture)",
            re.IGNORECASE,
        ),
        "facial recognition is out of scope — I'm defensive-only and won't "
        "identify people from images",
    ),
    (
        re.compile(
            r"track\s+(?:down\s+)?(?:a\s+|this\s+|that\s+|the\s+)?person"
            r"|locate\s+(?:a\s+|this\s+|that\s+|the\s+)?person"
            r"|surveil|stalk|follow\s+(?:a\s+|this\s+|that\s+|the\s+)?person"
            r"|find\s+(?:someone|a\s+person)'?s?\s+(?:location|whereabouts)",
            re.IGNORECASE,
        ),
        "tracking or locating a person is out of scope — I won't surveil people",
    ),
    (
        re.compile(
            r"write\s+(?:me\s+)?(?:an?\s+)?(?:exploit|malware|virus|ransomware|trojan)"
            r"|build\s+(?:an?\s+)?(?:exploit|malware|virus|ransomware|botnet)"
            r"|hack\s+into|break\s+into\s+(?:the\s+|a\s+)?(?:system|server|network|account)"
            r"|ddos|phishing\s+(?:kit|campaign|email)|sql\s+injection\s+attack"
            r"|offensive\s+cyber",
            re.IGNORECASE,
        ),
        "offensive cyber work is out of scope — I operate defensive-only",
    ),
)


class Orchestrator:
    """Coordinates one conversational turn end-to-end.

    Args:
        llm: The provider used for persona synthesis and conversation. Only the
            abstract :class:`LLMProvider` is depended upon.
        registry: The tool registry the research path dispatches through.
        memory: Per-session short-term conversation buffer.
        persona_path: Path to ``persona/friday.md``, injected as the system
            prompt for every synthesized reply.
    """

    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
        memory: ShortTermMemory,
        persona_path: str | Path,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._memory = memory
        self._persona_path = Path(persona_path)

    # -- persona ----------------------------------------------------------- #
    def _persona_text(self) -> str:
        """Read the persona spec, falling back to a minimal contract on error."""
        try:
            return self._persona_path.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - defensive; persona ships in-repo
            logger.warning("persona spec unreadable at %s: %s", self._persona_path, exc)
            return (
                "You are FRIDAY, a defensive-only local assistant. Be concise, "
                "honest, and direct. Never fabricate capability or data."
            )

    def _system_prompt(self) -> Message:
        owner = get_settings().owner_address
        persona = self._persona_text()
        content = (
            f"{persona}\n\n"
            f"---\nAddress the owner as '{owner}'. Answer first, keep it tight, "
            f"and never fabricate a capability, a fact, or a tool result you do "
            f"not have."
        )
        return Message(role="system", content=content)

    # -- refusal ----------------------------------------------------------- #
    @staticmethod
    def _refusal_reason(text: str) -> str | None:
        """Return a refusal reason if ``text`` asks for a cut capability."""
        for pattern, reason in _REFUSAL_RULES:
            if pattern.search(text):
                return reason
        return None

    def _decline(self, reason: str) -> str:
        """A short, honest, in-character decline naming ``reason``."""
        owner = get_settings().owner_address
        return f"Can't do that one, {owner} — {reason}."

    # -- synthesis --------------------------------------------------------- #
    async def _synthesize(self, history: list[Message], task: Message) -> str:
        """Call the LLM with persona + history + the turn's task message.

        Wraps provider failures in an honest, in-character message rather than
        leaking a traceback or faking a success.
        """
        messages = [self._system_prompt(), *history, task]
        try:
            response = await self._llm.complete(messages, tools=None)
        except ProviderError as exc:
            logger.warning("LLM synthesis failed: %s", exc)
            owner = get_settings().owner_address
            return (
                f"I'm having trouble reaching my language backend right now, "
                f"{owner}. That's a real outage, not me stalling — try again in "
                f"a moment."
            )
        text = response.text
        if not text or not text.strip():
            owner = get_settings().owner_address
            return f"I came back empty on that one, {owner}. Mind rephrasing?"
        return text.strip()

    # -- conversation ------------------------------------------------------ #
    async def _converse(self, state: GraphState, history: list[Message]) -> str:
        task = Message(role="user", content=state.user_input)
        return await self._synthesize(history, task)

    # -- research ---------------------------------------------------------- #
    async def _research(self, state: GraphState, history: list[Message]) -> str:
        """Minimal research path: optionally search, then synthesize honestly.

        Calls ``web_search`` through the registry respecting the research
        agent's ``allowed_tools``. Whatever the tool returns (success, empty, or
        a handled failure) is reported truthfully — no fabricated findings.
        """
        findings_block = ""
        try:
            result = await self._registry.execute(
                "web_search",
                {"query": state.user_input, "max_results": 5},
                allowed_tools=_RESEARCH_ALLOWED_TOOLS,
            )
            state.scratchpad["web_search_invoked"] = True
        except PermissionError as exc:
            # The research agent should always be allowed web_search; if not,
            # report honestly rather than guess.
            logger.warning("research path denied web_search: %s", exc)
            state.scratchpad["web_search_invoked"] = False
            findings_block = (
                "NOTE TO SELF: web search was not permitted, so you have no "
                "retrieved sources. Say so plainly; do not invent findings."
            )
        else:
            if result.ok:
                rows = result.data.get("results", [])
                state.scratchpad["web_search_results"] = rows
                if rows:
                    lines = [
                        f"- {r.get('title', '')} ({r.get('url', '')}): "
                        f"{r.get('snippet', '')}".strip()
                        for r in rows
                    ]
                    findings_block = "RETRIEVED SOURCES:\n" + "\n".join(lines)
                else:
                    findings_block = (
                        "NOTE TO SELF: the search returned no results. Say so "
                        "plainly; do not invent findings."
                    )
            else:
                err = result.error
                detail = err.message if err is not None else "unknown error"
                logger.warning("research web_search failed: %s", detail)
                findings_block = (
                    "NOTE TO SELF: the web search failed "
                    f"({detail}). Report that you couldn't retrieve sources; do "
                    "not fabricate findings."
                )

        task_content = (
            f"The owner asked: {state.user_input!r}\n\n"
            f"{findings_block}\n\n"
            "Synthesize a concise, answer-first reply from ONLY the retrieved "
            "sources above. If there are no sources, say so honestly and offer "
            "to dig further — do not invent facts or citations."
        )
        task = Message(role="user", content=task_content)
        return await self._synthesize(history, task)

    # -- clarify ----------------------------------------------------------- #
    def _clarify(self, state: GraphState) -> str:
        """Return a clarifying question — never a guess at the intent."""
        owner = get_settings().owner_address
        return (
            f"I want to get this right rather than guess, {owner} — could you "
            f"say a bit more about what you're after?"
        )

    # -- public entrypoint ------------------------------------------------- #
    async def handle(self, state: GraphState) -> GraphState:
        """Run one turn end-to-end, mutating and returning ``state``.

        The returned state carries the final ``mode``, the ``route`` decision,
        and the synthesized ``response``. The user turn and the assistant reply
        are recorded in short-term memory.
        """
        try:
            return await self._handle_inner(state)
        except FridayError as exc:
            # Map any domain error to an honest, in-character message; never
            # fake success. Surfaced here so a stray FridayError from a deeper
            # call site still produces a truthful reply.
            logger.warning("orchestrator caught FridayError: %s", exc)
            owner = get_settings().owner_address
            state.response = (
                f"Hit a snag I can't paper over, {owner}: {exc}. "
                f"That's the honest status."
            )
            return state

    async def _handle_inner(self, state: GraphState) -> GraphState:
        # 1. Load short-term history for the session.
        history = self._memory.history(state.session_id)

        # 2. Refuse out-of-scope asks before spending a model call.
        reason = self._refusal_reason(state.user_input)
        if reason is not None:
            state.mode = Mode.CONVERSATION
            state.response = self._decline(reason)
            self._record(state)
            return state

        # 3. Route the turn.
        decision = await route(state)
        state.route = decision
        state.mode = decision.mode

        # 4. Dispatch on mode.
        if decision.mode is Mode.CLARIFY:
            state.response = self._clarify(state)
        elif decision.mode is Mode.RESEARCH:
            state.response = await self._research(state, history)
        else:
            # CONVERSATION (and any non-clarify, non-research fallback).
            state.mode = Mode.CONVERSATION
            state.response = await self._converse(state, history)

        self._record(state)
        return state

    def _record(self, state: GraphState) -> None:
        """Persist the user turn and assistant reply to short-term memory."""
        self._memory.append(
            state.session_id, Message(role="user", content=state.user_input)
        )
        if state.response is not None:
            self._memory.append(
                state.session_id, Message(role="assistant", content=state.response)
            )

    # -- node-level API (used by core/modes.py + core/graph.py) ------------ #
    #
    # These mirror the steps of ``_handle_inner`` but as discrete, individually
    # callable units so the LangGraph node functions can drive the same logic
    # through an explicit ROUTING -> mode-node -> END flow. They share state via
    # the passed-in :class:`GraphState`.

    async def node_routing(self, state: GraphState) -> GraphState:
        """ROUTING node: refusal short-circuit, else classify into a mode.

        On an out-of-scope ask this sets ``mode=CONVERSATION`` and writes the
        decline directly to ``state.response`` (the conditional edge then routes
        straight to END via the conversation node, which is a no-op when a
        response is already present).
        """
        reason = self._refusal_reason(state.user_input)
        if reason is not None:
            state.mode = Mode.CONVERSATION
            state.response = self._decline(reason)
            return state

        decision = await route(state)
        state.route = decision
        state.mode = decision.mode
        # Normalize any non-clarify, non-research decision to CONVERSATION so the
        # graph's conditional edge has a closed set of targets.
        if decision.mode not in (Mode.CLARIFY, Mode.RESEARCH):
            state.mode = Mode.CONVERSATION
        return state

    async def node_conversation(self, state: GraphState) -> GraphState:
        """CONVERSATION node: inline persona answer (skipped if already replied)."""
        if state.response is None:
            history = self._memory.history(state.session_id)
            state.response = await self._converse(state, history)
        self._record(state)
        return state

    async def node_research(self, state: GraphState) -> GraphState:
        """RESEARCH node: minimal search-then-synthesize path."""
        if state.response is None:
            history = self._memory.history(state.session_id)
            state.response = await self._research(state, history)
        self._record(state)
        return state

    async def node_clarify(self, state: GraphState) -> GraphState:
        """CLARIFY node: return a clarifying question, never a guess."""
        if state.response is None:
            state.response = self._clarify(state)
        self._record(state)
        return state
