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

from friday.agents.base import AgentRegistry, AgentResult
from friday.config import get_settings
from friday.core.router import route
from friday.core.security import run_lockdown
from friday.core.state import GraphState, Mode
from friday.errors import FridayError, PermissionError, ProviderError
from friday.memory.short_term import ShortTermMemory
from friday.providers.llm import LLMProvider, Message
from friday.tools.registry import ToolRegistry
from friday.tools.security import AuditRecord

logger = logging.getLogger("friday.core.orchestrator")

# Tools the minimal Research path is allowed to reach this phase.
_RESEARCH_ALLOWED_TOOLS: frozenset[str] = frozenset({"web_search"})

# The specialist modes that dispatch to an :class:`Agent` in the registry, and
# the tool whose side-effecting/idempotent metadata gates the confirm-step for
# each. RESEARCH stays on the dedicated ``_research`` path (it is read-only), so
# it is intentionally absent here.
_MODE_TO_AGENT: dict[Mode, str] = {
    Mode.AUTOMATION: "automation",
    Mode.DEVICE_CONTROL: "device",
    Mode.ALERTING: "alerting",
}
# The side-effecting tool whose confirm-step gates a dispatched mode. Modes
# absent from this map are not confirmation-gated at the orchestrator level.
_MODE_TO_GATED_TOOL: dict[Mode, str] = {
    Mode.DEVICE_CONTROL: "home",
    Mode.ALERTING: "notify",
}

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
        agents: Optional registry of specialist agents. When present the
            orchestrator dispatches AUTOMATION / DEVICE_CONTROL / ALERTING turns
            to the matching agent; when absent those modes fall back to a plain
            persona conversation so the orchestrator is still constructible
            without the full agent graph (e.g. in narrow unit tests).
    """

    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
        memory: ShortTermMemory,
        persona_path: str | Path,
        agents: AgentRegistry | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._memory = memory
        self._persona_path = Path(persona_path)
        self._agents = agents

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

    # -- confirm-step (build-spec §12) ------------------------------------- #
    def _needs_confirmation(self, mode: Mode, state: GraphState) -> bool:
        """Whether a side-effecting dispatch must be confirmed before executing.

        A mode is confirmation-gated when it dispatches a tool that is
        ``side_effecting`` and not ``idempotent`` (read from the tool registry,
        so the gate tracks the tool's own metadata rather than a hard-coded
        list). An already-confirmed turn (``state.confirmed``) clears the gate.
        """
        if state.confirmed:
            return False
        tool_name = _MODE_TO_GATED_TOOL.get(mode)
        if tool_name is None:
            return False
        try:
            tool = self._registry.get(tool_name)
        except KeyError:  # pragma: no cover - defensive: tool always registered
            return False
        return bool(tool.side_effecting and not tool.idempotent)

    def _confirm_question(self, mode: Mode, state: GraphState) -> str:
        """A persona confirm question for a pending side-effecting action.

        The action stays *pending*: nothing is dispatched or executed. A
        confirming follow-up (``state.confirmed=True``) then proceeds.
        """
        owner = get_settings().owner_address
        what = self._pending_action_summary(mode, state)
        return (
            f"That's a real-world action, {owner}, so I'll confirm before I act: "
            f"{what} Want me to go ahead? Reply to confirm."
        )

    @staticmethod
    def _pending_action_summary(mode: Mode, state: GraphState) -> str:
        """A short human description of the side-effecting action awaiting confirm."""
        if mode is Mode.DEVICE_CONTROL:
            device = state.scratchpad.get("device")
            if isinstance(device, dict):
                action = device.get("action", "act on")
                device_id = device.get("device_id", "the device")
                return f"I'm about to {action} {device_id!r}."
            return "I'm about to control a device."
        if mode is Mode.ALERTING:
            alert = state.scratchpad.get("alert")
            if isinstance(alert, dict):
                subject = alert.get("subject", "an alert")
                target = alert.get("target", "the target")
                return f"I'm about to send the alert {subject!r} to {target}."
            return "I'm about to send a notification."
        return "I'm about to take a side-effecting action."

    # -- agent dispatch ---------------------------------------------------- #
    async def _dispatch_agent(self, mode: Mode, state: GraphState) -> AgentResult:
        """Look up the agent for ``mode`` and run it against ``state``.

        Raises :class:`KeyError` if the agent is not registered — a programmer
        error (``app.py`` populates the registry), surfaced honestly by
        :meth:`handle`'s ``FridayError`` guard would not catch it, so callers
        only invoke this when an agent registry is present.
        """
        assert self._agents is not None  # guarded by the caller
        agent_name = _MODE_TO_AGENT[mode]
        agent = self._agents.get(agent_name)
        result = await agent.run(state)
        state.scratchpad["agent"] = agent_name
        state.scratchpad["agent_confidence"] = result.confidence
        return result

    async def _handle_agent_mode(
        self, mode: Mode, state: GraphState, history: list[Message]
    ) -> str:
        """Resolve a specialist-agent turn: confirm-gate, dispatch, then persona.

        When the mode is confirmation-gated and the turn is unconfirmed, returns
        a persona confirm question WITHOUT dispatching the agent (nothing
        executes). Otherwise it runs the agent and synthesizes a tight,
        in-persona reply grounded in the agent's ``output``.
        """
        if self._agents is None or mode not in self._agents_for_mode():
            # No agent wired for this mode: fall back to a plain persona answer
            # rather than crash, keeping the orchestrator usable un-wired.
            return await self._converse(state, history)

        if self._needs_confirmation(mode, state):
            return self._confirm_question(mode, state)

        result = await self._dispatch_agent(mode, state)
        return await self._persona_wrap(state, history, result.output)

    def _agents_for_mode(self) -> frozenset[Mode]:
        """The modes this orchestrator can dispatch (agent present in registry)."""
        if self._agents is None:
            return frozenset()
        return frozenset(
            mode for mode, name in _MODE_TO_AGENT.items() if name in self._agents
        )

    async def _persona_wrap(
        self, state: GraphState, history: list[Message], draft: str
    ) -> str:
        """Synthesize a persona reply that conveys ``draft`` faithfully.

        The agent has already done the work and any honesty/anti-fabrication
        guard; the LLM only re-voices it in the FRIDAY persona. If synthesis
        fails or comes back empty, the agent's own ``draft`` is returned verbatim
        so the turn never loses the real result.
        """
        task = Message(
            role="user",
            content=(
                "Relay the following result to the owner in your voice — keep it "
                "answer-first and tight, change no facts, and add nothing the "
                f"result does not contain:\n\n{draft}"
            ),
        )
        messages = [self._system_prompt(), *history, task]
        try:
            response = await self._llm.complete(messages, tools=None)
        except ProviderError as exc:
            logger.warning("persona wrap failed; using agent draft: %s", exc)
            return draft
        text = response.text
        if not text or not text.strip():
            return draft
        return text.strip()

    # -- security lockdown (build-spec §9.9) ------------------------------- #
    async def _lockdown(self, state: GraphState) -> str:
        """Run the defensive lockdown subgraph and report the audit trail.

        This is deliberately NOT a chatty agent path: the audit trail is reported
        deterministically (no LLM call) so a lockdown is always truthful and
        never blocked on a model. The records are stashed on ``scratchpad`` for
        the API/audit view.
        """
        records: list[AuditRecord] = await run_lockdown(state)
        state.scratchpad["lockdown_audit"] = [r.model_dump() for r in records]
        owner = get_settings().owner_address
        lines = [f"Lockdown complete, {owner}. Audit trail:"]
        for record in records:
            status = "ok" if record.ok else "FAILED"
            lines.append(f"- {record.step} [{status}]: {record.detail}")
        return "\n".join(lines)

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
        elif decision.mode is Mode.SECURITY_LOCKDOWN:
            state.response = await self._lockdown(state)
        elif decision.mode in _MODE_TO_AGENT:
            state.response = await self._handle_agent_mode(
                decision.mode, state, history
            )
        else:
            # CONVERSATION (and any unrecognized fallback).
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
        # Keep the graph's conditional-edge targets a closed set: the modes with
        # a dedicated node (CLARIFY, RESEARCH, SECURITY_LOCKDOWN, and the
        # specialist-agent modes) pass through; everything else normalizes to
        # CONVERSATION.
        known = {Mode.CLARIFY, Mode.RESEARCH, Mode.SECURITY_LOCKDOWN, *_MODE_TO_AGENT}
        if decision.mode not in known:
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

    async def node_automation(self, state: GraphState) -> GraphState:
        """AUTOMATION node: dispatch the automation agent (no confirm-gate)."""
        return await self._node_agent_mode(Mode.AUTOMATION, state)

    async def node_device(self, state: GraphState) -> GraphState:
        """DEVICE_CONTROL node: confirm-gate then dispatch the device agent."""
        return await self._node_agent_mode(Mode.DEVICE_CONTROL, state)

    async def node_alerting(self, state: GraphState) -> GraphState:
        """ALERTING node: confirm-gate then dispatch the alerting agent."""
        return await self._node_agent_mode(Mode.ALERTING, state)

    async def _node_agent_mode(self, mode: Mode, state: GraphState) -> GraphState:
        """Shared node body for a specialist-agent mode."""
        if state.response is None:
            history = self._memory.history(state.session_id)
            state.mode = mode
            state.response = await self._handle_agent_mode(mode, state, history)
        self._record(state)
        return state

    async def node_security_lockdown(self, state: GraphState) -> GraphState:
        """SECURITY_LOCKDOWN node: run the lockdown subgraph, report the audit."""
        if state.response is None:
            state.mode = Mode.SECURITY_LOCKDOWN
            state.response = await self._lockdown(state)
        self._record(state)
        return state
