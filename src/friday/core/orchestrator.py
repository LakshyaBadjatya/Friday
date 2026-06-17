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

Phase 4 adds the durable-memory policy (build-spec §10): after an agent runs the
orchestrator applies a **write-consent** gate to its proposed :class:`MemoryWrite`
records — non-sensitive writes auto-commit to the long-term + vector stores when
``settings.memory_autowrite`` is set, while sensitive writes are never
auto-persisted and only land on an explicit confirmation. A **"forget X"**
command purges everything stored about ``X`` from both stores. Grounded recall
itself lives in the Knowledge agent (hybrid vector + long-term retrieval).

Honesty is structural: any :class:`~friday.errors.FridayError` is surfaced
plainly rather than masked as success, and tool failures are reported as
"couldn't retrieve" rather than fabricated.

This module never imports an LLM SDK — it depends only on the
:class:`~friday.providers.llm.LLMProvider` abstraction (grep-enforced by
``tests/unit/test_architecture.py``).
"""

from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote

from pydantic import BaseModel

from friday.agents.base import AgentRegistry, AgentResult
from friday.config import get_settings
from friday.core.confidence import ConfidenceScorer, signals_from_state
from friday.core.critic import SelfCritic
from friday.core.router import route
from friday.core.security import run_lockdown
from friday.core.state import GraphState, Mode
from friday.errors import FridayError, PermissionError, ProviderError
from friday.memory.compaction import Compactor
from friday.memory.long_term import LongTermStore
from friday.memory.short_term import ShortTermMemory
from friday.memory.vector import Chunk
from friday.models.budget import Budgeter
from friday.models.gateway import ModelGateway
from friday.observability.metrics import Metrics
from friday.observability.replay import TurnRecorder
from friday.observability.tracing import Tracer
from friday.observability.usage import UsageLedger
from friday.protocols.runner import ProtocolResult, ProtocolRunner
from friday.protocols.store import Protocol as ProtocolModel
from friday.providers.emotion import Emotion, emotion_hint
from friday.providers.llm import LLMProvider, LLMResponse, Message
from friday.roster import Persona, RosterRegistry
from friday.tools.registry import ToolRegistry
from friday.tools.security import AuditRecord

logger = logging.getLogger("friday.core.orchestrator")


@runtime_checkable
class ForgettableVectorStore(Protocol):
    """The slice of the vector-store contract the orchestrator depends on.

    Extends the Phase-2 ``add``/``query`` retrieval surface with the Phase-4
    ``forget`` operation (build-spec §10). Both
    :class:`~friday.memory.vector.InMemoryVectorStore` and
    :class:`~friday.memory.vector.SQLiteVectorStore` satisfy this structurally, so
    the orchestrator can index auto-committed writes and purge them on a
    "forget X" command without coupling to a concrete store.
    """

    def add(self, docs: list[tuple[str, str]]) -> None:
        """Index ``docs`` as ``(text, source_id)`` pairs."""
        ...

    def query(self, text: str, k: int = 4) -> list[Chunk]:
        """Return up to ``k`` chunks most relevant to ``text``, closest first."""
        ...

    def forget(self, query_or_source_id: str) -> int:
        """Drop documents matching ``query_or_source_id``; return the count."""
        ...


@runtime_checkable
class ProtocolStore(Protocol):
    """The slice of the protocol-store contract the orchestrator depends on.

    Only the read surface the trigger-phrase hook needs: list every protocol (to
    scan their trigger phrases) and look one up by name (for "run the <name>
    protocol"). :class:`~friday.protocols.store.SQLiteProtocolStore` satisfies this
    structurally, so the orchestrator depends on the contract, not the concrete
    store, and stays constructible un-wired in narrow unit tests.
    """

    def list_protocols(self) -> list[ProtocolModel]:
        """Return every stored protocol (insertion order)."""
        ...

    def get_by_name(self, name: str) -> ProtocolModel | None:
        """Return the protocol named ``name`` (case-insensitive) or ``None``."""
        ...


@runtime_checkable
class N8nServiceProtocol(Protocol):
    """The slice of the n8n-service contract the orchestrator depends on.

    Only :meth:`make_workflow` is needed: given a description and the turn's
    confirm flag, it returns either a ``needs_confirmation`` payload (n8n is down)
    or a drafted/imported workflow result. :class:`~friday.n8n.service.N8nService`
    satisfies this structurally, so the orchestrator depends on the contract — not
    the concrete service — and stays constructible un-wired in narrow unit tests
    (and never imports the ``n8n`` package).
    """

    async def make_workflow(
        self, description: str, *, confirmed: bool = False
    ) -> dict[str, Any]:
        """Draft (and optionally import) a workflow; confirm-gate the n8n start."""
        ...


class MemoryWrite(BaseModel):
    """A memory write an agent *proposes*; the orchestrator decides if it lands.

    Agents append these to :attr:`~friday.agents.base.AgentResult.memory_writes`
    (whose declared type is ``list[Any]``, so this extends that shape without
    changing the agent contract). The orchestrator's write-consent policy
    (build-spec §10) then commits them:

    * ``sensitive=False`` -> auto-committed to the long-term + vector stores when
      ``settings.memory_autowrite`` is true; dropped when it is false.
    * ``sensitive=True`` -> NEVER auto-persisted. The orchestrator holds the write
      pending and asks the owner to confirm; only a confirming follow-up
      (``state.confirmed``) commits it.

    ``source_id`` ties the stored fact/chunk back to its origin so a grounded
    answer can cite it (and so ``forget`` can target it).
    """

    text: str
    source_id: str
    sensitive: bool = False

# Tools the minimal Research path is allowed to reach this phase.
_RESEARCH_ALLOWED_TOOLS: frozenset[str] = frozenset({"web_search", "weather"})

# Live-data retrieval on the research path is tool-aware: a weather/temperature
# question is answered from the keyless ``weather`` tool (wttr.in) instead of the
# flaky search backend — a model shouldn't have to pick the right tool. The
# detector is deliberately narrow so ordinary research ("research the best vector
# database") still goes through web_search.
_WEATHER_RE = re.compile(
    r"\b(?:weather|forecast|temperature|how\s+hot|how\s+cold)\b", re.IGNORECASE
)
_LOCATION_AFTER_RE = re.compile(
    r"\b(?:in|at|for|near|around)\b\s+(.+)$", re.IGNORECASE
)
_WEATHER_NOISE_RE = re.compile(
    r"\b(?:what'?s?|what\s+is|how'?s?|how\s+is|the|a|current|currently|"
    r"right\s+now|now|today|tonight|tomorrow|this\s+week|weather|forecast|"
    r"temperature|temp|like|outside|in|at|for|of|please|tell\s+me)\b",
    re.IGNORECASE,
)


def _weather_location(query: str) -> str:
    """Best-effort place name from a weather question, or ``""`` if none found.

    Prefers the text after "in/at/for <place>"; otherwise strips the weather and
    time noise words. Offline and deterministic — no model call.
    """
    cleaned = query.strip().rstrip("?.! ").strip()
    after = _LOCATION_AFTER_RE.search(cleaned)
    candidate = after.group(1) if after else cleaned
    loc = _WEATHER_NOISE_RE.sub(" ", candidate)
    return re.sub(r"\s{2,}", " ", loc).strip(" ,.-")

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

# "Forget X" intent (build-spec §10): a light keyword check in the orchestrator
# extracts the topic ``X`` to forget. Ordered most-specific-first so the longer
# lead-in is tried before the bare "forget X". Each pattern captures ``X`` in its
# only group. The router has no FORGET mode, so this is detected up front.
_FORGET_RULES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:forget|delete|erase|wipe)\s+(?:everything|all|what)\s+(?:you\s+)?"
        r"(?:know|remember|have)\s+about\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:forget|delete|erase|wipe)\s+about\s+(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:forget|delete|erase|wipe)\s+(.+)",
        re.IGNORECASE,
    ),
)


# "run the <name> protocol" intent (Tier 1 voice protocols): an explicit, named
# way to fire a stored protocol regardless of its trigger phrase. The name is the
# only capture group; "the"/"my" lead-ins and the trailing "protocol"/"routine"
# keyword are optional so "run goodnight", "run the goodnight protocol", and
# "start my bedtime routine" all match. Detected up front (the router has no
# PROTOCOL mode).
_RUN_PROTOCOL_RULES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:run|start|execute|activate|begin)\s+(?:the\s+|my\s+)?"
        r"(.+?)\s+(?:protocol|routine)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:run|start|execute|activate|begin)\s+(?:the\s+|my\s+)?protocol\s+(.+)",
        re.IGNORECASE,
    ),
)


# "make a workflow on n8n <X>" / "n8n workflow <X>" intent (Tier 2 n8n): a light
# keyword check extracts the workflow description ``X``. Ordered most-specific-first
# so the fuller "make a workflow on n8n ..." lead-in is tried before the bare "n8n
# workflow ...". Each pattern captures the description in its only group. The router
# has no N8N mode, so this is detected up front (inert unless n8n is enabled +
# wired). The leading verb is optional ("make/create/build/set up a workflow").
_N8N_WORKFLOW_RULES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:make|create|build|set\s*up|draft)\s+(?:me\s+)?(?:an?\s+)?"
        r"(?:workflow|automation)\s+(?:on|in|with|using|for)\s+n8n\s+(?:that\s+|to\s+|which\s+|for\s+)?(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"n8n\s+(?:workflow|automation)\s+(?:that\s+|to\s+|which\s+|for\s+)?(.+)",
        re.IGNORECASE,
    ),
)


def _extract_n8n_description(text: str) -> str | None:
    """Return the workflow description ``X`` of an "n8n workflow X" command, or None.

    Matches the ordered :data:`_N8N_WORKFLOW_RULES` and returns the captured
    description with surrounding whitespace and a trailing period/quote stripped,
    so ``"Make a workflow on n8n that posts to Slack."`` yields ``"posts to
    Slack"``.
    """
    for pattern in _N8N_WORKFLOW_RULES:
        match = pattern.search(text)
        if match is not None:
            description = match.group(1).strip().strip("\"'").rstrip(".!?").strip()
            if description:
                return description
    return None


# Light "maps" intent: an "open maps" turn deep-links to the local ``/maps``
# globe; a "distance to <X>" / "show me distance to <X>" turn deep-links to
# ``/maps?to=<X>`` (the destination URL-encoded) so the page geocodes + measures
# it on load. Detected up front (the router has no MAPS mode) and answered
# deterministically — no LLM call, so the link is always exact and offline-safe.
# The ``distance`` rule is tried before the bare ``open maps`` so "show me
# distance to ..." is never mistaken for a plain open.
_MAPS_DISTANCE_RULES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:show\s+me\s+(?:the\s+)?|what'?s\s+the\s+|get\s+(?:me\s+)?(?:the\s+)?)?"
        r"distance\s+(?:to|from\s+here\s+to)\s+(.+)",
        re.IGNORECASE,
    ),
)
_OPEN_MAPS_RULE: re.Pattern[str] = re.compile(
    r"^\s*(?:open|show|launch|bring\s+up)\s+(?:the\s+)?maps?\b",
    re.IGNORECASE,
)


def _extract_maps_distance(text: str) -> str | None:
    """Return the destination ``X`` of a "distance to <X>" command, or ``None``.

    Matches the ordered :data:`_MAPS_DISTANCE_RULES` and returns the captured
    destination with surrounding whitespace and a trailing period/quote stripped,
    so ``"show me distance to New York City."`` yields ``"New York City"``.
    """
    for pattern in _MAPS_DISTANCE_RULES:
        match = pattern.search(text)
        if match is not None:
            target = match.group(1).strip().strip("\"'").rstrip(".!?").strip()
            if target:
                return target
    return None


def _is_open_maps(text: str) -> bool:
    """Whether ``text`` is a plain "open maps" command (no destination)."""
    return _OPEN_MAPS_RULE.search(text) is not None


def _extract_protocol_name(text: str) -> str | None:
    """Return the protocol name from a "run the <name> protocol" command, or None.

    Matches the ordered :data:`_RUN_PROTOCOL_RULES` and returns the captured name
    with surrounding whitespace and a trailing period/quote stripped, so
    ``"run the Goodnight protocol."`` yields ``"Goodnight"``.
    """
    for pattern in _RUN_PROTOCOL_RULES:
        match = pattern.search(text)
        if match is not None:
            name = match.group(1).strip().strip("\"'").rstrip(".!?").strip()
            if name:
                return name
    return None


def _extract_forget_target(text: str) -> str | None:
    """Return the topic ``X`` of a "forget X" command, or ``None`` if not one.

    Matches the ordered :data:`_FORGET_RULES` and returns the captured topic with
    surrounding whitespace and a trailing period/quote stripped, so
    ``"forget what you know about my address."`` yields ``"my address"``.
    """
    for pattern in _FORGET_RULES:
        match = pattern.search(text)
        if match is not None:
            target = match.group(1).strip().strip("\"'").rstrip(".!?").strip()
            if target:
                return target
    return None


# Address-by-name (Stage 2 roster): a turn that opens with a persona code-name —
# "GECKO, ..." (direct address) or "ask/tell/have VISION ..." (delegation form) —
# routes under that named persona's least-privilege tool scope + memory namespace.
# Addressing is by the EXPLICIT code-name (a leading all-caps token, or a delegation
# verb followed by the name), so an ordinary word that merely happens to match a
# name ("forge a plan", "track the build") never trips it. The name is the only
# capture group; the registry's case-insensitive lookup resolves it to a persona.
_ADDRESS_RULES: tuple[re.Pattern[str], ...] = (
    # Delegation: "ask/tell/have/get GECKO to ..." (name may be any case here, but
    # it must be a single bare token immediately after the verb).
    re.compile(
        r"^\s*(?:ask|tell|have|get)\s+([A-Za-z]+)\b(?:\s+(?:to|about|for|whether|if)\b)?",
        re.IGNORECASE,
    ),
    # Direct address: a leading ALL-CAPS code-name followed by a comma/colon (so
    # "GECKO, ..." addresses GECKO but "Gecko geckos are lizards" does not).
    re.compile(r"^\s*([A-Z]{2,})\s*[,:]"),
)


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
        long_term: Optional durable store for the write-consent policy and the
            "forget" command. When absent, agent-proposed writes are simply not
            persisted (the orchestrator stays constructible un-wired).
        vector: Optional persistent vector store. Auto-committed (and confirmed
            sensitive) writes are also indexed here so the Knowledge agent can
            retrieve them; ``forget`` removes them from here too.
        tracer: Optional per-request :class:`~friday.observability.tracing.Tracer`.
            When present, :meth:`handle` opens a trace per turn with route /
            dispatch / synth spans and stamps the turn's mode (build-spec §11).
            When absent (the default) no trace is opened, so orchestrator unit
            tests that don't pass one behave exactly as before.
        metrics: Optional :class:`~friday.observability.metrics.Metrics` counters.
            When present, each turn increments ``requests`` and the per-mode
            counter (and ``errors`` on a domain failure). No-op when absent.
        protocol_store: Optional durable
            :class:`~friday.protocols.store.SQLiteProtocolStore`. When wired *and*
            ``FRIDAY_ENABLE_PROTOCOLS`` is set, a turn whose text matches a stored
            ``trigger_phrase`` (or "run the <name> protocol") fires that protocol
            via ``protocol_runner`` before normal routing. When absent (the default)
            the hook is inert, so existing orchestrator tests behave unchanged.
        protocol_runner: Optional :class:`~friday.protocols.runner.ProtocolRunner`
            over the shared registry; runs a matched protocol's steps in order,
            honoring the confirm-step. Inert unless both it and ``protocol_store``
            are wired and the flag is on.
        critic: Optional :class:`~friday.core.critic.SelfCritic`. When wired *and*
            ``FRIDAY_ENABLE_SELF_CRITIQUE`` is set, the final synthesized reply is
            reviewed once before it is returned; if the review fails and offers a
            concrete revision, that revision replaces the reply (one bounded pass,
            never re-critiqued, and non-fatal — any critic error keeps the
            original). When absent or the flag is off, the critic is never called
            (no extra LLM cost), so existing orchestrator behavior is unchanged.
        n8n_service: Optional n8n service (an :class:`N8nServiceProtocol`). When
            wired *and* ``FRIDAY_ENABLE_N8N`` is set, a turn whose text matches
            "make a workflow on n8n <X>" / "n8n workflow <X>" drafts (and
            best-effort imports) a workflow, threading the turn's confirm flag (a
            docker auto-start is confirm-gated). When absent or the flag is off the
            hook is inert, so existing orchestrator tests behave unchanged.
        roster: Optional :class:`~friday.roster.RosterRegistry`. When wired, a turn
            that opens by addressing a persona code-name ("GECKO, ..." /
            "ask VISION to ...") is routed under that named persona's
            least-privilege tool scope + memory namespace (recorded on the
            scratchpad as ``persona`` / ``persona_scope`` / ``persona_namespace``)
            before normal routing. An un-addressed turn (or an unknown name) leaves
            the hook inert, so existing orchestrator behaviour is unchanged.
        confidence: Optional :class:`~friday.core.confidence.ConfidenceScorer`. When
            wired *and* ``FRIDAY_ENABLE_CONFIDENCE`` is set, the orchestrator stamps
            ``state.scratchpad["confidence"]`` after a synthesized reply and, when
            the blended confidence falls below ``confidence_note_threshold``, appends
            a one-line honest caveat (skipped for CLARIFY — a question carries no
            answer to score). When absent or the flag is off the hook is inert, so
            existing orchestrator behaviour is unchanged.
        budgeter: Optional :class:`~friday.models.budget.Budgeter`. When wired *and*
            ``FRIDAY_ENABLE_BUDGETER`` is set, each completion's token (and any
            priced dollar) usage is recorded per session and, once the turn crosses
            the downshift threshold, the gateway's active model is switched to
            ``budget_downshift_model_id`` (only when a ``gateway`` is wired and the
            id is non-empty — otherwise the signal is surfaced but no swap happens).
            When absent or the flag is off the hook is inert.
        gateway: Optional :class:`~friday.models.gateway.ModelGateway` — the thing
            whose active model the budgeter downshifts. ``None`` on the
            fake/single-provider build, in which case the budgeter still tallies but
            has nothing to swap.
        budget_downshift_model_id: The catalog id the budgeter downshifts the
            gateway to when a turn runs hot; empty (the default) keeps the current
            active model (the budgeter still surfaces the signal).
    """

    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
        memory: ShortTermMemory,
        persona_path: str | Path,
        agents: AgentRegistry | None = None,
        long_term: LongTermStore | None = None,
        vector: ForgettableVectorStore | None = None,
        tracer: Tracer | None = None,
        metrics: Metrics | None = None,
        usage_ledger: UsageLedger | None = None,
        turn_recorder: TurnRecorder | None = None,
        protocol_store: ProtocolStore | None = None,
        protocol_runner: ProtocolRunner | None = None,
        critic: SelfCritic | None = None,
        n8n_service: N8nServiceProtocol | None = None,
        roster: RosterRegistry | None = None,
        confidence: ConfidenceScorer | None = None,
        budgeter: Budgeter | None = None,
        gateway: ModelGateway | None = None,
        budget_downshift_model_id: str = "",
        compaction: Compactor | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._memory = memory
        self._persona_path = Path(persona_path)
        self._agents = agents
        self._long_term = long_term
        self._vector = vector
        self._tracer = tracer
        self._metrics = metrics
        self._usage_ledger = usage_ledger
        self._turn_recorder = turn_recorder
        self._protocol_store = protocol_store
        self._protocol_runner = protocol_runner
        self._critic = critic
        self._n8n_service = n8n_service
        self._roster = roster
        self._confidence = confidence
        # The per-turn budgeter (Wave 0): records each completion's usage and
        # surfaces ``should_downshift``. It only has something to downshift when a
        # :class:`~friday.models.gateway.ModelGateway` is wired (the thing whose
        # active model it swaps); on the fake/single-provider build ``gateway`` is
        # ``None`` so the downshift is a no-op and only the signal is stamped.
        self._budgeter = budgeter
        self._gateway = gateway
        self._budget_downshift_model_id = budget_downshift_model_id
        self._compaction = compaction
        # Sensitive writes proposed but not yet confirmed, keyed by session id.
        # They are held here (never persisted) until a confirming follow-up turn
        # commits them — the write-consent gate for sensitive data (§10).
        self._pending_writes: dict[str, list[MemoryWrite]] = {}

    # -- tracing ----------------------------------------------------------- #
    @contextlib.contextmanager
    def _span(self, name: str, **attrs: Any) -> Iterator[None]:
        """Open a tracer span for ``name`` when a tracer is wired, else a no-op.

        Centralizes the ``tracer is None`` guard so the turn body reads cleanly
        and behaves identically (no timing, no recording) when observability is
        not injected.
        """
        if self._tracer is None:
            yield
            return
        with self._tracer.span(name, **attrs):
            yield

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

    def _system_prompt(self, emotion: Emotion | None = None) -> Message:
        owner = get_settings().owner_address
        persona = self._persona_text()
        content = (
            f"{persona}\n\n"
            f"---\nAddress the owner as '{owner}'. Answer first, keep it tight, "
            f"and never fabricate a capability, a fact, or a tool result you do "
            f"not have."
        )
        if emotion is not None and get_settings().emotion_adapt:
            content += "\n\n" + emotion_hint(emotion)
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
    async def _synthesize(
        self, history: list[Message], task: Message,
        session_id: str | None = None, emotion: Emotion | None = None,
    ) -> str:
        """Call the LLM with persona + history + the turn's task message.

        Wraps provider failures in an honest, in-character message rather than
        leaking a traceback or faking a success. When a budgeter is wired and
        ``session_id`` is given, the completion's token usage is recorded against
        the session's turn budget and a hot turn downshifts the gateway's active
        model (see :meth:`_record_budget`).
        """
        messages = [self._system_prompt(emotion), *history, task]
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
        self._record_usage(response)
        if session_id is not None:
            self._record_budget(session_id, response)
        text = response.text
        if not text or not text.strip():
            owner = get_settings().owner_address
            return f"I came back empty on that one, {owner}. Mind rephrasing?"
        return text.strip()

    # -- usage/cost ledger (observability; cost dashboard) ----------------- #
    def _record_usage(self, response: LLMResponse) -> None:
        """Tally a completion's tokens into the process usage ledger.

        Always-on observability (like :class:`Metrics`), independent of the
        budgeter flag — so ``GET /admin/usage`` reflects real spend on every
        build. The model id is the gateway's active model when a gateway is
        wired, else the configured provider name (e.g. ``"fake"``). Free models
        record ``usd=0.0`` so only the token columns move. Inert (and never
        raises into the turn) when no ledger is wired.
        """
        if self._usage_ledger is None:
            return
        usage = response.usage
        model_id = (
            self._gateway.active_model_id
            if self._gateway is not None
            else get_settings().llm_provider
        )
        self._usage_ledger.record(
            model_id,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            usd=0.0,
        )

    # -- per-turn budgeter (Wave 0) ---------------------------------------- #
    def _start_budget_turn(self, session_id: str) -> None:
        """Zero the session's per-turn spend tally at the start of a turn.

        Inert unless ``FRIDAY_ENABLE_BUDGETER`` is set *and* a budgeter is wired.
        Resetting per turn keeps spend from leaking across turns; the budgeter also
        lazy-starts on first record, so this is the explicit reset, not a hard
        requirement.
        """
        if self._budgeter is None or not get_settings().enable_budgeter:
            return
        self._budgeter.start_turn(session_id)

    def _record_budget(self, session_id: str, response: LLMResponse) -> None:
        """Tally a completion's usage and downshift the gateway when a turn is hot.

        Inert unless ``FRIDAY_ENABLE_BUDGETER`` is set *and* a
        :class:`~friday.models.budget.Budgeter` is wired — so the flag-off /
        un-wired path keeps no tally and never swaps models. When active, the
        completion's total tokens are recorded against the session's turn budget
        (free models stay at ``usd=0.0`` — only the token cap governs), and once
        :meth:`Budgeter.should_downshift` trips AND a non-empty
        ``budget_downshift_model_id`` is configured, the gateway's active model is
        switched down a tier. The ``set_active`` call is guarded on a gateway being
        wired (the fake/single-provider build has none — the budgeter still tallies
        but has nothing to swap). Non-fatal by construction: it never raises into
        the turn.
        """
        if self._budgeter is None or not get_settings().enable_budgeter:
            return
        usage = response.usage
        total_tokens = usage.prompt_tokens + usage.completion_tokens
        self._budgeter.record(session_id, tokens=total_tokens, usd=0.0)
        if (
            self._gateway is not None
            and self._budget_downshift_model_id
            and self._budgeter.should_downshift(session_id)
        ):
            current = self._gateway.active_model_id
            target = self._budget_downshift_model_id
            if current != target:
                logger.info(
                    "budget downshift: session %s over budget, switching %s -> %s",
                    session_id,
                    current,
                    target,
                )
                self._gateway.set_active(target)

    # -- self-critique (Tier 2; build-spec post-spec) ---------------------- #
    async def _maybe_compact(self, state: GraphState) -> None:
        """Optionally fold an over-long session into a summary + recent tail.

        Inert unless ``FRIDAY_ENABLE_COMPACTION`` is set *and* a
        :class:`~friday.memory.compaction.Compactor` is wired, so the flag-off /
        un-wired path makes no extra model call. When it fires, the session's
        short-term buffer is replaced with a single summary message followed by the
        retained recent turns — the conversation is condensed, never dropped
        (compaction returns ``None`` — leaving the buffer untouched — when there is
        too little history or the summary pass fails). Non-fatal by construction.
        """
        if self._compaction is None or not get_settings().enable_compaction:
            return
        history = self._memory.history(state.session_id)
        result = await self._compaction.maybe_compact(history)
        if result is None:
            return
        self._memory.clear(state.session_id)
        self._memory.append(
            state.session_id,
            Message(
                role="system",
                content=f"[Earlier conversation summarized] {result.summary}",
            ),
        )
        for message in result.kept:
            self._memory.append(state.session_id, message)
        state.scratchpad["compaction"] = {"compacted": result.compacted_count}

    async def _maybe_critique(self, state: GraphState) -> None:
        """Optionally review the final synthesized reply once; revise if it fails.

        Inert unless ``FRIDAY_ENABLE_SELF_CRITIQUE`` is set *and* a
        :class:`~friday.core.critic.SelfCritic` is wired — so the flag-off /
        un-wired path never touches the critic's LLM (no extra model call). When
        active it runs exactly ONE review of ``state.response``; if the critique
        is ``not ok`` and offers a concrete ``revised`` draft, that draft replaces
        the response (the revision is NOT itself re-critiqued — one bounded pass).

        Non-fatal by construction: the critic swallows its own LLM/parse errors
        (returning a passing critique), so a failure here can only ever keep the
        original response. The outcome is recorded on the scratchpad for the trace.
        """
        if self._critic is None or not get_settings().enable_self_critique:
            return
        draft = state.response
        if draft is None:
            return
        # Open the "critique" span ONLY here — past the flag/wiring guard — so the
        # off/un-wired turn emits the exact same span set as before (no observable
        # change when the feature is inert).
        with self._span("critique"):
            critique = await self._critic.review(draft, user_text=state.user_input)
        revised = critique.revised if (not critique.ok and critique.revised) else None
        state.scratchpad["self_critique"] = {
            "ok": critique.ok,
            "issues": critique.issues,
            "revised": revised is not None,
        }
        if revised is not None:
            state.response = revised

    # -- calibrated confidence (Wave 0) ------------------------------------ #
    def _maybe_stamp_confidence(self, state: GraphState) -> None:
        """Optionally stamp a calibrated confidence score; caveat below threshold.

        Inert unless ``FRIDAY_ENABLE_CONFIDENCE`` is set *and* a
        :class:`~friday.core.confidence.ConfidenceScorer` is wired — so the flag-off
        / un-wired path stamps nothing and appends no caveat (existing behaviour
        unchanged). When active it blends the turn's signals (router confidence, any
        agent confidence, grounding, a web-search hit) into one
        :class:`~friday.core.confidence.ConfidenceScore`, stamps it onto
        ``state.scratchpad["confidence"]`` (model-dumped for the trace/HUD), and —
        only when the blended value falls below ``confidence_note_threshold`` AND a
        real reply is present — appends a one-line honest caveat.

        Called only after a real synthesized reply (CLARIFY is skipped, like
        critique, since a clarifying question carries no answer to score). Pure and
        deterministic: the scorer reads no settings and uses no clock.
        """
        if self._confidence is None or not get_settings().enable_confidence:
            return
        score = self._confidence.score(signals_from_state(state))
        state.scratchpad["confidence"] = score.model_dump()
        threshold = get_settings().confidence_note_threshold
        if score.value < threshold and state.response is not None:
            owner = get_settings().owner_address
            state.response = (
                f"{state.response}\n\n(Confidence is on the low side here, "
                f"{owner} — {score.rationale} Worth a second check.)"
            )

    # -- conversation ------------------------------------------------------ #
    async def _converse(self, state: GraphState, history: list[Message]) -> str:
        task = Message(role="user", content=state.user_input)
        return await self._synthesize(history, task, state.session_id,
                                      emotion=state.emotion)

    # -- research ---------------------------------------------------------- #
    async def _research(self, state: GraphState, history: list[Message]) -> str:
        """Minimal research path: optionally search, then synthesize honestly.

        Calls ``web_search`` through the registry respecting the research
        agent's ``allowed_tools``. Whatever the tool returns (success, empty, or
        a handled failure) is reported truthfully — no fabricated findings.
        """
        findings_block = ""
        # Tool-aware retrieval: a weather question uses the keyless wttr.in
        # ``weather`` tool (the search backend is unreliable); everything else
        # uses ``web_search`` exactly as before.
        location = (
            _weather_location(state.user_input)
            if _WEATHER_RE.search(state.user_input)
            else ""
        )
        if location and "weather" in _RESEARCH_ALLOWED_TOOLS:
            tool_name = "weather"
            tool_args: dict[str, object] = {"location": location}
        else:
            tool_name = "web_search"
            tool_args = {"query": state.user_input, "max_results": 5}
        try:
            result = await self._registry.execute(
                tool_name,
                tool_args,
                allowed_tools=_RESEARCH_ALLOWED_TOOLS,
            )
            state.scratchpad["web_search_invoked"] = tool_name == "web_search"
        except PermissionError as exc:
            # The research path should always be allowed its retrieval tool; if
            # not, report honestly rather than guess.
            logger.warning("research path denied %s: %s", tool_name, exc)
            state.scratchpad["web_search_invoked"] = False
            findings_block = (
                "NOTE TO SELF: retrieval was not permitted, so you have no "
                "retrieved sources. Say so plainly; do not invent findings."
            )
        else:
            if result.ok and tool_name == "weather":
                summary = str(result.data.get("summary", "")).strip()
                state.scratchpad["weather_result"] = result.data
                findings_block = (
                    "RETRIEVED WEATHER (live, from wttr.in):\n" + summary
                    if summary
                    else (
                        "NOTE TO SELF: the weather lookup returned nothing "
                        "usable. Say so plainly; do not invent a forecast."
                    )
                )
            elif result.ok:
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
                label = "weather lookup" if tool_name == "weather" else "web search"
                logger.warning("research %s failed: %s", tool_name, detail)
                findings_block = (
                    f"NOTE TO SELF: the {label} failed "
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
        return await self._synthesize(history, task, state.session_id,
                                      emotion=state.emotion)

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
        consent_prompt = self._apply_write_consent(state, result)
        if consent_prompt is not None:
            # A sensitive write is pending the owner's confirmation; surface the
            # consent question instead of (or alongside) the agent's reply.
            return consent_prompt
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
        messages = [self._system_prompt(state.emotion), *history, task]
        try:
            response = await self._llm.complete(messages, tools=None)
        except ProviderError as exc:
            logger.warning("persona wrap failed; using agent draft: %s", exc)
            return draft
        text = response.text
        if not text or not text.strip():
            return draft
        return text.strip()

    # -- write-consent policy (build-spec §10) ----------------------------- #
    def _apply_write_consent(
        self, state: GraphState, result: AgentResult
    ) -> str | None:
        """Commit an agent's proposed writes per the consent policy.

        Inspects ``result.memory_writes`` for :class:`MemoryWrite` records (other
        write shapes — e.g. the automation agent's bare step strings — are left
        for the audit trail and ignored here). For each :class:`MemoryWrite`:

        * non-sensitive -> auto-committed when ``settings.memory_autowrite`` is
          true, dropped otherwise;
        * sensitive -> NEVER auto-persisted; held pending and a persona confirm
          prompt is returned so the owner can authorize it.

        Returns the confirm prompt when at least one sensitive write is pending,
        else ``None`` (the caller then proceeds with the normal persona reply).
        """
        proposed = [w for w in result.memory_writes if isinstance(w, MemoryWrite)]
        if not proposed:
            return None

        autowrite = get_settings().memory_autowrite
        sensitive = [w for w in proposed if w.sensitive]
        nonsensitive = [w for w in proposed if not w.sensitive]

        if autowrite:
            for write in nonsensitive:
                self._commit_write(write)

        if sensitive:
            # Hold sensitive writes pending; do not persist anything sensitive.
            self._pending_writes.setdefault(state.session_id, []).extend(sensitive)
            return self._consent_question(sensitive)
        return None

    def _consent_question(self, writes: list[MemoryWrite]) -> str:
        """A persona confirm prompt for one or more pending sensitive writes."""
        owner = get_settings().owner_address
        if len(writes) == 1:
            what = f"“{writes[0].text}”"
        else:
            what = f"{len(writes)} sensitive item(s)"
        return (
            f"That's sensitive, {owner}, so I won't store it unless you say so: "
            f"want me to remember {what}? Reply to confirm and I'll commit it."
        )

    def _commit_pending_writes(self, state: GraphState) -> int:
        """Commit any sensitive writes pending for the session; return the count.

        Called when a follow-up turn arrives confirmed (``state.confirmed``). The
        pending writes (held un-persisted since the proposing turn) are committed
        to the long-term + vector stores and cleared.
        """
        pending = self._pending_writes.pop(state.session_id, [])
        for write in pending:
            self._commit_write(write)
        return len(pending)

    def _commit_write(self, write: MemoryWrite) -> None:
        """Persist one approved write to the long-term and vector stores.

        Idempotent w.r.t. wiring: a store that was not injected is simply skipped,
        so the orchestrator stays usable in narrow unit tests with no stores.
        """
        if self._long_term is not None:
            self._long_term.add_fact(
                write.text, write.source_id, sensitive=write.sensitive
            )
        if self._vector is not None:
            self._vector.add([(write.text, write.source_id)])

    # -- forget command (build-spec §10) ----------------------------------- #
    def _forget(self, target: str) -> str:
        """Remove everything FRIDAY knows about ``target`` and confirm in persona.

        Calls ``forget`` on both the long-term and vector stores (each skipped if
        not wired) and reports the total rows removed honestly — including the
        "nothing matched" case, which is stated plainly rather than dressed up.
        """
        owner = get_settings().owner_address
        removed = 0
        if self._long_term is not None:
            removed += self._long_term.forget(target)
        if self._vector is not None:
            removed += self._vector.forget(target)
        if removed == 0:
            return (
                f"I had nothing stored about “{target}”, {owner}, so there was "
                f"nothing to forget."
            )
        return (
            f"Done, {owner} — I've forgotten what I knew about “{target}” "
            f"({removed} record(s) removed)."
        )

    # -- address-by-name (Stage 2 roster) ---------------------------------- #
    def _match_persona(self, text: str) -> Persona | None:
        """Return the roster persona ``text`` addresses by code-name, or ``None``.

        Inert unless a :class:`~friday.roster.RosterRegistry` is wired. Matches the
        ordered :data:`_ADDRESS_RULES` (delegation form first, then the leading
        all-caps direct address) and resolves the captured token to a persona
        (case-insensitively). A token that is not a registered persona name yields
        ``None`` so the turn proceeds unchanged.
        """
        roster = self._roster
        if roster is None:
            return None
        for pattern in _ADDRESS_RULES:
            match = pattern.match(text)
            if match is None:
                continue
            name = match.group(1).strip()
            if not name:
                continue
            try:
                return roster.get(name)
            except KeyError:
                # The leading token is not a persona — not an address; stop trying
                # later (less-specific) patterns so a plain word never matches.
                return None
        return None

    def _apply_persona_scope(self, persona: Persona, state: GraphState) -> None:
        """Record the addressed persona's least-privilege scope onto ``state``.

        Stamps the scratchpad with the persona's ``name``, its sorted tool
        ``persona_scope`` (the least-privilege allow-list the turn runs under), and
        its ``persona_namespace`` (the memory namespace it reads/writes under), so
        the turn is routed under the named operator rather than the prime.
        """
        state.scratchpad["persona"] = persona.name
        state.scratchpad["persona_scope"] = sorted(persona.allowed_tools)
        state.scratchpad["persona_namespace"] = persona.memory_namespace

    # -- maps deep links (Tier 3) ------------------------------------------ #
    def _open_maps_reply(self) -> str:
        """A persona reply deep-linking to the local ``/maps`` globe."""
        owner = get_settings().owner_address
        return (
            f"Opening the map, {owner} — here's the globe: /maps. Say a place to "
            f"fly there, or ask for a distance."
        )

    def _distance_maps_reply(self, destination: str) -> str:
        """A persona reply deep-linking to ``/maps?to=<destination>`` (encoded).

        The destination is URL-encoded with :func:`urllib.parse.quote` (so a
        space, ``&``, or ``?`` in the place name never breaks the query) and the
        page geocodes + measures it on load. Reported deterministically (no LLM
        call), so the link is always exact.
        """
        owner = get_settings().owner_address
        link = f"/maps?to={quote(destination)}"
        return (
            f"On it, {owner} — pulling up the distance to {destination}: {link}"
        )

    # -- n8n workflows (Tier 2) -------------------------------------------- #
    async def _make_n8n_workflow(self, description: str, state: GraphState) -> str:
        """Draft (and maybe import) an n8n workflow and report it in persona.

        Threads ``state.confirmed`` into the service so a docker auto-start is
        gated by the confirm-step: when the service returns
        ``needs_confirmation`` (n8n is down, unconfirmed) a persona confirm
        question is surfaced and NOTHING is started or drafted; a confirming
        follow-up re-fires and proceeds. Otherwise the drafted/imported workflow
        and its setup notes are reported deterministically (no LLM call here — the
        draft already used one), so an n8n turn is always truthful.
        """
        assert self._n8n_service is not None  # guarded by the caller
        state.mode = Mode.AUTOMATION
        result = await self._n8n_service.make_workflow(
            description, confirmed=state.confirmed
        )
        state.scratchpad["n8n"] = result
        return self._n8n_reply(result)

    def _n8n_reply(self, result: dict[str, Any]) -> str:
        """A deterministic persona reply describing an n8n workflow outcome."""
        owner = get_settings().owner_address
        if result.get("kind") == "needs_confirmation":
            return (
                f"n8n isn't running, {owner}, so I can't draft against it yet. I "
                f"can start it with docker — that's a real-world action, so reply "
                f"to confirm and I'll bring it up, then build the workflow."
            )

        workflow = result.get("workflow")
        name = workflow.get("name") if isinstance(workflow, dict) else None
        label = f"“{name}”" if name else "the workflow"
        lines: list[str] = []
        if result.get("started"):
            lines.append(f"Started n8n via docker, {owner}, then drafted {label}.")
        elif result.get("imported"):
            lines.append(f"Done, {owner} — drafted and imported {label} into n8n.")
        else:
            lines.append(f"Done, {owner} — drafted {label}.")

        import_error = result.get("import_error")
        if import_error:
            lines.append(
                f"I couldn't import it automatically ({import_error}); the JSON is "
                f"ready for you to import by hand."
            )
        elif not result.get("imported"):
            lines.append(
                "I didn't import it (no n8n API key set); the JSON is ready to "
                "import in n8n."
            )

        setup_notes = result.get("setup_notes") or []
        if isinstance(setup_notes, list) and setup_notes:
            lines.append("Still to configure:")
            lines.extend(f"- {note}" for note in setup_notes)
        return "\n".join(lines)

    # -- voice protocols (Tier 1) ------------------------------------------ #
    def _match_protocol(self, text: str) -> ProtocolModel | None:
        """Return the enabled protocol ``text`` fires, or ``None``.

        Matching is inert unless protocols are enabled *and* both the store and
        runner are wired. Two ways to fire a protocol, most-specific first:

        1. An explicit "run the <name> protocol" command -> looked up by name.
        2. A stored ``trigger_phrase`` contained (case-insensitively, on a word
           boundary) in the user message -> the longest matching phrase wins, so a
           more specific trigger beats a shorter one.

        Only ``enabled`` protocols match; a disabled one is skipped.
        """
        if not get_settings().enable_protocols:
            return None
        store = self._protocol_store
        if store is None or self._protocol_runner is None:
            return None

        named = _extract_protocol_name(text)
        if named is not None:
            candidate = store.get_by_name(named)
            if candidate is not None and candidate.enabled:
                return candidate

        lowered = text.lower()
        best: ProtocolModel | None = None
        best_len = 0
        for protocol in store.list_protocols():
            if not protocol.enabled:
                continue
            phrase = protocol.trigger_phrase.strip().lower()
            if not phrase:
                continue
            if re.search(rf"\b{re.escape(phrase)}\b", lowered):
                # Compare like-for-like: the normalized phrase length, not the raw
                # (possibly whitespace-padded) incumbent length, so "longest phrase
                # wins" picks the genuinely most-specific protocol.
                if best is None or len(phrase) > best_len:
                    best = protocol
                    best_len = len(phrase)
        return best

    async def _run_protocol(
        self, protocol: ProtocolModel, state: GraphState
    ) -> str:
        """Run ``protocol`` via the runner, threading ``state.confirmed``.

        When a side-effecting step pauses on the confirm-step (and the turn is
        unconfirmed), a persona confirm question is surfaced and NO side-effecting
        step runs — a confirming follow-up re-fires the protocol and proceeds.
        Otherwise the completed run is reported honestly (including a step error).
        """
        assert self._protocol_runner is not None  # guarded by the caller
        state.mode = Mode.AUTOMATION
        result = await self._protocol_runner.run(
            protocol, confirmed=state.confirmed
        )
        state.scratchpad["protocol"] = protocol.name
        state.scratchpad["protocol_result"] = result.model_dump()
        return self._protocol_reply(protocol, result)

    def _protocol_reply(
        self, protocol: ProtocolModel, result: ProtocolResult
    ) -> str:
        """A deterministic persona reply describing a protocol run's outcome.

        Reported without an LLM call so a protocol run is always truthful and never
        blocked on a model: a confirm question on a paused side-effecting step, an
        honest failure on a stepped error, or a tidy "done" on a full run.
        """
        owner = get_settings().owner_address
        if result.needs_confirmation:
            paused = result.steps[-1].tool if result.steps else "a step"
            done = max(len(result.steps) - 1, 0)
            return (
                f"The {protocol.name} protocol has a real-world step "
                f"({paused!r}) I won't run without a nod, {owner}. "
                f"{done} step(s) done so far — reply to confirm and I'll finish it."
            )
        if not result.ran:
            failed = next((s for s in result.steps if not s.ok), None)
            where = failed.tool if failed is not None else "a step"
            detail = failed.error if failed is not None and failed.error else "unknown"
            return (
                f"The {protocol.name} protocol stopped at {where!r}, {owner} — "
                f"{detail}. I ran the steps before it and held the rest."
            )
        return (
            f"Done, {owner} — ran the {protocol.name} protocol "
            f"({len(result.steps)} step(s))."
        )

    # -- knowledge turn (grounded retrieval entrypoint) -------------------- #
    async def knowledge_turn(self, state: GraphState) -> GraphState:
        """Run a grounded Knowledge-agent turn and synthesize a persona reply.

        The deterministic router has no dedicated KNOWLEDGE mode, so this is the
        explicit entrypoint callers use to ask the Knowledge agent (which does
        hybrid vector + long-term retrieval and cites ``source_id``s). When no
        agent registry / knowledge agent is wired, it falls back to a plain
        persona conversation rather than crashing.
        """
        history = self._memory.history(state.session_id)
        state.mode = Mode.CONVERSATION
        if self._agents is None or "knowledge" not in self._agents:
            state.response = await self._converse(state, history)
            self._record(state)
            return state
        agent = self._agents.get("knowledge")
        result = await agent.run(state)
        state.scratchpad["agent"] = "knowledge"
        state.scratchpad["agent_confidence"] = result.confidence
        state.response = await self._persona_wrap(state, history, result.output)
        self._record(state)
        return state

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

        When a :class:`~friday.observability.tracing.Tracer` / metrics are wired
        (build-spec §11), the turn opens one trace (with route / dispatch / synth
        spans, stamped with the final mode) and increments the request, per-mode,
        and — on a domain failure — error counters. These emit points are no-ops
        when nothing is injected, so the turn semantics are otherwise unchanged.
        """
        if self._tracer is not None:
            self._tracer.start_trace(state.session_id)
        if self._metrics is not None:
            self._metrics.inc_requests()

        errored = False
        try:
            return await self._handle_inner(state)
        except FridayError as exc:
            # Map any domain error to an honest, in-character message; never
            # fake success. Surfaced here so a stray FridayError from a deeper
            # call site still produces a truthful reply.
            errored = True
            logger.warning("orchestrator caught FridayError: %s", exc)
            owner = get_settings().owner_address
            state.response = (
                f"Hit a snag I can't paper over, {owner}: {exc}. "
                f"That's the honest status."
            )
            return state
        finally:
            if self._metrics is not None:
                self._metrics.inc_mode(str(state.mode))
                if errored:
                    self._metrics.inc_errors()
            if self._tracer is not None:
                self._tracer.finish().mode = str(state.mode)
            if self._turn_recorder is not None:
                self._turn_recorder.record(
                    session_id=state.session_id,
                    user_input=state.user_input,
                    response=state.response,
                    mode=None if state.mode is None else str(state.mode),
                )

    async def _handle_inner(self, state: GraphState) -> GraphState:
        # 1. Load short-term history for the session.
        history = self._memory.history(state.session_id)

        # 1.0 Zero this session's per-turn spend tally (Wave 0). Inert unless the
        # budgeter is wired + the flag is on; the budgeter also lazy-starts, so a
        # skipped call is harmless — this is the explicit per-turn reset.
        self._start_budget_turn(state.session_id)

        # 1a. Address-by-name (Stage 2 roster): a turn that opens with a persona
        # code-name ("GECKO, ..." / "ask VISION to ...") routes under that named
        # persona's least-privilege tool scope + memory namespace. Recorded on the
        # scratchpad up front; inert (no scratchpad keys) when no roster is wired
        # or the turn is un-addressed, so existing behaviour is unchanged.
        persona = self._match_persona(state.user_input)
        if persona is not None:
            self._apply_persona_scope(persona, state)

        # 2. Refuse out-of-scope asks before spending a model call.
        reason = self._refusal_reason(state.user_input)
        if reason is not None:
            state.mode = Mode.CONVERSATION
            state.response = self._decline(reason)
            self._record(state)
            return state

        # 2a. A confirmed follow-up commits any sensitive writes held pending from
        # a prior turn (the write-consent gate for sensitive data, §10).
        if state.confirmed and self._pending_writes.get(state.session_id):
            committed = self._commit_pending_writes(state)
            owner = get_settings().owner_address
            state.mode = Mode.CONVERSATION
            state.response = (
                f"Stored, {owner} — committed {committed} item(s) to memory as "
                f"you confirmed."
            )
            self._record(state)
            return state

        # 2b. A voice-protocol trigger (a stored ``trigger_phrase`` or "run the
        # <name> protocol") fires that protocol through the runner, threading the
        # turn's confirm flag. Inert unless protocols are enabled + wired.
        protocol = self._match_protocol(state.user_input)
        if protocol is not None:
            state.response = await self._run_protocol(protocol, state)
            self._record(state)
            return state

        # 2b-ii. An "n8n workflow <X>" request drafts (and maybe imports) a
        # workflow through the n8n service, threading the turn's confirm flag (a
        # docker auto-start is confirm-gated). Inert unless n8n is enabled + wired.
        if self._n8n_service is not None and get_settings().enable_n8n:
            n8n_description = _extract_n8n_description(state.user_input)
            if n8n_description is not None:
                state.response = await self._make_n8n_workflow(
                    n8n_description, state
                )
                self._record(state)
                return state

        # 2c. A "forget X" command removes everything stored about X (§10).
        forget_target = _extract_forget_target(state.user_input)
        if forget_target is not None:
            state.mode = Mode.CONVERSATION
            state.response = self._forget(forget_target)
            self._record(state)
            return state

        # 2d. A light "maps" intent deep-links to the local Maps globe. The
        # ``distance to <X>`` form is checked BEFORE the bare ``open maps`` so
        # "show me distance to ..." is never mistaken for a plain open. Answered
        # deterministically (no LLM call); the router has no MAPS mode.
        maps_destination = _extract_maps_distance(state.user_input)
        if maps_destination is not None:
            state.mode = Mode.CONVERSATION
            state.response = self._distance_maps_reply(maps_destination)
            self._record(state)
            return state
        if _is_open_maps(state.user_input):
            state.mode = Mode.CONVERSATION
            state.response = self._open_maps_reply()
            self._record(state)
            return state

        # 3. Route the turn (timed as the "route" span).
        with self._span("route"):
            decision = await route(state)
            state.route = decision
            state.mode = decision.mode

        # 4. Dispatch on mode (timed as the "dispatch" span). The mode handler
        # produces the response — including any LLM synthesis embedded in the
        # research / agent paths.
        with self._span("dispatch", mode=str(decision.mode)):
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

        # 4a. Optionally self-critique the final synthesized reply (Tier 2). Only
        # the LLM-synthesized modes are reviewed — CLARIFY is a deterministic
        # question, so it needs no model review. Inert unless the flag is on AND a
        # critic is wired; bounded to one pass; non-fatal (errors keep the reply).
        if state.mode is not Mode.CLARIFY:
            await self._maybe_critique(state)

        # 4b. Optionally stamp a calibrated confidence score (Wave 0) and append a
        # one-line caveat below threshold. Skipped for CLARIFY (no answer to score),
        # exactly like critique; inert unless the flag is on AND a scorer is wired.
        if state.mode is not Mode.CLARIFY:
            self._maybe_stamp_confidence(state)

        # 5. Persist the turn (timed as the "synth" span — the final assembly +
        # short-term-memory write that completes the synthesized reply).
        with self._span("synth"):
            self._record(state)
        # Optionally fold an over-long session into a summary + recent tail (Wave
        # 1). Runs after the turn is recorded so the just-finished turn is part of
        # what may be summarized; inert unless the flag is on and a Compactor wired.
        await self._maybe_compact(state)
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
            await self._maybe_critique(state)
            self._maybe_stamp_confidence(state)
        self._record(state)
        return state

    async def node_research(self, state: GraphState) -> GraphState:
        """RESEARCH node: minimal search-then-synthesize path."""
        if state.response is None:
            history = self._memory.history(state.session_id)
            state.response = await self._research(state, history)
            await self._maybe_critique(state)
            self._maybe_stamp_confidence(state)
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
            await self._maybe_critique(state)
            self._maybe_stamp_confidence(state)
        self._record(state)
        return state

    async def node_security_lockdown(self, state: GraphState) -> GraphState:
        """SECURITY_LOCKDOWN node: run the lockdown subgraph, report the audit."""
        if state.response is None:
            state.mode = Mode.SECURITY_LOCKDOWN
            state.response = await self._lockdown(state)
        self._record(state)
        return state
