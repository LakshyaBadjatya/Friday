"""FastAPI application factory and lifespan wiring (Task 1.9).

:func:`create_app` builds the FastAPI app and, on startup, assembles the runtime
graph of dependencies:

* :class:`~friday.config.Settings` (via the cached ``get_settings``);
* the :class:`~friday.providers.llm.LLMProvider` — the real
  :class:`~friday.providers.llm.NvidiaNIMProvider` when
  ``settings.llm_provider == "nvidia"`` (and a key is present), otherwise a
  :class:`~friday.providers.llm.FakeLLM` so the app boots and tests run with zero
  network;
* a :class:`~friday.tools.registry.ToolRegistry` with the keyless
  :class:`~friday.tools.web_search.WebSearchTool` registered;
* per-session :class:`~friday.memory.short_term.ShortTermMemory`;
* the :class:`~friday.core.orchestrator.Orchestrator`, stashed on ``app.state``
  for the ``/chat`` route to use.

Structured logging is configured at startup. The factory is import-safe: building
the app performs no network I/O (the FakeLLM default and lazy provider clients
keep ``create_app()`` side-effect-light), which is what the Phase-0 boot test
relies on.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from friday.agents.alerting import AlertingAgent
from friday.agents.analysis import AnalysisAgent
from friday.agents.automation import AutomationAgent
from friday.agents.base import AgentRegistry
from friday.agents.device import DeviceAgent
from friday.agents.knowledge import KnowledgeAgent
from friday.api.middleware import AuthMiddleware, RateLimitMiddleware
from friday.api.routes_admin import router as admin_router
from friday.api.routes_briefing import router as briefing_router
from friday.api.routes_chat import router as chat_router
from friday.api.routes_health import router as health_router
from friday.api.routes_protocols import router as protocols_router
from friday.api.routes_rag import router as rag_router
from friday.api.routes_reminders import router as reminders_router
from friday.api.routes_schedules import router as schedules_router
from friday.api.routes_studio import STATIC_DIR as STUDIO_STATIC_DIR
from friday.api.routes_studio import router as studio_router
from friday.api.routes_voice import router as voice_router
from friday.api.ws import router as ws_router
from friday.briefing.service import BriefingService
from friday.config import Settings, get_settings
from friday.core.critic import DEFAULT_PERSONA_MARKERS, SelfCritic
from friday.core.orchestrator import Orchestrator
from friday.logging import configure_logging, get_logger
from friday.memory.long_term import SQLiteLongTermStore
from friday.memory.short_term import ShortTermMemory
from friday.memory.vector import SQLiteVectorStore
from friday.observability.audit import AuditLog
from friday.observability.metrics import Metrics
from friday.observability.tracing import Tracer
from friday.protocols.runner import ProtocolRunner
from friday.protocols.store import SQLiteProtocolStore
from friday.providers.embeddings import (
    EmbeddingProvider,
    FakeEmbeddings,
    NvidiaEmbeddings,
)
from friday.providers.llm import (
    FakeLLM,
    FallbackLLM,
    GeminiProvider,
    LLMProvider,
    NvidiaNIMProvider,
)
from friday.providers.stt import FakeSTT, FasterWhisperSTT, STTProvider
from friday.providers.tts import FakeTTS, TTSProvider, make_tts
from friday.rag.ingest import DocumentIngestor
from friday.reminders.store import SQLiteReminderStore
from friday.scheduler.engine import Scheduler
from friday.scheduler.store import SQLiteTriggerStore, Trigger
from friday.studio.generator import (
    MeshyText3D,
    ProceduralGenerator,
    StudioService,
    Text3DProvider,
)
from friday.tools.base import Tool
from friday.tools.home import HomeControlTool
from friday.tools.notify import NotifyTool, SentMessage
from friday.tools.registry import ToolRegistry
from friday.tools.reminders import (
    CompleteReminderTool,
    CreateReminderTool,
    ListRemindersTool,
)
from friday.tools.web_search import WebSearchTool

logger = get_logger("friday.app")

# Persona spec ships alongside the package; resolve relative to this file so the
# path is correct regardless of the process working directory.
_PERSONA_PATH = Path(__file__).resolve().parent / "persona" / "friday.md"


def _build_gemini_fallback(settings: Settings) -> GeminiProvider | None:
    """Build the Gemini secondary provider, or ``None`` if not configured.

    Returns a :class:`GeminiProvider` only when ``llm_fallback_provider`` is
    ``"gemini"`` *and* a Gemini key is present; otherwise ``None`` so the caller
    keeps the single-provider behaviour. Lazy client construction means this
    performs no network I/O.
    """
    if settings.llm_fallback_provider == "gemini" and settings.gemini_api_key is not None:
        logger.info("using Gemini LLM fallback", extra={"model": settings.gemini_model})
        return GeminiProvider(
            api_key=settings.gemini_api_key.get_secret_value(),
            base_url=settings.gemini_base_url,
            model=settings.gemini_model,
            timeout=settings.llm_timeout_seconds,
        )
    return None


def _build_llm(settings: Settings) -> LLMProvider:
    """Select the LLM provider from settings.

    NVIDIA NIM when explicitly configured *and* a key is present; otherwise the
    scripted :class:`FakeLLM` (empty script) so the app always boots without
    credentials or network. When the live primary is selected *and* a Gemini
    fallback is configured (``llm_fallback_provider == "gemini"`` with a key),
    the primary is wrapped in a :class:`FallbackLLM` whose secondary is the
    :class:`GeminiProvider`. The ``fake`` path is untouched by fallback config.
    """
    if settings.llm_provider == "nvidia" and settings.nvidia_api_key is not None:
        logger.info("using NVIDIA NIM LLM provider", extra={"model": settings.nvidia_model})
        primary: LLMProvider = NvidiaNIMProvider(
            api_key=settings.nvidia_api_key.get_secret_value(),
            base_url=settings.nvidia_base_url,
            model=settings.nvidia_model,
            timeout=settings.llm_timeout_seconds,
        )
        secondary = _build_gemini_fallback(settings)
        if secondary is not None:
            return FallbackLLM(primary=primary, secondary=secondary)
        return primary
    logger.info("using FakeLLM provider (no network)")
    return FakeLLM(responses=[])


def _build_embedder(settings: Settings) -> EmbeddingProvider:
    """Select the embedding provider from settings.

    Defaults to the deterministic, offline :class:`FakeEmbeddings` (tests / no
    credentials) so the app always boots without a key or network. The real
    :class:`NvidiaEmbeddings` adapter is chosen only when ``embedding_provider``
    is ``nvidia`` *and* a NVIDIA key is configured; it lazy-imports ``openai``
    inside ``providers/`` and is never required for the gate.
    """
    if (
        settings.embedding_provider == "nvidia"
        and settings.nvidia_api_key is not None
        and settings.embedding_model
    ):
        logger.info(
            "using NVIDIA NIM embeddings", extra={"model": settings.embedding_model}
        )
        return NvidiaEmbeddings(
            api_key=settings.nvidia_api_key.get_secret_value(),
            base_url=settings.nvidia_base_url,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            timeout=settings.llm_timeout_seconds,
        )
    logger.info("using FakeEmbeddings provider (deterministic, no network)")
    return FakeEmbeddings(dim=settings.embedding_dim)


def _build_long_term(settings: Settings) -> SQLiteLongTermStore:
    """Build the local-first SQLite long-term store, ensuring its dir exists.

    ``data/`` is gitignored; a missing parent directory is created so a fresh
    checkout boots. ``":memory:"`` paths are left untouched (no directory).
    """
    path = settings.memory_db_path
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SQLiteLongTermStore(path)


def _build_vector(
    settings: Settings, embedder: EmbeddingProvider
) -> SQLiteVectorStore:
    """Build the persistent SQLite vector store alongside the long-term DB.

    The vector store lives in a sibling file (``*.vec.db``) next to the
    long-term database so both share the ``data/`` directory; ``":memory:"``
    stays ephemeral. Sized to ``settings.embedding_dim`` to match the embedder.
    """
    db_path = settings.memory_db_path
    if db_path == ":memory:":
        vec_path = ":memory:"
    else:
        base = Path(db_path)
        vec_path = str(base.with_suffix(base.suffix + ".vec"))
        base.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteVectorStore(vec_path, embedder=embedder, dim=settings.embedding_dim)


def _build_reminder_store(settings: Settings) -> SQLiteReminderStore:
    """Build the local-first SQLite reminder store alongside the long-term DB.

    Reuses ``memory_db_path``: the reminders live in a sibling file (``*.rem.db``)
    next to the long-term database so they share the gitignored ``data/``
    directory; ``":memory:"`` stays ephemeral. The store's clock is left at its
    wall-clock default (only ``created_at`` uses it); tested paths inject their
    own clock and ``due()`` is driven by a passed timestamp, never the wall clock.
    """
    db_path = settings.memory_db_path
    if db_path == ":memory:":
        rem_path = ":memory:"
    else:
        base = Path(db_path)
        rem_path = str(base.with_suffix(base.suffix + ".rem"))
        base.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteReminderStore(rem_path)


def _build_trigger_store(settings: Settings) -> SQLiteTriggerStore:
    """Build the local-first SQLite trigger store alongside the long-term DB.

    Reuses ``memory_db_path``: the triggers live in a sibling file (``*.sched.db``)
    next to the long-term database so they share the gitignored ``data/``
    directory; ``":memory:"`` stays ephemeral. ``due()`` is driven by a passed
    ``now`` datetime, never the wall clock.
    """
    db_path = settings.memory_db_path
    if db_path == ":memory:":
        sched_path = ":memory:"
    else:
        base = Path(db_path)
        sched_path = str(base.with_suffix(base.suffix + ".sched"))
        base.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteTriggerStore(sched_path)


def _build_protocol_store(settings: Settings) -> SQLiteProtocolStore:
    """Build the local-first SQLite protocol store alongside the long-term DB.

    Reuses ``memory_db_path``: the protocols live in a sibling file (``*.proto.db``)
    next to the long-term database so they share the gitignored ``data/``
    directory; ``":memory:"`` stays ephemeral. Steps are persisted as JSON in the
    store; the store is connection-per-call for file paths (thread-safe).
    """
    db_path = settings.memory_db_path
    if db_path == ":memory:":
        proto_path = ":memory:"
    else:
        base = Path(db_path)
        proto_path = str(base.with_suffix(base.suffix + ".proto"))
        base.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteProtocolStore(proto_path)


def _registered_tool_names(registry: ToolRegistry) -> frozenset[str]:
    """The names of every tool registered in ``registry``.

    Used to seed the protocol runner's allow-list so a protocol may invoke only
    tools that already exist in the shared registry. The names are probed through
    the registry's public ``get`` against the same set ``_build_registry``
    registers, so this never reaches into registry internals and stays correct as
    that builder evolves: a name that is not actually registered is dropped.
    """
    candidates = (
        "web_search",
        "notify",
        "home",
        "create_reminder",
        "list_reminders",
        "complete_reminder",
    )
    present: set[str] = set()
    for name in candidates:
        try:
            registry.get(name)
        except KeyError:  # pragma: no cover - all candidates are registered
            continue
        present.add(name)
    return frozenset(present)


def _build_protocol_runner(registry: ToolRegistry) -> ProtocolRunner:
    """Assemble the :class:`ProtocolRunner` over the shared tool registry.

    The runner's ``allowed_tools`` is the set of *registered* tool names, so a
    protocol may invoke only tools that already exist in the shared registry (no
    arbitrary execution) and every step passes the registry's permission /
    validation / confirm-step gates.
    """
    return ProtocolRunner(registry, _registered_tool_names(registry))


def _build_critic(llm: LLMProvider) -> SelfCritic:
    """Build the :class:`SelfCritic` over the live LLM + persona banned markers.

    The critic reviews the final persona reply once before it is returned. It is
    always constructed and injected (so the wiring is uniform), but the
    orchestrator only *invokes* it when ``FRIDAY_ENABLE_SELF_CRITIQUE`` is on — so
    the off-by-default build pays no extra LLM call. The banned-tone markers come
    from the persona spec (:data:`DEFAULT_PERSONA_MARKERS`), the same list the
    deterministic scan flags. Construction performs no I/O.
    """
    return SelfCritic(llm, persona_markers=list(DEFAULT_PERSONA_MARKERS))


def _make_due_reminders_action(
    reminder_store: SQLiteReminderStore, notify: NotifyTool
) -> Callable[[Trigger], Awaitable[None]]:
    """Build the ``due_reminders`` scheduler action over the shared stores.

    Reads the reminders due as of ``utcnow`` from the *shared*
    :class:`SQLiteReminderStore` (the same one the reminder tools/routes write to)
    and emits each via the :class:`NotifyTool` fake sink + a log line, so an
    enabled trigger surfaces overdue reminders proactively. ``utcnow`` is read
    here in the background action only — the tested ``tick(now)`` unit stays
    clock-injected; this action is exercised through the run-now route, not for
    timing.
    """

    async def _due_reminders(trigger: Trigger) -> None:
        now_iso = datetime.now(UTC).isoformat()
        due = reminder_store.due(now_iso)
        for reminder in due:
            notify.sink.append(
                SentMessage(
                    channel="webhook",
                    target="scheduler",
                    subject="Reminder due",
                    body=reminder.text,
                )
            )
        logger.info(
            "scheduler due_reminders fired",
            extra={"trigger": trigger.name, "count": len(due)},
        )

    return _due_reminders


async def _noop_action(trigger: Trigger) -> None:
    """A placeholder scheduler action that does nothing but log (e.g. briefing).

    A registered no-op so a trigger can be created/exercised end-to-end before a
    real briefing action is wired; it is also the default action the API tests
    use to validate the CRUD + run-now surface without side effects.
    """
    logger.info("scheduler noop fired", extra={"trigger": trigger.name})


def _build_briefing_service(
    settings: Settings,
    reminder_store: SQLiteReminderStore,
    audit: AuditLog,
    metrics: Metrics,
    llm: LLMProvider,
) -> BriefingService:
    """Assemble the :class:`BriefingService` over the shared local stores.

    Pure assembly from the *shared* runtime pieces — the same reminder store the
    reminder tools/routes use, the process-wide audit log + metrics — plus the
    live LLM for an optional, non-fatal natural-language summary. The greeting
    addresses ``owner_address`` and the recent-activity section is sized to
    ``briefing_recent_activity``. Constructing the service performs no I/O.
    """
    return BriefingService(
        reminder_store,
        audit_log=audit,
        metrics=metrics,
        llm=llm,
        owner_address=settings.owner_address,
        recent_activity_limit=settings.briefing_recent_activity,
    )


def _make_briefing_action(
    briefing_service: BriefingService, notify: NotifyTool
) -> Callable[[Trigger], Awaitable[None]]:
    """Build the ``briefing`` scheduler action over the shared briefing service.

    Builds the briefing for ``utcnow`` from the *shared* :class:`BriefingService`
    and emits it via the :class:`NotifyTool` fake sink (greeting + each section
    line) + a log line, so an enabled trigger surfaces a morning/EOD briefing
    proactively. ``utcnow`` is read here in the background action only — the
    tested ``BriefingService.build(now)`` unit stays clock-injected; this action
    is exercised through the run-now route, not for timing. Any LLM error is
    already swallowed inside ``build`` (structured-only fallback), so this never
    raises.
    """

    async def _briefing(trigger: Trigger) -> None:
        briefing = await briefing_service.build(datetime.now(UTC))
        body_lines = [briefing.greeting]
        for section in briefing.sections:
            body_lines.append(f"{section.title}:")
            body_lines.extend(f"- {item}" for item in section.items)
        notify.sink.append(
            SentMessage(
                channel="webhook",
                target="scheduler",
                subject="Briefing",
                body="\n".join(body_lines),
            )
        )
        logger.info(
            "scheduler briefing fired",
            extra={"trigger": trigger.name, "sections": len(briefing.sections)},
        )

    return _briefing


def _build_scheduler(
    trigger_store: SQLiteTriggerStore,
    reminder_store: SQLiteReminderStore,
    registry: ToolRegistry,
    briefing_service: BriefingService,
) -> Scheduler:
    """Assemble the :class:`Scheduler` and register the default actions.

    Registers ``"due_reminders"`` (emit due reminders via the shared notify tool's
    sink/log), ``"briefing"`` (build + emit the proactive briefing via the same
    sink), and a ``"noop"`` placeholder. The notify tool is pulled from the shared
    :class:`ToolRegistry` so the scheduler emits into the *same* sink the alerting
    agent uses (one auditable place for everything that would have been sent); a
    fresh :class:`NotifyTool` is used as a fallback if absent.
    """
    scheduler = Scheduler(trigger_store)
    try:
        notify = registry.get("notify")
    except KeyError:  # pragma: no cover - notify is always registered
        notify = None
    notify_tool = notify if isinstance(notify, NotifyTool) else NotifyTool()
    scheduler.register_action(
        "due_reminders", _make_due_reminders_action(reminder_store, notify_tool)
    )
    scheduler.register_action(
        "briefing", _make_briefing_action(briefing_service, notify_tool)
    )
    scheduler.register_action("noop", _noop_action)
    return scheduler


def _build_registry(
    reminder_store: SQLiteReminderStore,
    audit: AuditLog | None = None,
    metrics: Metrics | None = None,
) -> ToolRegistry:
    """Build the tool registry with every Phase-2 tool plus the reminder tools.

    Registers the keyless web search tool plus the side-effecting notify and home
    tools (their own flag/allow-list/confirm gates keep them safe), and the three
    Tier-1 reminder tools (``create_reminder``/``list_reminders``/
    ``complete_reminder``) backed by ``reminder_store``. The reminder tools write
    local personal data only, so they are non-side-effecting and skip the
    confirm-step. Each tool satisfies the ``Tool`` protocol structurally, but a
    concrete ``args_model`` (``type[SomeArgs]``) trips the protocol's invariant
    ``type[BaseModel]`` field under nominal checking; cast to the protocol to
    register.

    When an :class:`~friday.observability.audit.AuditLog` / :class:`Metrics` are
    passed (the app wires the process-wide instances), every ``execute`` records a
    redacted tool-call audit row and bumps the ``tool_calls`` counter (build-spec
    §11). They default to ``None`` so a bare registry behaves exactly as before.
    """
    registry = ToolRegistry(audit=audit, metrics=metrics)
    registry.register(cast(Tool, WebSearchTool()))
    registry.register(cast(Tool, NotifyTool()))
    registry.register(cast(Tool, HomeControlTool()))
    registry.register(cast(Tool, CreateReminderTool(reminder_store)))
    registry.register(cast(Tool, ListRemindersTool(reminder_store)))
    registry.register(cast(Tool, CompleteReminderTool(reminder_store)))
    return registry


def _build_agents(
    settings: Settings,
    registry: ToolRegistry,
    llm: LLMProvider,
    vector: SQLiteVectorStore,
    long_term: SQLiteLongTermStore,
) -> AgentRegistry:
    """Construct each specialist agent with its dependencies and register it.

    * ``analysis`` — evidence-grounded synthesis over the web-search tool + LLM.
    * ``knowledge`` — hybrid grounded retrieval over the persistent SQLite vector
      store + recent long-term facts, citing each chunk's ``source_id``.
    * ``automation`` — bounded multi-step job executor; also reaches the Tier-1
      reminder tools through the shared registry on reminder-shaped requests.
    * ``device`` — confirm-gated, allow-listed home control via the ``home`` tool.
    * ``alerting`` — deduped/rate-limited notifications via the ``notify`` tool;
      "now" comes from the injected wall clock (the agent windows on it).
    """
    agents = AgentRegistry()
    agents.register(AnalysisAgent(registry, llm=llm))
    agents.register(
        KnowledgeAgent(
            store=vector, memory=ShortTermMemory(), long_term=long_term
        )
    )
    agents.register(AutomationAgent(registry=registry))
    agents.register(DeviceAgent(registry))
    agents.register(
        AlertingAgent(registry, clock=time.monotonic, settings=settings)
    )
    return agents


@dataclass
class AppRuntime:
    """The shared runtime graph assembled at startup.

    Bundles the orchestrator with the long-lived pieces the ``/admin`` routes read
    back (build-spec §11): the process-wide observability stores
    (:class:`Tracer` / :class:`AuditLog` / :class:`Metrics`), the shared tool
    :class:`~friday.tools.registry.ToolRegistry`, the shared per-session
    :class:`~friday.memory.short_term.ShortTermMemory`, the durable
    :class:`~friday.memory.long_term.SQLiteLongTermStore`, and the mutable
    runtime feature-flag override holder. ``create_app`` stashes each of these on
    ``app.state`` so the admin views and the orchestrator emit/observe the *same*
    instances.
    """

    orchestrator: Orchestrator
    tracer: Tracer
    audit: AuditLog
    metrics: Metrics
    registry: ToolRegistry
    short_term: ShortTermMemory
    long_term: SQLiteLongTermStore
    flag_overrides: dict[str, bool]
    #: The live LLM provider, shared with the studio's procedural generator so
    #: the (free) 3D scene generation uses the same provider as the chat loop.
    llm: LLMProvider
    #: The shared persistent vector store — the same one the Knowledge agent
    #: retrieves from, reused by personal RAG so ingested docs are answerable.
    vector: SQLiteVectorStore
    #: The shared reminder store — the same one the reminder tools and the
    #: ``/reminders`` routes read/write, so an agent-created and an HTTP-created
    #: reminder land in the same place.
    reminder_store: SQLiteReminderStore
    #: The shared scheduled-trigger store — the same one the ``/schedules`` routes
    #: and the background tick loop read/write.
    trigger_store: SQLiteTriggerStore
    #: The scheduler (action registry + ``tick``) over ``trigger_store``; its
    #: registered ``due_reminders``/``briefing`` actions reuse the shared stores.
    scheduler: Scheduler
    #: The shared briefing service — assembles the digest from the same reminder
    #: store + audit log + metrics the rest of the runtime uses; the ``/briefing``
    #: route and the scheduler ``briefing`` action both build through it.
    briefing: BriefingService
    #: The shared protocol store — the same one the ``/protocols`` routes and the
    #: orchestrator's trigger-phrase hook read/write.
    protocol_store: SQLiteProtocolStore
    #: The protocol runner over the shared registry; its ``allowed_tools`` is the
    #: set of registered tool names, so a protocol runs only registered tools.
    protocol_runner: ProtocolRunner


def build_runtime(settings: Settings) -> AppRuntime:
    """Assemble the full runtime graph from ``settings``.

    Constructs the process-wide observability stores first, injects the audit +
    metrics into the shared tool registry (so every tool call is audited and
    counted) and the tracer + metrics into the orchestrator (so every turn opens a
    trace and bumps the request/by-mode counters), and returns everything bundled
    in an :class:`AppRuntime` so the admin API can read the same instances.
    """
    tracer = Tracer()
    audit = AuditLog()
    metrics = Metrics()
    flag_overrides: dict[str, bool] = {}

    reminder_store = _build_reminder_store(settings)
    trigger_store = _build_trigger_store(settings)
    registry = _build_registry(reminder_store, audit=audit, metrics=metrics)
    llm = _build_llm(settings)
    embedder = _build_embedder(settings)
    long_term = _build_long_term(settings)
    vector = _build_vector(settings, embedder)
    agents = _build_agents(settings, registry, llm, vector, long_term)
    briefing = _build_briefing_service(
        settings, reminder_store, audit, metrics, llm
    )
    scheduler = _build_scheduler(
        trigger_store, reminder_store, registry, briefing
    )
    protocol_store = _build_protocol_store(settings)
    protocol_runner = _build_protocol_runner(registry)
    critic = _build_critic(llm)
    short_term = ShortTermMemory()
    orchestrator = Orchestrator(
        llm=llm,
        registry=registry,
        memory=short_term,
        persona_path=_PERSONA_PATH,
        agents=agents,
        long_term=long_term,
        vector=vector,
        tracer=tracer,
        metrics=metrics,
        protocol_store=protocol_store,
        protocol_runner=protocol_runner,
        critic=critic,
    )
    return AppRuntime(
        orchestrator=orchestrator,
        tracer=tracer,
        audit=audit,
        metrics=metrics,
        registry=registry,
        short_term=short_term,
        long_term=long_term,
        flag_overrides=flag_overrides,
        llm=llm,
        vector=vector,
        reminder_store=reminder_store,
        trigger_store=trigger_store,
        scheduler=scheduler,
        briefing=briefing,
        protocol_store=protocol_store,
        protocol_runner=protocol_runner,
    )


def build_orchestrator(settings: Settings) -> Orchestrator:
    """Assemble a fully-wired :class:`Orchestrator` from ``settings``.

    Thin wrapper over :func:`build_runtime` for callers that only need the
    orchestrator (the observability stores are still constructed and injected, but
    not surfaced). The shared tool registry, LLM provider, local-first SQLite
    long-term + persistent vector stores (with the configured embedder), and the
    populated agent registry are wired so AUTOMATION / DEVICE_CONTROL / ALERTING
    turns dispatch to their specialist agents and SECURITY_LOCKDOWN runs the
    lockdown subgraph.
    """
    return build_runtime(settings).orchestrator


def _build_voice_stt(settings: Settings) -> STTProvider:
    """Select the STT provider for voice turns.

    When the LLM provider is the offline ``fake`` (tests / no credentials) we use
    :class:`FakeSTT` so the app boots and exercises with zero models/network.
    Otherwise the real :class:`FasterWhisperSTT` is chosen; it lazy-imports
    ``faster-whisper`` and only fails at transcription time if the optional voice
    extras are not installed (``make install-voice``).
    """
    if settings.llm_provider == "fake":
        return FakeSTT()
    return FasterWhisperSTT()


def _build_voice_tts(settings: Settings) -> TTSProvider:
    """Select the TTS provider for voice turns.

    Offline (``fake`` LLM) defaults to :class:`FakeTTS`; otherwise the
    config-selected real adapter (``FRIDAY_TTS_PROVIDER``) via :func:`make_tts`.
    The real adapters lazy-load their heavy deps, so this stays import-light.
    """
    if settings.llm_provider == "fake":
        return FakeTTS()
    return make_tts(settings)


def _wire_voice(app: FastAPI, settings: Settings) -> None:
    """Stash the voice STT/TTS adapters on ``app.state`` when voice is enabled.

    The ``/voice`` route reads these off ``app.state`` (falling back to fakes),
    so wiring them only when ``enable_voice`` is set keeps the offline default
    untouched and never constructs a heavy adapter unless asked for.
    """
    if not settings.enable_voice:
        return
    app.state.voice_stt = _build_voice_stt(settings)
    app.state.voice_tts = _build_voice_tts(settings)


def _build_studio_hifi(settings: Settings) -> Text3DProvider | None:
    """Build the optional high-fidelity text-to-3D provider, or ``None``.

    Returns a lazy, keyless-safe :class:`MeshyText3D` only when
    ``studio_hifi_provider == "meshy"`` *and* a Meshy key is present; otherwise
    ``None`` so the studio uses the free procedural-only path. Construction
    performs no network I/O (the ``httpx`` client is lazy in ``generate_mesh``).
    """
    if settings.studio_hifi_provider == "meshy" and settings.meshy_api_key is not None:
        logger.info("using Meshy hi-fi text-to-3D provider")
        return MeshyText3D(
            api_key=settings.meshy_api_key.get_secret_value(),
            model=settings.studio_hifi_model,
            timeout=settings.llm_timeout_seconds,
        )
    return None


def _build_studio_service(settings: Settings, llm: LLMProvider) -> StudioService:
    """Assemble the :class:`StudioService`: procedural (live LLM) + optional hi-fi.

    The procedural generator drives the *same* live LLM as the chat loop (free by
    default), so 3D scene generation needs no extra credentials. The hi-fi adapter
    is wired only when configured + keyed; otherwise the service falls back to
    procedural at request time (never paywalls the user).
    """
    return StudioService(ProceduralGenerator(llm), hifi=_build_studio_hifi(settings))


def _wire_studio(app: FastAPI, settings: Settings, llm: LLMProvider) -> None:
    """Stash the :class:`StudioService` on ``app.state`` when studio is enabled.

    The ``/studio`` route reads ``app.state.studio``; building it only when
    ``enable_studio`` is set keeps the offline default untouched and constructs no
    hi-fi adapter unless asked for. The StaticFiles mount is added separately in
    :func:`create_app` (a mount cannot be added per-request).
    """
    if not settings.enable_studio:
        return
    app.state.studio = _build_studio_service(settings, llm)


def _wire_rag(app: FastAPI, settings: Settings, runtime: AppRuntime) -> None:
    """Stash a :class:`DocumentIngestor` on ``app.state`` when RAG is enabled.

    The ingestor reuses the *shared* runtime stores — the same persistent vector
    store the Knowledge agent retrieves from and the same long-term store — so an
    ingested document is immediately answerable via the existing knowledge path
    with citations. Building it only when ``enable_rag`` is set keeps the offline
    default untouched (the ``/rag`` routes self-guard on the flag and 404 when
    off). No retrieval logic is duplicated; this only adds the write/forget seam.
    """
    if not settings.enable_rag:
        return
    app.state.rag_ingestor = DocumentIngestor(
        runtime.vector,
        runtime.long_term,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )


def _wire_reminders(app: FastAPI, settings: Settings, runtime: AppRuntime) -> None:
    """Stash the shared reminder store on ``app.state`` when reminders are enabled.

    The ``/reminders`` routes read ``app.state.reminder_store`` — the *same*
    store the registered reminder tools (and so the Automation agent) write to —
    so an HTTP-created reminder and an agent-created one share state. Building it
    only when ``enable_reminders`` is set keeps the offline default untouched (the
    routes self-guard on the flag and 404 when off).
    """
    if not settings.enable_reminders:
        return
    app.state.reminder_store = runtime.reminder_store


def _wire_scheduler(app: FastAPI, settings: Settings, runtime: AppRuntime) -> None:
    """Stash the shared trigger store + scheduler on ``app.state`` when enabled.

    The ``/schedules`` routes read ``app.state.trigger_store`` and
    ``app.state.scheduler`` — the *same* store the background tick loop drives and
    the *same* scheduler whose ``due_reminders``/``noop`` actions are registered —
    so an HTTP-created trigger and the loop operate on one store with one action
    registry. Building it only when ``enable_scheduler`` is set keeps the offline
    default untouched (the routes self-guard on the flag and 404 when off, and the
    background loop is started only in the lifespan when enabled).
    """
    if not settings.enable_scheduler:
        return
    app.state.trigger_store = runtime.trigger_store
    app.state.scheduler = runtime.scheduler


def _wire_briefing(app: FastAPI, settings: Settings, runtime: AppRuntime) -> None:
    """Stash the shared briefing service on ``app.state`` when briefing is enabled.

    The ``/briefing`` route reads ``app.state.briefing`` — the *same*
    :class:`BriefingService` the scheduler ``briefing`` action builds through —
    so the on-demand HTTP briefing and the proactive scheduled one assemble from
    one set of local stores. Building it only when ``enable_briefing`` is set
    keeps the offline default untouched (the route self-guards on the flag and
    404s when off).
    """
    if not settings.enable_briefing:
        return
    app.state.briefing = runtime.briefing


def _wire_protocols(app: FastAPI, settings: Settings, runtime: AppRuntime) -> None:
    """Stash the shared protocol store + runner on ``app.state`` when enabled.

    The ``/protocols`` routes read ``app.state.protocol_store`` and
    ``app.state.protocol_runner`` — the *same* store the orchestrator's
    trigger-phrase hook reads and the *same* runner (over the shared registry)
    both fire through — so an HTTP-created protocol and a spoken trigger operate on
    one store with one runner. Building it only when ``enable_protocols`` is set
    keeps the offline default untouched (the routes self-guard on the flag and 404
    when off).
    """
    if not settings.enable_protocols:
        return
    app.state.protocol_store = runtime.protocol_store
    app.state.protocol_runner = runtime.protocol_runner


def _install_runtime(app: FastAPI, settings: Settings) -> None:
    """Assemble the runtime graph and stash every shared piece on ``app.state``.

    The orchestrator, the process-wide observability stores (tracer / audit /
    metrics), the shared tool registry, the short-term + long-term memory, and the
    mutable runtime flag-override holder all land on ``app.state`` so the
    ``/admin`` routes read back the exact instances the turn loop emits into.
    """
    runtime = build_runtime(settings)
    app.state.settings = settings
    app.state.orchestrator = runtime.orchestrator
    app.state.tracer = runtime.tracer
    app.state.audit = runtime.audit
    app.state.metrics = runtime.metrics
    app.state.registry = runtime.registry
    app.state.short_term = runtime.short_term
    app.state.long_term = runtime.long_term
    app.state.flag_overrides = runtime.flag_overrides
    _wire_studio(app, settings, runtime.llm)
    _wire_rag(app, settings, runtime)
    _wire_reminders(app, settings, runtime)
    _wire_scheduler(app, settings, runtime)
    _wire_briefing(app, settings, runtime)
    _wire_protocols(app, settings, runtime)


def _start_scheduler_loop(
    app: FastAPI, settings: Settings
) -> asyncio.Task[None] | None:
    """Start the background scheduler ``run_loop`` as a task when enabled.

    Returns the created :class:`asyncio.Task` (so the lifespan can cancel it on
    shutdown) or ``None`` when the scheduler is disabled — keeping the offline
    default free of any background work. The loop is a thin wrapper over the
    tested ``tick(now)``; its cadence is ``scheduler_tick_seconds``.
    """
    if not settings.enable_scheduler:
        return None
    scheduler = getattr(app.state, "scheduler", None)
    if not isinstance(scheduler, Scheduler):  # pragma: no cover - startup guard
        return None
    logger.info(
        "starting scheduler loop",
        extra={"tick_seconds": settings.scheduler_tick_seconds},
    )
    return asyncio.create_task(
        scheduler.run_loop(settings.scheduler_tick_seconds)
    )


async def _stop_scheduler_loop(task: asyncio.Task[None] | None) -> None:
    """Cancel the background scheduler task cleanly (no-op when ``None``)."""
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def create_app() -> FastAPI:
    """Construct the FRIDAY FastAPI application.

    The runtime graph (orchestrator + observability stores) is built in the
    lifespan startup so configuration is read once per process and the dependency
    graph is shared across requests.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        configure_logging(json_logs=settings.log_json, level=settings.log_level)
        logger.info("FRIDAY starting up", extra={"llm_provider": settings.llm_provider})
        _install_runtime(app, settings)
        _wire_voice(app, settings)
        scheduler_task = _start_scheduler_loop(app, settings)
        try:
            yield
        finally:
            await _stop_scheduler_loop(scheduler_task)
            logger.info("FRIDAY shutting down")

    app = FastAPI(title="FRIDAY", version="0.1.0", lifespan=lifespan)

    # Gateway hardening (Phase 6). ``add_middleware`` is LIFO, so the *last*
    # registered runs *outermost* (first per request). We add rate-limit first
    # and auth second, so the request order is: AuthMiddleware -> then
    # RateLimitMiddleware -> then the route. Rationale: reject an unauthenticated
    # caller (401) before it can consume a rate-limit slot, so a bad token can't
    # exhaust a client's budget. Both gates self-exempt ``/health`` and self-
    # disable via settings (``require_auth`` / ``rate_limit_enabled``), keeping
    # the local-first default open. The rate-limit clock is injectable via
    # ``app.state.rate_limit_clock`` (default ``time.monotonic``) so rate-limit
    # tests advance "now" deterministically without the wall clock.
    gateway_settings = get_settings()
    app.state.rate_limit_clock = time.monotonic
    app.add_middleware(RateLimitMiddleware, settings=gateway_settings)
    app.add_middleware(AuthMiddleware, settings=gateway_settings)

    app.include_router(chat_router)
    app.include_router(health_router)
    # Voice endpoints are always registered but self-guard on FRIDAY_ENABLE_VOICE
    # (404 / socket refusal when off), so the offline default exposes no voice UX.
    app.include_router(voice_router)
    app.include_router(ws_router)
    # The admin/observability control plane (Phase 5, Stage 2A).
    app.include_router(admin_router)
    # The 3D Studio (Phase 7) — always registered but self-guards on
    # FRIDAY_ENABLE_STUDIO (404 when off), so the offline default exposes no
    # studio surface. The StaticFiles mount is added below only when enabled.
    app.include_router(studio_router)
    # Personal RAG (Tier 1) — always registered but self-guards on
    # FRIDAY_ENABLE_RAG (404 when off), so the offline default exposes no RAG
    # surface. The DocumentIngestor is wired onto app.state only when enabled.
    app.include_router(rag_router)
    # Reminders & tasks (Tier 1) — always registered but self-guards on
    # FRIDAY_ENABLE_REMINDERS (404 when off), so the offline default exposes no
    # reminder surface. The shared reminder store is wired onto app.state only
    # when enabled.
    app.include_router(reminders_router)
    # Scheduled triggers (Tier 1) — always registered but self-guards on
    # FRIDAY_ENABLE_SCHEDULER (404 when off), so the offline default exposes no
    # scheduler surface. The shared trigger store + scheduler are wired onto
    # app.state only when enabled, and the background tick loop starts only in the
    # lifespan when enabled.
    app.include_router(schedules_router)
    # Proactive briefing (Tier 1) — always registered but self-guards on
    # FRIDAY_ENABLE_BRIEFING (404 when off), so the offline default exposes no
    # briefing surface. The shared briefing service is wired onto app.state only
    # when enabled.
    app.include_router(briefing_router)
    # Voice protocols (Tier 1) — always registered but self-guards on
    # FRIDAY_ENABLE_PROTOCOLS (404 when off), so the offline default exposes no
    # protocol surface. The shared protocol store + runner are wired onto
    # app.state only when enabled.
    app.include_router(protocols_router)

    # Build the runtime eagerly too so a TestClient that does not trigger the
    # lifespan (or any direct create_app() user) still has a working app. The
    # lifespan rebuild is harmless and keeps per-process config fresh.
    settings = get_settings()
    _install_runtime(app, settings)
    _wire_voice(app, settings)
    _mount_studio_static(app, settings)

    return app


def _mount_studio_static(app: FastAPI, settings: Settings) -> None:
    """Mount the studio static assets at ``/studio/static`` when enabled + present.

    The frontend assets (``src/friday/studio/static``) are produced by the Stage-2
    agent in parallel; the mount is added only when the studio flag is on *and* the
    directory exists, so the backend-only build (no frontend files yet) still boots
    and its tests pass. The route ``GET /studio`` independently serves
    ``index.html`` via a FileResponse.
    """
    if not settings.enable_studio:
        return
    if not STUDIO_STATIC_DIR.is_dir():
        logger.info(
            "studio enabled but static dir missing; skipping StaticFiles mount",
            extra={"static_dir": str(STUDIO_STATIC_DIR)},
        )
        return
    app.mount(
        "/studio/static",
        StaticFiles(directory=STUDIO_STATIC_DIR),
        name="studio-static",
    )
