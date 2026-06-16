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
from friday.api.routes_comms import router as comms_router
from friday.api.routes_email import router as email_router
from friday.api.routes_ensemble import router as ensemble_router
from friday.api.routes_graph import router as graph_router
from friday.api.routes_health import router as health_router
from friday.api.routes_journal import router as journal_router
from friday.api.routes_meetings import router as meetings_router
from friday.api.routes_memory import router as memory_router
from friday.api.routes_models import router as models_router
from friday.api.routes_n8n import router as n8n_router
from friday.api.routes_perception import router as perception_router
from friday.api.routes_planner import router as planner_router
from friday.api.routes_plugins import router as plugins_router
from friday.api.routes_protocols import router as protocols_router
from friday.api.routes_rag import router as rag_router
from friday.api.routes_reminders import router as reminders_router
from friday.api.routes_roster import router as roster_router
from friday.api.routes_schedules import router as schedules_router
from friday.api.routes_studio import STATIC_DIR as STUDIO_STATIC_DIR
from friday.api.routes_studio import router as studio_router
from friday.api.routes_study import router as study_router
from friday.api.routes_system import router as system_router
from friday.api.routes_voice import router as voice_router
from friday.api.ws import router as ws_router
from friday.briefing.service import BriefingService
from friday.broker import Broker, HashChainedAudit
from friday.config import Settings, get_settings
from friday.core.confidence import ConfidenceScorer
from friday.core.critic import DEFAULT_PERSONA_MARKERS, SelfCritic
from friday.core.ensemble import Ensemble
from friday.core.orchestrator import ForgettableVectorStore, Orchestrator
from friday.core.planner import Planner
from friday.desktop import AuditedDesktop, DesktopAction, FakeDesktop
from friday.errors import ProviderError
from friday.family import router as family_router
from friday.graph.extractor import EntityExtractor
from friday.graph.store import SQLiteGraphStore
from friday.hud import router as hud_router
from friday.integrations import calendar_router
from friday.journal.service import JournalService
from friday.journal.store import SQLiteJournalStore
from friday.logging import configure_logging, get_logger
from friday.maps import router as maps_router
from friday.market import router as market_router
from friday.meetings.capture import MeetingCapture
from friday.meetings.store import SQLiteMeetingStore
from friday.memory.compaction import Compactor
from friday.memory.contradiction import ContradictionDetector
from friday.memory.long_term import LongTermStore, SQLiteLongTermStore
from friday.memory.pg import PgVectorStore, PostgresLongTermStore
from friday.memory.short_term import ShortTermMemory
from friday.memory.vector import SQLiteVectorStore
from friday.models.budget import Budgeter
from friday.models.catalog import ModelCatalog
from friday.models.gateway import ModelGateway
from friday.n8n.client import N8nClient
from friday.n8n.drafter import WorkflowDrafter
from friday.n8n.service import N8nService
from friday.observability.audit import AuditLog
from friday.observability.metrics import Metrics
from friday.observability.tracing import Tracer
from friday.perception.clipboard import FakeClipboard
from friday.perception.ocr import FakeOCR
from friday.perception.screen import (
    FakeScreen,
    PerceptionService,
)
from friday.perception.vision import FakeVision
from friday.plugins.loader import PluginInfo, load_into
from friday.presence import router as presence_router
from friday.proactive import AnomalyDetector, Foresight
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
    OpenCodeProvider,
    OpenRouterProvider,
)
from friday.providers.offline import select_llm
from friday.providers.redacting import RedactingLLM
from friday.providers.stt import FakeSTT, FasterWhisperSTT, STTProvider
from friday.providers.tts import FakeTTS, TTSProvider, make_tts
from friday.pwa import router as pwa_router
from friday.rag.ingest import DocumentIngestor
from friday.reminders.store import SQLiteReminderStore
from friday.roster import ROSTER, RosterRegistry
from friday.roster.custom import merge_personas, parse_custom_operators
from friday.roster.definitions import ROSTER_PERSONAS
from friday.scheduler.engine import Scheduler
from friday.scheduler.store import SQLiteTriggerStore, Trigger
from friday.secrets import (
    EnvVault,
    FileVault,
    KeyringVault,
    MemoryVault,
    SecretVault,
    SecretVaultError,
    scan_for_plaintext_secrets,
)
from friday.security.egress import EgressPolicy
from friday.studio.generator import (
    MeshyText3D,
    ProceduralGenerator,
    StudioService,
    Text3DProvider,
)
from friday.study.store import SQLiteStudyStore
from friday.system.monitor import PsutilSampler, SystemMonitor
from friday.tools.agent_reach import AgentReachTool
from friday.tools.ask_user import AskUserTool
from friday.tools.base import Tool
from friday.tools.browser_tool import BrowserTool
from friday.tools.capabilities import CapabilitiesTool
from friday.tools.dossier import DossierTool
from friday.tools.downloads_butler import DownloadsButlerTool
from friday.tools.home import HomeControlTool
from friday.tools.infofeed import InfofeedTool
from friday.tools.media import MediaTool
from friday.tools.notify import NotifyTool, SentMessage
from friday.tools.registry import ToolRegistry
from friday.tools.reminders import (
    CompleteReminderTool,
    CreateReminderTool,
    ListRemindersTool,
)
from friday.tools.system_exec import FindFilesTool, OpenAppTool, RunCommandTool
from friday.tools.weather import WeatherTool
from friday.tools.web_search import WebSearchTool
from friday.voice.voiceprint import FakeVoiceprint, OwnerIdentity

# The system-automation tool names added to the Automation agent's allow-list
# (and registered) only when ``enable_system_automation`` is set.
_SYSTEM_AUTOMATION_TOOLS: frozenset[str] = frozenset(
    {"run_command", "find_files", "open_app"}
)

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


# The reliable gateway fallback model id (a free-tier NVIDIA model). When a
# primary model raises, the gateway retries this once — so a flaky free model
# never sinks a turn.
_GATEWAY_FALLBACK_MODEL_ID = "nvidia:meta/llama-3.1-8b-instruct"


def _build_providers(settings: Settings) -> dict[str, LLMProvider]:
    """Construct the per-provider LLM clients from whichever keys are set.

    Builds one :class:`~friday.providers.llm.LLMProvider` per provider that has a
    usable credential — OpenRouter / OpenCode from their own keys, NVIDIA from the
    NVIDIA key, Gemini from the Gemini key. Provider clients are lazy (the
    ``openai`` :class:`~openai.AsyncOpenAI` client performs no I/O at
    construction), so this is side-effect-free. The keys are unwrapped from their
    :class:`~pydantic.SecretStr` only here, into the provider clients — never
    logged. Providers without a key are simply absent, so the catalog only offers
    models the build can actually serve.
    """
    providers: dict[str, LLMProvider] = {}
    if settings.openrouter_api_key is not None:
        providers["openrouter"] = OpenRouterProvider(
            api_key=settings.openrouter_api_key.get_secret_value(),
            base_url=settings.openrouter_base_url,
            timeout=settings.llm_timeout_seconds,
        )
    if settings.opencode_api_key is not None:
        providers["opencode"] = OpenCodeProvider(
            api_key=settings.opencode_api_key.get_secret_value(),
            base_url=settings.opencode_base_url,
            timeout=settings.llm_timeout_seconds,
        )
    if settings.nvidia_api_key is not None:
        providers["nvidia"] = NvidiaNIMProvider(
            api_key=settings.nvidia_api_key.get_secret_value(),
            base_url=settings.nvidia_base_url,
            model=settings.nvidia_model,
            timeout=settings.llm_timeout_seconds,
        )
    if settings.gemini_api_key is not None:
        providers["gemini"] = GeminiProvider(
            api_key=settings.gemini_api_key.get_secret_value(),
            base_url=settings.gemini_base_url,
            model=settings.gemini_model,
            timeout=settings.llm_timeout_seconds,
        )
    return providers


def _use_gateway(settings: Settings, providers: dict[str, LLMProvider]) -> bool:
    """Whether the multi-model gateway should back the orchestrator's LLM.

    The gateway is selected when ``FRIDAY_LLM_PROVIDER == "gateway"`` (explicit
    opt-in), OR — for convenience — when an OpenRouter/OpenCode key is present and
    the provider is not the offline ``fake`` (so a build with a multi-model key
    gets the gateway without extra config). The ``fake`` build never selects the
    gateway, so the offline default keeps its scripted single-provider path and
    every ``/models`` route stays ``404``.
    """
    if settings.llm_provider == "fake":
        return False
    if settings.llm_provider == "gateway":
        return True
    return "openrouter" in providers or "opencode" in providers


def _build_gateway(
    settings: Settings, providers: dict[str, LLMProvider]
) -> tuple[ModelGateway, ModelCatalog] | None:
    """Build the model catalog + gateway, or ``None`` when no gateway is wired.

    Returns ``None`` (no gateway) on the offline ``fake`` build or when no
    multi-model provider key is present (:func:`_use_gateway`), so the
    single-provider/fake paths — and their tests — are untouched and ``/models``
    routes ``404``. Otherwise a :class:`ModelCatalog` scoped to the available
    providers is built and fronted by a :class:`ModelGateway` over the same
    providers dict, with the configured ``default_model_id`` active and the
    reliable NVIDIA model as the single retry fallback. Construction performs no
    network I/O.
    """
    if not _use_gateway(settings, providers) or not providers:
        return None
    catalog = ModelCatalog(available_providers=set(providers))
    gateway = ModelGateway(
        providers,
        catalog,
        default_model_id=settings.default_model_id,
        fallback_model_id=_GATEWAY_FALLBACK_MODEL_ID,
    )
    logger.info(
        "using multi-model gateway as the LLM",
        extra={
            "providers": sorted(providers),
            "active": settings.default_model_id,
        },
    )
    return gateway, catalog


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


def _select_long_term(settings: Settings) -> LongTermStore:
    """Select the durable long-term store: Postgres when enabled, else SQLite.

    When ``enable_postgres`` is set, the server-backed
    :class:`~friday.memory.pg.PostgresLongTermStore` (over ``postgres_dsn``)
    replaces the local-first :class:`SQLiteLongTermStore`; both satisfy the same
    :class:`~friday.memory.long_term.LongTermStore` contract, so every downstream
    consumer (the orchestrator's write-consent / forget, the briefing/journal
    services, the agents) is unchanged. The Postgres adapter is LAZY: it imports
    ``psycopg`` and validates the DSN only in its constructor, raising a clear
    :class:`~friday.errors.FridayError` (install ``psycopg`` + set
    ``FRIDAY_POSTGRES_DSN``) when either is absent — so the off-by-default path
    never touches the driver and a misconfigured opt-in fails fast and loud.
    """
    if settings.enable_postgres:
        dsn = (
            settings.postgres_dsn.get_secret_value()
            if settings.postgres_dsn is not None
            else None
        )
        logger.info("using Postgres long-term store")
        return PostgresLongTermStore(dsn)
    return _build_long_term(settings)


def _select_vector(
    settings: Settings, embedder: EmbeddingProvider
) -> ForgettableVectorStore:
    """Select the persistent vector store: pgvector when Postgres is on, else SQLite.

    Mirrors :func:`_select_long_term`: when ``enable_postgres`` is set the
    server-backed :class:`~friday.memory.pg.PgVectorStore` (over ``postgres_dsn``,
    sized to ``embedding_dim``) replaces the local-first
    :class:`SQLiteVectorStore`; both satisfy the same
    :class:`~friday.memory.vector.VectorStore` contract (and the orchestrator's
    ``forget`` surface), so the Knowledge agent's retrieval and personal-RAG
    indexing are unchanged. The pgvector adapter is LAZY (imports ``psycopg`` +
    validates the DSN only in its constructor), so the off-by-default path never
    touches the driver and a misconfigured opt-in raises a clear
    :class:`~friday.errors.FridayError`.
    """
    if settings.enable_postgres:
        dsn = (
            settings.postgres_dsn.get_secret_value()
            if settings.postgres_dsn is not None
            else None
        )
        logger.info("using pgvector vector store")
        return PgVectorStore(dsn, embedder, settings.embedding_dim)
    return _build_vector(settings, embedder)


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


def _build_meeting_store(settings: Settings) -> SQLiteMeetingStore:
    """Build the local-first SQLite meeting store alongside the long-term DB.

    Reuses ``memory_db_path``: the meeting notes live in a sibling file
    (``*.meet.db``) next to the long-term database so they share the gitignored
    ``data/`` directory; ``":memory:"`` stays ephemeral. Action items are
    persisted as JSON in the store; the store is connection-per-call for file
    paths (thread-safe).
    """
    db_path = settings.memory_db_path
    if db_path == ":memory:":
        meet_path = ":memory:"
    else:
        base = Path(db_path)
        meet_path = str(base.with_suffix(base.suffix + ".meet"))
        base.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteMeetingStore(meet_path)


def _build_graph_store(settings: Settings) -> SQLiteGraphStore:
    """Build the local-first SQLite knowledge-graph store alongside the long-term DB.

    Reuses ``memory_db_path``: the graph lives in a sibling file (``*.graph.db``)
    next to the long-term database so they share the gitignored ``data/``
    directory; ``":memory:"`` stays ephemeral. Entities are keyed on
    ``(name, type)`` (idempotent upsert); the store is connection-per-call for file
    paths (thread-safe). Built eagerly (cheap, no I/O beyond schema init) and
    surfaced on the runtime; the ``/graph`` route self-guards on the flag.
    """
    db_path = settings.memory_db_path
    if db_path == ":memory:":
        graph_path = ":memory:"
    else:
        base = Path(db_path)
        graph_path = str(base.with_suffix(base.suffix + ".graph"))
        base.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteGraphStore(graph_path)


def _build_journal_store(settings: Settings) -> SQLiteJournalStore:
    """Build the local-first SQLite journal store alongside the long-term DB.

    Reuses ``memory_db_path``: the journal entries live in a sibling file
    (``*.journal.db``) next to the long-term database so they share the gitignored
    ``data/`` directory; ``":memory:"`` stays ephemeral. Entries are upserted by
    date; the store is connection-per-call for file paths (thread-safe). Built
    eagerly (cheap, no I/O beyond schema init) and surfaced on the runtime; the
    ``/journal`` route self-guards on the flag.
    """
    db_path = settings.memory_db_path
    if db_path == ":memory:":
        journal_path = ":memory:"
    else:
        base = Path(db_path)
        journal_path = str(base.with_suffix(base.suffix + ".journal"))
        base.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteJournalStore(journal_path)


def _build_study_store(settings: Settings) -> SQLiteStudyStore:
    """Build the local-first SQLite study store alongside the long-term DB.

    Reuses ``memory_db_path``: the flashcards + study sessions live in a sibling
    file (``*.study.db``) next to the long-term database so they share the
    gitignored ``data/`` directory; ``":memory:"`` stays ephemeral. The store's
    clock is left at its wall-clock default (it stamps a session's ``at`` and a
    reviewed card's ``due_at``); tested paths inject their own clock and
    ``due_cards()`` is driven by a passed ``now``, never the wall clock. Built
    eagerly (cheap, no I/O beyond schema init) and surfaced on the runtime; the
    ``/study`` route self-guards on the flag.
    """
    db_path = settings.memory_db_path
    if db_path == ":memory:":
        study_path = ":memory:"
    else:
        base = Path(db_path)
        study_path = str(base.with_suffix(base.suffix + ".study"))
        base.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteStudyStore(study_path)


def _build_journal_service(
    reminder_store: SQLiteReminderStore,
    audit: AuditLog,
    metrics: Metrics,
    llm: LLMProvider,
    owner_address: str,
) -> JournalService:
    """Assemble the :class:`JournalService` over the shared local stores.

    Pure assembly from the *shared* runtime pieces — the same reminder store the
    reminder tools/routes use (so the day's completed reminders are counted), the
    process-wide audit log + metrics — plus the live LLM for an optional, non-fatal
    natural-language summary. Constructing the service performs no I/O.
    """
    return JournalService(
        audit,
        reminder_store=reminder_store,
        metrics=metrics,
        llm=llm,
        owner_address=owner_address,
    )


def _make_journal_action(
    journal_service: JournalService, journal_store: SQLiteJournalStore
) -> Callable[[Trigger], Awaitable[None]]:
    """Build the ``journal`` scheduler action over the shared service + store.

    Builds the journal entry for ``utcnow`` from the *shared* :class:`JournalService`
    and saves it (upsert by date) into the *shared* :class:`SQLiteJournalStore`, so
    an enabled trigger writes an end-of-day journal proactively. ``utcnow`` is read
    here in the background action only — the tested ``JournalService.build_entry(day)``
    unit stays clock-injected; this action is exercised through the run-now route,
    not for timing. Any LLM error is already swallowed inside ``build_entry``
    (deterministic-summary fallback), so this never raises.
    """

    async def _journal(trigger: Trigger) -> None:
        entry = await journal_service.build_entry(datetime.now(UTC))
        journal_store.save(entry)
        logger.info(
            "scheduler journal fired",
            extra={
                "trigger": trigger.name,
                "date": entry.date,
                "events": entry.event_count,
            },
        )

    return _journal


def _build_rag_ingestor(
    settings: Settings, vector: ForgettableVectorStore, long_term: LongTermStore
) -> DocumentIngestor | None:
    """Build the shared :class:`DocumentIngestor`, or ``None`` when RAG is off.

    The ingestor reuses the *shared* runtime stores — the same persistent vector
    store the Knowledge agent retrieves from and the same long-term store — so an
    ingested document (a RAG upload or a meeting transcript) is immediately
    answerable via the existing knowledge path with citations. Built only when
    ``enable_rag`` is set, so the offline default constructs no write seam; the
    one instance is shared by both the ``/rag`` routes and meeting capture.
    """
    if not settings.enable_rag:
        return None
    return DocumentIngestor(
        vector,
        long_term,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )


def _build_meeting_stt(settings: Settings) -> STTProvider:
    """Select the STT provider for meeting capture (safe to call eagerly).

    Offline (``fake`` LLM) -> :class:`FakeSTT` (tests / no credentials). With a
    real LLM the real :class:`FasterWhisperSTT` is preferred, but its construction
    (which lazy-imports the optional ``faster-whisper`` voice extra) is wrapped:
    if the extra is missing the runtime degrades to :class:`FakeSTT` rather than
    failing startup, so the offline build (and a real-LLM build without the voice
    extra) still boots. The ``/meetings`` route self-guards on the flag, so this
    only affects whether a real transcript is produced once meetings are enabled.
    """
    if settings.llm_provider == "fake":
        return FakeSTT()
    try:
        return FasterWhisperSTT()
    except ProviderError:
        logger.info(
            "faster-whisper not installed; meeting capture uses FakeSTT "
            "(install voice extras for real transcription)"
        )
        return FakeSTT()


def _build_meeting_capture(
    settings: Settings,
    llm: LLMProvider,
    ingestor: DocumentIngestor | None,
) -> MeetingCapture:
    """Assemble the :class:`MeetingCapture` pipeline over the shared providers.

    Drives the meeting STT provider (:func:`_build_meeting_stt` — ``FakeSTT``
    offline / without the voice extra, real Whisper otherwise) and the live LLM
    (the same one the chat loop uses) for the one non-fatal summary pass. The RAG
    ingestor is passed through when available (RAG enabled) so a captured
    transcript is also indexed for retrieval; otherwise capture still works, just
    without indexing. Construction performs no network I/O.
    """
    return MeetingCapture(_build_meeting_stt(settings), llm, ingestor=ingestor)


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
        "run_command",
        "find_files",
        "open_app",
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


def _build_n8n_service(settings: Settings, llm: LLMProvider) -> N8nService:
    """Assemble the :class:`N8nService` over the live LLM + an n8n REST client.

    The client targets ``n8n_base_url`` with the (optional) ``n8n_api_key`` (a
    :class:`~pydantic.SecretStr` whose value is unwrapped only here, into the
    client's ``X-N8N-API-KEY`` header — never logged). The drafter drives the
    *same* live LLM the chat loop uses for one NON-FATAL draft pass. The docker
    auto-start argv is built from the compose file + service name
    (``docker compose -f <file> up -d <service>``) and spawned argv-only by the
    service behind the confirm-step. Always constructed (uniform wiring); the
    ``/n8n`` route and orchestrator hook only become reachable when
    ``enable_n8n`` is on. Construction performs no network I/O (the ``httpx``
    client is created per-call inside :class:`N8nClient`).
    """
    api_key = (
        settings.n8n_api_key.get_secret_value()
        if settings.n8n_api_key is not None
        else None
    )
    client = N8nClient(
        settings.n8n_base_url,
        api_key,
        timeout=settings.llm_timeout_seconds,
    )
    start_cmd = [
        "docker",
        "compose",
        "-f",
        settings.n8n_docker_compose_file,
        "up",
        "-d",
        settings.n8n_docker_service,
    ]
    return N8nService(client, WorkflowDrafter(llm), start_cmd=start_cmd)


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


def _build_system_monitor(settings: Settings) -> SystemMonitor:
    """Assemble the :class:`SystemMonitor` over the real :class:`PsutilSampler`.

    The sampler lazy-imports ``psutil`` (only when a live sample is taken), so this
    construction performs no reads and the module imports even where ``psutil`` is
    absent. The four breach thresholds come from settings
    (``sys_cpu_threshold``/``sys_mem_threshold``/``sys_disk_threshold``/
    ``sys_temp_threshold``); the ``/system`` route and the scheduler
    ``"system_check"`` action both read through this one monitor. Built eagerly
    (cheap) and surfaced on the runtime; the route + action self-guard on the flag.
    """
    return SystemMonitor(
        PsutilSampler(),
        cpu_threshold=settings.sys_cpu_threshold,
        mem_threshold=settings.sys_mem_threshold,
        disk_threshold=settings.sys_disk_threshold,
        temp_threshold=settings.sys_temp_threshold,
    )


# Upper bound on the rolling CPU history the proactive anomaly detector scores;
# keeps memory bounded across an unbounded run of scheduled ticks.
_ANOMALY_HISTORY_LIMIT = 64


def _make_system_check_action(
    monitor: SystemMonitor,
    notify: NotifyTool,
    anomaly_detector: AnomalyDetector | None = None,
) -> Callable[[Trigger], Awaitable[None]]:
    """Build the ``system_check`` scheduler action over the shared monitor.

    Samples the *shared* :class:`SystemMonitor` once and emits each breached
    threshold (:class:`~friday.system.monitor.Alert`) via the :class:`NotifyTool`
    fake sink (one message per alert, carrying the alert's human-readable line) +
    a summary log line, so an enabled trigger surfaces a resource breach
    proactively. When the host is healthy (no alerts) nothing is sent. The
    sampling is the action's only side seam; ``check()`` itself is the
    deterministic, fake-sampler-tested unit.

    **Proactive anomaly detection (Stage 2).** When an
    :class:`~friday.proactive.AnomalyDetector` is wired (``enable_proactive``), the
    action ALSO keeps a bounded rolling history of the sampled CPU utilisation and
    flags the latest reading when it is a causal rolling-z-score outlier — so a
    *spike* (a sharp deviation from recent history) is surfaced even when it stays
    below the static breach threshold. This is purely additive to the existing
    threshold alerts; off (no detector) the behaviour is exactly as before.
    """
    cpu_history: list[float] = []

    async def _system_check(trigger: Trigger) -> None:
        alerts = monitor.check()
        for alert in alerts:
            notify.sink.append(
                SentMessage(
                    channel="webhook",
                    target="scheduler",
                    subject="System alert",
                    body=alert.message,
                )
            )
        anomaly_flagged = False
        if anomaly_detector is not None:
            cpu = monitor.stats().cpu_percent
            cpu_history.append(cpu)
            if len(cpu_history) > _ANOMALY_HISTORY_LIMIT:
                del cpu_history[0]
            anomalies = anomaly_detector.detect(cpu_history)
            # Only the latest reading matters for a proactive, real-time flag.
            latest = cpu_history[-1] if cpu_history else None
            for anomaly in anomalies:
                if anomaly.index == len(cpu_history) - 1:
                    anomaly_flagged = True
                    notify.sink.append(
                        SentMessage(
                            channel="webhook",
                            target="scheduler",
                            subject="System anomaly",
                            body=(
                                f"CPU spike detected: {latest:g}% is a "
                                f"{anomaly.zscore:.1f}-sigma outlier vs recent history."
                            ),
                        )
                    )
        logger.info(
            "scheduler system_check fired",
            extra={
                "trigger": trigger.name,
                "alerts": len(alerts),
                "anomaly": anomaly_flagged,
            },
        )

    return _system_check


def _build_scheduler(
    trigger_store: SQLiteTriggerStore,
    reminder_store: SQLiteReminderStore,
    registry: ToolRegistry,
    briefing_service: BriefingService,
    journal_service: JournalService,
    journal_store: SQLiteJournalStore,
    system_monitor: SystemMonitor,
    anomaly_detector: AnomalyDetector | None = None,
) -> Scheduler:
    """Assemble the :class:`Scheduler` and register the default actions.

    Registers ``"due_reminders"`` (emit due reminders via the shared notify tool's
    sink/log), ``"briefing"`` (build + emit the proactive briefing via the same
    sink), ``"journal"`` (build + save the day's journal entry into the shared
    journal store), ``"system_check"`` (sample the shared system monitor and emit
    any breached thresholds via the same sink), and a ``"noop"`` placeholder. The
    notify tool is pulled from the shared :class:`ToolRegistry` so the scheduler
    emits into the *same* sink the alerting agent uses (one auditable place for
    everything that would have been sent); a fresh :class:`NotifyTool` is used as a
    fallback if absent.
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
    scheduler.register_action(
        "journal", _make_journal_action(journal_service, journal_store)
    )
    scheduler.register_action(
        "system_check",
        _make_system_check_action(system_monitor, notify_tool, anomaly_detector),
    )
    scheduler.register_action("noop", _noop_action)
    return scheduler


class _DesktopAuditAdapter:
    """Adapt the hash-chained ledger to the desktop :class:`AuditSink` shape.

    :class:`~friday.desktop.AuditedDesktop` records each action to a sink whose
    contract is ``append(DesktopAction)``, whereas
    :class:`~friday.broker.HashChainedAudit` appends a plain ``dict``. This thin
    adapter bridges the two: it serializes each :class:`~friday.desktop.DesktopAction`
    to a redaction-safe record (``kind`` + ``args``) and appends it to the ledger,
    so EVERY desktop action lands one tamper-evident, hash-chained row BEFORE it
    executes (frictionless but fully audited). The ledger redacts sensitive-keyed
    values itself before hashing, so a typed password never reaches disk verbatim.
    """

    def __init__(self, ledger: HashChainedAudit) -> None:
        self._ledger = ledger

    def append(self, action: DesktopAction) -> None:
        """Record one desktop action to the hash-chained ledger."""
        self._ledger.append(
            {"desktop_action": action.kind, "args": dict(action.args)}
        )


def _build_desktop(
    settings: Settings, hash_audit: HashChainedAudit
) -> AuditedDesktop | None:
    """Build the frictionless, fully-audited desktop controller, or ``None``.

    Returns an :class:`~friday.desktop.AuditedDesktop` wrapping a deterministic
    :class:`~friday.desktop.FakeDesktop` (no real mouse/keyboard/display) ONLY when
    ``enable_desktop`` is set; off by default it returns ``None`` so the offline
    build constructs no desktop seam. The wrapper is FRICTIONLESS (no per-action
    prompt) but FULLY AUDITED — every action is appended to the shared hash-chained
    ledger (via :class:`_DesktopAuditAdapter`) BEFORE it executes. The real
    ``pyautogui`` adapter is wired only when explicitly chosen (it lazy-imports its
    backend); the default fake keeps this construction side-effect-free.
    """
    if not settings.enable_desktop:
        return None
    logger.info("desktop control enabled (audited, frictionless; fake backend)")
    return AuditedDesktop(
        FakeDesktop(), audit_sink=_DesktopAuditAdapter(hash_audit)
    )


def _build_owner_identity(settings: Settings) -> OwnerIdentity | None:
    """Build advisory owner recognition over a deterministic voiceprint, or ``None``.

    Returns an :class:`~friday.voice.voiceprint.OwnerIdentity` over a deterministic
    :class:`~friday.voice.voiceprint.FakeVoiceprint` ONLY when ``enable_voiceprint``
    is set; off by default it returns ``None`` so no voiceprint seam is built.
    Owner recognition is ADVISORY by default (it never blocks — only opt-in callers
    gate on it). The real ``resemblyzer`` backend stays OPTIONAL/LAZY (excluded from
    the uv lock); the fake makes this construction model-free and I/O-free. The
    profile is enrolled on a stable owner-marker sample so the identity is usable
    out of the box without a saved enrollment.
    """
    if not settings.enable_voiceprint:
        return None
    verifier = FakeVoiceprint()
    # A stable, deterministic owner marker; the real build loads a saved profile
    # from an EnrollmentStore. Enrolling here keeps the advisory identity usable.
    profile = verifier.enroll([b"FRIDAY_OWNER_VOICEPRINT"])
    logger.info("voiceprint owner recognition enabled (advisory; fake backend)")
    return OwnerIdentity(verifier, profile)


def _build_anomaly_detector(settings: Settings) -> AnomalyDetector | None:
    """Build the proactive anomaly detector, or ``None`` when proactive is off.

    Returns a pure, deterministic :class:`~friday.proactive.AnomalyDetector` ONLY
    when ``enable_proactive`` is set; off by default it returns ``None`` so the
    scheduler's ``system_check`` action keeps its plain threshold-only behaviour.
    When built it is wired into the ``system_check`` action so a spike in the
    sampled metric history (a causal rolling z-score outlier) is ALSO flagged —
    additive to the existing breach alerts.
    """
    if not settings.enable_proactive:
        return None
    return AnomalyDetector()


def _build_foresight(settings: Settings) -> Foresight | None:
    """Build the proactive foresight engine, or ``None`` when proactive is off.

    Returns a deterministic, rule-based :class:`~friday.proactive.Foresight` ONLY
    when ``enable_proactive`` is set; off by default it returns ``None``. It imports
    no LLM SDK and takes every input injected, so construction is side-effect-free.
    """
    if not settings.enable_proactive:
        return None
    return Foresight()


def _build_budgeter(settings: Settings) -> Budgeter | None:
    """Build the per-turn cost/latency budgeter, or ``None`` when it is off.

    Returns a pure, offline :class:`~friday.models.budget.Budgeter` ONLY when
    ``enable_budgeter`` is set; off by default it returns ``None`` so the turn loop
    keeps no spend tally and never downshifts. When built, the caps are injected
    here (dependency injection — the budgeter reads no settings itself): the hard
    per-turn token ceiling, an optional dollar ceiling, and the fraction of the
    token cap at/beyond which a downshift trips. The orchestrator records each
    completion's usage and consults :meth:`Budgeter.should_downshift` to swap the
    gateway's active model down a tier; the budgeter only has something to downshift
    when a :class:`~friday.models.gateway.ModelGateway` is wired (the orchestrator
    guards the ``set_active`` call on the gateway being present).
    """
    if not settings.enable_budgeter:
        return None
    return Budgeter(
        max_tokens=settings.budget_max_tokens_per_turn,
        max_usd=settings.budget_max_usd_per_turn,
        downshift_at=settings.budget_downshift_at,
    )


def _build_compactor(settings: Settings, llm: LLMProvider) -> Compactor | None:
    """Build the session-compaction helper, or ``None`` when compaction is off.

    Returns a :class:`~friday.memory.compaction.Compactor` over the live LLM ONLY
    when ``enable_compaction`` is set; off by default it returns ``None`` so the
    orchestrator's turn loop makes no extra model call and never rewrites the
    short-term buffer. Uses the Compactor's default thresholds.
    """
    if not settings.enable_compaction:
        return None
    return Compactor(llm)


def _build_confidence(settings: Settings) -> ConfidenceScorer | None:
    """Build the calibrated confidence scorer, or ``None`` when it is off.

    Returns a pure, deterministic :class:`~friday.core.confidence.ConfidenceScorer`
    ONLY when ``enable_confidence`` is set; off by default it returns ``None`` so the
    orchestrator stamps no confidence and appends no caveat (existing behaviour
    unchanged). The scorer takes no constructor args and reads no settings — the
    flag and ``confidence_note_threshold`` are read by the orchestrator, which
    stamps ``state.scratchpad["confidence"]`` after a synthesized reply and appends
    a one-line honest caveat when the blended confidence falls below the threshold.
    """
    if not settings.enable_confidence:
        return None
    return ConfidenceScorer()


class _BrokerSecretAdapter:
    """Adapt a :class:`~friday.secrets.SecretVault` to the broker's resolver shape.

    The broker's secret-provider contract is ``get(name) -> str`` (a marker is
    always resolved to a concrete value), whereas a :class:`SecretVault` returns
    ``str | None`` (``None`` when the secret is absent so callers can layer
    vaults). This thin adapter bridges the two: it forwards to the wrapped vault
    and raises a clear :class:`~friday.secrets.SecretVaultError` when the requested
    secret is missing — so a ``{{secret:NAME}}`` marker for an unset credential
    fails loud at injection time rather than silently injecting ``None`` into a
    tool's arguments.
    """

    def __init__(self, vault: SecretVault) -> None:
        self._vault = vault

    def get(self, name: str) -> str:
        value = self._vault.get(name)
        if value is None:
            raise SecretVaultError(
                f"secret {name!r} is not set in the configured vault"
            )
        return value


def _build_audit_ledger(settings: Settings) -> HashChainedAudit:
    """Build the tamper-evident, hash-chained audit ledger over ``audit_ledger_path``.

    The ledger is an append-only JSONL file; every tool the shared registry
    executes appends ONE hash-chained record (the tamper-evident system-of-record),
    additive to the in-memory observability :class:`AuditLog`. A missing parent
    directory is created on first append by the ledger itself; ``data/`` is
    gitignored. Construction performs no I/O (the file is opened only on append /
    verify), so this is safe to call eagerly at startup.
    """
    return HashChainedAudit(settings.audit_ledger_path)


def _build_egress_policy(settings: Settings) -> EgressPolicy | None:
    """Build the fail-closed egress allow-list, or ``None`` when the firewall is off.

    Returns an :class:`~friday.security.egress.EgressPolicy` over
    ``egress_allowlist`` ONLY when ``enable_egress_firewall`` is set; off by default
    it returns ``None`` so the broker's dispatch is unchanged. The policy is pure
    (no I/O); the broker consults it before executing a tool whose args carry an
    outbound URL, denying a call to a host that is not on the allow-list.
    """
    if not settings.enable_egress_firewall:
        return None
    return EgressPolicy(settings.egress_allowlist)


def _build_secret_vault(settings: Settings) -> SecretVault:
    """Construct the secret backend selected by ``secret_vault`` (lazy / safe).

    ``env`` (default) reads ``FRIDAY_<NAME>`` from the process environment;
    ``memory`` is in-process only; ``file`` is a ``0600`` JSON dev fallback stored
    next to the long-term DB's ``data/`` directory; ``keyring`` wraps the OPTIONAL
    OS-keychain backend (``keyring`` is lazy-imported only here, when selected —
    it is kept out of the core lock). A misconfigured/absent ``keyring`` raises a
    clear :class:`~friday.secrets.SecretVaultError` from its constructor, so the
    opt-in fails fast and loud while the default ``env`` path never touches the
    optional dependency. The vault is the broker's ``{{secret:NAME}}`` resolver.
    """
    backend = settings.secret_vault
    if backend == "memory":
        logger.info("using in-memory secret vault")
        return MemoryVault()
    if backend == "file":
        # Co-locate the dev secret file with the gitignored data/ dir; fall back to
        # data/ when the DB is purely in-memory (tests).
        db_path = settings.memory_db_path
        data_dir = Path(db_path).parent if db_path != ":memory:" else Path("data")
        path = str(data_dir / "secrets.json")
        logger.info("using file secret vault", extra={"path": path})
        return FileVault(path)
    if backend == "keyring":
        logger.info("using OS keyring secret vault")
        return KeyringVault()
    logger.info("using environment secret vault")
    return EnvVault()


def _run_secret_self_check(settings: Settings, repo_root: str) -> None:
    """Scan ``repo_root`` for plaintext secrets and LOG a WARNING per finding.

    Warn-only by design: this NEVER raises and NEVER refuses to boot — it is a
    default-safe nudge to move a committed credential into the vault. Skipped
    entirely when ``enable_secret_self_check`` is off. The scan reads only
    committed source files (``.py`` / ``.env``-family, excluding the git-ignored
    ``.env``); any unreadable file is silently skipped by the scanner.
    """
    if not settings.enable_secret_self_check:
        return
    findings = scan_for_plaintext_secrets(repo_root)
    for finding in findings:
        logger.warning(
            "possible plaintext secret in tracked source",
            extra={"file": finding.file, "line": finding.line, "kind": finding.kind},
        )
    if findings:
        logger.warning(
            "secret self-check found %d possible plaintext secret(s); "
            "move them into the secret vault (boot continues)",
            len(findings),
        )


# Bind hosts that mean "this machine only" — reaching FRIDAY over one of these
# requires already being on the host, so an open gateway behind them is not
# network-exposed and the exposure nudge stays silent.
_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})


def _warn_if_exposed_without_auth(settings: Settings) -> None:
    """LOG a prominent WARNING when FRIDAY is bound to the network with no auth.

    The "everywhere" safety nudge: when the configured uvicorn bind host
    (``bind_host`` / ``FRIDAY_BIND_HOST``) is NON-loopback (so the gateway is
    reachable from other machines) AND ``require_auth`` is off (so any caller is
    accepted), one prominent WARNING is logged that FRIDAY is exposed without auth,
    pointing the operator at ``FRIDAY_REQUIRE_AUTH`` + ``FRIDAY_API_KEYS``.

    ADVISORY by design — like the plaintext-secret self-check, this NEVER raises
    and NEVER refuses to boot: binding ``0.0.0.0`` for local dev (or a LAN demo)
    must keep working. On the loopback default (or with auth on) it is SILENT.
    """
    host = settings.bind_host.strip().lower()
    if host in _LOOPBACK_HOSTS or settings.require_auth:
        return
    logger.warning(
        "FRIDAY is bound to %s WITHOUT auth — the gateway is exposed to the "
        "network and will accept ANY caller. Set FRIDAY_REQUIRE_AUTH=true (with "
        "FRIDAY_API_KEYS) to require a bearer key, or bind 127.0.0.1 for "
        "local-only access (boot continues).",
        settings.bind_host,
        extra={"bind_host": settings.bind_host, "require_auth": settings.require_auth},
    )


def _build_registry(
    settings: Settings,
    reminder_store: SQLiteReminderStore,
    audit: AuditLog | None = None,
    metrics: Metrics | None = None,
    hash_audit: HashChainedAudit | None = None,
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

    The read-only :class:`~friday.tools.agent_reach.AgentReachTool` is registered
    ONLY when ``enable_agent_reach`` is set (it is configured with the Jina base +
    timeout from settings); off by default it is absent from the registry, so the
    offline build's tool surface is unchanged.

    When an :class:`~friday.observability.audit.AuditLog` / :class:`Metrics` are
    passed (the app wires the process-wide instances), every ``execute`` records a
    redacted tool-call audit row and bumps the ``tool_calls`` counter (build-spec
    §11). They default to ``None`` so a bare registry behaves exactly as before.

    When a hash-chained :class:`~friday.broker.HashChainedAudit` ledger is passed
    (the app wires the process-wide instance), every ``execute`` ADDITIONALLY
    appends one tamper-evident record to the on-disk ledger (the system-of-record
    verified by ``GET /admin/audit/verify``). This is purely additive — the
    in-memory ``audit`` row above is unchanged — and defaults to ``None``.
    """
    registry = ToolRegistry(audit=audit, metrics=metrics, hash_audit=hash_audit)
    registry.register(cast(Tool, WebSearchTool()))
    registry.register(cast(Tool, NotifyTool()))
    registry.register(cast(Tool, HomeControlTool()))
    registry.register(cast(Tool, CreateReminderTool(reminder_store)))
    registry.register(cast(Tool, ListRemindersTool(reminder_store)))
    registry.register(cast(Tool, CompleteReminderTool(reminder_store)))
    if settings.enable_agent_reach:
        registry.register(
            cast(
                Tool,
                AgentReachTool(
                    jina_base=settings.agent_reach_jina_base,
                    timeout=settings.agent_reach_timeout,
                ),
            )
        )
    # System-automation tools (Tier 2) — registered ONLY when the flag is on, so
    # the offline default's tool surface is unchanged. Their own gates keep them
    # safe: run_command/open_app are side-effecting + non-idempotent (the registry
    # confirm-step gates them) and find_files is read-only + root-confined.
    if settings.enable_system_automation:
        registry.register(cast(Tool, RunCommandTool()))
        registry.register(cast(Tool, FindFilesTool()))
        registry.register(cast(Tool, OpenAppTool()))
    return registry


# The read-only idea-batch tool names, by which agent they are added to. The
# Capabilities + AskUser tools reach EVERY agent (planning + clarification are
# universal); the dossier reaches the Knowledge agent (it stitches graph + memory);
# infofeed + browser + weather reach the Research (analysis) agent (read-only web
# reach — weather is the keyless wttr.in current-conditions lookup).
_EXTRA_TOOLS_FOR_ALL: frozenset[str] = frozenset({"capabilities", "ask_user"})
_EXTRA_TOOLS_FOR_KNOWLEDGE: frozenset[str] = frozenset({"entity_dossier"})
_EXTRA_TOOLS_FOR_RESEARCH: frozenset[str] = frozenset(
    {"infofeed", "browser", "weather"}
)


def _register_extra_tools(
    settings: Settings,
    registry: ToolRegistry,
    graph_store: SQLiteGraphStore,
    long_term: LongTermStore,
) -> None:
    """Register the read-only idea-batch tools into the shared registry (flag ON).

    Registered only when ``enable_extra_tools`` is set (default ON) — they are all
    read-only (``side_effecting=False``), so they add no real-world action, only a
    capability map, a clarification pause, an entity dossier, an RSS/Atom read, a
    keyless page read, and a keyless current-weather lookup (wttr.in). The
    capabilities tool reflects over the SAME shared
    registry (so its map always matches what is registered), and the dossier reads
    the shared knowledge graph + long-term store with the registered ``web_search``
    as its optional, best-effort fallback (consulted only when the local graph /
    memory turn up nothing). Off, none are registered, so the offline tool surface
    is unchanged. The companion :func:`_extend_agent_extra_tools` adds the matching
    names to the fitting agents' allow-lists.
    """
    if not settings.enable_extra_tools:
        return
    searcher = None
    try:
        searcher = registry.get("web_search")
    except KeyError:  # pragma: no cover - web_search is always registered
        searcher = None
    registry.register(cast(Tool, CapabilitiesTool(registry)))
    registry.register(cast(Tool, AskUserTool()))
    registry.register(
        cast(Tool, DossierTool(graph_store, long_term, searcher=searcher))
    )
    registry.register(cast(Tool, InfofeedTool()))
    registry.register(cast(Tool, BrowserTool()))
    registry.register(cast(Tool, WeatherTool()))


def _register_side_effecting_extra_tools(
    settings: Settings, registry: ToolRegistry
) -> None:
    """Register the side-effecting idea-batch tools behind their OWN readiness flags.

    These are NOT gated by ``enable_extra_tools`` — each rides its own flag so the
    owner opts in deliberately. ``downloads_butler`` (organize a folder) is
    side-effecting + non-idempotent, so a real (non-dry-run) move is gated by the
    registry's confirm-step; ``media`` (play/pause/next/prev/volume) is
    side-effecting but idempotent transport, so it dispatches straight through. Both
    default off, so the offline build registers neither.
    """
    if settings.enable_downloads_butler:
        registry.register(cast(Tool, DownloadsButlerTool()))
    if settings.enable_media_control:
        registry.register(cast(Tool, MediaTool()))


def _extend_agent_extra_tools(settings: Settings, agents: AgentRegistry) -> None:
    """Add the read-only idea-batch tools to the fitting agents' allow-lists (flag ON).

    Mirrors :func:`_register_extra_tools`: capabilities + ask_user reach EVERY
    registered agent; the dossier reaches the Knowledge agent; infofeed + browser +
    weather reach the Research (analysis) agent. Off (``enable_extra_tools`` unset)
    the
    allow-lists are the class defaults, so the offline build is unchanged. The
    registry still enforces each allow-list, so an agent can reach only the tools
    added here.
    """
    if not settings.enable_extra_tools:
        return
    for name in ("analysis", "knowledge", "automation", "device", "alerting"):
        if name in agents:
            agent = agents.get(name)
            agent.allowed_tools = agent.allowed_tools | _EXTRA_TOOLS_FOR_ALL
    if "knowledge" in agents:
        knowledge = agents.get("knowledge")
        knowledge.allowed_tools = (
            knowledge.allowed_tools | _EXTRA_TOOLS_FOR_KNOWLEDGE
        )
    if "analysis" in agents:
        analysis = agents.get("analysis")
        analysis.allowed_tools = (
            analysis.allowed_tools | _EXTRA_TOOLS_FOR_RESEARCH
        )


def _build_agents(
    settings: Settings,
    registry: ToolRegistry,
    llm: LLMProvider,
    vector: ForgettableVectorStore,
    long_term: LongTermStore,
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

    When ``enable_agent_reach`` is set, ``"agent_reach"`` is added to the Research
    (analysis) and Knowledge agents' instance-level ``allowed_tools`` so they may
    reach the registered read-only tool (full-page read + transcribe). Off by
    default the allow-lists are the class defaults, so the offline build is
    unchanged.
    """
    agents = AgentRegistry()
    analysis = AnalysisAgent(registry, llm=llm)
    knowledge = KnowledgeAgent(
        store=vector, memory=ShortTermMemory(), long_term=long_term
    )
    if settings.enable_agent_reach:
        analysis.allowed_tools = analysis.allowed_tools | {"agent_reach"}
        knowledge.allowed_tools = knowledge.allowed_tools | {"agent_reach"}
    agents.register(analysis)
    agents.register(knowledge)
    automation = AutomationAgent(registry=registry)
    if settings.enable_system_automation:
        # The Automation agent may reach the registered system-automation tools;
        # the registry's permission/confirm gates still apply to every call.
        automation.allowed_tools = (
            automation.allowed_tools | _SYSTEM_AUTOMATION_TOOLS
        )
    agents.register(automation)
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
    #: The shared durable long-term store — :class:`SQLiteLongTermStore` by
    #: default, or :class:`~friday.memory.pg.PostgresLongTermStore` when
    #: ``enable_postgres`` is on (the contract is identical either way).
    long_term: LongTermStore
    flag_overrides: dict[str, bool]
    #: The live LLM provider, shared with the studio's procedural generator so
    #: the (free) 3D scene generation uses the same provider as the chat loop.
    llm: LLMProvider
    #: The shared persistent vector store — the same one the Knowledge agent
    #: retrieves from, reused by personal RAG so ingested docs are answerable.
    #: :class:`SQLiteVectorStore` by default, or
    #: :class:`~friday.memory.pg.PgVectorStore` when ``enable_postgres`` is on.
    vector: ForgettableVectorStore
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
    #: The shared journal service — aggregates a day's events (audit + reminders +
    #: metrics) into a deterministic entry; the ``/journal/build`` route and the
    #: scheduler ``journal`` action both build through it.
    journal_service: JournalService
    #: The shared journal store — the same one the ``/journal`` routes and the
    #: scheduler ``journal`` action read/write (upsert by date).
    journal_store: SQLiteJournalStore
    #: The shared protocol store — the same one the ``/protocols`` routes and the
    #: orchestrator's trigger-phrase hook read/write.
    protocol_store: SQLiteProtocolStore
    #: The protocol runner over the shared registry; its ``allowed_tools`` is the
    #: set of registered tool names, so a protocol runs only registered tools.
    protocol_runner: ProtocolRunner
    #: The shared RAG document ingestor, or ``None`` when RAG is disabled. The same
    #: instance the ``/rag`` routes and meeting capture write through, so an
    #: ingested doc or meeting transcript is answerable via the Knowledge path.
    rag_ingestor: DocumentIngestor | None
    #: The shared meeting-notes store — the same one the ``/meetings`` routes
    #: read/write.
    meeting_store: SQLiteMeetingStore
    #: The shared meeting-capture pipeline (shared STT + live LLM + the RAG
    #: ingestor if available); the ``/meetings/capture`` route drives it.
    meeting_capture: MeetingCapture
    #: The shared knowledge-graph store — the same one the ``/graph`` routes
    #: read/write; entity cards read long-term facts from the shared long_term.
    graph_store: SQLiteGraphStore
    #: The entity extractor over the live LLM; ``POST /graph/extract`` drives it for
    #: one NON-FATAL extraction pass that upserts into ``graph_store``.
    graph_extractor: EntityExtractor
    #: The shared study store — the same one the ``/study`` routes read/write
    #: (spaced-repetition flashcards + logged study sessions).
    study_store: SQLiteStudyStore
    #: The shared hardware/system monitor over the real ``PsutilSampler`` — the
    #: same one the ``/system`` routes read and the scheduler ``system_check``
    #: action samples; its thresholds come from settings.
    system_monitor: SystemMonitor
    #: The shared n8n service (REST client + LLM drafter + docker auto-start) — the
    #: same one the ``/n8n`` routes drive and the orchestrator's "make a workflow
    #: on n8n" hook reaches; only reachable when ``enable_n8n`` is on.
    n8n_service: N8nService
    #: The tamper-evident, hash-chained audit ledger (the security spine's
    #: system-of-record). Every executed tool call appends one record here in
    #: addition to the in-memory ``audit`` log; ``GET /admin/audit/verify`` walks
    #: the chain. Always constructed (uniform wiring) over ``audit_ledger_path``.
    hash_audit: HashChainedAudit
    #: The action broker (validate → classify → deny-by-default gate → secret
    #: injection → execute → hash-chained audit) over the shared registry +
    #: ledger + secret vault. Always constructed (uniform wiring) but only
    #: interposed into dispatch when ``enable_broker`` is on; with the flag off the
    #: orchestrator keeps the plain registry path, so dispatch is unchanged.
    broker: Broker
    #: The persona roster registry (FRIDAY + eight least-privilege specialists) —
    #: always built (no flag). Surfaced for ``GET /roster`` and injected into the
    #: orchestrator so an addressed turn ("GECKO, ...") routes under the named
    #: persona's least-privilege scope + memory namespace.
    roster: RosterRegistry
    #: The frictionless, fully-audited desktop controller (an
    #: :class:`~friday.desktop.AuditedDesktop` over a :class:`FakeDesktop`), or
    #: ``None`` when ``enable_desktop`` is off. Every action is appended to the
    #: hash-chained ledger before it runs.
    desktop: AuditedDesktop | None
    #: Advisory owner recognition (an :class:`~friday.voice.voiceprint.OwnerIdentity`
    #: over a :class:`FakeVoiceprint`), or ``None`` when ``enable_voiceprint`` is
    #: off. Advisory by default — it never blocks.
    owner_identity: OwnerIdentity | None
    #: The proactive anomaly detector wired into the scheduler's ``system_check``
    #: action so a CPU spike is flagged, or ``None`` when ``enable_proactive`` is
    #: off.
    anomaly_detector: AnomalyDetector | None
    #: The proactive, rule-based foresight engine surfaced for suggestions, or
    #: ``None`` when ``enable_proactive`` is off.
    foresight: Foresight | None
    #: The multi-model gateway fronting the configured providers as one
    #: :class:`~friday.providers.llm.LLMProvider`, or ``None`` on the
    #: single-provider/fake builds. When present it IS the orchestrator's ``llm``
    #: and backs the ``/models`` control surface (list / switch / compare).
    gateway: ModelGateway | None
    #: The model catalog scoped to the available providers, or ``None`` when no
    #: gateway is wired. The ``/models`` routes list + validate ids against it.
    model_catalog: ModelCatalog | None
    #: The per-turn cost/latency budgeter wired into the orchestrator's turn loop,
    #: or ``None`` when ``enable_budgeter`` is off. When present (and a gateway is
    #: wired) a hot turn downshifts the gateway's active model.
    budgeter: Budgeter | None
    #: The calibrated confidence scorer wired into the orchestrator, or ``None``
    #: when ``enable_confidence`` is off. When present the orchestrator stamps
    #: ``scratchpad["confidence"]`` after synthesis and caveats below threshold.
    confidence: ConfidenceScorer | None


def build_runtime(settings: Settings, *, repo_root: str = ".") -> AppRuntime:
    """Assemble the full runtime graph from ``settings``.

    Constructs the process-wide observability stores first, injects the audit +
    metrics into the shared tool registry (so every tool call is audited and
    counted) and the tracer + metrics into the orchestrator (so every turn opens a
    trace and bumps the request/by-mode counters), and returns everything bundled
    in an :class:`AppRuntime` so the admin API can read the same instances.

    The Stage-1 security spine is wired here too: a tamper-evident, hash-chained
    :class:`~friday.broker.HashChainedAudit` ledger (over ``audit_ledger_path``)
    is injected into the shared registry so every executed tool call ALSO appends
    one tamper-evident record (additive — the in-memory audit is untouched); a
    secret vault is constructed per ``secret_vault`` (lazy; ``env`` default); the
    plaintext-secret self-check runs over ``repo_root`` when enabled (warn-only,
    never refusing boot); and an action :class:`~friday.broker.Broker` over the
    registry + ledger + vault is constructed and surfaced. Routing dispatch THROUGH
    the broker is opt-in via ``enable_broker`` (off by default → dispatch
    unchanged); ``repo_root`` is the directory the self-check scans (the process
    working directory by default; tests pin a tmp path).
    """
    tracer = Tracer()
    audit = AuditLog()
    metrics = Metrics()
    flag_overrides: dict[str, bool] = {}

    # Security spine: the tamper-evident ledger + secret vault are built first so
    # the registry can append a hash-chained record per tool call and the broker
    # can resolve ``{{secret:NAME}}`` markers. The self-check is warn-only.
    hash_audit = _build_audit_ledger(settings)
    secret_vault = _build_secret_vault(settings)
    _run_secret_self_check(settings, repo_root)

    reminder_store = _build_reminder_store(settings)
    trigger_store = _build_trigger_store(settings)
    registry = _build_registry(
        settings, reminder_store, audit=audit, metrics=metrics, hash_audit=hash_audit
    )
    # The action broker over the shared registry + ledger + vault. Always built
    # (uniform wiring) and surfaced on the runtime, but only interposed into
    # dispatch when ``enable_broker`` is on — off by default, the orchestrator
    # keeps the plain registry path, so existing dispatch behaviour is unchanged.
    broker = Broker(
        registry,
        hash_audit,
        secret_provider=_BrokerSecretAdapter(secret_vault),
        egress_policy=_build_egress_policy(settings),
    )
    # The multi-model gateway: built from whichever provider keys are set when a
    # gateway is selected (``FRIDAY_LLM_PROVIDER == "gateway"`` or an
    # OpenRouter/OpenCode key on a non-fake build), else ``None``. When present it
    # IS the live LLM (a drop-in ``LLMProvider`` that resolves a turn to the active
    # model and falls back on error); otherwise the existing single-provider /
    # fallback / fake selection stands, exactly as before. The gateway + catalog
    # are also surfaced for the ``/models`` control surface.
    providers = _build_providers(settings)
    gateway_pair = _build_gateway(settings, providers)
    if gateway_pair is not None:
        gateway, model_catalog = gateway_pair
        primary_llm: LLMProvider = gateway
    else:
        gateway = None
        model_catalog = None
        primary_llm = _build_llm(settings)
    # Then engage strict offline mode when configured: ``select_llm`` swaps in the
    # network-free OfflineLLM when ``enable_offline_mode`` is on, so every consumer
    # (orchestrator, agents, briefing/journal/studio/meeting/n8n) shares one
    # provider and no outbound LLM call is ever attempted. Off by default,
    # ``primary_llm`` (the gateway or the single provider) passes through.
    llm = select_llm(settings, primary_llm)
    # PII redaction (security spine): when enabled, wrap the selected provider so
    # high-confidence PII is scrubbed from outbound messages before any real
    # provider sees them. Off by default, so the provider passes through unwrapped.
    if settings.enable_pii_redaction:
        llm = RedactingLLM(llm)
    embedder = _build_embedder(settings)
    # Postgres-or-SQLite store selection (lazy; the PG adapters only import the
    # driver / validate the DSN when ``enable_postgres`` is on).
    long_term = _select_long_term(settings)
    vector = _select_vector(settings, embedder)
    # The knowledge-graph store is built BEFORE the registry's read-only idea-batch
    # tools because the entity dossier reads it (graph + long-term) — both are
    # already constructed here, so the dossier is wired against the SAME shared
    # stores the rest of the runtime uses.
    graph_store = _build_graph_store(settings)
    graph_extractor = EntityExtractor(llm)
    # Stage-2 idea-batch tools: the read-only ones (capabilities / ask_user /
    # entity_dossier / infofeed / browser) are registered into the shared registry
    # when ``enable_extra_tools`` is on (default), and the side-effecting ones
    # (downloads_butler / media) ride their own readiness flags. Their own gates
    # (read-only, or the registry confirm-step) keep them safe.
    _register_extra_tools(settings, registry, graph_store, long_term)
    _register_side_effecting_extra_tools(settings, registry)
    agents = _build_agents(settings, registry, llm, vector, long_term)
    # Add the read-only idea-batch tools to the fitting agents' allow-lists
    # (capabilities/ask_user to all; entity_dossier to knowledge; infofeed/browser
    # to research). Inert when ``enable_extra_tools`` is off (class defaults stand).
    _extend_agent_extra_tools(settings, agents)
    briefing = _build_briefing_service(
        settings, reminder_store, audit, metrics, llm
    )
    journal_store = _build_journal_store(settings)
    journal_service = _build_journal_service(
        reminder_store, audit, metrics, llm, settings.owner_address
    )
    system_monitor = _build_system_monitor(settings)
    # Stage-2 perception extras: the desktop controller, advisory owner recognition,
    # and the proactive engines — each ``None`` unless its own flag is set, so the
    # offline default constructs no seam. The anomaly detector is wired into the
    # scheduler's ``system_check`` action so a CPU spike is flagged proactively.
    desktop = _build_desktop(settings, hash_audit)
    owner_identity = _build_owner_identity(settings)
    anomaly_detector = _build_anomaly_detector(settings)
    foresight = _build_foresight(settings)
    scheduler = _build_scheduler(
        trigger_store,
        reminder_store,
        registry,
        briefing,
        journal_service,
        journal_store,
        system_monitor,
        anomaly_detector,
    )
    protocol_store = _build_protocol_store(settings)
    protocol_runner = _build_protocol_runner(registry)
    rag_ingestor = _build_rag_ingestor(settings, vector, long_term)
    meeting_store = _build_meeting_store(settings)
    meeting_capture = _build_meeting_capture(settings, llm, rag_ingestor)
    study_store = _build_study_store(settings)
    critic = _build_critic(llm)
    n8n_service = _build_n8n_service(settings, llm)
    # The persona roster (FRIDAY + eight least-privilege specialists). Always built
    # (no flag); the canonical pre-built ROSTER is reused so the registry instance
    # is shared with ``GET /roster`` and the orchestrator's address-by-name hook.
    # When ``custom_operators`` are configured, the canonical built-ins are merged
    # with the parsed customs into a fresh registry (built-ins win any name
    # collision). Parsing is wrapped so a malformed value never crashes boot: any
    # error is logged and the unmodified ROSTER stands.
    roster: RosterRegistry = ROSTER
    if settings.custom_operators:
        try:
            customs = parse_custom_operators(settings.custom_operators)
            merged = merge_personas(ROSTER_PERSONAS, customs)
            roster = RosterRegistry(merged)
            logger.info(
                "merged custom operators into the roster",
                extra={"count": len(merged) - len(ROSTER_PERSONAS)},
            )
        except Exception as exc:  # noqa: BLE001 - malformed config must not crash boot
            logger.warning("skipping invalid FRIDAY_CUSTOM_OPERATORS: %s", exc)
            roster = ROSTER
    # The per-turn budgeter + calibrated confidence scorer (Wave 0). Each is
    # ``None`` unless its own flag is set, so the offline default constructs
    # neither seam. The budgeter governs gateway downshift; the confidence scorer
    # is injected into the orchestrator, which stamps + caveats post-synthesis.
    budgeter = _build_budgeter(settings)
    confidence = _build_confidence(settings)
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
        n8n_service=n8n_service,
        roster=roster,
        confidence=confidence,
        budgeter=budgeter,
        gateway=gateway,
        budget_downshift_model_id=settings.budget_downshift_model_id,
        compaction=_build_compactor(settings, llm),
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
        journal_service=journal_service,
        journal_store=journal_store,
        protocol_store=protocol_store,
        protocol_runner=protocol_runner,
        rag_ingestor=rag_ingestor,
        meeting_store=meeting_store,
        meeting_capture=meeting_capture,
        graph_store=graph_store,
        graph_extractor=graph_extractor,
        study_store=study_store,
        system_monitor=system_monitor,
        n8n_service=n8n_service,
        hash_audit=hash_audit,
        broker=broker,
        roster=roster,
        desktop=desktop,
        owner_identity=owner_identity,
        anomaly_detector=anomaly_detector,
        foresight=foresight,
        gateway=gateway,
        model_catalog=model_catalog,
        budgeter=budgeter,
        confidence=confidence,
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


def _wire_ensemble(app: FastAPI, settings: Settings, llm: LLMProvider) -> None:
    """Stash an :class:`Ensemble` on ``app.state`` when ensemble/debate is enabled.

    The ``/ensemble`` route reads ``app.state.ensemble`` and 404s when absent. Built
    over the same LLM the chat loop uses; off by default so the offline build
    constructs no debate seam.
    """
    if not settings.enable_ensemble:
        return
    app.state.ensemble = Ensemble(llm)


def _wire_planner(app: FastAPI, settings: Settings, llm: LLMProvider) -> None:
    """Stash a :class:`Planner` on ``app.state`` when the planner is enabled.

    The ``/planner`` route reads ``app.state.planner`` and 404s when absent. Built
    over the same LLM; off by default so the offline build constructs no planner.
    """
    if not settings.enable_planner:
        return
    app.state.planner = Planner(llm)


def _wire_contradiction(app: FastAPI, settings: Settings, llm: LLMProvider) -> None:
    """Stash a :class:`ContradictionDetector` on ``app.state`` when enabled.

    The ``/memory/contradiction`` route reads ``app.state.contradiction_detector``
    and 404s when absent. Built over the same LLM; off by default so the offline
    build constructs no detector.
    """
    if not settings.enable_contradiction:
        return
    app.state.contradiction_detector = ContradictionDetector(llm)


def _wire_rag(app: FastAPI, settings: Settings, runtime: AppRuntime) -> None:
    """Stash a :class:`DocumentIngestor` on ``app.state`` when RAG is enabled.

    The ingestor reuses the *shared* runtime stores — the same persistent vector
    store the Knowledge agent retrieves from and the same long-term store — so an
    ingested document is immediately answerable via the existing knowledge path
    with citations. Building it only when ``enable_rag`` is set keeps the offline
    default untouched (the ``/rag`` routes self-guard on the flag and 404 when
    off). No retrieval logic is duplicated; this only adds the write/forget seam.
    """
    if not settings.enable_rag or runtime.rag_ingestor is None:
        return
    app.state.rag_ingestor = runtime.rag_ingestor


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


def _wire_journal(app: FastAPI, settings: Settings, runtime: AppRuntime) -> None:
    """Stash the shared journal service + store on ``app.state`` when enabled.

    The ``/journal`` routes read ``app.state.journal_service`` — the *same*
    :class:`JournalService` the scheduler ``journal`` action builds through — and
    ``app.state.journal_store`` — the *same* :class:`SQLiteJournalStore` that
    action saves into — so an on-demand HTTP build and the proactive scheduled one
    assemble + persist through one service + store. Building it only when
    ``enable_journal`` is set keeps the offline default untouched (the routes
    self-guard on the flag and 404 when off).
    """
    if not settings.enable_journal:
        return
    app.state.journal_service = runtime.journal_service
    app.state.journal_store = runtime.journal_store


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


def _wire_meetings(app: FastAPI, settings: Settings, runtime: AppRuntime) -> None:
    """Stash the shared meeting capture + store on ``app.state`` when enabled.

    The ``/meetings`` routes read ``app.state.meeting_capture`` (the shared
    pipeline over the shared STT + live LLM + the RAG ingestor when available) and
    ``app.state.meeting_store`` (the shared SQLite notes store). Building it only
    when ``enable_meetings`` is set keeps the offline default untouched (the routes
    self-guard on the flag and 404 when off).
    """
    if not settings.enable_meetings:
        return
    app.state.meeting_capture = runtime.meeting_capture
    app.state.meeting_store = runtime.meeting_store


def _wire_graph(app: FastAPI, settings: Settings, runtime: AppRuntime) -> None:
    """Stash the shared graph store + extractor on ``app.state`` when enabled.

    The ``/graph`` routes read ``app.state.graph_store`` (the shared SQLite
    entity/relation store) and ``app.state.graph_extractor`` (the entity extractor
    over the live LLM); the entity card additionally reads long-term facts from the
    already-stashed ``app.state.long_term``. Building it only when
    ``enable_knowledge_graph`` is set keeps the offline default untouched (the
    routes self-guard on the flag and 404 when off).
    """
    if not settings.enable_knowledge_graph:
        return
    app.state.graph_store = runtime.graph_store
    app.state.graph_extractor = runtime.graph_extractor


def _wire_study(app: FastAPI, settings: Settings, runtime: AppRuntime) -> None:
    """Stash the shared study store on ``app.state`` when study is enabled.

    The ``/study`` routes read ``app.state.study_store`` — the shared
    :class:`SQLiteStudyStore` for spaced-repetition flashcards + logged study
    sessions. Building it only when ``enable_study`` is set keeps the offline
    default untouched (the routes self-guard on the flag and 404 when off).
    """
    if not settings.enable_study:
        return
    app.state.study_store = runtime.study_store


def _wire_system(app: FastAPI, settings: Settings, runtime: AppRuntime) -> None:
    """Stash the shared system monitor on ``app.state`` when monitoring is enabled.

    The ``/system`` routes read ``app.state.system_monitor`` — the *same*
    :class:`~friday.system.monitor.SystemMonitor` (over the real ``PsutilSampler``)
    the scheduler ``system_check`` action samples — so the on-demand HTTP stats /
    check and the proactive scheduled breach alert read one monitor with one set of
    thresholds. Building it only when ``enable_system_monitor`` is set keeps the
    offline default untouched (the routes self-guard on the flag and 404 when off).
    """
    if not settings.enable_system_monitor:
        return
    app.state.system_monitor = runtime.system_monitor


def _wire_n8n(app: FastAPI, settings: Settings, runtime: AppRuntime) -> None:
    """Stash the shared n8n service on ``app.state`` when n8n is enabled.

    The ``/n8n`` routes read ``app.state.n8n_service`` — the *same*
    :class:`~friday.n8n.service.N8nService` the orchestrator's "make a workflow on
    n8n" hook reaches — so an HTTP-driven draft and a spoken request operate on one
    service (one REST client + drafter + docker auto-start). Building it only when
    ``enable_n8n`` is set keeps the offline default untouched (the routes
    self-guard on the flag and 404 when off).
    """
    if not settings.enable_n8n:
        return
    app.state.n8n_service = runtime.n8n_service


def _build_perception_service() -> PerceptionService:
    """Assemble the :class:`PerceptionService` from the offline fakes.

    Built entirely from fakes (:class:`FakeVision` / :class:`FakeOCR` /
    :class:`FakeClipboard` / :class:`FakeScreen`) so the perception surface boots
    with zero heavy libraries (opencv/ultralytics, pytesseract/pillow, pyperclip,
    mss are all kept OUT of the uv lock). The real adapters are constructed only
    lazily/when explicitly configured (each lazy-imports its backend and raises a
    ``make install-perception`` hint when absent), so this default construction
    performs no capture, clipboard access, or model load. Privacy-heavy by design:
    the whole surface is gated behind ``enable_perception``, so this is only wired
    onto ``app.state`` when the flag is on.
    """
    return PerceptionService(
        vision=FakeVision(),
        ocr=FakeOCR(),
        clipboard=FakeClipboard(),
        screen=FakeScreen(),
    )


def _wire_perception(app: FastAPI, settings: Settings) -> None:
    """Stash the :class:`PerceptionService` on ``app.state`` when enabled.

    The ``/perception`` routes read ``app.state.perception``; building it only
    when ``enable_perception`` is set keeps the offline default untouched (the
    routes self-guard on the flag and 404 when off) and — since perception can
    READ THE SCREEN AND CLIPBOARD — never constructs the seam unless explicitly
    asked for. The service is composed from fakes (no heavy library) by default.
    """
    if not settings.enable_perception:
        return
    app.state.perception = _build_perception_service()


def _wire_plugins(app: FastAPI, settings: Settings, runtime: AppRuntime) -> None:
    """Load owner-supplied plugins into the shared registry when enabled.

    The built-in tools are already registered (``build_runtime`` -> ``_build_registry``
    runs *before* this), so :func:`~friday.plugins.loader.load_into` registers each
    plugin's tools into the *same* shared :class:`ToolRegistry` while rejecting any
    name that collides with a built-in — guaranteeing a plugin can never shadow a
    built-in. The resulting :class:`PluginInfo` list is stashed on
    ``app.state.plugins`` so the ``/plugins`` route can report what loaded (and what
    failed). Loading is per-plugin isolated, so a broken plugin is captured and
    skipped, never crashing startup. Skipped entirely when ``enable_plugins`` is
    off (the route self-guards on the flag and 404s when off).
    """
    if not settings.enable_plugins:
        app.state.plugins = []
        return
    plugins: list[PluginInfo] = load_into(runtime.registry, settings.plugins_dir)
    app.state.plugins = plugins
    logger.info(
        "plugins loaded",
        extra={
            "count": len(plugins),
            "errors": sum(1 for info in plugins if info.error is not None),
            "plugins_dir": settings.plugins_dir,
        },
    )


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
    # The in-memory tool-call audit log, exposed so the /protocols/learn route can
    # fold recent tool-calls into a draft macro.
    app.state.audit = runtime.audit
    app.state.flag_overrides = runtime.flag_overrides
    # Security spine: the hash-chained ledger (read back by GET /admin/audit/verify)
    # and the action broker. The ledger is the tamper-evident system-of-record the
    # shared registry appends to per tool call; the broker is exposed for opt-in
    # routing (``enable_broker``) but is not interposed into dispatch by default.
    app.state.hash_audit = runtime.hash_audit
    app.state.broker = runtime.broker
    # Stage-2 roster: always surfaced (no flag) so ``GET /roster`` reads the SAME
    # registry the orchestrator's address-by-name hook uses.
    app.state.roster = runtime.roster
    # Stage-2 perception extras — surfaced for completeness; each is ``None`` unless
    # its own flag is set, so the offline default exposes no desktop/voiceprint/
    # proactive seam.
    app.state.desktop = runtime.desktop
    app.state.owner_identity = runtime.owner_identity
    app.state.anomaly_detector = runtime.anomaly_detector
    app.state.foresight = runtime.foresight
    # The multi-model gateway + catalog back the ``/models`` control surface (and
    # the per-turn ``model`` override in ``/chat``). Both are ``None`` on the
    # single-provider/fake builds, so those routes self-guard to 404.
    app.state.gateway = runtime.gateway
    app.state.model_catalog = runtime.model_catalog
    _wire_studio(app, settings, runtime.llm)
    _wire_ensemble(app, settings, runtime.llm)
    _wire_planner(app, settings, runtime.llm)
    _wire_contradiction(app, settings, runtime.llm)
    _wire_rag(app, settings, runtime)
    _wire_reminders(app, settings, runtime)
    _wire_scheduler(app, settings, runtime)
    _wire_briefing(app, settings, runtime)
    _wire_journal(app, settings, runtime)
    _wire_protocols(app, settings, runtime)
    _wire_meetings(app, settings, runtime)
    _wire_graph(app, settings, runtime)
    _wire_study(app, settings, runtime)
    _wire_system(app, settings, runtime)
    _wire_n8n(app, settings, runtime)
    _wire_perception(app, settings)
    # Plugins load LAST so every built-in tool is already registered (built-ins
    # win name collisions); the loaded PluginInfo list lands on app.state.plugins.
    _wire_plugins(app, settings, runtime)


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
        _warn_if_exposed_without_auth(settings)
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
    # Multi-model gateway control surface — always registered but self-guards on
    # the presence of a gateway (built only when OpenRouter/OpenCode keys exist or
    # FRIDAY_LLM_PROVIDER == "gateway"); with no gateway every /models route is
    # 404, so the offline fake build exposes no model surface.
    app.include_router(models_router)
    app.include_router(ensemble_router)
    app.include_router(planner_router)
    app.include_router(memory_router)
    # Voice endpoints are always registered but self-guard on FRIDAY_ENABLE_VOICE
    # (404 / socket refusal when off), so the offline default exposes no voice UX.
    app.include_router(voice_router)
    app.include_router(ws_router)
    # The admin/observability control plane (Phase 5, Stage 2A).
    app.include_router(admin_router)
    # The persona roster listing (Stage 2) — always available (no flag); a pure
    # read-only listing of FRIDAY + the eight least-privilege specialists.
    app.include_router(roster_router)
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
    # Auto-journaling (Tier 2) — always registered but self-guards on
    # FRIDAY_ENABLE_JOURNAL (404 when off), so the offline default exposes no
    # journal surface. The shared journal service + store are wired onto app.state
    # only when enabled, and the scheduler "journal" action saves into the same
    # store.
    app.include_router(journal_router)
    # Voice protocols (Tier 1) — always registered but self-guards on
    # FRIDAY_ENABLE_PROTOCOLS (404 when off), so the offline default exposes no
    # protocol surface. The shared protocol store + runner are wired onto
    # app.state only when enabled.
    app.include_router(protocols_router)
    # Meeting capture (Tier 1) — always registered but self-guards on
    # FRIDAY_ENABLE_MEETINGS (404 when off), so the offline default exposes no
    # meeting surface. The shared meeting capture pipeline + notes store are wired
    # onto app.state only when enabled.
    app.include_router(meetings_router)
    # Knowledge graph / entity cards (Tier 2) — always registered but self-guards
    # on FRIDAY_ENABLE_KNOWLEDGE_GRAPH (404 when off), so the offline default
    # exposes no graph surface. The shared graph store + entity extractor are wired
    # onto app.state only when enabled.
    app.include_router(graph_router)
    # Study / productivity (Tier 2) — always registered but self-guards on
    # FRIDAY_ENABLE_STUDY (404 when off), so the offline default exposes no study
    # surface. The shared study store is wired onto app.state only when enabled.
    app.include_router(study_router)
    # Hardware / system monitoring (Tier 2) — always registered but self-guards on
    # FRIDAY_ENABLE_SYSTEM_MONITOR (404 when off), so the offline default exposes no
    # system surface. The shared SystemMonitor (over the real PsutilSampler) is
    # wired onto app.state only when enabled, and the scheduler "system_check"
    # action samples the same monitor.
    app.include_router(system_router)
    # Plugins / extensions (Tier 2) — always registered but self-guards on
    # FRIDAY_ENABLE_PLUGINS (404 when off), so the offline default exposes no
    # plugin surface. The plugins are loaded into the shared registry and the
    # resulting PluginInfo list is stashed on app.state only when enabled.
    app.include_router(plugins_router)
    # Perception (Tier 2; PRIVACY-HEAVY) — always registered but self-guards on
    # FRIDAY_ENABLE_PERCEPTION (404 when off), so the offline default exposes no
    # perception surface. The PerceptionService (built from fakes by default) is
    # wired onto app.state only when enabled; perception can read the screen and
    # clipboard, so it never constructs that seam unless the flag is on.
    app.include_router(perception_router)
    # n8n integration (Tier 2) — always registered but self-guards on
    # FRIDAY_ENABLE_N8N (404 when off), so the offline default exposes no n8n
    # surface. The shared N8nService (REST client + LLM drafter + docker
    # auto-start) is wired onto app.state only when enabled.
    app.include_router(n8n_router)
    # Tier-3 feature fan-out. Each router self-guards on its own ``FRIDAY_ENABLE_*``
    # flag (404 when off, carrying the feature's own "disabled" detail), so they
    # are all included UNCONDITIONALLY here — the offline default exposes none of
    # them as live surfaces, yet every route is registered. No app.state wiring is
    # required: each route reads its config lazily from get_settings() and (where a
    # backend is needed) builds a per-request httpx client from a SecretStr token.
    #
    # Maps (Photorealistic 3D globe) — FRIDAY_ENABLE_MAPS.
    app.include_router(maps_router)
    # Presence (which known devices are nearby) — FRIDAY_ENABLE_PRESENCE.
    app.include_router(presence_router)
    # Market data (Dhan quotes) — FRIDAY_ENABLE_MARKET_DATA.
    app.include_router(market_router)
    # Calendar (Google Calendar v3) — FRIDAY_ENABLE_CALENDAR.
    app.include_router(calendar_router)
    # Email (Gmail inbox/draft) — FRIDAY_ENABLE_EMAIL.
    app.include_router(email_router)
    # Comms (Twilio SMS/WhatsApp) — FRIDAY_ENABLE_COMMS.
    app.include_router(comms_router)
    # HUD (arc-reactor cockpit page) — FRIDAY_ENABLE_HUD.
    app.include_router(hud_router)
    # Family sharing (consent-enforced) — FRIDAY_ENABLE_FAMILY_SHARING.
    app.include_router(family_router)
    # The PWA shell (manifest + service worker + offline page) — ALWAYS included,
    # no feature flag: installing a Progressive Web App is harmless and the static
    # shell carries no secrets, so a fresh install is reachable + installable even
    # on the offline default. Served at ROOT scope so the service worker controls
    # the whole origin (the dashboard/HUD).
    app.include_router(pwa_router)

    # Build the runtime eagerly too so a TestClient that does not trigger the
    # lifespan (or any direct create_app() user) still has a working app. The
    # lifespan rebuild is harmless and keeps per-process config fresh.
    settings = get_settings()
    _warn_if_exposed_without_auth(settings)
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
