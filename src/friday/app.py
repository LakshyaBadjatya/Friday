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

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
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
from friday.api.routes_chat import router as chat_router
from friday.api.routes_health import router as health_router
from friday.api.routes_rag import router as rag_router
from friday.api.routes_studio import STATIC_DIR as STUDIO_STATIC_DIR
from friday.api.routes_studio import router as studio_router
from friday.api.routes_voice import router as voice_router
from friday.api.ws import router as ws_router
from friday.config import Settings, get_settings
from friday.core.orchestrator import Orchestrator
from friday.logging import configure_logging, get_logger
from friday.memory.long_term import SQLiteLongTermStore
from friday.memory.short_term import ShortTermMemory
from friday.memory.vector import SQLiteVectorStore
from friday.observability.audit import AuditLog
from friday.observability.metrics import Metrics
from friday.observability.tracing import Tracer
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
from friday.studio.generator import (
    MeshyText3D,
    ProceduralGenerator,
    StudioService,
    Text3DProvider,
)
from friday.tools.base import Tool
from friday.tools.home import HomeControlTool
from friday.tools.notify import NotifyTool
from friday.tools.registry import ToolRegistry
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


def _build_registry(
    audit: AuditLog | None = None, metrics: Metrics | None = None
) -> ToolRegistry:
    """Build the tool registry with every Phase-2 tool registered.

    Registers the keyless web search tool plus the side-effecting notify and home
    tools (their own flag/allow-list/confirm gates keep them safe). Each tool
    satisfies the ``Tool`` protocol structurally, but a concrete ``args_model``
    (``type[SomeArgs]``) trips the protocol's invariant ``type[BaseModel]`` field
    under nominal checking; cast to the protocol to register.

    When an :class:`~friday.observability.audit.AuditLog` / :class:`Metrics` are
    passed (the app wires the process-wide instances), every ``execute`` records a
    redacted tool-call audit row and bumps the ``tool_calls`` counter (build-spec
    §11). They default to ``None`` so a bare registry behaves exactly as before.
    """
    registry = ToolRegistry(audit=audit, metrics=metrics)
    registry.register(cast(Tool, WebSearchTool()))
    registry.register(cast(Tool, NotifyTool()))
    registry.register(cast(Tool, HomeControlTool()))
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
    * ``automation`` — bounded multi-step job executor (no tools).
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
    agents.register(AutomationAgent())
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

    registry = _build_registry(audit=audit, metrics=metrics)
    llm = _build_llm(settings)
    embedder = _build_embedder(settings)
    long_term = _build_long_term(settings)
    vector = _build_vector(settings, embedder)
    agents = _build_agents(settings, registry, llm, vector, long_term)
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
        yield
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
