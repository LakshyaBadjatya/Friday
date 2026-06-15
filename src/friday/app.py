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
from pathlib import Path
from typing import cast

from fastapi import FastAPI

from friday.agents.alerting import AlertingAgent
from friday.agents.analysis import AnalysisAgent
from friday.agents.automation import AutomationAgent
from friday.agents.base import AgentRegistry
from friday.agents.device import DeviceAgent
from friday.agents.knowledge import KnowledgeAgent
from friday.api.routes_chat import router as chat_router
from friday.api.routes_health import router as health_router
from friday.api.routes_voice import router as voice_router
from friday.api.ws import router as ws_router
from friday.config import Settings, get_settings
from friday.core.orchestrator import Orchestrator
from friday.logging import configure_logging, get_logger
from friday.memory.long_term import SQLiteLongTermStore
from friday.memory.short_term import ShortTermMemory
from friday.memory.vector import SQLiteVectorStore
from friday.providers.embeddings import (
    EmbeddingProvider,
    FakeEmbeddings,
    NvidiaEmbeddings,
)
from friday.providers.llm import FakeLLM, LLMProvider, NvidiaNIMProvider
from friday.providers.stt import FakeSTT, FasterWhisperSTT, STTProvider
from friday.providers.tts import FakeTTS, TTSProvider, make_tts
from friday.tools.base import Tool
from friday.tools.home import HomeControlTool
from friday.tools.notify import NotifyTool
from friday.tools.registry import ToolRegistry
from friday.tools.web_search import WebSearchTool

logger = get_logger("friday.app")

# Persona spec ships alongside the package; resolve relative to this file so the
# path is correct regardless of the process working directory.
_PERSONA_PATH = Path(__file__).resolve().parent / "persona" / "friday.md"


def _build_llm(settings: Settings) -> LLMProvider:
    """Select the LLM provider from settings.

    NVIDIA NIM when explicitly configured *and* a key is present; otherwise the
    scripted :class:`FakeLLM` (empty script) so the app always boots without
    credentials or network.
    """
    if settings.llm_provider == "nvidia" and settings.nvidia_api_key is not None:
        logger.info("using NVIDIA NIM LLM provider", extra={"model": settings.nvidia_model})
        return NvidiaNIMProvider(
            api_key=settings.nvidia_api_key.get_secret_value(),
            base_url=settings.nvidia_base_url,
            model=settings.nvidia_model,
            timeout=settings.llm_timeout_seconds,
        )
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


def _build_registry() -> ToolRegistry:
    """Build the tool registry with every Phase-2 tool registered.

    Registers the keyless web search tool plus the side-effecting notify and home
    tools (their own flag/allow-list/confirm gates keep them safe). Each tool
    satisfies the ``Tool`` protocol structurally, but a concrete ``args_model``
    (``type[SomeArgs]``) trips the protocol's invariant ``type[BaseModel]`` field
    under nominal checking; cast to the protocol to register.
    """
    registry = ToolRegistry()
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


def build_orchestrator(settings: Settings) -> Orchestrator:
    """Assemble a fully-wired :class:`Orchestrator` from ``settings``.

    Builds the shared tool registry, the LLM provider, the local-first SQLite
    long-term + persistent vector stores (with the configured embedder), and the
    populated agent registry. The stores are injected into both the Knowledge
    agent (hybrid grounding) and the orchestrator (write-consent + forget), so
    AUTOMATION / DEVICE_CONTROL / ALERTING turns dispatch to their specialist
    agents and SECURITY_LOCKDOWN runs the lockdown subgraph.
    """
    registry = _build_registry()
    llm = _build_llm(settings)
    embedder = _build_embedder(settings)
    long_term = _build_long_term(settings)
    vector = _build_vector(settings, embedder)
    agents = _build_agents(settings, registry, llm, vector, long_term)
    return Orchestrator(
        llm=llm,
        registry=registry,
        memory=ShortTermMemory(),
        persona_path=_PERSONA_PATH,
        agents=agents,
        long_term=long_term,
        vector=vector,
    )


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


def create_app() -> FastAPI:
    """Construct the FRIDAY FastAPI application.

    The orchestrator is built in the lifespan startup so configuration is read
    once per process and the dependency graph is shared across requests.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        configure_logging(json_logs=settings.log_json, level=settings.log_level)
        logger.info("FRIDAY starting up", extra={"llm_provider": settings.llm_provider})
        app.state.settings = settings
        app.state.orchestrator = build_orchestrator(settings)
        _wire_voice(app, settings)
        yield
        logger.info("FRIDAY shutting down")

    app = FastAPI(title="FRIDAY", version="0.1.0", lifespan=lifespan)
    app.include_router(chat_router)
    app.include_router(health_router)
    # Voice endpoints are always registered but self-guard on FRIDAY_ENABLE_VOICE
    # (404 / socket refusal when off), so the offline default exposes no voice UX.
    app.include_router(voice_router)
    app.include_router(ws_router)

    # Build the orchestrator eagerly too so a TestClient that does not trigger
    # the lifespan (or any direct create_app() user) still has a working app.
    # The lifespan rebuild is harmless and keeps per-process config fresh.
    settings = get_settings()
    app.state.settings = settings
    app.state.orchestrator = build_orchestrator(settings)
    _wire_voice(app, settings)

    return app
