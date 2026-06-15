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

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

from fastapi import FastAPI

from friday.api.routes_chat import router as chat_router
from friday.config import Settings, get_settings
from friday.core.orchestrator import Orchestrator
from friday.logging import configure_logging, get_logger
from friday.memory.short_term import ShortTermMemory
from friday.providers.llm import FakeLLM, LLMProvider, NvidiaNIMProvider
from friday.tools.base import Tool
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
        )
    logger.info("using FakeLLM provider (no network)")
    return FakeLLM(responses=[])


def _build_registry() -> ToolRegistry:
    """Build the tool registry with the keyless web search tool registered."""
    registry = ToolRegistry()
    # ``WebSearchTool`` satisfies the ``Tool`` protocol structurally, but its
    # concrete ``args_model`` (``type[WebSearchArgs]``) trips the protocol's
    # invariant ``type[BaseModel]`` field under nominal checking; cast to the
    # protocol to register it.
    registry.register(cast(Tool, WebSearchTool()))
    return registry


def build_orchestrator(settings: Settings) -> Orchestrator:
    """Assemble a fully-wired :class:`Orchestrator` from ``settings``."""
    return Orchestrator(
        llm=_build_llm(settings),
        registry=_build_registry(),
        memory=ShortTermMemory(),
        persona_path=_PERSONA_PATH,
    )


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
        yield
        logger.info("FRIDAY shutting down")

    app = FastAPI(title="FRIDAY", version="0.1.0", lifespan=lifespan)
    app.include_router(chat_router)

    # Build the orchestrator eagerly too so a TestClient that does not trigger
    # the lifespan (or any direct create_app() user) still has a working app.
    # The lifespan rebuild is harmless and keeps per-process config fresh.
    settings = get_settings()
    app.state.settings = settings
    app.state.orchestrator = build_orchestrator(settings)

    return app
