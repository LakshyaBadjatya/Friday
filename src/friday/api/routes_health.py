"""``GET /health`` — a cheap liveness probe.

Reports that the process is up and which LLM provider/model it is configured for,
*without* making any LLM call. The provider and model are read from the
:class:`~friday.config.Settings` stashed on ``app.state`` at startup (falling back
to the cached process settings), so the probe never touches the network and never
constructs a completion.

``model`` is the configured NVIDIA model when the provider is ``nvidia``; for the
offline ``fake`` provider it is ``null`` (no model is meaningful there).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from friday.config import Settings, get_settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Liveness payload: status plus the configured provider/model."""

    status: str
    llm_provider: str
    model: str | None


def _settings_for(request: Request) -> Settings:
    """Prefer the startup settings on ``app.state``; fall back to the cache."""
    settings = getattr(request.app.state, "settings", None)
    if isinstance(settings, Settings):
        return settings
    return get_settings()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Return liveness and the configured LLM provider/model (no LLM call)."""
    settings = _settings_for(request)
    model = settings.nvidia_model if settings.llm_provider == "nvidia" else None
    return HealthResponse(
        status="ok",
        llm_provider=settings.llm_provider,
        model=model,
    )
