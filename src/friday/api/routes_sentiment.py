# © Lakshya Badjatya — Author
"""``/sentiment`` — flagged offline sentiment analysis (Wave C; default off).

A single ``POST /sentiment`` that scores text mood with the dependency-free
lexicon analyzer (:mod:`friday.nlp.sentiment`). Gated behind
``FRIDAY_ENABLE_SENTIMENT`` (read lazily off
:func:`~friday.config.get_settings` so the router works mounted on a bare
``FastAPI()`` before ``app.py`` wiring exists); when the flag is off the route is
``404`` so the feature simply does not exist for callers (mirroring ``/hud`` and
``/maps``).

The analyzer is pure and stateless, so it is a module-level singleton — no
``app.state`` wiring, no network, no LLM SDK.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from friday.config import get_settings
from friday.nlp.sentiment import SentimentAnalyzer

router = APIRouter()

#: Stateless analyzer reused across requests (pure lexicon scorer).
_ANALYZER = SentimentAnalyzer()


class SentimentRequest(BaseModel):
    """The ``POST /sentiment`` body: the text to score."""

    text: str = Field(min_length=1, max_length=10_000)


def _enabled() -> bool:
    """Whether the sentiment surface is enabled, read lazily from settings."""
    return bool(getattr(get_settings(), "enable_sentiment", False))


@router.post("/sentiment", response_model=None)
async def analyze_sentiment(body: SentimentRequest) -> JSONResponse:
    """Score the text's sentiment; ``404`` when the feature is disabled."""
    if not _enabled():
        return JSONResponse(status_code=404, content={"detail": "sentiment disabled"})
    result = _ANALYZER.analyze(body.text)
    return JSONResponse(status_code=200, content=result.model_dump())
