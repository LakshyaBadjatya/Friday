"""Unit tests for ``friday.notify`` — smart Telegram share + daily digest.

Importers/callers under test: ``friday.notify.smart_share`` (used by the
``POST /siri/telegram`` route) and ``build_digest`` / ``_news_block`` /
``_weather_block`` (used by the ``/siri/digest`` route). Network is stubbed via
``monkeypatch`` on ``friday.notify._get`` so the tests are offline/deterministic.
No persisted schema. Verbatim instruction: "the bot should enhance the message
and only share required things from the transcript … ask me boss, what to send …
news headlines at morning six … weather forecast … raining and thunderstorming,
and alerts if there are any."
"""

from __future__ import annotations

import asyncio
import json

import friday.notify as notify
from friday.providers.llm import LLMResponse


class _StubLLM:
    """Minimal LLMProvider stub returning a fixed completion text."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def complete(self, messages, tools=None, *, model=None):  # noqa: ANN001, ANN002
        return LLMResponse(text=self._text)


def test_smart_share_short_text_passes_through() -> None:
    msg, question = asyncio.run(notify.smart_share(None, "ping Aryan"))
    assert msg == "ping Aryan"
    assert question is None


def test_smart_share_empty_asks() -> None:
    msg, question = asyncio.run(notify.smart_share(None, "   "))
    assert msg is None
    assert question and "send" in question.lower()


def test_smart_share_extracts_key_content() -> None:
    llm = _StubLLM("Kinetic energy: KE = ½ m v²")
    long = "hey can you send my friend the kinetic energy formula we discussed " * 2
    msg, question = asyncio.run(notify.smart_share(llm, long))
    assert msg == "Kinetic energy: KE = ½ m v²"
    assert question is None


def test_smart_share_ask_when_vague() -> None:
    llm = _StubLLM("ASK: Which formula should I send?")
    long = "send that thing to telegram please it was important " * 2
    msg, question = asyncio.run(notify.smart_share(llm, long))
    assert msg is None
    assert question == "Which formula should I send?"


def test_news_block_parses_titles(monkeypatch) -> None:  # noqa: ANN001
    rss = (
        b"<rss><channel><title>Top stories</title>"
        b"<item><title>Big thing happened - The Hindu</title></item>"
        b"<item><title>Another &amp; bigger thing - TOI</title></item>"
        b"</channel></rss>"
    )
    monkeypatch.setattr(notify, "_get", lambda url: rss)
    block = notify._news_block(5)
    assert block is not None
    assert "Big thing happened" in block
    assert "Another & bigger thing" in block  # entity unescaped, publisher stripped
    assert "Top stories" not in block  # channel title skipped


def test_weather_block_flags_thunderstorm(monkeypatch) -> None:  # noqa: ANN001
    j1 = {
        "current_condition": [
            {
                "temp_C": "28",
                "FeelsLikeC": "31",
                "weatherDesc": [{"value": "Cloudy"}],
            }
        ],
        "weather": [
            {
                "maxtempC": "33",
                "mintempC": "24",
                "hourly": [
                    {"chanceofrain": "70", "chanceofthunder": "60"},
                    {"chanceofrain": "20", "chanceofthunder": "10"},
                ],
            }
        ],
        "nearest_area": [{"areaName": [{"value": "Kota"}]}],
    }
    monkeypatch.setattr(notify, "_get", lambda url: json.dumps(j1).encode())
    block = notify._weather_block("25.17", "75.84")
    assert block is not None
    assert "Kota" in block
    assert "High 33°C, low 24°C" in block
    assert "Thunderstorm likely" in block
    assert "Rain likely" in block


def test_build_digest_composes(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(notify, "_weather_block", lambda lat, lon: "🌤 weather here")
    monkeypatch.setattr(notify, "_news_block", lambda: "📰 news here")
    out = asyncio.run(notify.build_digest("1", "2"))
    assert "Good morning" in out
    assert "weather here" in out
    assert "news here" in out


def test_build_digest_handles_dead_sources(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(notify, "_weather_block", lambda lat, lon: None)
    monkeypatch.setattr(notify, "_news_block", lambda: None)
    out = asyncio.run(notify.build_digest("", ""))
    assert "try again later" in out
