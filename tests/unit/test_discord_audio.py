"""Unit tests for turning what was said in a voice channel into words.

No network and no audio: the HTTP call is replaced, because what these pin is
the *request* and the retry, not the transcription. A dropped transcription is
the quietest failure in the whole system — she simply does not reply, which is
indistinguishable from nobody having spoken — so the things worth pinning are
the ones that decide whether an answer is attempted at all.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from friday.discord import audio


def _capture(monkeypatch: pytest.MonkeyPatch, outcomes: list[Any]) -> list[dict]:
    """Answer the transcription call from ``outcomes``, recording each request."""
    seen: list[dict] = []
    calls = {"n": 0}

    class _Resp:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def read(self) -> bytes:
            return self._payload

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_: object) -> None:
            return None

    def _fake_urlopen(request: Any, timeout: float = 0) -> Any:
        seen.append({"url": request.full_url,
                     "body": json.loads(request.data.decode())})
        outcome = outcomes[min(calls["n"], len(outcomes) - 1)]
        calls["n"] += 1
        if isinstance(outcome, Exception):
            raise outcome
        return _Resp(json.dumps(
            {"candidates": [{"content": {"parts": [{"text": outcome}]}}]}
        ).encode())

    monkeypatch.setattr(audio.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(audio.time, "sleep", lambda _s: None)
    return seen


def _http_error(code: int) -> Exception:
    return audio.urllib.error.HTTPError(
        "u", code, "err", {}, None  # type: ignore[arg-type]
    )


def _settings(model: str = "") -> Any:
    class _S:
        gemini_api_key = type("_K", (), {"get_secret_value": lambda self: "k"})()
        stt_model = model

    return _S()


def _speech() -> bytes:
    return b"\x00" * (audio.MIN_SPEECH_BYTES + 2)


def test_a_busy_model_is_asked_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """Someone spoke; losing that to a transient 503 just looks like silence."""
    seen = _capture(monkeypatch, [_http_error(503), "en|hello there"])
    assert audio._gemini_stt("k", "m", b"RIFF") == "en|hello there"
    assert len(seen) == 2


def test_a_rejected_request_is_not_repeated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 400 will be a 400 again, and the speaker is still waiting."""
    seen = _capture(monkeypatch, [_http_error(400)])
    assert audio._gemini_stt("k", "m", b"RIFF") is None
    assert len(seen) == 1


def test_it_gives_up_rather_than_retrying_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _capture(monkeypatch, [_http_error(429)])
    assert audio._gemini_stt("k", "m", b"RIFF") is None
    assert len(seen) == audio._ATTEMPTS


@pytest.mark.asyncio
async def test_the_configured_model_is_used_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _capture(monkeypatch, ["en|hi"])
    await audio.transcribe(_settings("some-other-model"), _speech())
    assert "some-other-model" in seen[0]["url"]


@pytest.mark.asyncio
async def test_an_unset_model_falls_back_to_a_current_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STT_MODEL is unset in every environment, so the fallback IS the model.

    It named 2.5-flash long after the rest of the code had moved on, purely
    because nothing pointed at it. Benchmarked before changing: on a synthesised
    clip 2.5 dropped "right now" from "what is open on my PC right now" on both
    runs, and 3.5 transcribed it exactly, and faster.
    """
    seen = _capture(monkeypatch, ["en|hi"])
    await audio.transcribe(_settings(), _speech())
    assert "gemini-3.5-flash" in seen[0]["url"]
    assert "gemini-2.5" not in seen[0]["url"]


@pytest.mark.asyncio
async def test_language_comes_back_with_the_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture(monkeypatch, ["pl|dzien dobry"])
    assert await audio.transcribe(_settings(), _speech()) == ("pl", "dzien dobry")


@pytest.mark.asyncio
async def test_a_reply_without_the_pipe_is_still_taken_as_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The words are good even when the format instruction was ignored."""
    _capture(monkeypatch, ["hello there"])
    assert await audio.transcribe(_settings(), _speech()) == ("en", "hello there")
