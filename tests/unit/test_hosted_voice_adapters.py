"""Unit tests for the two voice adapters that need no local model.

``FasterWhisperSTT`` and ``PiperTTSProvider`` both need something on local disk —
weights, a binary — which a small container does not have. :class:`GeminiSTT` and
:class:`EdgeTTSProvider` are the pair that can answer ``POST /voice`` from a
hosted deployment, so what matters here is that they work without any of that:
no model is loaded, no audio device is touched, and the network is faked.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from friday.config import Settings
from friday.errors import ProviderError
from friday.providers.stt import GeminiSTT, STTProvider, make_stt
from friday.providers.tts import (
    EdgeTTSProvider,
    TTSProvider,
    VoiceConfig,
    _edge_rate,
    make_tts,
)


class _FakeResponse:
    """The slice of ``httpx.Response`` the adapter actually uses."""

    def __init__(self, payload: Any, status: int = 200, text: str = "") -> None:
        self._payload = payload
        self.status_code = status
        self.text = text

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise _fake_httpx().HTTPStatusError("boom", response=self)


def _fake_httpx() -> ModuleType:
    """A stand-in ``httpx`` whose error types the adapter can catch."""
    module = sys.modules.get("_fake_httpx_module")
    if module is not None:
        return module

    module = ModuleType("_fake_httpx_module")

    class HTTPError(Exception):
        pass

    class HTTPStatusError(HTTPError):
        def __init__(self, message: str, response: Any) -> None:
            super().__init__(message)
            self.response = response

    module.HTTPError = HTTPError  # type: ignore[attr-defined]
    module.HTTPStatusError = HTTPStatusError  # type: ignore[attr-defined]
    sys.modules["_fake_httpx_module"] = module
    return module


def _install_httpx(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> dict[str, Any]:
    """Install a fake ``httpx`` that returns ``response``, recording the request."""
    seen: dict[str, Any] = {}
    base = _fake_httpx()

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            seen["client_kwargs"] = kwargs

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
            seen["url"] = url
            seen.update(kwargs)
            return response

    module = ModuleType("httpx")
    module.AsyncClient = FakeClient  # type: ignore[attr-defined]
    module.HTTPError = base.HTTPError  # type: ignore[attr-defined]
    module.HTTPStatusError = base.HTTPStatusError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", module)
    return seen


def _gemini_body(text: str) -> dict[str, Any]:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


# --- GeminiSTT ---------------------------------------------------------------


def test_gemini_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        GeminiSTT()


def test_gemini_reads_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k-test")
    assert isinstance(GeminiSTT(), STTProvider)


async def test_gemini_transcribes_and_sends_the_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _install_httpx(monkeypatch, _FakeResponse(_gemini_body("turn on the lights")))

    result = await GeminiSTT(api_key="k-test").transcribe(b"RIFFfake", lang="en")

    assert result.text == "turn on the lights"
    assert result.lang == "en"
    # The audio must actually travel, base64'd, with its container declared.
    part = seen["json"]["contents"][0]["parts"][1]["inline_data"]
    assert part["mime_type"] == "audio/wav"
    assert part["data"] == "UklGRmZha2U="
    assert seen["headers"]["x-goog-api-key"] == "k-test"


async def test_gemini_strips_surrounding_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_httpx(monkeypatch, _FakeResponse(_gemini_body("  hello  \n")))
    result = await GeminiSTT(api_key="k").transcribe(b"a", lang=None)
    assert result.text == "hello"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"candidates": []},
        {"candidates": [{"finishReason": "SAFETY"}]},
        {"candidates": [{"content": {}}]},
        {"candidates": [{"content": {"parts": []}}]},
        "not a dict at all",
    ],
)
async def test_gemini_treats_a_missing_answer_as_silence(
    monkeypatch: pytest.MonkeyPatch, body: Any
) -> None:
    """A blocked or empty candidate list is a response shape, not a crash.

    This is the difference between the voice route replying "she said nothing"
    and it raising a KeyError from inside a request handler.
    """
    _install_httpx(monkeypatch, _FakeResponse(body))
    result = await GeminiSTT(api_key="k").transcribe(b"a", lang=None)
    assert result.text == ""


async def test_gemini_maps_an_http_error_to_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_httpx(monkeypatch, _FakeResponse({}, status=429, text="rate limited"))
    with pytest.raises(ProviderError, match="429"):
        await GeminiSTT(api_key="k").transcribe(b"a", lang=None)


# --- EdgeTTSProvider ---------------------------------------------------------


def _install_edge_tts(
    monkeypatch: pytest.MonkeyPatch, chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    class Communicate:
        def __init__(self, text: str, voice: str, **kwargs: Any) -> None:
            seen["text"] = text
            seen["voice"] = voice
            seen.update(kwargs)

        async def stream(self) -> Any:
            for chunk in chunks:
                yield chunk

    module = ModuleType("edge_tts")
    module.Communicate = Communicate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "edge_tts", module)
    return seen


async def test_edge_concatenates_only_the_audio_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _install_edge_tts(
        monkeypatch,
        [
            {"type": "audio", "data": b"ID3"},
            {"type": "WordBoundary", "offset": 0},
            {"type": "audio", "data": b"rest"},
        ],
    )

    audio = await EdgeTTSProvider().synthesize("hello", VoiceConfig())

    assert audio == b"ID3rest"
    assert seen["voice"] == "en-GB-SoniaNeural"


async def test_edge_lets_an_explicit_voice_win(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_edge_tts(monkeypatch, [{"type": "audio", "data": b"x"}])
    await EdgeTTSProvider().synthesize("hi", VoiceConfig(voice_id="en-IN-NeerjaNeural"))
    assert seen["voice"] == "en-IN-NeerjaNeural"


async def test_edge_treats_the_default_placeholder_as_no_opinion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VoiceConfig's default voice_id is the string "default", not a real voice."""
    seen = _install_edge_tts(monkeypatch, [{"type": "audio", "data": b"x"}])
    await EdgeTTSProvider(voice="en-US-AriaNeural").synthesize("hi", VoiceConfig())
    assert seen["voice"] == "en-US-AriaNeural"


async def test_edge_raises_rather_than_returning_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_edge_tts(monkeypatch, [{"type": "WordBoundary", "offset": 0}])
    with pytest.raises(ProviderError, match="no audio"):
        await EdgeTTSProvider().synthesize("hi", VoiceConfig())


@pytest.mark.parametrize(
    ("speed", "expected"),
    [(1.0, "+0%"), (1.2, "+20%"), (0.85, "-15%")],
)
def test_edge_rate_is_a_signed_percentage(speed: float, expected: str) -> None:
    """edge-tts rejects a bare number, so 1.0 must render as "+0%"."""
    assert _edge_rate(speed) == expected


async def test_edge_passes_the_rate_through(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_edge_tts(monkeypatch, [{"type": "audio", "data": b"x"}])
    await EdgeTTSProvider().synthesize("hi", VoiceConfig(speed=1.1))
    assert seen["rate"] == "+10%"


# --- factories ---------------------------------------------------------------


def test_make_stt_selects_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k-test")
    settings = Settings(stt_provider="gemini", gemini_api_key="k-test")
    assert isinstance(make_stt(settings), GeminiSTT)


def test_make_stt_rejects_an_unknown_name() -> None:
    with pytest.raises(ProviderError, match="unknown FRIDAY_STT_PROVIDER"):
        make_stt(Settings(stt_provider="whisper-large-turbo-v9"))


def test_make_tts_selects_edge() -> None:
    assert isinstance(make_tts(Settings(tts_provider="edge")), EdgeTTSProvider)


def test_make_tts_still_rejects_an_unknown_name() -> None:
    with pytest.raises(ProviderError, match="unknown FRIDAY_TTS_PROVIDER"):
        make_tts(Settings(tts_provider="nope"))


def test_both_satisfy_their_protocols(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert isinstance(GeminiSTT(), STTProvider)
    assert isinstance(EdgeTTSProvider(), TTSProvider)


def test_settings_default_keeps_the_local_adapter() -> None:
    """Changing the default would break every existing local voice install."""
    assert Settings().stt_provider == "faster-whisper"
