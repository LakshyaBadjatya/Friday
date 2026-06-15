"""Unit tests for the real TTS adapters and the :func:`make_tts` factory.

These tests verify the lazy contract from Phase 3 Stage A1:

* Importing ``friday.providers.tts`` must NOT require ``httpx``/``piper`` to be
  present as heavy voice extras (the module import stays light).
* :class:`PiperTTSProvider` raises a helpful :class:`ProviderError` when the
  ``piper`` binary is absent (``shutil.which`` returns ``None``).
* :class:`ElevenLabsTTSProvider` raises a :class:`ProviderError` when no API key
  is available, and lazily POSTs via ``httpx`` (mocked) when it is.
* :func:`make_tts` selects ``piper`` | ``elevenlabs`` | ``fake`` and rejects
  unknown values.
* :class:`FakeTTS` still returns non-empty bytes.

No real binary is executed, no real network call is made: the piper binary is
simulated via ``shutil.which`` / ``subprocess.run`` monkeypatches and the
ElevenLabs HTTP call via a fake ``httpx`` module.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from friday.config import Settings
from friday.errors import ProviderError
from friday.providers.tts import (
    ElevenLabsTTS,
    ElevenLabsTTSProvider,
    FakeTTS,
    PiperTTS,
    PiperTTSProvider,
    TTSProvider,
    VoiceConfig,
    make_tts,
)


# --------------------------------------------------------------------------- #
# Module import is light
# --------------------------------------------------------------------------- #
def test_module_exposes_real_adapters_and_factory() -> None:
    module = importlib.import_module("friday.providers.tts")
    assert hasattr(module, "PiperTTSProvider")
    assert hasattr(module, "ElevenLabsTTSProvider")
    assert hasattr(module, "make_tts")


def test_no_heavy_import_at_module_top_level() -> None:
    spec = importlib.util.find_spec("friday.providers.tts")
    assert spec is not None and spec.origin is not None
    text = open(spec.origin, encoding="utf-8").read()
    for line in text.splitlines():
        stripped = line.lstrip()
        # ``import httpx`` / ``import subprocess`` etc. must be inside a function
        # (indented), never at column 0.
        if line.startswith(("import httpx", "from httpx", "import subprocess")):
            pytest.fail(f"heavy import at module top level: {stripped!r}")


# --------------------------------------------------------------------------- #
# PiperTTSProvider: helpful error when the binary is missing
# --------------------------------------------------------------------------- #
async def test_piper_missing_binary_raises_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    provider = PiperTTSProvider()
    with pytest.raises(ProviderError) as exc:
        await provider.synthesize("hello", VoiceConfig())
    assert "make install-voice" in str(exc.value)


async def test_piper_synthesizes_when_binary_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/piper")

    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        return SimpleNamespace(stdout=b"RIFFpiper-wav-bytes", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = PiperTTSProvider(model_path="/models/voice.onnx")
    out = await provider.synthesize("hi there", VoiceConfig(voice_id="x", speed=2.0))
    assert out == b"RIFFpiper-wav-bytes"
    assert captured["input"] == b"hi there"
    # Model path and length-scale (1/speed) flow into the command.
    assert "/models/voice.onnx" in captured["cmd"]
    assert "--length_scale" in captured["cmd"]
    assert "0.5" in captured["cmd"]


def test_piper_provider_is_tts_provider() -> None:
    assert isinstance(PiperTTSProvider(), TTSProvider)


# --------------------------------------------------------------------------- #
# ElevenLabsTTSProvider: missing key, and a mocked successful call
# --------------------------------------------------------------------------- #
def test_elevenlabs_missing_key_raises_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(ProviderError) as exc:
        ElevenLabsTTSProvider()
    assert "ELEVENLABS_API_KEY" in str(exc.value)


def test_elevenlabs_reads_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test")
    # Construction succeeds (no network yet).
    provider = ElevenLabsTTSProvider()
    assert isinstance(provider, TTSProvider)


class _FakeResponse:
    def __init__(self) -> None:
        self.content = b"ID3-mp3-bytes"

    def raise_for_status(self) -> None:
        return None


class _FakeAsyncClient:
    last_post: dict[str, Any] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        _FakeAsyncClient.last_post = {"url": url, **kwargs}
        return _FakeResponse()


def _install_fake_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("httpx")
    module.AsyncClient = _FakeAsyncClient  # type: ignore[attr-defined]

    class _HTTPError(Exception):
        pass

    class _HTTPStatusError(_HTTPError):
        pass

    module.HTTPError = _HTTPError  # type: ignore[attr-defined]
    module.HTTPStatusError = _HTTPStatusError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", module)


async def test_elevenlabs_synthesizes_via_mocked_httpx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test")
    _install_fake_httpx(monkeypatch)
    provider = ElevenLabsTTSProvider(api_key="sk-explicit")
    out = await provider.synthesize("speak", VoiceConfig(voice_id="rachel"))
    assert out == b"ID3-mp3-bytes"
    posted = _FakeAsyncClient.last_post
    assert posted["url"].endswith("/rachel")
    assert posted["headers"]["xi-api-key"] == "sk-explicit"
    assert posted["json"]["text"] == "speak"


async def test_elevenlabs_missing_httpx_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test")
    # Ensure any cached httpx is removed and the import fails.
    monkeypatch.delitem(sys.modules, "httpx", raising=False)
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "httpx":
            raise ImportError("No module named 'httpx' (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    provider = ElevenLabsTTSProvider()
    with pytest.raises(ProviderError) as exc:
        await provider.synthesize("hi", VoiceConfig())
    assert "make install-voice" in str(exc.value)


# --------------------------------------------------------------------------- #
# make_tts factory
# --------------------------------------------------------------------------- #
def _settings(provider: str) -> Settings:
    return Settings(_env_file=None, tts_provider=provider)


def test_make_tts_fake() -> None:
    assert isinstance(make_tts(_settings("fake")), FakeTTS)


def test_make_tts_piper() -> None:
    assert isinstance(make_tts(_settings("piper")), PiperTTSProvider)


def test_make_tts_elevenlabs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test")
    assert isinstance(make_tts(_settings("elevenlabs")), ElevenLabsTTSProvider)


def test_make_tts_is_case_insensitive() -> None:
    assert isinstance(make_tts(_settings("PIPER")), PiperTTSProvider)


def test_make_tts_unknown_provider_raises() -> None:
    with pytest.raises(ProviderError) as exc:
        make_tts(_settings("espeak"))
    assert "espeak" in str(exc.value)


# --------------------------------------------------------------------------- #
# Fakes / Phase-0 stubs preserved
# --------------------------------------------------------------------------- #
async def test_fake_tts_still_returns_nonempty_bytes() -> None:
    out = await FakeTTS().synthesize("hello", VoiceConfig())
    assert isinstance(out, bytes)
    assert len(out) > 0


def test_phase0_stubs_preserved() -> None:
    assert isinstance(PiperTTS(), TTSProvider)
    assert isinstance(ElevenLabsTTS(), TTSProvider)
