"""Unit tests for the real STT adapter (:class:`FasterWhisperSTT`).

These tests verify the lazy-import contract from Phase 3 Stage A1:

* Importing ``friday.providers.stt`` must NOT require ``faster_whisper`` (the
  heavy optional dependency stays out of the uv lock).
* Constructing :class:`FasterWhisperSTT` when ``faster_whisper`` is absent must
  raise a :class:`ProviderError` carrying the ``make install-voice`` hint.
* The existing :class:`FakeSTT` still returns a non-empty transcript.

No real model is loaded, no audio device or network is touched: the missing
library is simulated by monkeypatching ``builtins.__import__``.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from collections.abc import Iterator
from types import ModuleType
from typing import Any

import pytest

from friday.errors import ProviderError
from friday.providers.stt import (
    FakeSTT,
    FasterWhisperSTT,
    STTProvider,
    Transcript,
    WhisperSTT,
)


def _block_import(monkeypatch: pytest.MonkeyPatch, blocked: str) -> None:
    """Make ``import blocked`` (and submodules) raise :class:`ImportError`."""
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> ModuleType:
        if name == blocked or name.startswith(blocked + "."):
            raise ImportError(f"No module named {blocked!r} (simulated)")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


# --------------------------------------------------------------------------- #
# Module import is light
# --------------------------------------------------------------------------- #
def test_module_imports_without_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-importing the module with faster_whisper blocked must succeed.

    The original cached module is restored afterwards so the freshly imported
    copy (with new class identities) does not leak into other tests.
    """
    original = sys.modules.get("friday.providers.stt")
    _block_import(monkeypatch, "faster_whisper")
    monkeypatch.delitem(sys.modules, "friday.providers.stt", raising=False)
    try:
        module = importlib.import_module("friday.providers.stt")
        # The class object must be present even though the heavy lib is absent.
        assert hasattr(module, "FasterWhisperSTT")
        assert hasattr(module, "FakeSTT")
    finally:
        if original is not None:
            sys.modules["friday.providers.stt"] = original


def test_faster_whisper_not_imported_at_module_top_level() -> None:
    """faster_whisper must not be a top-level import of the stt module."""
    import friday.providers.stt as stt_module

    source = (
        importlib.util.find_spec("friday.providers.stt").origin  # type: ignore[union-attr]
    )
    assert source is not None
    text = open(source, encoding="utf-8").read()
    # Allowed only inside a function body (indented); never at column 0.
    for line in text.splitlines():
        if line.startswith("import faster_whisper") or line.startswith(
            "from faster_whisper"
        ):
            pytest.fail("faster_whisper is imported at module top level")
    assert stt_module is not None


# --------------------------------------------------------------------------- #
# FasterWhisperSTT: helpful error when the lib is missing
# --------------------------------------------------------------------------- #
def test_faster_whisper_missing_lib_raises_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_import(monkeypatch, "faster_whisper")
    with pytest.raises(ProviderError) as exc:
        FasterWhisperSTT()
    assert "make install-voice" in str(exc.value)


def test_faster_whisper_error_mentions_requirements_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_import(monkeypatch, "faster_whisper")
    with pytest.raises(ProviderError) as exc:
        FasterWhisperSTT(model_size="small")
    assert "requirements-voice.txt" in str(exc.value)


# --------------------------------------------------------------------------- #
# FasterWhisperSTT: works against a fake faster_whisper module
# --------------------------------------------------------------------------- #
class _FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeInfo:
    language = "en"


class _FakeWhisperModel:
    last_kwargs: dict[str, Any] = {}

    def __init__(self, model_size: str, **kwargs: Any) -> None:
        self.model_size = model_size

    def transcribe(self, path: str, language: str | None = None) -> Any:
        return ([_FakeSegment("hello "), _FakeSegment("world")], _FakeInfo())


@pytest.fixture
def fake_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    module = ModuleType("faster_whisper")
    module.WhisperModel = _FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    yield
    # monkeypatch restores sys.modules automatically.


async def test_faster_whisper_transcribes_with_fake_model(
    fake_faster_whisper: None,
) -> None:
    stt = FasterWhisperSTT(model_size="base")
    result = await stt.transcribe(b"RIFFfakewav", lang="en")
    assert isinstance(result, Transcript)
    assert result.text == "hello world"
    assert result.lang == "en"


async def test_faster_whisper_autodetects_language(
    fake_faster_whisper: None,
) -> None:
    stt = FasterWhisperSTT()
    result = await stt.transcribe(b"RIFFfakewav", lang=None)
    # Falls back to the model-reported language when no hint is given.
    assert result.lang == "en"


def test_faster_whisper_is_stt_provider(fake_faster_whisper: None) -> None:
    assert isinstance(FasterWhisperSTT(), STTProvider)


# --------------------------------------------------------------------------- #
# FakeSTT untouched
# --------------------------------------------------------------------------- #
async def test_fake_stt_still_returns_nonempty_text() -> None:
    result = await FakeSTT().transcribe(b"\x00\x01", None)
    assert isinstance(result, Transcript)
    assert result.text


def test_phase0_whisper_stub_preserved() -> None:
    """The Phase-0 placeholder adapter is retained for backwards compat."""
    assert isinstance(WhisperSTT(), STTProvider)
