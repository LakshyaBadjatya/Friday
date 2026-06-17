"""Shared pytest fixtures for the FRIDAY test suite.

Provides the provider fakes (``fake_llm``, ``fake_stt``, ``fake_tts``) and an
env-isolated ``settings`` fixture so unit tests never read the developer's real
``.env`` or process environment.

``fake_llm`` imports :class:`friday.providers.llm.FakeLLM` lazily (inside the
fixture body) so test collection never breaks while that slice is still being
built; the import error only surfaces in a test that actually requests the
fixture.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from friday.config import Settings, get_settings
from friday.providers.stt import FakeSTT
from friday.providers.tts import FakeTTS

if TYPE_CHECKING:
    from friday.providers.llm import FakeLLM


# Environment variables that could otherwise bleed real configuration into the
# isolated settings fixture.
_LEAKY_ENV_PREFIXES = ("FRIDAY_", "NVIDIA_")


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Clear the process-global ``get_settings`` lru-cache around every test.

    ``friday.config.get_settings`` is ``@lru_cache``d, so a :class:`Settings`
    built by one test (e.g. an auth/RBAC test that sets ``FRIDAY_REQUIRE_AUTH``)
    would otherwise survive in the cache and leak into the next test's
    ``create_app()`` — wiring auth middleware where none is expected and turning a
    ``404`` into a ``401``. Clearing before *and* after each test makes every test
    read settings fresh from its own environment, killing this order-dependent
    flakiness. Clearing is always safe: it just forces the next ``get_settings()``
    to re-read the current env (which tests set before they call it).
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fake_llm() -> FakeLLM:
    """An empty scripted :class:`FakeLLM`.

    Tests append responses (or re-construct with a script) as needed. Imported
    lazily so collection survives even if the LLM slice is not yet present.
    """
    from friday.providers.llm import FakeLLM

    return FakeLLM(responses=[])


@pytest.fixture
def fake_stt() -> FakeSTT:
    """A :class:`FakeSTT` returning a deterministic non-empty transcript."""
    return FakeSTT()


@pytest.fixture
def fake_tts() -> FakeTTS:
    """A :class:`FakeTTS` returning deterministic non-empty audio bytes."""
    return FakeTTS()


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """An env-isolated :class:`Settings` instance.

    Strips ``FRIDAY_*`` / ``NVIDIA_*`` from the environment and disables
    ``.env`` loading so the returned settings reflect only in-code defaults,
    keeping tests deterministic across machines.
    """
    import os

    for key in list(os.environ):
        if key.upper().startswith(_LEAKY_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    yield Settings(_env_file=None)
