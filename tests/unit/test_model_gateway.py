# © Lakshya Badjatya — Author
"""Unit tests for the multi-model gateway (``friday.models.gateway``).

The gateway fronts the configured providers as a single drop-in
:class:`~friday.providers.llm.LLMProvider`: it resolves a turn to the active model
(or a per-call override), falls back once on a primary :class:`ProviderError`,
fans out a side-by-side compare (never raising), and asks a judge model to pick
the best answer (non-fatal). All tests use fakes — zero network, no LLM SDK.
"""

from __future__ import annotations

import pytest

from friday.errors import ProviderError
from friday.models.catalog import ModelCatalog
from friday.models.gateway import CompareResult, ModelGateway
from friday.providers.llm import (
    FakeLLM,
    LLMProvider,
    LLMResponse,
    Message,
    ToolSpec,
    Usage,
)


def _resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], usage=Usage())


class RecordingProvider(LLMProvider):
    """A fake provider that records the per-call ``model`` and returns a canned text."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.seen_models: list[str | None] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> LLMResponse:
        self.seen_models.append(model)
        return _resp(self.text)


class BoomProvider(LLMProvider):
    """A fake provider that always raises :class:`ProviderError`."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> LLMResponse:
        self.calls += 1
        raise ProviderError("boom")


def _catalog() -> ModelCatalog:
    return ModelCatalog(available_providers={"openrouter", "opencode", "nvidia"})


def _messages() -> list[Message]:
    return [Message(role="user", content="hi")]


# --------------------------------------------------------------------------- #
# active model + complete
# --------------------------------------------------------------------------- #
async def test_active_model_id_defaults_to_init() -> None:
    gw = ModelGateway(
        providers={"openrouter": RecordingProvider("ok")},
        catalog=_catalog(),
        default_model_id="openrouter:google/gemma-4-31b-it:free",
    )
    assert gw.active_model_id == "openrouter:google/gemma-4-31b-it:free"


async def test_complete_uses_active_model_and_its_provider() -> None:
    provider = RecordingProvider("from-router")
    gw = ModelGateway(
        providers={"openrouter": provider},
        catalog=_catalog(),
        default_model_id="openrouter:google/gemma-4-31b-it:free",
    )
    r = await gw.complete(_messages())
    assert r.text == "from-router"
    # The gateway resolves the active id -> the provider's model slug.
    assert provider.seen_models == ["google/gemma-4-31b-it:free"]


async def test_complete_per_call_model_override() -> None:
    provider = RecordingProvider("ok")
    gw = ModelGateway(
        providers={"openrouter": provider},
        catalog=_catalog(),
        default_model_id="openrouter:google/gemma-4-31b-it:free",
    )
    await gw.complete(_messages(), model="openrouter:openai/gpt-oss-120b:free")
    assert provider.seen_models == ["openai/gpt-oss-120b:free"]


async def test_set_active_changes_the_model() -> None:
    provider = RecordingProvider("ok")
    gw = ModelGateway(
        providers={"openrouter": provider},
        catalog=_catalog(),
        default_model_id="openrouter:google/gemma-4-31b-it:free",
    )
    gw.set_active("openrouter:qwen/qwen3-coder:free")
    assert gw.active_model_id == "openrouter:qwen/qwen3-coder:free"
    await gw.complete(_messages())
    assert provider.seen_models == ["qwen/qwen3-coder:free"]


async def test_complete_falls_back_on_primary_error() -> None:
    primary = BoomProvider()
    fallback = RecordingProvider("from-nvidia")
    gw = ModelGateway(
        providers={"openrouter": primary, "nvidia": fallback},
        catalog=_catalog(),
        default_model_id="openrouter:google/gemma-4-31b-it:free",
        fallback_model_id="nvidia:meta/llama-3.1-8b-instruct",
    )
    r = await gw.complete(_messages())
    assert r.text == "from-nvidia"
    assert primary.calls == 1
    assert fallback.seen_models == ["meta/llama-3.1-8b-instruct"]


async def test_complete_raises_when_no_fallback() -> None:
    gw = ModelGateway(
        providers={"openrouter": BoomProvider()},
        catalog=_catalog(),
        default_model_id="openrouter:google/gemma-4-31b-it:free",
    )
    with pytest.raises(ProviderError):
        await gw.complete(_messages())


async def test_complete_raises_when_fallback_same_as_active() -> None:
    primary = BoomProvider()
    gw = ModelGateway(
        providers={"openrouter": primary},
        catalog=_catalog(),
        default_model_id="openrouter:google/gemma-4-31b-it:free",
        fallback_model_id="openrouter:google/gemma-4-31b-it:free",
    )
    with pytest.raises(ProviderError):
        await gw.complete(_messages())
    # No second attempt — fallback is the same model.
    assert primary.calls == 1


async def test_complete_unknown_model_raises_provider_error() -> None:
    gw = ModelGateway(
        providers={"openrouter": RecordingProvider("ok")},
        catalog=_catalog(),
        default_model_id="openrouter:google/gemma-4-31b-it:free",
    )
    with pytest.raises(ProviderError):
        await gw.complete(_messages(), model="nope:does-not-exist")


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
async def test_compare_fans_out_to_three_providers() -> None:
    gw = ModelGateway(
        providers={
            "openrouter": RecordingProvider("router-answer"),
            "opencode": RecordingProvider("opencode-answer"),
            "nvidia": RecordingProvider("nvidia-answer"),
        },
        catalog=_catalog(),
        default_model_id="openrouter:google/gemma-4-31b-it:free",
    )
    model_ids = [
        "openrouter:google/gemma-4-31b-it:free",
        "opencode:mimo-v2.5-free",
        "nvidia:meta/llama-3.1-8b-instruct",
    ]
    results = await gw.compare(_messages(), model_ids)
    assert len(results) == 3
    assert all(isinstance(r, CompareResult) for r in results)
    by_id = {r.model_id: r for r in results}
    assert by_id[model_ids[0]].text == "router-answer"
    assert by_id[model_ids[0]].ok is True
    assert by_id[model_ids[0]].error is None
    assert by_id[model_ids[0]].latency_ms >= 0
    assert by_id[model_ids[1]].text == "opencode-answer"
    assert by_id[model_ids[2]].text == "nvidia-answer"
    # Order is preserved across the fan-out.
    assert [r.model_id for r in results] == model_ids


async def test_compare_captures_error_without_raising() -> None:
    gw = ModelGateway(
        providers={
            "openrouter": RecordingProvider("good"),
            "opencode": BoomProvider(),
        },
        catalog=_catalog(),
        default_model_id="openrouter:google/gemma-4-31b-it:free",
    )
    model_ids = ["openrouter:google/gemma-4-31b-it:free", "opencode:mimo-v2.5-free"]
    results = await gw.compare(_messages(), model_ids)
    by_id = {r.model_id: r for r in results}
    assert by_id[model_ids[0]].ok is True
    errored = by_id[model_ids[1]]
    assert errored.ok is False
    assert errored.text is None
    assert errored.error is not None
    assert "boom" in errored.error


async def test_compare_unknown_model_is_captured_not_raised() -> None:
    gw = ModelGateway(
        providers={"openrouter": RecordingProvider("good")},
        catalog=_catalog(),
        default_model_id="openrouter:google/gemma-4-31b-it:free",
    )
    results = await gw.compare(_messages(), ["nope:missing"])
    assert len(results) == 1
    assert results[0].ok is False
    assert results[0].error is not None


async def test_compare_uses_clock_for_latency() -> None:
    # A single-model compare measures exactly one start/end pair, so the injected
    # clock's delta (1.0 -> 1.5 == 0.5s) maps to a deterministic 500ms latency.
    ticks = iter([1.0, 1.5])

    def clock() -> float:
        return next(ticks)

    gw = ModelGateway(
        providers={"openrouter": RecordingProvider("a")},
        catalog=_catalog(),
        default_model_id="openrouter:google/gemma-4-31b-it:free",
        clock=clock,
    )
    results = await gw.compare(
        _messages(),
        ["openrouter:google/gemma-4-31b-it:free"],
    )
    assert len(results) == 1
    assert results[0].latency_ms == 500


# --------------------------------------------------------------------------- #
# judge
# --------------------------------------------------------------------------- #
async def test_judge_returns_best_model_id() -> None:
    # The judge provider echoes the winning id.
    judge_provider = FakeLLM(
        responses=[_resp("opencode:mimo-v2.5-free")]
    )
    gw = ModelGateway(
        providers={
            "openrouter": RecordingProvider("a"),
            "opencode": judge_provider,
        },
        catalog=_catalog(),
        default_model_id="openrouter:google/gemma-4-31b-it:free",
    )
    results = [
        CompareResult(
            model_id="openrouter:google/gemma-4-31b-it:free",
            label="Gemma",
            text="weak",
            latency_ms=10,
            ok=True,
            error=None,
        ),
        CompareResult(
            model_id="opencode:mimo-v2.5-free",
            label="MiMo",
            text="strong",
            latency_ms=20,
            ok=True,
            error=None,
        ),
    ]
    best = await gw.judge("which is best?", results, judge_model_id="opencode:mimo-v2.5-free")
    assert best == "opencode:mimo-v2.5-free"


async def test_judge_non_fatal_on_error() -> None:
    gw = ModelGateway(
        providers={"opencode": BoomProvider()},
        catalog=_catalog(),
        default_model_id="openrouter:google/gemma-4-31b-it:free",
    )
    results = [
        CompareResult(
            model_id="opencode:mimo-v2.5-free",
            label="MiMo",
            text="x",
            latency_ms=1,
            ok=True,
            error=None,
        )
    ]
    best = await gw.judge("q", results, judge_model_id="opencode:mimo-v2.5-free")
    assert best is None


async def test_judge_unknown_model_is_none() -> None:
    gw = ModelGateway(
        providers={"openrouter": RecordingProvider("a")},
        catalog=_catalog(),
        default_model_id="openrouter:google/gemma-4-31b-it:free",
    )
    results = [
        CompareResult(
            model_id="openrouter:google/gemma-4-31b-it:free",
            label="Gemma",
            text="a",
            latency_ms=1,
            ok=True,
            error=None,
        )
    ]
    best = await gw.judge("q", results, judge_model_id="nope:missing")
    assert best is None
