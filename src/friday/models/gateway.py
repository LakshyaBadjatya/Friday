# © Lakshya Badjatya — Author
"""The multi-model gateway — a drop-in :class:`LLMProvider` over many models.

:class:`ModelGateway` fronts the configured providers (keyed by provider name) as a
single :class:`~friday.providers.llm.LLMProvider`, so the orchestrator can keep
depending only on the ``complete`` contract while gaining:

* an *active model* (and a per-call override) resolved through the
  :class:`~friday.models.catalog.ModelCatalog` to the provider + model slug that
  serves it;
* a single *fallback* on a primary :class:`~friday.errors.ProviderError`, retried
  once via the fallback model's provider;
* a side-by-side *compare* that fans out to several models concurrently, timing
  each and capturing ``ok``/``text``/``error`` per model — it NEVER raises;
* an LLM-as-*judge* pass that asks a judge model to name the best answer —
  non-fatal (any error -> ``None``).

This module imports no LLM SDK: it depends only on the
:class:`~friday.providers.llm.LLMProvider` contract, so the ``openai`` import
stays confined to :mod:`friday.providers.llm`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from pydantic import BaseModel

from friday.errors import ProviderError
from friday.models.catalog import ModelCatalog, ModelInfo
from friday.providers.llm import LLMProvider, LLMResponse, Message, ToolSpec

#: How long a rate-limited model is left alone. Long enough that a depleted
#: free-tier quota stops costing a round trip on every single message, short
#: enough that an hourly window reopening is noticed within a few minutes.
_COOLDOWN = 300.0


def _is_rate_limited(exc: Exception) -> bool:
    """Whether a provider error means "out of quota" rather than "broke"."""
    detail = str(exc).lower()
    return "429" in detail or "rate limit" in detail or "quota" in detail

logger = logging.getLogger("friday.models.gateway")


class CompareResult(BaseModel):
    """One model's result from a side-by-side :meth:`ModelGateway.compare`.

    Captures the outcome per model so a failed model does not sink the whole
    comparison: ``ok`` is ``False`` with ``text=None`` and a human-readable
    ``error`` when that model raised, otherwise ``ok`` is ``True`` with the
    completion ``text`` (which may itself be ``None`` if the model returned no
    text). ``latency_ms`` is the wall-clock cost measured via the injected clock.
    """

    model_id: str
    label: str
    text: str | None
    latency_ms: int
    ok: bool
    error: str | None = None


class ModelGateway(LLMProvider):
    """A drop-in :class:`LLMProvider` that routes across many catalogued models.

    ``providers`` maps a provider name (``"openrouter"``/``"opencode"``/
    ``"nvidia"``/...) to a constructed :class:`LLMProvider`; the gateway resolves a
    ``provider:model`` id through ``catalog`` to pick the provider and pass the
    model slug via the provider's per-call ``model`` keyword. ``clock`` is injected
    (defaulting to :func:`time.monotonic`) so :meth:`compare` latency is
    deterministic in tests.
    """

    def __init__(
        self,
        providers: dict[str, LLMProvider],
        catalog: ModelCatalog,
        *,
        default_model_id: str,
        fallback_model_id: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._providers = dict(providers)
        self._catalog = catalog
        self._active_model_id = default_model_id
        self._fallback_model_id = fallback_model_id
        self._clock = clock
        #: ``{model_id: monotonic deadline}`` for models resting off a 429.
        self._rate_limited_until: dict[str, float] = {}

    @property
    def active_model_id(self) -> str:
        """The model id resolved for a turn when no per-call override is given."""
        return self._active_model_id

    def set_active(self, model_id: str) -> None:
        """Set the active model id used by subsequent default :meth:`complete` calls."""
        self._active_model_id = model_id

    def _resolve(self, model_id: str) -> tuple[ModelInfo, LLMProvider]:
        """Resolve a ``provider:model`` id to its :class:`ModelInfo` + provider.

        Raises :class:`ProviderError` when the id is not catalogued or no provider
        is wired for it — so an unknown/unavailable model surfaces as the same
        typed failure the orchestrator already handles, never a bare ``KeyError``.
        """
        info = self._catalog.get(model_id)
        if info is None:
            raise ProviderError(f"unknown model id: {model_id!r}")
        provider = self._providers.get(info.provider)
        if provider is None:
            raise ProviderError(
                f"no provider wired for {info.provider!r} (model {model_id!r})"
            )
        return info, provider

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> LLMResponse:
        """Complete via the resolved (or active) model, falling back once on error.

        Resolves ``model`` (or the active id) to its provider and calls it with the
        catalogued model slug. On a :class:`ProviderError`, if a distinct
        ``fallback_model_id`` is configured, retries exactly once via the
        fallback's provider; otherwise the original error propagates.
        """
        target_id = model if model is not None else self._active_model_id
        fallback_id = self._fallback_model_id

        # A model that just told us it is out of quota is not going to have any
        # a second later. The primary here is a free tier that returns 429 for
        # hours once the daily allowance is gone, and every message was paying
        # for that refusal before falling back — which is why replies got
        # steadily slower as the day went on rather than all at once.
        if (
            model is None
            and fallback_id is not None
            and fallback_id != target_id
            and self._resting(target_id)
        ):
            fb_info, fb_provider = self._resolve(fallback_id)
            return await fb_provider.complete(messages, tools, model=fb_info.model)

        info, provider = self._resolve(target_id)
        try:
            return await provider.complete(messages, tools, model=info.model)
        except ProviderError as exc:
            if fallback_id is None or fallback_id == target_id:
                raise
            if _is_rate_limited(exc):
                # Logged once per cooldown rather than once per message, so a
                # exhausted quota reads as one event instead of a wall.
                self._rest(target_id)
                logger.warning(
                    "gateway model %s is rate-limited; using %s for %ds",
                    target_id, fallback_id, _COOLDOWN,
                )
            else:
                logger.warning(
                    "gateway model %s failed, falling back to %s: %s",
                    target_id,
                    fallback_id,
                    exc,
                )
            fb_info, fb_provider = self._resolve(fallback_id)
            return await fb_provider.complete(messages, tools, model=fb_info.model)

    def _resting(self, model_id: str) -> bool:
        """Whether this model is still inside its rate-limit cooldown."""
        until = self._rate_limited_until.get(model_id, 0.0)
        if until <= self._clock():
            self._rate_limited_until.pop(model_id, None)
            return False
        return True

    def _rest(self, model_id: str) -> None:
        """Stop calling this model for a while."""
        self._rate_limited_until[model_id] = self._clock() + _COOLDOWN

    async def _compare_one(
        self,
        messages: list[Message],
        model_id: str,
    ) -> CompareResult:
        """Run one model for :meth:`compare`, capturing outcome + latency.

        NEVER raises: an unknown model, an unwired provider, or a provider error
        all land as ``ok=False`` with a human-readable ``error`` so one bad model
        cannot sink the whole comparison.
        """
        info = self._catalog.get(model_id)
        label = info.label if info is not None else model_id
        start = self._clock()
        try:
            _, provider = self._resolve(model_id)
            # ``info`` is guaranteed non-None here (resolve would have raised).
            assert info is not None
            response = await provider.complete(messages, None, model=info.model)
            latency_ms = int((self._clock() - start) * 1000)
            return CompareResult(
                model_id=model_id,
                label=label,
                text=response.text,
                latency_ms=latency_ms,
                ok=True,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001 - compare must never raise
            latency_ms = int((self._clock() - start) * 1000)
            return CompareResult(
                model_id=model_id,
                label=label,
                text=None,
                latency_ms=latency_ms,
                ok=False,
                error=str(exc),
            )

    async def compare(
        self,
        messages: list[Message],
        model_ids: list[str],
    ) -> list[CompareResult]:
        """Fan out ``messages`` to each model concurrently; return one result each.

        Each model runs via :meth:`_compare_one` (which never raises), so the
        returned list has exactly one :class:`CompareResult` per id, in the same
        order, with errors captured rather than propagated.
        """
        tasks = [self._compare_one(messages, mid) for mid in model_ids]
        return list(await asyncio.gather(*tasks))

    async def judge(
        self,
        question: str,
        results: list[CompareResult],
        *,
        judge_model_id: str,
    ) -> str | None:
        """Ask the judge model to name the best non-empty answer; ``None`` on any error.

        Builds a prompt listing each non-empty candidate answer (by model id) and
        asks the judge model to reply with the id of the best one. Non-fatal: an
        unknown judge model, a provider error, or an unparseable reply all yield
        ``None`` so the compare surface degrades to "no verdict" rather than
        raising.
        """
        candidates = [r for r in results if r.ok and r.text]
        if not candidates:
            return None
        valid_ids = {r.model_id for r in candidates}
        listing = "\n\n".join(
            f"[{r.model_id}]\n{r.text}" for r in candidates
        )
        prompt = (
            "You are judging several model answers to the same question. "
            "Reply with ONLY the id (in square brackets, e.g. "
            f"{next(iter(valid_ids))}) of the single best answer.\n\n"
            f"Question: {question}\n\nAnswers:\n{listing}"
        )
        messages = [Message(role="user", content=prompt)]
        try:
            info, provider = self._resolve(judge_model_id)
            response = await provider.complete(messages, None, model=info.model)
        except ProviderError:
            return None
        verdict = response.text
        if not verdict:
            return None
        # Match the first candidate id that appears verbatim in the verdict, so a
        # reply like "[opencode:mimo-v2.5-free] is best" still resolves cleanly.
        for candidate in candidates:
            if candidate.model_id in verdict:
                return candidate.model_id
        return None
