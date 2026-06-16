# © Lakshya Badjatya — Author
"""The model catalog — the verified free/free-tier LLM roster + lookup.

A pure, offline data layer over the models FRIDAY has verified working. Each
entry is a :class:`ModelInfo` keyed on a ``provider:model`` id; :class:`ModelCatalog`
filters the listing to the providers actually available at runtime (so the UI only
offers models the build has a key for) while still resolving any catalogued id by
:meth:`~ModelCatalog.get` (so the gateway can attempt a configured-but-unlisted
model). This module imports no LLM SDK and performs no I/O.
"""

from __future__ import annotations

from pydantic import BaseModel, model_validator


class ModelInfo(BaseModel):
    """One catalogued model: a ``provider:model`` id plus display metadata.

    ``id`` is always ``f"{provider}:{model}"`` (validated below) so a single
    string round-trips to the provider that serves it and the model slug that
    provider expects.
    """

    id: str
    provider: str
    model: str
    label: str
    free: bool

    @model_validator(mode="after")
    def _id_matches_provider_model(self) -> ModelInfo:
        expected = f"{self.provider}:{self.model}"
        if self.id != expected:
            raise ValueError(
                f"ModelInfo id {self.id!r} must equal provider:model {expected!r}"
            )
        return self


def _model(provider: str, model: str, label: str, *, free: bool = True) -> ModelInfo:
    """Construct a :class:`ModelInfo`, deriving its id from provider + model."""
    return ModelInfo(
        id=f"{provider}:{model}",
        provider=provider,
        model=model,
        label=label,
        free=free,
    )


#: The verified working models. Every entry has been confirmed reachable; the
#: free OpenRouter/OpenCode models are ``free=True`` and the NVIDIA fast model is
#: free-tier (also ``free=True``). The first OpenRouter entry is the fast default
#: active model; the NVIDIA model is the reliable gateway fallback.
DEFAULT_CATALOG: tuple[ModelInfo, ...] = (
    # OpenRouter (free)
    _model("openrouter", "openai/gpt-oss-20b:free", "GPT-OSS 20B"),
    _model("openrouter", "google/gemma-4-31b-it:free", "Gemma 4 31B IT"),
    _model("openrouter", "nvidia/nemotron-nano-9b-v2:free", "Nemotron Nano 9B v2"),
    _model("openrouter", "openai/gpt-oss-120b:free", "GPT-OSS 120B"),
    _model("openrouter", "qwen/qwen3-coder:free", "Qwen3 Coder"),
    _model(
        "openrouter",
        "meta-llama/llama-3.3-70b-instruct:free",
        "Llama 3.3 70B Instruct",
    ),
    _model(
        "openrouter",
        "nousresearch/hermes-3-llama-3.1-405b:free",
        "Hermes 3 Llama 3.1 405B",
    ),
    # OpenCode Zen (free)
    _model("opencode", "mimo-v2.5-free", "MiMo v2.5"),
    _model("opencode", "nemotron-3-ultra-free", "Nemotron 3 Ultra"),
    _model("opencode", "deepseek-v4-flash-free", "DeepSeek v4 Flash"),
    # NVIDIA NIM (free-tier; reliable fallback)
    _model("nvidia", "meta/llama-3.1-8b-instruct", "Llama 3.1 8B Instruct"),
)


class ModelCatalog:
    """Lookup over :data:`DEFAULT_CATALOG`, scoped to available providers.

    ``available_providers`` is the set of provider keys the runtime actually has
    a usable provider for (e.g. ``{"openrouter", "nvidia"}``). :meth:`list_models`
    and :meth:`ids` return only entries whose provider is available, so a UI never
    offers a model the build cannot serve; :meth:`get` resolves from the FULL
    catalog regardless of availability, so the gateway can still describe (and
    attempt) a configured id whose provider was not wired.
    """

    def __init__(
        self,
        available_providers: set[str],
        catalog: tuple[ModelInfo, ...] = DEFAULT_CATALOG,
    ) -> None:
        self._available = set(available_providers)
        self._catalog = catalog
        self._by_id = {info.id: info for info in catalog}

    def list_models(self) -> list[ModelInfo]:
        """Return the catalogued models whose provider is available."""
        return [
            info for info in self._catalog if info.provider in self._available
        ]

    def get(self, model_id: str) -> ModelInfo | None:
        """Resolve a ``provider:model`` id to its :class:`ModelInfo`, or ``None``.

        Resolves from the full catalog (availability-independent) so the gateway
        can still look up the provider/model slug for a configured id.
        """
        return self._by_id.get(model_id)

    def ids(self) -> list[str]:
        """Return the ids of the available (listed) models, in catalog order."""
        return [info.id for info in self.list_models()]
