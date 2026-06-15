"""Typed application configuration via ``pydantic-settings``.

This is the single reader of environment variables. Secret-bearing fields use
:class:`pydantic.SecretStr` so they never leak into ``repr``/``str`` output or
logs. Most settings are read from ``FRIDAY_``-prefixed env vars; the
provider-native ``NVIDIA_*`` vars are read unprefixed (matching the upstream
OpenAI-compatible convention) via explicit validation aliases.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings sourced from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="FRIDAY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM provider ---
    llm_provider: Literal["nvidia", "fake"] = "fake"
    nvidia_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("NVIDIA_API_KEY", "nvidia_api_key"),
    )
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        validation_alias=AliasChoices("NVIDIA_BASE_URL", "nvidia_base_url"),
    )
    nvidia_model: str = Field(
        default="meta/llama-3.3-70b-instruct",
        validation_alias=AliasChoices("NVIDIA_MODEL", "nvidia_model"),
    )
    # Per-request LLM timeout in seconds. The provider client retries 0 times,
    # so this is the hard wall-clock budget for a single completion before it
    # surfaces as a ``ProviderError`` (env: ``FRIDAY_LLM_TIMEOUT_SECONDS``).
    # Defaults to 120s to absorb NVIDIA cold-start latency on heavier JSON
    # generation (e.g. the 3D studio) without spuriously timing out.
    llm_timeout_seconds: float = 120.0

    # --- LLM fallback provider ---
    # Which secondary provider :class:`FallbackLLM` uses when the primary fails.
    # ``none`` (default) keeps the single-provider behaviour; ``gemini`` wraps
    # the primary in a fallback to Gemini's OpenAI-compatible endpoint, but only
    # when a Gemini key is present (env: ``FRIDAY_LLM_FALLBACK_PROVIDER``).
    llm_fallback_provider: Literal["none", "gemini"] = "none"
    # Gemini credentials/config, read from the provider-native ``GEMINI_*`` env
    # vars unprefixed (matching the OpenAI-compatible convention) via aliases.
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "gemini_api_key"),
    )
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        validation_alias=AliasChoices("GEMINI_BASE_URL", "gemini_base_url"),
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        validation_alias=AliasChoices("GEMINI_MODEL", "gemini_model"),
    )

    # --- Routing ---
    route_min_confidence: float = 0.55

    # --- Memory ---
    memory_autowrite: bool = True
    # Local-first durable store. ``data/`` is gitignored; tests use ":memory:"
    # or a tmp_path file and never touch this real path.
    memory_db_path: str = "data/friday.db"

    # --- Embeddings (persistent vector store) ---
    # ``fake`` is the deterministic, offline default (tests, no key, no network);
    # ``nvidia`` selects the real NIM ``/embeddings`` adapter (providers/ only).
    embedding_provider: Literal["fake", "nvidia"] = "fake"
    # NVIDIA embedding model id; only used when ``embedding_provider == "nvidia"``.
    embedding_model: str = ""
    # Vector dimensionality the store is sized to; the fake honors it directly,
    # the NVIDIA adapter records it for store sizing.
    embedding_dim: int = 64

    # --- Persona ---
    owner_address: str = "Boss"

    # --- Feature flags (default off) ---
    enable_voice: bool = False
    enable_home: bool = False
    tts_provider: str = "piper"
    wake_word_engine: str = "openwakeword"

    # --- 3D Studio (Phase 7; default off) ---
    # The whole studio feature (router + static UI) is gated behind this flag; off
    # by default so the offline build exposes no studio surface (route -> 404).
    enable_studio: bool = False
    # Optional high-fidelity external text-to-3D backend. ``none`` (default) keeps
    # the free procedural-only path; ``meshy`` enables the lazy Meshy adapter, but
    # only when ``meshy_api_key`` is present — otherwise generation falls back to
    # the free procedural Scene path (never paywalls the user).
    studio_hifi_provider: Literal["none", "meshy"] = "none"
    # Meshy credentials, read from the provider-native ``MESHY_API_KEY`` env var
    # unprefixed via an alias (matching the NVIDIA/Gemini convention).
    meshy_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("MESHY_API_KEY", "meshy_api_key"),
    )
    # Optional Meshy model id; empty lets the provider use its default.
    studio_hifi_model: str = ""

    # --- Device control ---
    # Allow-list of device ids the home/device tools may actuate. Read from
    # ``FRIDAY_DEVICE_ALLOWLIST`` as a comma-separated string (e.g.
    # ``light.kitchen,switch.fan``) and split into a list; empty means no device
    # is actuatable.
    # ``NoDecode`` keeps pydantic-settings from JSON-decoding the raw env value so
    # the comma-splitting ``field_validator`` below receives the plain string.
    device_allowlist: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # --- Gateway hardening (Phase 6) ---
    # When true, every route except ``/health`` requires an ``Authorization:
    # Bearer <key>`` header whose key is in ``api_keys``; missing/invalid -> 401.
    # Default off keeps the local-first gateway open (no credentials needed).
    require_auth: bool = False
    # Accepted bearer keys. Read from ``FRIDAY_API_KEYS`` as a comma-separated
    # string (e.g. ``key1,key2``) and split into a list (same ``NoDecode`` +
    # before-validator pattern as ``device_allowlist``); empty means no key is
    # accepted (so ``require_auth`` with no keys rejects everything).
    api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # Fixed-window rate limit per client (bearer key if present, else client IP):
    # at most ``rate_limit_requests`` requests per ``rate_limit_window_seconds``;
    # over the limit -> 429 with ``Retry-After``. ``/health`` is exempt. Toggle
    # the whole gate off via ``rate_limit_enabled``.
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 60
    rate_limit_window_seconds: float = 60.0

    # --- Alerting ---
    # Identical alerts within this window collapse to a single send (dedupe +
    # rate-limit). Time is injected at the call site, never read from the clock
    # here, so behaviour is deterministic in tests.
    alert_rate_limit_seconds: float = 300.0
    alert_dedupe: bool = True

    # --- Logging ---
    log_level: str = "INFO"
    log_json: bool = True

    @field_validator("device_allowlist", mode="before")
    @classmethod
    def _split_device_allowlist(cls, value: object) -> object:
        """Comma-split a raw ``FRIDAY_DEVICE_ALLOWLIST`` string into a list.

        Pydantic-settings would otherwise try to JSON-decode a ``list`` env var;
        we accept a plain comma-separated string (whitespace-trimmed, empties
        dropped) so ``"a, b ,c"`` -> ``["a", "b", "c"]`` and ``""`` -> ``[]``.
        A value that is already a list/tuple is passed through unchanged.
        """
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_api_keys(cls, value: object) -> object:
        """Comma-split a raw ``FRIDAY_API_KEYS`` string into a list of keys.

        Mirrors :meth:`_split_device_allowlist`: a plain comma-separated string
        (whitespace-trimmed, empties dropped) so ``"k1, k2 ,k3"`` ->
        ``["k1", "k2", "k3"]`` and ``""`` -> ``[]``; a value that is already a
        list/tuple is passed through unchanged.
        """
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()
