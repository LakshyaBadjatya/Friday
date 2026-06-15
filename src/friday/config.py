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
    llm_timeout_seconds: float = 60.0

    # --- Routing ---
    route_min_confidence: float = 0.55

    # --- Memory ---
    memory_autowrite: bool = True

    # --- Persona ---
    owner_address: str = "Boss"

    # --- Feature flags (default off) ---
    enable_voice: bool = False
    enable_home: bool = False
    tts_provider: str = "piper"
    wake_word_engine: str = "openwakeword"

    # --- Device control ---
    # Allow-list of device ids the home/device tools may actuate. Read from
    # ``FRIDAY_DEVICE_ALLOWLIST`` as a comma-separated string (e.g.
    # ``light.kitchen,switch.fan``) and split into a list; empty means no device
    # is actuatable.
    # ``NoDecode`` keeps pydantic-settings from JSON-decoding the raw env value so
    # the comma-splitting ``field_validator`` below receives the plain string.
    device_allowlist: Annotated[list[str], NoDecode] = Field(default_factory=list)

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


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()
