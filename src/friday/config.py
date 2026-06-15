"""Typed application configuration via ``pydantic-settings``.

This is the single reader of environment variables. Secret-bearing fields use
:class:`pydantic.SecretStr` so they never leak into ``repr``/``str`` output or
logs. Most settings are read from ``FRIDAY_``-prefixed env vars; the
provider-native ``NVIDIA_*`` vars are read unprefixed (matching the upstream
OpenAI-compatible convention) via explicit validation aliases.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # --- Logging ---
    log_level: str = "INFO"
    log_json: bool = True


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()
