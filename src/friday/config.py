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

    # --- Personal RAG (Tier 1; default off) ---
    # Gates the whole ``/rag`` ingestion surface; off by default so the offline
    # build exposes no RAG routes (each -> 404). When on, ingested documents are
    # chunked into the shared vector store and become answerable via the existing
    # Knowledge path with citations. PDF reading stays optional/lazy (``pypdf``).
    enable_rag: bool = False
    # Target chunk size (characters) and inter-chunk overlap the ingestor uses to
    # split a document; overlap keeps a fact spanning a boundary retrievable.
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 120

    # --- Reminders & tasks (Tier 1; default off) ---
    # Gates the whole ``/reminders`` REST surface; off by default so the offline
    # build exposes no reminder routes (each -> 404). When on, reminders are
    # stored in a sibling SQLite file alongside ``memory_db_path`` and are also
    # creatable/listable/completable by the Automation agent via the tool
    # registry. Reuses ``memory_db_path`` (no new path setting).
    enable_reminders: bool = False

    # --- Meeting capture (Tier 1; default off) ---
    # Gates the whole ``/meetings`` REST surface; off by default so the offline
    # build exposes no meeting routes (each -> 404). When on, a meeting's audio is
    # transcribed (FakeSTT unless voice is configured), summarized by one
    # NON-FATAL LLM pass (any error -> transcript-only notes), optionally ingested
    # into the shared vector store (so the meeting is answerable via Knowledge),
    # and persisted in a sibling SQLite file alongside ``memory_db_path``.
    enable_meetings: bool = False

    # --- Agent reach (Tier 1; default off) ---
    # Gates the read-only ``agent_reach`` tool (full-page reader + media
    # transcription); off by default so the tool is unregistered and the
    # Research/Knowledge agents' allow-lists are unchanged. When on, ``read_url``
    # fetches a FULL page as clean markdown via the keyless Jina Reader
    # (``agent_reach_jina_base``, no binary needed) and ``transcribe`` shells out
    # to the isolated ``agent-reach`` CLI (installed via ``uv tool``, NOT a FRIDAY
    # dependency); a missing CLI degrades cleanly with an install hint. Read-only:
    # never fabricates on error.
    enable_agent_reach: bool = False
    # Base of the keyless Jina Reader endpoint; ``read_url`` issues
    # ``GET {base}{url}`` and expects clean markdown back.
    agent_reach_jina_base: str = "https://r.jina.ai/"
    # Per-request wall-clock budget (seconds) shared by the Jina GET and the CLI run.
    agent_reach_timeout: float = 60.0

    # --- Scheduled triggers (Tier 1; default off) ---
    # Gates the whole ``/schedules`` REST surface *and* the background tick loop;
    # off by default so the offline build exposes no scheduler routes (each ->
    # 404) and starts no background work. When on, time-based triggers
    # (interval/once/daily/weekly) fire named actions (e.g. ``due_reminders``),
    # persisted in a sibling SQLite file alongside ``memory_db_path``.
    enable_scheduler: bool = False
    # How often the background ``run_loop`` ticks (seconds) when the scheduler is
    # enabled. Only used by the un-unit-tested wall-clock loop; the tested
    # ``tick(now)`` unit takes ``now`` injected, so this never affects test timing.
    scheduler_tick_seconds: float = 30.0

    # --- Proactive briefing (Tier 1; default off) ---
    # Gates the whole ``/briefing`` surface *and* the scheduler ``briefing``
    # action; off by default so the offline build exposes no briefing route
    # (-> 404) and the registered action is inert unless a trigger fires it. When
    # on, the briefing is a deterministic digest assembled from the shared local
    # stores (reminders + audit + metrics) with an optional, non-fatal LLM
    # summary; no new store or path is introduced.
    enable_briefing: bool = False
    # How many recent tool-call audit rows the briefing's recent-activity section
    # summarizes (one line each).
    briefing_recent_activity: int = 5

    # --- Auto-journaling (Tier 2; default off) ---
    # Gates the whole ``/journal`` surface *and* the scheduler ``journal`` action;
    # off by default so the offline build exposes no journal route (-> 404) and the
    # registered action is inert unless a trigger fires it. When on, a day's events
    # (tool-call audit rows + reminders completed + a metrics line) are aggregated
    # into a deterministic :class:`~friday.journal.service.JournalEntry` and saved
    # (upsert by date) in a sibling SQLite file alongside ``memory_db_path``; an
    # optional, non-fatal LLM narration falls back to a deterministic summary.
    enable_journal: bool = False

    # --- Voice protocols (Tier 1; default off) ---
    # Gates the whole ``/protocols`` REST surface *and* the orchestrator's
    # trigger-phrase hook; off by default so the offline build exposes no protocol
    # routes (each -> 404) and the orchestrator never matches a protocol. When on,
    # named routines (an ordered list of registered tool calls) are persisted in a
    # sibling SQLite file alongside ``memory_db_path`` and fired by one trigger,
    # honoring the existing confirm-step on any side-effecting step.
    enable_protocols: bool = False

    # --- Self-critique loop (Tier 2; default off) ---
    # Gates the orchestrator's post-synthesis self-review. Off by default so the
    # turn loop makes no extra LLM call. When on, the final persona reply is
    # reviewed once (a deterministic banned-tone scan plus one LLM verdict pass);
    # if it fails and a concrete correction is offered, that revision replaces the
    # reply — one bounded pass (the revision is never re-critiqued) and non-fatal
    # (any critic error keeps the original response).
    enable_self_critique: bool = False

    # --- Plugins / extensions (Tier 2; default off) ---
    # Gates the whole plugin surface (the loader *and* the ``/plugins`` route).
    # Off by default so the offline build loads no third-party code and exposes no
    # plugin route (-> 404). When on, every ``*.py`` in ``plugins_dir`` is loaded
    # at startup and its ``get_tools()`` tools are registered into the shared tool
    # registry (after the built-ins, which win any name collision). Plugins are
    # TRUSTED local code the owner drops in (arbitrary Python by design); a broken
    # plugin is captured and skipped, never crashing startup.
    enable_plugins: bool = False
    # Directory scanned for plugin ``*.py`` files, relative to the process working
    # directory (so a local ``plugins/`` works out of the box). Only used when
    # ``enable_plugins`` is on.
    plugins_dir: str = "plugins"

    # --- Knowledge graph / entity cards (Tier 2; default off) ---
    # Gates the whole ``/graph`` REST surface (entities list, entity card,
    # extract). Off by default so the offline build exposes no graph routes (each
    # -> 404) and builds no extraction seam. When on, a tiny knowledge graph of
    # entities + relations is persisted in a sibling SQLite file alongside
    # ``memory_db_path``; ``POST /graph/extract`` runs one NON-FATAL LLM pass to
    # pull entities/relations from a note (any error -> empty result, never
    # raises), and an entity card stitches an entity together with its relations
    # and the long-term facts that mention it.
    enable_knowledge_graph: bool = False

    # --- Study / productivity (Tier 2; default off) ---
    # Gates the whole ``/study`` REST surface (flashcards + study sessions). Off by
    # default so the offline build exposes no study routes (each -> 404). When on,
    # spaced-repetition flashcards (scheduled by a pure SM-2 core) and logged study
    # sessions are persisted in a sibling SQLite file alongside ``memory_db_path``.
    # ``GET /study/review`` returns the cards due for utcnow; ``POST
    # /study/review/{id}`` applies SM-2 for the given recall grade (0..5) and
    # reschedules the card. Reuses ``memory_db_path`` (no new path setting).
    enable_study: bool = False

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
