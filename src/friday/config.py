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
    llm_provider: Literal[
        "nvidia", "fake", "openrouter", "opencode", "gateway"
    ] = "fake"
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
    llm_fallback_provider: Literal[
        "none", "gemini", "openrouter", "opencode", "gateway"
    ] = "none"
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

    # --- OpenRouter / OpenCode (multi-model free-tier providers) ---
    # Both expose an OpenAI-compatible ``/chat/completions`` surface and broker
    # many upstream models (a roster of free ones each). Keys are read from the
    # provider-native ``OPENROUTER_API_KEY`` / ``OPENCODE_API_KEY`` env vars
    # unprefixed (matching the NVIDIA/Gemini convention) via aliases; both are
    # :class:`SecretStr` so they never leak into repr/str/logs.
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "openrouter_api_key"),
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias=AliasChoices("OPENROUTER_BASE_URL", "openrouter_base_url"),
    )
    opencode_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENCODE_API_KEY", "opencode_api_key"),
    )
    opencode_base_url: str = Field(
        default="https://opencode.ai/zen/v1",
        validation_alias=AliasChoices("OPENCODE_BASE_URL", "opencode_base_url"),
    )

    # --- Model catalog / gateway ---
    # The active model the gateway resolves a turn to when no per-call override is
    # given (a ``provider:model`` id from the catalog). Defaults to a fast,
    # verified free OpenRouter model.
    default_model_id: str = "openrouter:google/gemma-4-31b-it:free"
    # The models the side-by-side compare fans out to. Read from
    # ``FRIDAY_COMPARE_MODEL_IDS`` as a comma-separated string (same ``NoDecode`` +
    # before-validator pattern as ``device_allowlist``) so each ``provider:model``
    # id is kept whole; empty means no compare set.
    compare_model_ids: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "openrouter:openai/gpt-oss-20b:free",
            "openrouter:google/gemma-4-31b-it:free",
            "opencode:mimo-v2.5-free",
            "nvidia:meta/llama-3.1-8b-instruct",
        ]
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

    # --- Hardware / system monitoring (Tier 2; default off) ---
    # Gates the whole ``/system`` REST surface (stats + check) *and* the scheduler
    # ``system_check`` action; off by default so the offline build exposes no
    # system routes (each -> 404) and the registered action is inert unless a
    # trigger fires it. When on, a :class:`~friday.system.monitor.SystemMonitor`
    # over the real :class:`~friday.system.monitor.PsutilSampler` reports the
    # host's CPU / memory / disk utilisation (plus optional temperature + load),
    # and the four thresholds below decide when a metric breach raises an alert.
    enable_system_monitor: bool = False
    # Breach thresholds for ``GET /system/check`` and the scheduler action: a
    # metric value strictly ABOVE its threshold raises one alert (boundary-equal
    # is healthy). CPU/memory are utilisation percentages, disk a usage
    # percentage, temperature degrees Celsius.
    sys_cpu_threshold: float = 90.0
    sys_mem_threshold: float = 90.0
    sys_disk_threshold: float = 95.0
    sys_temp_threshold: float = 85.0

    # --- System automation (Tier 2; default off) ---
    # Gates the three system-automation tools (``run_command`` / ``find_files`` /
    # ``open_app``). Off by default so the offline build registers none of them and
    # the Automation agent's allow-list is unchanged. When on, the tools are
    # registered behind the registry's existing permission / confirm-step gates
    # (``run_command`` and ``open_app`` are side-effecting + non-idempotent, so the
    # confirm-step gates them; ``find_files`` is read-only). Every execution path is
    # argv-only (``create_subprocess_exec`` — never a shell), output is capped and a
    # timeout enforced, and file search is confined to ``system_automation_root``.
    enable_system_automation: bool = False
    # Root the file-search tool is confined to: ``find_files`` rejects any pattern
    # or root that resolves OUTSIDE this directory (path-traversal guard). Defaults
    # to the process working directory.
    system_automation_root: str = "."
    # Optional allow-list of command basenames ``run_command`` may execute. Read
    # from ``FRIDAY_SYSTEM_EXEC_ALLOWLIST`` as a comma-separated string (same
    # ``NoDecode`` + before-validator pattern as ``device_allowlist``); empty means
    # no allow-list is enforced (any command may run, still argv-only / no shell).
    system_exec_allowlist: Annotated[list[str], NoDecode] = Field(
        default_factory=list
    )
    # Per-command wall-clock budget (seconds) for ``run_command`` / ``open_app``;
    # a command exceeding it is killed and surfaced as a timeout error.
    system_exec_timeout: float = 30.0

    # --- n8n integration (Tier 2; default off) ---
    # Gates the whole ``/n8n`` REST surface *and* the orchestrator's "make a
    # workflow on n8n <X>" hook; off by default so the offline build exposes no
    # n8n routes (each -> 404) and the orchestrator never reaches n8n. When on,
    # FRIDAY drafts a MINIMAL valid n8n workflow JSON (one NON-FATAL LLM pass; a
    # safe single-Manual-Trigger stub on failure) and, if a key is set, imports it
    # into a running n8n via its REST API. n8n itself can be auto-started via
    # ``docker compose up -d <service>`` — gated behind the confirm-step (a
    # side-effecting action), spawned argv-only (``create_subprocess_exec`` — never
    # a shell). The API key is a :class:`SecretStr` (never logged).
    enable_n8n: bool = False
    # Base URL of the local n8n instance the client probes / imports into.
    n8n_base_url: str = "http://localhost:5678"
    # n8n REST API key, read from ``FRIDAY_N8N_API_KEY``. A :class:`SecretStr` so it
    # never leaks into repr/str/logs; sent ONLY as the ``X-N8N-API-KEY`` header.
    # When unset, drafting still works but import is skipped/errors clearly.
    n8n_api_key: SecretStr | None = None
    # The compose file + service name used to build the docker auto-start argv:
    # ``["docker", "compose", "-f", <file>, "up", "-d", <service>]``.
    n8n_docker_compose_file: str = "docker-compose.yml"
    n8n_docker_service: str = "n8n"

    # --- Perception (Tier 2; default off; PRIVACY-HEAVY) ---
    # Gates the whole ``/perception`` REST surface (vision / OCR / clipboard /
    # screen). Off by default so the offline build exposes no perception routes
    # (each -> 404) and constructs no capture/clipboard seam. PRIVACY note: when
    # enabled, perception can READ YOUR SCREEN AND CLIPBOARD — only turn it on if
    # you intend FRIDAY to observe them. The heavy backends (opencv/ultralytics,
    # pytesseract/pillow, pyperclip, mss) are kept OUT of the uv lock and lazy-
    # imported by the real adapters (``make install-perception``); the app wires
    # the fakes by default, so the offline build needs no heavy library.
    enable_perception: bool = False

    # --- Maps / geolocation (Tier 3; default off) ---
    # Gates the Google Maps surface (directions / places / geocoding). Off by
    # default so the offline build wires no Maps client. When on, the adapter talks
    # to the Google Maps web APIs with ``google_maps_api_key`` (a :class:`SecretStr`,
    # never logged; sent only as the documented ``key`` query parameter) over httpx.
    enable_maps: bool = False
    # Google Maps API key, read from ``FRIDAY_GOOGLE_MAPS_API_KEY``. A
    # :class:`SecretStr` so it never leaks into repr/str/logs.
    google_maps_api_key: SecretStr | None = None

    # --- Presence detection (Tier 3; default off) ---
    # Gates the presence surface (which known devices are currently on the LAN).
    # Off by default so the offline build runs no scan. When on, presence maps the
    # MAC addresses seen on the network to friendly names from
    # ``presence_known_devices``.
    enable_presence: bool = False
    # Known devices as comma-separated ``MAC=Name`` entries (e.g.
    # ``AA:BB:CC:DD:EE:FF=Phone,11:22:33:44:55:66=Laptop``). Read from
    # ``FRIDAY_PRESENCE_KNOWN_DEVICES``; ``NoDecode`` + the before-validator below
    # comma-split it (mirroring ``device_allowlist``), keeping each ``MAC=Name``
    # entry whole; empty means no devices are tracked.
    presence_known_devices: Annotated[list[str], NoDecode] = Field(
        default_factory=list
    )

    # --- Market data (Tier 3; default off) ---
    # Gates the market-data surface (quotes / holdings via the Dhan broker API).
    # Off by default so the offline build wires no broker client. When on, the
    # adapter talks to Dhan over httpx using the two credentials below.
    enable_market_data: bool = False
    # Dhan client id and access token, read from ``FRIDAY_DHAN_CLIENT_ID`` /
    # ``FRIDAY_DHAN_ACCESS_TOKEN``. Both :class:`SecretStr` so they never leak into
    # repr/str/logs; sent only as the documented Dhan auth headers.
    dhan_client_id: SecretStr | None = None
    dhan_access_token: SecretStr | None = None

    # --- Calendar (Tier 3; default off) ---
    # Gates the Google Calendar surface. Off by default so the offline build wires
    # no calendar client. When on, the adapter talks to the Google Calendar API
    # over httpx using ``google_oauth_token``.
    enable_calendar: bool = False
    # Google OAuth bearer token, read from ``FRIDAY_GOOGLE_OAUTH_TOKEN``. A
    # :class:`SecretStr` so it never leaks into repr/str/logs; sent only as the
    # ``Authorization: Bearer`` header.
    google_oauth_token: SecretStr | None = None

    # --- Email (Tier 3; default off) ---
    # Gates the Gmail surface (read / send). Off by default so the offline build
    # wires no mail client. When on, the adapter talks to the Gmail API over httpx
    # using ``gmail_oauth_token``.
    enable_email: bool = False
    # Gmail OAuth bearer token, read from ``FRIDAY_GMAIL_OAUTH_TOKEN``. A
    # :class:`SecretStr` so it never leaks into repr/str/logs; sent only as the
    # ``Authorization: Bearer`` header.
    gmail_oauth_token: SecretStr | None = None

    # --- Comms / messaging (Tier 3; default off) ---
    # Gates the Twilio messaging surface (SMS / voice). Off by default so the
    # offline build wires no Twilio client. When on, the adapter talks to the
    # Twilio REST API over httpx using the SID/token below (HTTP basic auth) and
    # sends from ``twilio_from_number``.
    enable_comms: bool = False
    # Twilio account SID and auth token, read from ``FRIDAY_TWILIO_ACCOUNT_SID`` /
    # ``FRIDAY_TWILIO_AUTH_TOKEN``. Both :class:`SecretStr` so they never leak into
    # repr/str/logs; used only as the Twilio HTTP basic-auth credentials.
    twilio_account_sid: SecretStr | None = None
    twilio_auth_token: SecretStr | None = None
    # The Twilio sender phone number (``From``), read from
    # ``FRIDAY_TWILIO_FROM_NUMBER``; empty means no sender is configured.
    twilio_from_number: str = ""

    # --- Postgres (Tier 3; default off) ---
    # Gates the optional Postgres backend. Off by default so the local-first build
    # uses the SQLite stores only. When on, ``postgres_dsn`` selects the Postgres
    # connection.
    enable_postgres: bool = False
    # Postgres DSN, read from ``FRIDAY_POSTGRES_DSN``. A :class:`SecretStr` (it may
    # embed a password) so it never leaks into repr/str/logs.
    postgres_dsn: SecretStr | None = None

    # --- Offline mode (default off) ---
    # When on, FRIDAY runs in a strict offline mode (no outbound network). Off by
    # default keeps the normal behaviour.
    enable_offline_mode: bool = False

    # --- HUD (default off) ---
    # Gates the heads-up-display surface. Off by default so the offline build
    # exposes no HUD.
    enable_hud: bool = False

    # --- Family sharing (default off) ---
    # Gates the family-sharing surface (shared reminders / lists across members).
    # Off by default so the build keeps a single-owner scope.
    enable_family_sharing: bool = False

    # --- Roster personas (Stage 2; always available) ---
    # The persona roster (FRIDAY + eight least-privilege specialists) is exposed
    # via ``GET /roster`` and drives the orchestrator's address-by-name hook. It
    # is always available (no flag) — it adds no side-effecting surface, only a
    # read-only listing and a least-privilege scope when a turn is addressed to a
    # named persona ("GECKO, ..." / "ask VISION to ...").

    # --- Idea-batch read-only tools (Stage 2; default ON) ---
    # Registers the read-only idea-batch tools (capabilities / ask_user /
    # entity_dossier / infofeed / browser) into the shared registry and adds them
    # to the fitting agents' allow-lists. Default ON because they are all
    # read-only (``side_effecting=False``) and add no real-world action — they only
    # reflect, ask, read feeds, or fetch readable page text. The side-effecting
    # idea-batch tools (downloads_butler / media) are NOT gated by this flag; they
    # ride their own readiness flags below and still pass the registry confirm-step.
    enable_extra_tools: bool = True
    # When set, the side-effecting downloads-butler (organize a folder) and media
    # (play/pause/volume) tools are also registered. Off by default so the offline
    # build registers no filesystem/media-mutating tool. Both go through the
    # registry's confirm-step (downloads_butler is side-effecting + non-idempotent,
    # so a real move is confirm-gated; media is idempotent transport).
    enable_downloads_butler: bool = False
    enable_media_control: bool = False

    # --- Desktop control (Stage 2; default off) ---
    # Gates desktop control (mouse/keyboard/screenshots via the
    # :class:`~friday.desktop.AuditedDesktop` over a :class:`FakeDesktop` by
    # default). Off by default so the offline build constructs no desktop seam.
    # When on, the wrapper is FRICTIONLESS (no per-action prompt) but FULLY AUDITED
    # — every action is recorded to the hash-chained ledger before it executes. The
    # real ``pyautogui`` backend stays OPTIONAL/LAZY (excluded from the uv lock).
    enable_desktop: bool = False

    # --- Voiceprint / owner recognition (Stage 2; default off) ---
    # Gates speaker verification (owner recognition). Off by default so no
    # voiceprint identity is constructed. When on, an
    # :class:`~friday.voice.voiceprint.OwnerIdentity` over a deterministic
    # :class:`FakeVoiceprint` is built and surfaced (ADVISORY by default — it never
    # blocks). The real ``resemblyzer`` backend stays OPTIONAL/LAZY (excluded from
    # the uv lock).
    enable_voiceprint: bool = False

    # --- Proactive intelligence (Stage 2; default off) ---
    # Gates proactive intelligence (anomaly detection + rule-based foresight). Off
    # by default so no proactive seam is constructed. When on, an
    # :class:`~friday.proactive.AnomalyDetector` is wired into the scheduler's
    # ``system_check`` action so a metric spike in the sampled history is flagged,
    # and a :class:`~friday.proactive.Foresight` is surfaced for suggestions. Both
    # are pure/deterministic and import no LLM SDK.
    enable_proactive: bool = False

    # --- Per-turn budgeter (Wave 0; cost/latency governor; default OFF) ---
    # Gates the per-turn cost/latency budgeter. Off by default so the turn loop
    # keeps no spend tally and never downshifts. When on, the orchestrator caps a
    # turn's token (and optional dollar) spend and, once a turn runs hot, can
    # downshift the gateway's active model to a cheaper/smaller tier. The budgeter
    # itself (:class:`~friday.models.budget.Budgeter`) is pure/offline — it reads
    # no settings and uses no clock; ``app.py`` injects the caps below.
    enable_budgeter: bool = False  # flag-gate the per-turn cost/latency budgeter (default OFF)
    # Hard token ceiling per conversation turn.
    budget_max_tokens_per_turn: int = 8000
    # Optional dollar ceiling per turn; None = unpriced (free models).
    budget_max_usd_per_turn: float | None = None
    # Catalog id to switch the gateway to when over budget; empty = keep current
    # active model (the budgeter still surfaces the signal for metrics/HUD).
    budget_downshift_model_id: str = ""
    # Fraction of the token cap at/beyond which ``should_downshift`` trips (0.0-1.0).
    budget_downshift_at: float = 0.8

    # --- Calibrated confidence (Wave 0; default OFF) ---
    # Gates the calibrated confidence scorer (:mod:`friday.core.confidence`). Off
    # by default so the orchestrator stamps no confidence and appends no caveat.
    # When on, after a synthesized reply the orchestrator stamps
    # ``state.scratchpad["confidence"]`` and, when the blended confidence falls
    # below ``confidence_note_threshold``, appends a one-line honest caveat. The
    # scorer is pure/deterministic and reads no settings itself.
    enable_confidence: bool = False  # gate the calibrated confidence scorer; default OFF
    # Below this blended confidence (0..1), the orchestrator may append a one-line
    # confidence caveat to the reply.
    confidence_note_threshold: float = 0.45

    # --- Custom operators (Wave 0; roster extension; default empty) ---
    # Extra personas merged into the always-on roster alongside the built-ins.
    # Read from ``FRIDAY_CUSTOM_OPERATORS`` as comma-separated
    # ``NAME|Title|tools|namespace|prompt`` entries (each entry itself contains
    # pipes, so the before-validator below splits ONLY on commas). ``NoDecode``
    # keeps pydantic-settings from JSON-decoding the raw env value. Empty (default)
    # adds no extra operator, so the roster is identical to today; a custom whose
    # name collides with a built-in is dropped (built-ins always win), and a
    # malformed value is logged and skipped at boot (never crashing).
    custom_operators: Annotated[list[str], NoDecode] = Field(default_factory=list)

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

    # --- Security spine (Stage 1) ---
    # Path to the tamper-evident, hash-chained audit ledger (an append-only JSONL
    # file). Every tool the shared registry executes appends ONE hash-chained
    # record here in addition to the in-memory observability ``AuditLog`` the
    # ``/admin/audit`` view reads — the ledger is the tamper-evident
    # system-of-record, verifiable via ``GET /admin/audit/verify``. ``data/`` is
    # gitignored; tests pin a tmp path and never touch this real file.
    audit_ledger_path: str = "data/audit.jsonl"
    # Which secret backend the runtime constructs (``friday.secrets``):
    # ``env`` (default) reads ``FRIDAY_<NAME>`` from the process environment;
    # ``keyring`` uses the OS keychain via the OPTIONAL ``keyring`` package
    # (lazy-imported only when selected — kept out of the core lock); ``file`` is a
    # ``0600`` JSON dev fallback alongside ``data/``; ``memory`` is in-process only.
    # The vault is the broker's secret provider for ``{{secret:NAME}}`` injection.
    secret_vault: Literal["env", "keyring", "file", "memory"] = "env"
    # When true, startup scans the repo for plaintext secret-looking literals
    # (``friday.secrets.scan_for_plaintext_secrets``) and LOGS a WARNING per
    # finding. It is warn-only and NEVER refuses to boot — a default-safe nudge to
    # move a committed credential into the vault, not a hard gate.
    enable_secret_self_check: bool = True
    # When true, side-effecting tool calls are routed THROUGH the broker
    # (validate → classify → deny-by-default gate → secret-inject → execute →
    # hash-chained audit). Default OFF keeps dispatch on the plain registry path,
    # so the existing default-on behaviour is unchanged until explicitly enabled.
    enable_broker: bool = False
    # When true AND the broker is on, the broker denies any tool call whose args
    # carry an outbound URL to a host not on ``egress_allowlist`` (fail-closed: an
    # empty allow-list blocks all such URLs). Off by default so the broker's
    # behaviour is unchanged until explicitly enabled.
    enable_egress_firewall: bool = False
    # Hosts the egress firewall permits (subdomains of a listed host included).
    # Read from ``FRIDAY_EGRESS_ALLOWLIST`` as a comma-separated string (same
    # ``NoDecode`` + before-validator pattern as ``device_allowlist``); empty
    # (default) blocks every outbound URL when the firewall is on.
    egress_allowlist: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # When true, the selected LLM provider is wrapped so high-confidence PII
    # (emails, payment-card numbers, IPv4 addresses, phone numbers) is scrubbed
    # from outbound messages before any real provider sees them. Off by default; a
    # deliberate privacy/utility trade-off (the model also stops seeing the PII).
    enable_pii_redaction: bool = False

    # --- Gateway hardening (Phase 6) ---
    # The uvicorn bind host the CLI ``serve`` default uses *and* the host the
    # startup exposure check reads. Defaults to the loopback ``127.0.0.1`` so the
    # local-first gateway is reachable only from the machine itself. When this is
    # set to a non-loopback address (e.g. ``0.0.0.0``) AND ``require_auth`` is off,
    # startup logs a prominent WARNING that FRIDAY is exposed to the network with
    # no auth — an advisory nudge (boot is NEVER refused, so local dev keeps
    # working). Read from ``FRIDAY_BIND_HOST``.
    bind_host: str = "127.0.0.1"
    # When true, every route except ``/health`` requires an ``Authorization:
    # Bearer <key>`` header whose key is in ``api_keys``; missing/invalid -> 401.
    # Default off keeps the local-first gateway open (no credentials needed).
    require_auth: bool = False
    # Accepted bearer keys. Read from ``FRIDAY_API_KEYS`` as a comma-separated
    # string (e.g. ``key1,key2``) and split into a list (same ``NoDecode`` +
    # before-validator pattern as ``device_allowlist``); empty means no key is
    # accepted (so ``require_auth`` with no keys rejects everything).
    api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # Role-based access control (default off). When on AND ``require_auth`` is on,
    # a validated bearer key whose role is not ``owner`` is denied the ``/admin``
    # surface (403). Off by default so auth stays a plain key check.
    enable_rbac: bool = False
    # Per-key role assignments. Read from ``FRIDAY_API_ROLES`` as comma-separated
    # ``key=role`` entries (e.g. ``s3cret=owner,guest=member``); the role is one of
    # ``owner`` (full access) / ``member`` (no admin). An unmapped valid key has no
    # role and is denied admin (deny-by-default). Same ``NoDecode`` + before-validator
    # comma-split as ``device_allowlist`` (each entry keeps its ``=``).
    api_roles: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # Fixed-window rate limit per client (bearer key if present, else client IP):
    # at most ``rate_limit_requests`` requests per ``rate_limit_window_seconds``;
    # over the limit -> 429 with ``Retry-After``. ``/health`` is exempt. Toggle
    # the whole gate off via ``rate_limit_enabled``.
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 60
    rate_limit_window_seconds: float = 60.0

    # --- Multi-agent Brain (Wave 1; default off) ---
    # Gates the ensemble/debate surface (``POST /ensemble/debate``): several named
    # operators each draft an answer and one synthesis pass fuses them. Off by
    # default so the route 404s; when on it drives the same LLM the chat loop uses.
    enable_ensemble: bool = False
    # Gates the planner surface (``POST /planner/plan``): decompose a goal into a
    # DAG of steps and render it for confirmation (planning only — execution stays
    # a separate, broker-gated action). Off by default so the route 404s.
    enable_planner: bool = False
    # Gates context compaction: once a session's short-term history grows past a
    # threshold, the orchestrator folds the older turns into one summary message
    # and keeps a recent tail (one bounded, non-fatal LLM pass). Off by default so
    # the turn loop makes no extra call and the buffer is never rewritten.
    enable_compaction: bool = False
    # Gates the contradiction-check surface (``POST /memory/contradiction``):
    # check whether a candidate fact conflicts with stored memory (one bounded,
    # non-fatal LLM pass over the relevant long-term facts). Off by default so the
    # route 404s.
    enable_contradiction: bool = False
    # Gates the auto-tagging surface (``POST /memory/tag``): suggest normalized
    # topic tags for a piece of text (one bounded, non-fatal LLM pass). Off by
    # default so the route 404s.
    enable_autotag: bool = False
    # Gates the second-brain export surface (``GET /export``): render the
    # long-term facts as an Obsidian-style Markdown note. Off by default so the
    # route 404s.
    enable_kb_export: bool = False

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

    @field_validator("presence_known_devices", mode="before")
    @classmethod
    def _split_presence_known_devices(cls, value: object) -> object:
        """Comma-split a raw ``FRIDAY_PRESENCE_KNOWN_DEVICES`` string into a list.

        Mirrors :meth:`_split_device_allowlist`: a plain comma-separated string of
        ``MAC=Name`` entries (whitespace-trimmed, empties dropped) so
        ``"AA:BB=Phone, 11:22=Laptop"`` -> ``["AA:BB=Phone", "11:22=Laptop"]`` and
        ``""`` -> ``[]``. Each entry is kept whole (the ``=`` is not split here); a
        value that is already a list/tuple is passed through unchanged.
        """
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("system_exec_allowlist", mode="before")
    @classmethod
    def _split_system_exec_allowlist(cls, value: object) -> object:
        """Comma-split a raw ``FRIDAY_SYSTEM_EXEC_ALLOWLIST`` string into a list.

        Mirrors :meth:`_split_device_allowlist`: a plain comma-separated string
        (whitespace-trimmed, empties dropped) so ``"ls, echo ,cat"`` ->
        ``["ls", "echo", "cat"]`` and ``""`` -> ``[]``; a value that is already a
        list/tuple is passed through unchanged.
        """
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("compare_model_ids", mode="before")
    @classmethod
    def _split_compare_model_ids(cls, value: object) -> object:
        """Comma-split a raw ``FRIDAY_COMPARE_MODEL_IDS`` string into a list of ids.

        Mirrors :meth:`_split_device_allowlist`: a plain comma-separated string of
        ``provider:model`` ids (whitespace-trimmed, empties dropped) so
        ``"openrouter:a:free, opencode:b"`` -> ``["openrouter:a:free",
        "opencode:b"]`` and ``""`` -> ``[]``. Each id is kept whole (the ``:`` is
        not split here); a value that is already a list/tuple is passed through
        unchanged.
        """
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("custom_operators", mode="before")
    @classmethod
    def _split_custom_operators(cls, value: object) -> object:
        """Comma-split a raw ``FRIDAY_CUSTOM_OPERATORS`` string into a list of entries.

        Mirrors :meth:`_split_compare_model_ids`: a plain comma-separated string of
        ``NAME|Title|tools|namespace|prompt`` entries (whitespace-trimmed, empties
        dropped) so ``"A|..., B|..."`` -> ``["A|...", "B|..."]`` and ``""`` ->
        ``[]``. The split is on COMMAS ONLY — each entry keeps its internal pipes,
        so the roster parser receives whole mini-format entries. A value that is
        already a list/tuple is passed through unchanged.
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

    @field_validator("egress_allowlist", mode="before")
    @classmethod
    def _split_egress_allowlist(cls, value: object) -> object:
        """Comma-split a raw ``FRIDAY_EGRESS_ALLOWLIST`` string into a list of hosts.

        Mirrors :meth:`_split_device_allowlist`: a plain comma-separated string
        (whitespace-trimmed, empties dropped) so ``"a.com, b.com"`` ->
        ``["a.com", "b.com"]`` and ``""`` -> ``[]``; a value that is already a
        list/tuple is passed through unchanged.
        """
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("api_roles", mode="before")
    @classmethod
    def _split_api_roles(cls, value: object) -> object:
        """Comma-split a raw ``FRIDAY_API_ROLES`` string into ``key=role`` entries.

        Mirrors :meth:`_split_device_allowlist`: split on COMMAS only (each entry
        keeps its internal ``=``), whitespace-trimmed, empties dropped, so
        ``"k1=owner, k2=member"`` -> ``["k1=owner", "k2=member"]`` and ``""`` ->
        ``[]``; a value that is already a list/tuple is passed through unchanged.
        """
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()
