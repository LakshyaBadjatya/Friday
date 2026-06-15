# Changelog

All notable changes to FRIDAY are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-16

The first stable release of FRIDAY — a local-first, provider-abstracted personal
AI operating system. Every capability beyond the core loop is behind a feature
flag and ships **off by default**, so the offline build stays fast, dependency-light,
and requires no API keys.

### Added

#### Core and orchestrator
- Deterministic turn router that classifies each request into a conversation mode
  (`CONVERSATION`, `RESEARCH`, `AUTOMATION`, `DEVICE_CONTROL`, `ALERTING`,
  `SECURITY_LOCKDOWN`, `CLARIFY`) and dispatches to the right specialist — routing
  happens before synthesis, so the decision stands even when the language backend
  is unreachable.
- An orchestrator over a typed mode graph that drives classify → retrieve → act →
  synthesize, with an honest-refusal path instead of fabricated answers.
- `FastAPI` surface with a `POST /chat` turn endpoint and a no-LLM `GET /health`
  liveness probe that reports the configured provider/model.
- Provider-abstracted LLM boundary: NVIDIA NIM for real replies with an optional
  Gemini fallback, plus a deterministic `FakeLLM` that keeps the full test suite
  green with zero network and no keys.

#### Persona roster
- An eight-specialist roster behind the FRIDAY prime persona — EDITH (security),
  ORACLE (automation), GECKO (finance), KAREN (communications), VERONICA (content),
  JOCASTA (memory), VISION (research), and FORGE (development/system) — each a
  least-privilege, owner-scoped operator the prime can delegate to.
- Frozen, dependency-free persona value objects and a case-insensitive registry
  the orchestrator can inject or override.

#### Memory and recall
- Short-term conversational memory plus a persistent **SQLite** long-term store
  for facts, tasks, and audit history.
- A vector store for semantic recall over remembered content.
- **Reciprocal Rank Fusion (RRF)** hybrid recall that fuses vector (semantic) and
  keyword (lexical) rankings into one ordered result set.
- A memory category/tier taxonomy that tags what kind of thing each memory is.
- **Personal RAG**: ingest a file or note (with optional, lazy PDF reading) so it
  becomes immediately answerable through the normal knowledge path, with citations.
- A lightweight personal **knowledge graph** of people, projects, and organizations,
  with entity cards that answer "what do you know about X?".

#### Voice, Studio, and Perception
- A full voice pipeline — wake word → microphone capture → Whisper STT →
  orchestrator → TTS — with **barge-in**, behind heavy backends kept out of the
  core install and lazy-imported.
- A **3D Studio** that turns a text or voice description into a validated JSON
  scene graph rendered by a no-build Three.js frontend, with GLB/STL/OBJ export.
  Model output is validated JSON, never executable code.
- A **perception** subsystem — object detection, screen/clipboard OCR, clipboard
  access, and screen capture — composed into a single screen-describe pass. Privacy-
  heavy and off by default.

#### Proactive intelligence
- **Reminders and tasks**: a durable, SQLite-backed reminder store with optional
  due dates and simple recurrence.
- A clock-injectable **scheduler** firing `interval` / `once` / `daily` / `weekly`
  triggers that run named actions and survive restarts.
- A proactive **briefing** digest of due/overdue/upcoming reminders, recent
  activity, and a metrics summary, available on demand and as a scheduled action.
- **Auto-journaling**: a deterministic per-day digest of FRIDAY's activity, with
  optional non-fatal LLM narration.
- **Anomaly detection** via a causal rolling z-score, and rule-based **foresight**
  that turns recent events into short, explainable suggestions.

#### Security and trust
- An **action broker** that fail-closes between intent and execution — validating
  arguments, classifying reversibility, enforcing a deny-by-default permission gate,
  injecting secrets at call time without surfacing them, and requiring a confirm
  step on side-effecting actions.
- A tamper-evident, **hash-chained audit** ledger of tool-call outcomes.
- A **secret vault** with pluggable backends (env, file, OS keyring, in-memory),
  `SecretStr`-typed secret fields redacted from logs, and a plaintext-secret scanner.

#### Integrations
- **Maps**: a flagged, no-build photorealistic 3D globe with fly-to and distance
  voice commands; the Maps API key is fetched at runtime, never baked into the page.
- **Market data**: live quotes and daily OHLCV history over the Dhan broker REST API,
  returning real data or an honest error — never a fabricated number.
- **Calendar, email, and communications**: thin `httpx` adapters over third-party
  REST APIs, each flagged and OAuth/bearer-token secured.
- **n8n**: draft a minimal valid workflow by description and optionally import it
  into a running n8n instance.
- **Voice protocols** ("agent-reach"): named, ordered routines of registered tool
  calls fired by a single trigger, honoring the broker confirm step with no arbitrary
  code execution.
- **Presence**: BLE proximity detection mapping nearby devices to friendly names,
  with arrival/departure transitions.

#### Operations
- Per-request **tracing**, a tool-call **audit log**, and **metrics**, surfaced
  through an `/admin` API.
- A **Streamlit dashboard** operator console over the admin endpoints.
- An installable, offline-capable **PWA** shell (web manifest, root-scope service
  worker, offline fallback).
- A **`friday` CLI** for driving the system from the terminal.
- **Docker** packaging (`Dockerfile`, `docker-compose.yml`) for containerized runs.
- A **CI** gate running lint, type checks, and the full test suite.

[1.0.0]: https://github.com/LakshyaBadjatya/Friday/releases/tag/v1.0.0
