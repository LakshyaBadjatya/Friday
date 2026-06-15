# FRIDAY

FRIDAY is a local-first, provider-abstracted personal AI assistant you run on your own
machine. Phases 0 and 1 are done: there is a real, runnable core chat loop — a FastAPI
`POST /chat` endpoint that drives an orchestrator through a deterministic intent router,
short-term per-session memory, a typed tool layer with permission checks, and a keyless
`web_search` tool — all wearing the FRIDAY persona (confident, dry, honest, calls you
"Boss"). The LLM sits behind a provider interface: an NVIDIA NIM adapter powers real
replies, while a `FakeLLM` keeps the entire test suite green with zero network and no API
keys. Nothing here phones home unless you point it at NVIDIA on purpose.

## Requirements

- **Python 3.12+** (developed and tested on 3.14)
- **[uv](https://docs.astral.sh/uv/)** for environments and dependencies

## Setup

```bash
make install            # uv sync --all-groups  (creates the venv, installs everything)
cp .env.example .env    # then edit .env
```

Open `.env` and choose your provider:

```bash
# fake   -> no network, no key, deterministic (great for dev/tests)
# nvidia -> real replies via NVIDIA NIM
FRIDAY_LLM_PROVIDER=fake

# only needed when FRIDAY_LLM_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-...
NVIDIA_MODEL=meta/llama-3.3-70b-instruct
```

`.env` is **gitignored** and is the only place real secrets live. `.env.example` documents
every variable with no real values. Secret fields are typed `SecretStr` and are redacted
from logs, so a key never lands in your console or log files.

## Run

```bash
make run     # uv run uvicorn friday.app:create_app --factory --reload
```

That serves the app on `http://127.0.0.1:8000`. Check it is alive (a cheap liveness
probe that reports the configured provider/model and makes **no** LLM call):

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"ok","llm_provider":"fake","model":null}
```

Then send it a turn:

```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'content-type: application/json' \
  -d '{"session_id":"s1","text":"hello"}'
```

A representative response (NVIDIA provider, persona reply):

```json
{
  "text": "Evening, Boss. All systems nominal.",
  "mode": "CONVERSATION",
  "route": {
    "mode": "CONVERSATION",
    "agent": null,
    "rationale": "conversational phrasing: hello",
    "confidence": 0.9
  },
  "audio": null
}
```

`mode` is the active conversation mode (`CONVERSATION`, `RESEARCH`, `CLARIFY`, …). `route`
is the router's deterministic decision for the turn — it is populated even when the
language backend is unreachable, because routing happens before synthesis. `audio` is
always `null` this phase; voice is a later flag.

If you set `FRIDAY_LLM_PROVIDER=fake` with the default empty script, FRIDAY still routes
your turn correctly and then honestly tells you the language backend is unavailable rather
than fabricating an answer — that is the persona working as designed, not a crash.

> **Heads-up on the NVIDIA free tier:** the first request after a cold start can take
> roughly 30 seconds while the model spins up. Subsequent turns are fast. If it consistently
> stalls, double-check the model name and key in `.env`. The cap is controlled by
> `FRIDAY_LLM_TIMEOUT_SECONDS` (default `60`); past it the call fails cleanly with an honest
> error instead of hanging.

## Make targets

| Target          | What it does                                                        |
|-----------------|---------------------------------------------------------------------|
| `make install`  | `uv sync --all-groups` — create the venv and install all deps       |
| `make test`     | Run the full test suite (`uv run pytest -q`)                        |
| `make lint`     | `uv run ruff check src tests`                                       |
| `make type`     | `uv run mypy` (`--strict`) over `src`                              |
| `make run`      | Serve the app via uvicorn with `--reload`                           |
| `make gate-0`   | Lint + types + unit tests — the Phase 0 foundation gate             |
| `make gate-1`   | Lint + types + the full suite — the Phase 1 core-loop gate          |

## What's built vs. deferred

**Built (Phases 0 + 1):**

- **Phase 0 — foundation:** typed `Settings` (env + feature flags, `SecretStr` keys),
  structured JSON logging with correlation ids and secret redaction, the `FridayError`
  hierarchy, the `LLMProvider` contract with `FakeLLM` + `FallbackLLM`, the NVIDIA NIM
  adapter (the only file allowed to import the `openai` SDK — enforced by a grep test), and
  STT/TTS provider interfaces with fakes.
- **Phase 1 — core loop:** `GraphState` / `Mode` / `RouteDecision`, the deterministic
  `route()`, short-term session memory, the typed `Tool` protocol + registry with
  permission gating, the keyless `web_search` tool, the orchestrator (memory → route →
  dispatch → persona synthesis → honest refusal), the FRIDAY persona spec, the mode graph,
  and `POST /chat` on the FastAPI factory.

**Deferred (behind flags / future phases):**

- Voice (STT/TTS, wake word) — interfaces exist, real adapters and the `/chat` `audio`
  field are off behind `FRIDAY_ENABLE_VOICE`.
- The other specialist agents — only the conversation and minimal research paths are wired;
  `agents/base.py` is the protocol the rest will implement.
- Security lockdown / hardening.
- Durable memory (Postgres + vector store) — today's memory is in-process and session-scoped.
- A dashboard / UI.
- Docker packaging.

## Project layout

```
src/friday/
  config.py            # typed Settings: env + feature flags, SecretStr keys
  logging.py           # structured JSON logging + correlation id + redaction
  errors.py            # FridayError hierarchy
  app.py               # FastAPI factory + lifespan wiring
  api/
    routes_chat.py     # POST /chat
  core/
    state.py           # Mode, RouteDecision, GraphState
    router.py          # route() -> RouteDecision
    orchestrator.py    # memory -> route -> dispatch -> persona -> refusal
    modes.py           # mode node functions
    graph.py           # mode-loop assembly
  agents/
    base.py            # Agent protocol + AgentResult
  tools/
    base.py            # Tool protocol, ToolResult, ToolError
    registry.py        # typed registry + permission gating
    web_search.py      # keyless web search tool
  memory/
    short_term.py      # in-process, session-scoped conversation buffer
  providers/
    llm.py             # LLMProvider, FakeLLM, FallbackLLM, NvidiaNIMProvider
    stt.py             # STTProvider + FakeSTT (real adapter deferred)
    tts.py             # TTSProvider + FakeTTS (real adapter deferred)
  persona/
    friday.md          # the FRIDAY persona spec
tests/
  conftest.py          # fake-provider fixtures
  unit/                # per-module unit tests
  integration/         # /chat core-loop end-to-end (against fakes)
```

The full design and the phased implementation plan live under
[`docs/superpowers/`](docs/superpowers/).

## Gate status

ruff clean · `mypy --strict` clean · **120 tests passing** (gate-1 green).
