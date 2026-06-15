# FRIDAY

FRIDAY is a local-first, provider-abstracted personal AI assistant you run on your own
machine. A FastAPI `POST /chat` endpoint drives an orchestrator that classifies each turn
with a deterministic router and dispatches to five specialist agents (analysis, knowledge,
automation, device, alerting), a defensive security-lockdown flow, short-term memory and
persistent SQLite-backed long-term + vector memory, a typed tool layer with permission
checks and a confirm-step on anything side-effecting — all wearing the FRIDAY persona
(confident, dry, honest, calls you "Boss"). On top of that core: an optional voice pipeline
(wake word → Whisper → TTS, with barge-in), a 3D Studio (describe a model, explore it with
hand gestures and voice, download GLB/STL/OBJ), per-request tracing + a tool-call audit + an
admin API with a Streamlit dashboard, and gateway auth + rate-limiting + container packaging.

The LLM sits behind a provider interface: NVIDIA NIM powers real replies with Gemini as a
fallback, while a `FakeLLM` keeps the entire test suite green with zero network and no API
keys. Nothing phones home unless you point it at a real provider on purpose. Every non-core
capability is behind a feature flag, default **off**.

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

`mode` is the active conversation mode (`CONVERSATION`, `RESEARCH`, `AUTOMATION`,
`DEVICE_CONTROL`, `ALERTING`, `SECURITY_LOCKDOWN`, `CLARIFY`, …). `route` is the router's
deterministic decision for the turn — it is populated even when the language backend is
unreachable, because routing happens before synthesis. `audio` is `null` on the text path;
spoken audio comes from the voice pipeline (below), off by default.

If you set `FRIDAY_LLM_PROVIDER=fake` with the default empty script, FRIDAY still routes
your turn correctly and then honestly tells you the language backend is unavailable rather
than fabricating an answer — that is the persona working as designed, not a crash.

> **Heads-up on the NVIDIA free tier:** the first request after a cold start can take
> roughly 30 seconds while the model spins up. Subsequent turns are fast. If it consistently
> stalls, double-check the model name and key in `.env`. The cap is controlled by
> `FRIDAY_LLM_TIMEOUT_SECONDS` (default `120`); past it the call fails cleanly with an honest
> error instead of hanging. With `FRIDAY_LLM_FALLBACK_PROVIDER=gemini` and a Gemini key, a
> timed-out NVIDIA call falls through to Gemini before giving up.

## Voice (optional, off by default)

FRIDAY ships a full voice pipeline — wake word → microphone capture → Whisper STT
→ orchestrator → TTS, with **barge-in** (start talking over FRIDAY and playback
stops) — but it is **off by default** and its heavy backends are kept out of the
core install so `uv sync` and the test suite stay fast and dependency-light.

Turn it on and install the extras:

```bash
export FRIDAY_ENABLE_VOICE=true   # the master flag; everything voice is gated on it
make install-voice                # uv pip install -r requirements-voice.txt
```

`make install-voice` pulls in the optional backends (`faster-whisper`,
`openwakeword`, `piper-tts`, `sounddevice`) listed in `requirements-voice.txt`.
These are **not** in `pyproject.toml`/the uv lock; the real adapters lazy-import
them and raise a clear "run `make install-voice`" error if they're missing, so the
package still imports and tests still pass without them. Live voice also needs a
**working microphone** (sounddevice / PortAudio).

When `FRIDAY_ENABLE_VOICE` is set, two endpoints come alive (they return `404` /
refuse the socket while the flag is off):

- `POST /voice` — one spoken turn: send audio as base64 (`{"audio_b64": "..."}`)
  or a multipart file upload; get back `{transcript, text, mode, audio_b64}`.
- `WS /ws/voice` — a minimal websocket scaffold for streaming + barge-in
  signaling (full duplex streaming UX lands in a later tier).

Pick a TTS backend with `FRIDAY_TTS_PROVIDER` (`piper` | `elevenlabs` | `fake`);
`elevenlabs` needs `ELEVENLABS_API_KEY`. With the offline `fake` LLM provider the
voice path uses `FakeSTT`/`FakeTTS` so it runs end-to-end with zero models.

## Perception (optional, off by default, privacy-heavy)

FRIDAY can **see**: object detection (YOLO), screen/clipboard OCR (Tesseract), the
system clipboard, and full-screen capture — composed into a single
`describe_screen()` pass (capture → OCR + detect on the same image). It is **off by
default** and is **privacy-heavy**: when enabled it can **read your screen and
clipboard**, so only turn it on if you intend FRIDAY to observe them.

```bash
export FRIDAY_ENABLE_PERCEPTION=true   # the master flag; everything perception is gated on it
make install-perception                # uv pip install -r requirements-perception.txt
```

`make install-perception` pulls in the optional backends (`opencv-python`,
`ultralytics`, `pytesseract`, `pillow`, `mss`, `pyperclip`) listed in
`requirements-perception.txt`. These are **not** in `pyproject.toml`/the uv lock;
the real adapters lazy-import them and raise a clear "run `make install-perception`"
error if they're missing, so the package still imports and tests still pass without
them. OCR also needs the **`tesseract` binary** on the host (e.g.
`apt install tesseract-ocr`). The app wires deterministic **fake** providers by
default, so the offline build needs **no heavy library and performs no real
capture**.

When `FRIDAY_ENABLE_PERCEPTION` is set, five endpoints come alive (they return
`404` while the flag is off):

- `POST /perception/vision` — `{ "image_b64": "..." }` → `{detections, count}`.
- `POST /perception/ocr` — `{ "image_b64": "..." }` → `{text}`.
- `GET  /perception/clipboard` → `{text}`; `POST /perception/clipboard`
  `{ "text": "..." }` → `{ok}`.
- `POST /perception/screen` — capture the screen and describe it →
  `{ocr_text, detections}`.

## Dashboard (optional)

FRIDAY ships a small [Streamlit](https://streamlit.io/) operator console that reads the
live admin surface: current state, the conversation log, the tool-call audit, per-request
traces, metrics, and feature-flag toggles. It is a **separate UI process** — not part of
the `friday` package, never imported by the test suite, and its deps (`streamlit`,
`requests`) are kept **out** of `pyproject.toml` / the uv lock (so `uv sync` and the gate
stay fast). The gate tests the admin endpoints the dashboard consumes, not the UI itself.

```bash
make install-dashboard   # uv pip install -r requirements-dashboard.txt
make run                 # start the API on http://127.0.0.1:8000 (in one shell)
make dashboard           # uv run streamlit run dashboard/app.py (in another)
```

Set the **API base URL** in the sidebar (default `http://127.0.0.1:8000`). If the API is
down the dashboard shows a friendly banner and never crashes — start it with `make run`
and refresh. See [`dashboard/README.md`](dashboard/README.md) for the panel-by-endpoint
map.

## 3D Studio (optional, off by default)

FRIDAY ships a **3D Studio**: describe a model by **text or voice**, watch it built
live in an interactive [Three.js](https://threejs.org/) canvas you explore with
**hand gestures** (webcam, via [MediaPipe Hands](https://developers.google.com/mediapipe))
and **voice** (the browser's Web Speech API), then **download** it as GLB / STL / OBJ.

It is **off by default** behind a flag and adds **no Node build step** — the page
loads Three.js + MediaPipe from CDNs via an importmap and is served by FastAPI as a
plain static file (like the dashboard, it is a UI surface that the gate
syntax-checks rather than browser-tests).

```bash
export FRIDAY_ENABLE_STUDIO=true   # the master flag; the route is 404 while it's off
make run                           # serve the API on http://127.0.0.1:8000
```

Then open **`http://127.0.0.1:8000/studio`** and **allow camera + microphone** when
prompted (both are optional — deny either and the Studio still works, it just shows
a hint and disables that input).

**Safety:** the model never emits JavaScript. The `POST /studio/generate` endpoint
returns a **validated JSON scene-graph** (a pydantic-checked `Scene`); the browser
maps that trusted-shape data to Three.js meshes. **No model output is ever
`eval`'d.**

**Free by default:** the *Fast* quality uses FRIDAY's existing LLM to author the
scene procedurally — no extra cost or key. *Hi-Fi* is an optional external mesh
provider that is lazy + flagged and needs a **free-tier key**; if the key is absent
or over quota it **falls back to the free procedural path** rather than erroring you
into a paywall.

### Cheat-sheet

| Hand gesture (webcam)        | Action                          |
|------------------------------|---------------------------------|
| **Pinch** (thumb + index)    | Zoom                            |
| **Open hand, move**          | Rotate the model                |
| **Two-hand spread**          | Scale                           |
| **Index point**              | Highlight the nearest part      |

| Voice command (mic)                       | Action                       |
|-------------------------------------------|------------------------------|
| “**make / create / build** a red sphere”  | Generate a new model         |
| “**rotate**” · “**reset**”                | Rotate · reset the view      |
| “**zoom in**” · “**zoom out**”            | Dolly the camera             |
| “**wireframe**”                           | Toggle wireframe             |
| “**color it blue**”                       | Recolor model/selection      |
| “**download glb / stl / obj**”            | Export the current scene     |

The same actions are available as on-screen buttons (prompt box, mic toggle,
Fast/Hi-Fi quality switch, GLB/STL/OBJ downloads). A status HUD shows the live
gesture, voice state, last heard phrase, and API connection. If the API, camera, or
mic is unavailable the Studio shows a friendly note and never hard-crashes.

## Continuous integration

Every push and pull request runs the same gate in GitHub Actions
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)): a single `gate` job on
`ubuntu-latest` that installs uv, pins **Python 3.12**, runs `uv sync --all-groups`,
then executes **ruff** (`uv run ruff check src tests`), **`mypy --strict`** (`uv run
mypy`), and the **full pytest suite** (`uv run pytest -q`) as separate steps so a
failure pinpoints which check broke. Any non-zero step fails the build.

## Make targets

| Target              | What it does                                                        |
|---------------------|---------------------------------------------------------------------|
| `make install`      | `uv sync --all-groups` — create the venv and install all deps       |
| `make install-voice`| `uv pip install -r requirements-voice.txt` — optional voice backends|
| `make install-perception`| `uv pip install -r requirements-perception.txt` — optional perception backends|
| `make install-dashboard`| `uv pip install -r requirements-dashboard.txt` — optional UI deps|
| `make dashboard`    | `uv run streamlit run dashboard/app.py` — launch the operator console|
| `make test`         | Run the full test suite (`uv run pytest -q`)                        |
| `make lint`         | `uv run ruff check src tests`                                       |
| `make type`         | `uv run mypy` (`--strict`) over `src`                              |
| `make run`          | Serve the app via uvicorn with `--reload`                           |
| `make docker-build` | `docker build -t friday .` — build the container image              |
| `make docker-up`    | `docker compose up -d` — start the local container stack            |
| `make docker-down`  | `docker compose down` — stop the stack (the data volume is kept)    |
| `make gate-0` … `gate-7` | Lint + types + tests — the per-phase gates (0 foundation, 1 core loop, 2 agents, 3 voice, 4 memory, 5 dashboard, 6 hardening, 7 studio) |

## Deployment

FRIDAY is **local-first**: the default deployment is a single container running the
FastAPI app against a SQLite database on a persistent named volume. No cloud account
and no external database are required. Full details and verification steps live in
[`docs/DEPLOY.md`](docs/DEPLOY.md).

```bash
cp .env.example .env     # then edit: provider, keys, flags (see below)
make docker-build        # docker build -t friday .
make docker-up           # docker compose up -d
curl -s http://localhost:8000/health
make docker-down         # docker compose down (the friday-data volume is preserved)
```

The image is multi-stage and uv-based on `python:3.12-slim` (3.12 is the safe
container base; local dev is on 3.14), installs production deps from the committed
`uv.lock`, runs as a **non-root** user, exposes `8000`, and serves
`uvicorn friday.app:create_app --factory`. **No secrets are baked in** — `.env` is
excluded by `.dockerignore` and supplied at run time via compose's `env_file`. The
SQLite db persists in the named volume `friday-data` mounted at `/app/data`, and the
compose healthcheck probes `/health` (which is exempt from both auth and
rate-limiting). A commented-out `postgres`/pgvector service documents the future
durable-memory swap; the local-first default stays single-service.

> **Honest note (binding):** `docker build` / `docker compose up` were **not**
> executed or verified in the environment that produced these files — **Docker is
> not installed there.** The `Dockerfile`/`docker-compose.yml`/`.dockerignore` were
> written and syntactically validated only (compose parsed with PyYAML). Verify the
> build and runtime on a host that has Docker; the steps are in `docs/DEPLOY.md`.

**Cloud (deferred, per spec §15):** a hosted deployment is a *later* target with
**no Terraform/Pulumi, no multi-cloud, no Kubernetes**. When it's needed, ship this
same image to a **single** managed container runtime — Cloud Run / ECS-Fargate /
Azure Container Apps — choosing one, not all. Prove the loops locally first.

## What's built vs. deferred

**Built:**

- **Phase 0 — foundation:** typed `Settings` (env + feature flags, `SecretStr` keys),
  structured JSON logging with correlation ids and secret redaction, the `FridayError`
  hierarchy, the `LLMProvider` contract with `FakeLLM` + `FallbackLLM`, the NVIDIA NIM +
  Gemini adapters (the only files allowed to import the `openai` SDK — enforced by a grep
  test), and STT/TTS provider interfaces with fakes.
- **Phase 1 — core loop:** `GraphState` / `Mode` / `RouteDecision`, the deterministic
  `route()`, short-term session memory, the typed `Tool` protocol + registry with permission
  gating, the keyless `web_search` tool, the orchestrator (memory → route → dispatch →
  persona synthesis → honest refusal), the FRIDAY persona spec, the mode graph, and
  `POST /chat`. Plus hardening: a per-call LLM timeout, `GET /health`, and `/chat` input
  validation.
- **Phase 2 — agents + tools:** five specialist agents (analysis with confidence tags and
  no fabricated probabilities, knowledge with grounded citations, automation with a hard
  step cap, device with an allowlist, alerting with dedupe + rate-limit), the `notify` /
  `home` / `security` tools, the **confirm-step** required before any side-effecting
  non-idempotent tool runs, and the **security lockdown** subgraph (revoke → kill → notify,
  audited).
- **Phase 3 — voice:** wake word, Whisper STT, Piper/ElevenLabs TTS, the capture→STT→
  orchestrator→TTS pipeline with **barge-in**. Off behind `FRIDAY_ENABLE_VOICE`.
- **Phase 4 — memory:** persistent **SQLite** long-term store (facts/tasks/audit) and a
  SQLite vector store behind a `VectorStore` interface, an `EmbeddingProvider` (NVIDIA real
  / deterministic fake), a **write-consent** policy (sensitive writes need confirmation), and
  a **"forget X"** command that wipes facts + vectors.
- **Phase 5 — observability + dashboard:** per-request **tracing**, a **tool-call audit**,
  **metrics**, an `/admin` API (flags/state/audit/traces/metrics), and a Streamlit dashboard.
- **Phase 6 — hardening + packaging:** gateway **auth** + **rate-limiting** (both off by
  default, `/health` exempt), a load/error-budget smoke, and a `Dockerfile` + `docker-compose.yml`.
- **3D Studio:** describe a model → validated JSON scene-graph → interactive Three.js canvas
  with MediaPipe hand-tracking + Web Speech voice + GLB/STL/OBJ export. Off behind
  `FRIDAY_ENABLE_STUDIO`.
- **Gemini fallback:** NVIDIA primary with Gemini as the `FallbackLLM` secondary.

**Deferred / documented swaps:**

- **Real voice/3D models on this machine** — the heavy backends (`faster-whisper`,
  `openwakeword`, `piper`, MediaPipe) ship as optional installs, not core deps; the logic is
  fully tested against fakes and a validated scene-graph.
- **Durable memory on Postgres + pgvector and the Chroma vector backend** — wired as lazy,
  flagged adapter stubs; the local-first default is SQLite (a commented-out `postgres`
  service documents the swap).
- **`docker compose up` was not run here** (Docker isn't installed in this build env) — the
  files are written and the compose parses; verify on a Docker host (see Deployment).
- **A hosted cloud deployment** — a single managed container runtime later (no
  Terraform/multi-cloud/Kubernetes); see the Deployment section.
- **Gemini fallback is wired but quota-gated** — the provided key authenticates but returns
  `429` on a zero free-tier quota until billing is enabled on it.

## Project layout

```
src/friday/
  config.py · logging.py · errors.py · app.py     # settings, logging, errors, FastAPI factory
  api/
    routes_chat.py · routes_health.py · routes_admin.py · routes_voice.py · routes_studio.py
    ws.py · middleware.py                          # websocket scaffold; auth + rate-limit
  core/
    state.py · router.py · orchestrator.py · modes.py · graph.py · security.py
  agents/
    base.py · analysis.py · knowledge.py · automation.py · device.py · alerting.py
  tools/
    base.py · registry.py · web_search.py · notify.py · home.py · security.py
  memory/
    short_term.py · long_term.py · vector.py       # in-process + SQLite long-term + vector
  providers/
    llm.py · embeddings.py · stt.py · tts.py        # NVIDIA/Gemini, embeddings, voice fakes+adapters
  voice/
    wake_word.py · capture.py · vad.py · pipeline.py · fixtures.py
  studio/
    scene.py · generator.py · static/              # JSON scene-graph + the Three.js canvas
  observability/
    tracing.py · audit.py · metrics.py
  persona/friday.md
dashboard/app.py                                   # Streamlit operator console (separate process)
Dockerfile · docker-compose.yml                    # container packaging
tests/  unit/ · integration/ · conftest.py
```

The full design, the per-phase implementation plans, and the original build spec live under
[`docs/superpowers/`](docs/superpowers/) and [`FRIDAY-build-spec.md`](FRIDAY-build-spec.md).

## Gate status

ruff clean · `mypy --strict` clean (56 source files) · **433 tests passing** · SDK-isolation
guard green. Gates 0–7 all green; built phase-by-phase via TDD, each behind a feature flag.
