# FRIDAY

A local-first, provider-abstracted personal AI assistant. Phase 0 (Foundation) and
Phase 1 (Core loop) deliver a typed configuration layer, structured logging, provider
interfaces with fakes, a NVIDIA NIM LLM adapter, and a runnable core loop
(FastAPI `/chat` → orchestrator → intent router → short-term memory → keyless
`web_search` tool → persona response).

## Principles

- **Local-first** — runs on one machine, no cloud account required to develop or test.
- **Provider-abstracted** — no LLM/STT/TTS SDK is imported in `agents/` or `core/`
  (enforced by a grep test). The only file allowed to import `openai` is
  `src/friday/providers/llm.py`.
- **Feature flags default off** — only the core loop and one tool are enabled.
- **Typed everywhere** — pydantic v2 at every boundary; `mypy --strict` on `src/`.
- **TDD per module** — failing test → minimal implementation → green.

## Requirements

- Python >= 3.12 (developed and tested on 3.14)
- [uv](https://docs.astral.sh/uv/) for environment and dependency management

## Setup

```bash
make install            # uv sync --all-groups
cp .env.example .env     # then fill in secrets (NVIDIA_API_KEY, etc.)
```

`.env` is gitignored and is the only place real secrets live. `.env.example`
documents every variable with no real values.

## Common commands

| Command       | Action                                              |
|---------------|-----------------------------------------------------|
| `make install`| Sync dependencies (`uv sync --all-groups`)          |
| `make test`   | Run the full test suite                             |
| `make lint`   | `ruff check src tests`                              |
| `make fmt`    | `ruff format src tests`                             |
| `make type`   | `mypy --strict` on `src`                           |
| `make run`    | Run the FastAPI app via uvicorn (`--reload`)        |
| `make gate-0` | Lint + types + unit tests (Phase 0 gate)            |
| `make gate-1` | Lint + types + full suite (Phase 1 gate)            |

## Configuration

All configuration is read from the environment (and `.env`) by
`src/friday/config.py` via `pydantic-settings`. Secret fields are typed
`SecretStr` and are redacted from logs. The default LLM provider is `fake`, so the
entire test suite runs green with zero network access or API keys. Set
`FRIDAY_LLM_PROVIDER=nvidia` and provide `NVIDIA_API_KEY` to use the real
NVIDIA NIM adapter.

## Layout

```
src/friday/
  config.py · logging.py · errors.py · app.py
  api/routes_chat.py
  core/{state,router,orchestrator,modes,graph}.py
  agents/base.py
  tools/{base,registry,web_search}.py
  memory/short_term.py
  providers/{llm,stt,tts}.py
  persona/friday.md
tests/{conftest.py, unit/, integration/}
```
