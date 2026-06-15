# FRIDAY Dashboard (optional UI)

A small [Streamlit](https://streamlit.io/) operator console for a running FRIDAY
API. It is a **separate UI process** — not part of the `friday` package, never
imported by the test suite, and its dependencies (`streamlit`, `requests`) are
kept out of `pyproject.toml` / the uv lock. The dashboard only ever talks to the
API over HTTP, so the gate tests the admin endpoints it consumes, not the UI.

## Install & run

```bash
make install-dashboard   # uv pip install -r requirements-dashboard.txt
make run                 # start the FRIDAY API on http://127.0.0.1:8000 (separate shell)
make dashboard           # uv run streamlit run dashboard/app.py
```

Streamlit opens a browser tab. Set the **API base URL** in the sidebar (default
`http://127.0.0.1:8000`); the sidebar shows whether the API is reachable.

## Panels

| Tab               | Reads / writes            | Shows                                            |
|-------------------|---------------------------|-------------------------------------------------|
| Live state        | `GET /admin/state`        | Active sessions, current modes, memory stats     |
| Conversation      | `POST /chat`              | Send a turn, render the reply, keep a session log |
| Tool-call audit   | `GET /admin/audit`        | Recent tool calls with redacted args             |
| Traces            | `GET /admin/traces`       | Recent per-request spans with timings            |
| Metrics           | `GET /admin/metrics`      | Counter snapshot (requests, tool calls, errors)  |
| Feature flags     | `GET` / `POST /admin/flags` | List flags; toggle boolean flags live          |

## Robustness

If the API is down or unreachable, the dashboard shows a friendly banner and
**never crashes** — every HTTP call is funneled through guarded helpers that
catch connection errors and non-2xx responses. Start the API (`make run`), then
refresh.
