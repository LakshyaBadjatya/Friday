"""FRIDAY dashboard — a Streamlit operator console for the admin API.

This is a **separate UI process**, not part of the ``friday`` package and never
imported by the test suite. It talks to a running FRIDAY API purely over HTTP
(``requests``), so the only dependencies are ``streamlit`` + ``requests`` from
``requirements-dashboard.txt`` (``make install-dashboard``) — neither is in
``pyproject.toml`` / the uv lock. Run it with ``make dashboard`` once the API is
up (``make run``).

It reads everything from the Phase-5 admin surface:

* **Live state**       ``GET  /admin/state``    active sessions, modes, memory
* **Conversation**     ``POST /chat``           send a turn, keep a session log
* **Tool-call audit**  ``GET  /admin/audit``    recent tool calls (args redacted)
* **Traces**           ``GET  /admin/traces``   recent per-request spans + timings
* **Metrics**          ``GET  /admin/metrics``  counter snapshot
* **Feature flags**    ``GET/POST /admin/flags`` list + toggle runtime overrides

Design rule: a down or misbehaving API must **never** crash the UI. Every call
goes through :func:`api_get` / :func:`api_post`, which return a tagged result and
surface a friendly message instead of a stack trace.
"""

from __future__ import annotations

import uuid
from typing import Any

import requests
import streamlit as st

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 10  # seconds; keep the UI responsive even if the API stalls


# --------------------------------------------------------------------------- #
# HTTP plumbing — every network touch is funneled through these two helpers so
# a dead API degrades to a friendly banner instead of a traceback.
# --------------------------------------------------------------------------- #
def api_get(base_url: str, path: str) -> tuple[bool, Any]:
    """GET ``{base_url}{path}``; return ``(ok, payload_or_error_message)``."""
    try:
        resp = requests.get(f"{base_url}{path}", timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        return False, _friendly_error(base_url, exc)
    return _interpret(resp)


def api_post(base_url: str, path: str, payload: dict[str, Any]) -> tuple[bool, Any]:
    """POST JSON to ``{base_url}{path}``; return ``(ok, payload_or_error)``."""
    try:
        resp = requests.post(f"{base_url}{path}", json=payload, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        return False, _friendly_error(base_url, exc)
    return _interpret(resp)


def _interpret(resp: requests.Response) -> tuple[bool, Any]:
    """Turn an HTTP response into ``(ok, json_or_message)`` without raising."""
    if resp.status_code >= 400:
        detail = _safe_json(resp)
        return False, f"API returned {resp.status_code}: {detail}"
    return True, _safe_json(resp)


def _safe_json(resp: requests.Response) -> Any:
    """Decode JSON if possible, else fall back to the raw text body."""
    try:
        return resp.json()
    except ValueError:
        return resp.text


def _friendly_error(base_url: str, exc: Exception) -> str:
    """A human message for a connection-level failure (API down, etc.)."""
    if isinstance(exc, requests.exceptions.ConnectTimeout | requests.exceptions.ReadTimeout):
        return f"Timed out talking to the API at {base_url}. Is it overloaded or asleep?"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return (
            f"Could not reach the FRIDAY API at {base_url}. "
            "Start it with `make run`, then refresh."
        )
    return f"Request to {base_url} failed: {exc}"


def render_api_down(message: str) -> None:
    """Standard 'API is unreachable' banner used by every panel."""
    st.warning(message)
    st.caption("The dashboard is read-only chrome; the API is the source of truth.")


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state() -> None:
    """Seed Streamlit session state (chat log + a stable session id)."""
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []  # list[dict]: {role, text, mode?, route?}
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"dashboard-{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------- #
# Panels — each is defensive: it handles a down API and unexpected shapes.
# --------------------------------------------------------------------------- #
def panel_live_state(base_url: str) -> None:
    """**Live state** — active sessions, current modes, memory sizes."""
    st.subheader("Live state")
    if st.button("Refresh state", key="refresh_state"):
        pass  # any rerun re-fetches; the button is just an affordance
    ok, data = api_get(base_url, "/admin/state")
    if not ok:
        render_api_down(str(data))
        return
    if not isinstance(data, dict):
        st.json(data)
        return

    sessions = data.get("sessions")
    if isinstance(sessions, list) and sessions:
        st.caption("Active sessions")
        st.dataframe(sessions, use_container_width=True)
    modes = data.get("modes")
    if modes is not None:
        st.caption("Current modes")
        st.json(modes)
    memory = data.get("memory")
    if memory is not None:
        st.caption("Memory stats")
        st.json(memory)
    # Always show the raw payload too so nothing is hidden by our assumptions.
    with st.expander("Raw /admin/state payload"):
        st.json(data)


def panel_conversation(base_url: str) -> None:
    """**Conversation** — POST a turn to ``/chat`` and keep a session log."""
    st.subheader("Conversation")
    st.caption(f"Session id: `{st.session_state.session_id}`")

    cols = st.columns([4, 1])
    with cols[0]:
        text = st.text_input("Say something to FRIDAY", key="chat_input")
    with cols[1]:
        st.write("")  # vertical alignment with the text box
        send = st.button("Send", key="chat_send", use_container_width=True)

    if send and text.strip():
        ok, data = api_post(
            base_url,
            "/chat",
            {"session_id": st.session_state.session_id, "text": text},
        )
        st.session_state.chat_log.append({"role": "you", "text": text})
        if ok and isinstance(data, dict):
            st.session_state.chat_log.append(
                {
                    "role": "friday",
                    "text": data.get("text", ""),
                    "mode": data.get("mode"),
                    "route": data.get("route"),
                }
            )
        else:
            st.session_state.chat_log.append({"role": "error", "text": str(data)})

    if st.button("Clear log", key="chat_clear"):
        st.session_state.chat_log = []

    st.divider()
    for entry in st.session_state.chat_log:
        role = entry.get("role")
        if role == "you":
            st.markdown(f"**You:** {entry['text']}")
        elif role == "friday":
            suffix = f"  _( {entry['mode']} )_" if entry.get("mode") else ""
            st.markdown(f"**FRIDAY:**{suffix} {entry['text']}")
            if entry.get("route"):
                with st.expander("route decision"):
                    st.json(entry["route"])
        else:
            st.error(entry["text"])


def panel_audit(base_url: str) -> None:
    """**Tool-call audit** — recent tool calls with redacted args."""
    st.subheader("Tool-call audit")
    ok, data = api_get(base_url, "/admin/audit")
    if not ok:
        render_api_down(str(data))
        return
    rows = _extract_rows(data, keys=("audit", "tool_calls", "rows", "items"))
    if not rows:
        st.info("No tool calls recorded yet.")
        with st.expander("Raw /admin/audit payload"):
            st.json(data)
        return
    st.dataframe(rows, use_container_width=True)
    # Security audit, if the endpoint also returns one alongside tool calls.
    if isinstance(data, dict):
        sec = data.get("security") or data.get("security_audit")
        if sec:
            st.caption("Security audit")
            st.dataframe(sec, use_container_width=True)


def panel_traces(base_url: str) -> None:
    """**Traces** — recent per-request traces with spans + timings."""
    st.subheader("Traces")
    ok, data = api_get(base_url, "/admin/traces")
    if not ok:
        render_api_down(str(data))
        return
    traces = _extract_rows(data, keys=("traces", "rows", "items"))
    if not traces:
        st.info("No traces yet — send a turn on the Conversation tab, then refresh.")
        with st.expander("Raw /admin/traces payload"):
            st.json(data)
        return
    for trace in traces:
        if not isinstance(trace, dict):
            st.json(trace)
            continue
        cid = trace.get("correlation_id", "?")
        mode = trace.get("mode")
        header = f"trace {cid}" + (f" · {mode}" if mode else "")
        with st.expander(header):
            spans = trace.get("spans") or []
            table = [
                {
                    "span": s.get("name"),
                    "start": s.get("start"),
                    "end": s.get("end"),
                    "ms": _duration_ms(s),
                    "attrs": s.get("attrs"),
                }
                for s in spans
                if isinstance(s, dict)
            ]
            if table:
                st.dataframe(table, use_container_width=True)
            else:
                st.json(trace)


def panel_metrics(base_url: str) -> None:
    """**Metrics** — the counter snapshot from ``/admin/metrics``."""
    st.subheader("Metrics")
    ok, data = api_get(base_url, "/admin/metrics")
    if not ok:
        render_api_down(str(data))
        return
    if not isinstance(data, dict):
        st.json(data)
        return

    scalar = {k: v for k, v in data.items() if isinstance(v, int | float)}
    if scalar:
        cols = st.columns(max(1, min(len(scalar), 4)))
        for i, (name, value) in enumerate(scalar.items()):
            cols[i % len(cols)].metric(name.replace("_", " "), value)
    nested = {k: v for k, v in data.items() if not isinstance(v, int | float)}
    if nested:
        st.caption("By-mode / breakdown counters")
        st.json(nested)
    if not scalar and not nested:
        st.info("No metrics reported.")


def panel_flags(base_url: str) -> None:
    """**Feature flags** — list flags and toggle runtime overrides."""
    st.subheader("Feature flags")
    ok, data = api_get(base_url, "/admin/flags")
    if not ok:
        render_api_down(str(data))
        return

    flags = _normalize_flags(data)
    if not flags:
        st.info("No feature flags reported.")
        with st.expander("Raw /admin/flags payload"):
            st.json(data)
        return

    st.caption("Boolean flags toggle live; non-boolean values are shown read-only.")
    for name, value in flags.items():
        if isinstance(value, bool):
            new_value = st.checkbox(name, value=value, key=f"flag_{name}")
            if new_value != value:
                post_ok, post_data = api_post(
                    base_url, "/admin/flags", {"name": name, "value": new_value}
                )
                if post_ok:
                    st.success(f"Set `{name}` = {new_value}")
                    st.rerun()
                else:
                    render_api_down(str(post_data))
        else:
            st.text_input(name, value=str(value), key=f"flag_{name}", disabled=True)


# --------------------------------------------------------------------------- #
# Small shape-tolerant helpers
# --------------------------------------------------------------------------- #
def _extract_rows(data: Any, keys: tuple[str, ...]) -> list[Any]:
    """Pull a list of rows whether the payload is a bare list or wrapped."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, list):
                return value
    return []


def _normalize_flags(data: Any) -> dict[str, Any]:
    """Accept ``{name: value}`` or ``{"flags": {...}}`` / list of pairs."""
    if isinstance(data, dict):
        inner = data.get("flags")
        if isinstance(inner, dict):
            return inner
        if isinstance(inner, list):
            return _pairs_to_dict(inner)
        # Bare mapping of flag -> value, skipping obvious envelope keys.
        return {k: v for k, v in data.items() if k not in {"flags", "ok", "status"}}
    if isinstance(data, list):
        return _pairs_to_dict(data)
    return {}


def _pairs_to_dict(items: list[Any]) -> dict[str, Any]:
    """Turn ``[{"name":..., "value":...}, ...]`` into a flat dict."""
    out: dict[str, Any] = {}
    for item in items:
        if isinstance(item, dict) and "name" in item:
            out[str(item["name"])] = item.get("value")
    return out


def _duration_ms(span: dict[str, Any]) -> float | None:
    """Best-effort span duration in ms from numeric start/end timestamps."""
    start, end = span.get("start"), span.get("end")
    if isinstance(start, int | float) and isinstance(end, int | float):
        return round((end - start) * 1000, 3)
    return None


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """Wire the sidebar + tabs and dispatch to each panel."""
    st.set_page_config(page_title="FRIDAY Dashboard", page_icon="🛰️", layout="wide")
    init_state()

    st.title("FRIDAY — operator console")

    with st.sidebar:
        st.header("Connection")
        base_url = st.text_input("API base URL", value=DEFAULT_BASE_URL).rstrip("/")
        ok, health = api_get(base_url, "/health")
        if ok:
            provider = health.get("llm_provider") if isinstance(health, dict) else None
            st.success(f"API reachable · provider: {provider or 'unknown'}")
        else:
            st.error("API unreachable")
            st.caption(str(health))
        st.caption("Start the API with `make run`, then use the tabs.")

    state_tab, chat_tab, audit_tab, traces_tab, metrics_tab, flags_tab = st.tabs(
        ["Live state", "Conversation", "Tool-call audit", "Traces", "Metrics", "Feature flags"]
    )
    with state_tab:
        panel_live_state(base_url)
    with chat_tab:
        panel_conversation(base_url)
    with audit_tab:
        panel_audit(base_url)
    with traces_tab:
        panel_traces(base_url)
    with metrics_tab:
        panel_metrics(base_url)
    with flags_tab:
        panel_flags(base_url)


if __name__ == "__main__":
    main()
