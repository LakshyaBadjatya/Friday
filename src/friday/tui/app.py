# © Lakshya Badjatya — Author
"""A dependency-light terminal cockpit for FRIDAY — a stdlib REPL surface.

Drives the **in-process** :class:`~friday.core.orchestrator.Orchestrator` (no
running server needed): each non-command line is one turn through ``handle``, so
the TUI answers with the same brain the HTTP ``/chat`` route uses — offline on
the ``fake`` build, live when providers are configured.

The line grammar is parsed by the pure :func:`parse_line`, and one turn is run by
the pure-ish :func:`step` (its only effect is the injected orchestrator call), so
the command surface is unit-testable without a terminal. :func:`run_tui` is the
thin I/O loop on top, with ``input``/``print`` injected so even it can be driven
headlessly in a test.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from friday.core.state import GraphState

BANNER = "FRIDAY — terminal cockpit. Type a message, or :help for commands."

HELP = (
    "commands:\n"
    "  <text>      ask FRIDAY (one turn through the core loop)\n"
    "  :help, :h   show this help\n"
    "  :clear      clear the screen\n"
    "  :quit, :q   exit\n"
)


def parse_line(line: str) -> tuple[str, str]:
    """Map an input line to ``(kind, argument)``.

    ``kind`` is one of ``ask``/``help``/``clear``/``quit``/``noop``/``unknown``.
    A leading ``:`` marks a command; everything else is an ``ask`` carrying the
    raw text. Blank lines are ``noop`` so the loop just re-prompts.
    """
    text = line.strip()
    if not text:
        return ("noop", "")
    low = text.lower()
    if low in (":q", ":quit", ":exit"):
        return ("quit", "")
    if low in (":help", ":h", "?"):
        return ("help", "")
    if low == ":clear":
        return ("clear", "")
    if text.startswith(":"):
        return ("unknown", text[1:])
    return ("ask", text)


def render_reply(mode: object, response: str | None) -> str:
    """Render a turn's reply with a ``[MODE]`` tag (mode name, not the enum repr)."""
    name = getattr(mode, "name", None) or (str(mode) if mode is not None else "")
    tag = f"[{name}] " if name else ""
    return tag + (response or "(no reply)")


class _Orchestratorish(Protocol):
    """Structural type: anything with an async ``handle(state) -> state``."""

    async def handle(self, state: GraphState) -> GraphState: ...


async def step(
    orchestrator: _Orchestratorish, session_id: str, line: str
) -> tuple[str, str]:
    """Process one input line; return ``(action, output)``.

    ``action`` is ``quit`` / ``clear`` / ``continue`` (the loop acts on it);
    ``output`` is the text to show (possibly empty). An ``ask`` runs exactly one
    orchestrator turn and renders its reply. Pure except for that one awaited call.
    """
    kind, arg = parse_line(line)
    if kind == "quit":
        return ("quit", "")
    if kind == "help":
        return ("continue", HELP)
    if kind == "clear":
        return ("clear", "")
    if kind == "noop":
        return ("continue", "")
    if kind == "unknown":
        return ("continue", f"unknown command: :{arg} (try :help)")
    state = GraphState(session_id=session_id, user_input=arg)
    result = await orchestrator.handle(state)
    return ("continue", render_reply(result.mode, result.response))


async def run_tui(
    orchestrator: _Orchestratorish,
    *,
    session_id: str = "tui",
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Run the REPL until EOF/``:quit``; return a process exit code (always 0).

    ``input_fn``/``output_fn`` are injected so the loop can be exercised
    headlessly; in production they are the builtin ``input``/``print``.
    """
    output_fn(BANNER)
    output_fn(HELP)
    while True:
        try:
            line = input_fn("friday> ")
        except (EOFError, KeyboardInterrupt):  # Ctrl-D / Ctrl-C exits cleanly
            output_fn("")
            return 0
        action, out = await step(orchestrator, session_id, line)
        if action == "quit":
            output_fn("Standing by, Boss.")
            return 0
        if action == "clear":
            output_fn("\033[2J\033[H")
            continue
        if out:
            output_fn(out)
