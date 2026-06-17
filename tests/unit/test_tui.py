# © Lakshya Badjatya — Author
"""Unit tests for the terminal cockpit (TUI) command surface + REPL loop."""

from __future__ import annotations

from friday.cli import _handle_tui, build_parser
from friday.core.state import GraphState, Mode
from friday.tui.app import HELP, parse_line, render_reply, run_tui, step


class _FakeOrchestrator:
    """Echoes the input back as a CONVERSATION reply; records calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def handle(self, state: GraphState) -> GraphState:
        self.calls.append(state.user_input)
        state.mode = Mode.CONVERSATION
        state.response = f"echo: {state.user_input}"
        return state


def test_parse_line_classifies_commands_and_text() -> None:
    assert parse_line("") == ("noop", "")
    assert parse_line("   ") == ("noop", "")
    assert parse_line(":q") == ("quit", "")
    assert parse_line(":QUIT") == ("quit", "")
    assert parse_line(":help") == ("help", "")
    assert parse_line("?") == ("help", "")
    assert parse_line(":clear") == ("clear", "")
    assert parse_line(":bogus") == ("unknown", "bogus")
    assert parse_line("what's 2+2") == ("ask", "what's 2+2")


def test_render_reply_tags_with_mode_name() -> None:
    assert render_reply(Mode.CONVERSATION, "hi") == "[CONVERSATION] hi"
    assert render_reply(None, "hi") == "hi"
    assert render_reply(Mode.CONVERSATION, None) == "[CONVERSATION] (no reply)"


async def test_step_runs_a_turn_for_plain_text() -> None:
    orch = _FakeOrchestrator()
    action, out = await step(orch, "s1", "ping")
    assert action == "continue"
    assert out == "[CONVERSATION] echo: ping"
    assert orch.calls == ["ping"]


async def test_step_handles_commands_without_calling_orchestrator() -> None:
    orch = _FakeOrchestrator()
    assert await step(orch, "s", ":quit") == ("quit", "")
    assert (await step(orch, "s", ":help"))[1] == HELP
    assert (await step(orch, "s", ":clear"))[0] == "clear"
    assert "unknown command" in (await step(orch, "s", ":nope"))[1]
    assert await step(orch, "s", "") == ("continue", "")
    assert orch.calls == []  # no turn ran for any command


async def test_run_tui_loops_until_quit() -> None:
    orch = _FakeOrchestrator()
    lines = iter(["hello", ":quit"])
    out: list[str] = []
    code = await run_tui(orch, input_fn=lambda _prompt: next(lines), output_fn=out.append)
    assert code == 0
    assert any("echo: hello" in line for line in out)
    assert any("Standing by" in line for line in out)


async def test_run_tui_exits_on_eof() -> None:
    orch = _FakeOrchestrator()

    def _raise_eof(_prompt: str) -> str:
        raise EOFError

    code = await run_tui(orch, input_fn=_raise_eof, output_fn=lambda _s: None)
    assert code == 0


def test_cli_registers_tui_subcommand() -> None:
    args = build_parser().parse_args(["tui", "--session", "demo"])
    assert args.func is _handle_tui
    assert args.session == "demo"
