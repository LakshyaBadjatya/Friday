"""Unit tests for :class:`friday.broker.Broker`.

Exercises the broker pipeline against fakes (a fake registry + a fake secret
provider): argument validation, reversibility classification, the fail-closed
permission gate (deny-by-default + confirmation for irreversible actions),
secret injection that never leaks into the result or the audit, and the
hash-chained audit row written per dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from friday.broker import Broker, HashChainedAudit
from friday.tools.base import ToolResult


class EchoArgs(BaseModel):
    """Args for the fake echo tool."""

    text: str


class FakeTool:
    """A configurable fake :class:`~friday.tools.base.Tool`.

    Records the args it was actually invoked with so tests can assert both that
    it ran (or did not) and that injected secrets reached it unredacted.
    """

    def __init__(
        self,
        name: str,
        *,
        side_effecting: bool,
        idempotent: bool,
        args_model: type[BaseModel] = EchoArgs,
    ) -> None:
        self.name = name
        self.description = f"fake {name}"
        self.args_model = args_model
        self.required_permission = name
        self.idempotent = idempotent
        self.side_effecting = side_effecting
        self.called_with: Any | None = None

    async def __call__(self, args: Any) -> ToolResult:
        self.called_with = args
        return ToolResult(ok=True, data={"echo": getattr(args, "text", None)})


class FakeRegistry:
    """A minimal registry exposing :meth:`get` over a name->tool map."""

    def __init__(self, tools: dict[str, FakeTool]) -> None:
        self._tools = tools

    def get(self, name: str) -> FakeTool:
        return self._tools[name]


class FakeSecretProvider:
    """A fake secret store recording which names were resolved."""

    def __init__(self, secrets: dict[str, str]) -> None:
        self._secrets = secrets
        self.requested: list[str] = []

    def get(self, name: str) -> str:
        self.requested.append(name)
        return self._secrets[name]


def _broker(
    tmp_path: Path,
    tools: dict[str, FakeTool],
    *,
    secrets: dict[str, str] | None = None,
) -> tuple[Broker, HashChainedAudit, FakeSecretProvider | None]:
    audit = HashChainedAudit(tmp_path / "audit.jsonl")
    registry = FakeRegistry(tools)
    provider = FakeSecretProvider(secrets) if secrets is not None else None
    broker = Broker(registry, audit, secret_provider=provider)
    return broker, audit, provider


async def test_reversible_read_only_runs(tmp_path: Path) -> None:
    tool = FakeTool("read", side_effecting=False, idempotent=True)
    broker, audit, _ = _broker(tmp_path, {"read": tool})

    result = await broker.dispatch(
        "read", {"text": "hi"}, allowed_tools={"read"}
    )

    assert result.ok is True
    assert result.data["echo"] == "hi"
    assert tool.called_with is not None
    # Exactly one hash-chained audit row, and the chain verifies.
    entries = audit.entries()
    assert len(entries) == 1
    assert entries[0].record["tool"] == "read"
    assert entries[0].record["ok"] is True
    ok, broken_at = audit.verify()
    assert ok is True and broken_at is None


async def test_deny_by_default_when_not_allowed(tmp_path: Path) -> None:
    tool = FakeTool("read", side_effecting=False, idempotent=True)
    broker, audit, _ = _broker(tmp_path, {"read": tool})

    result = await broker.dispatch("read", {"text": "hi"}, allowed_tools=set())

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "denied"
    # The tool must never have run.
    assert tool.called_with is None
    # The denial is still audited.
    entries = audit.entries()
    assert len(entries) == 1
    assert entries[0].record["ok"] is False
    assert entries[0].record["decision"] == "denied"


async def test_irreversible_needs_confirmation(tmp_path: Path) -> None:
    # side_effecting and not idempotent -> irreversible.
    tool = FakeTool("send", side_effecting=True, idempotent=False)
    broker, audit, _ = _broker(tmp_path, {"send": tool})

    result = await broker.dispatch(
        "send", {"text": "hi"}, allowed_tools={"send"}
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "needs_confirmation"
    assert result.data.get("needs_confirmation") is True
    assert tool.called_with is None
    entries = audit.entries()
    assert entries[-1].record["decision"] == "needs_confirmation"


async def test_irreversible_runs_when_confirmed(tmp_path: Path) -> None:
    tool = FakeTool("send", side_effecting=True, idempotent=False)
    broker, _, _ = _broker(tmp_path, {"send": tool})

    result = await broker.dispatch(
        "send", {"text": "hi"}, allowed_tools={"send"}, confirmed=True
    )

    assert result.ok is True
    assert tool.called_with is not None


async def test_idempotent_side_effect_is_reversible_no_confirm(tmp_path: Path) -> None:
    # side_effecting but idempotent -> classified reversible -> no confirm gate.
    tool = FakeTool("upsert", side_effecting=True, idempotent=True)
    broker, _, _ = _broker(tmp_path, {"upsert": tool})

    result = await broker.dispatch(
        "upsert", {"text": "hi"}, allowed_tools={"upsert"}
    )

    assert result.ok is True
    assert tool.called_with is not None


async def test_bad_args_rejected_before_execution(tmp_path: Path) -> None:
    tool = FakeTool("read", side_effecting=False, idempotent=True)
    broker, audit, _ = _broker(tmp_path, {"read": tool})

    # Missing required "text".
    result = await broker.dispatch("read", {}, allowed_tools={"read"})

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "bad_args"
    assert tool.called_with is None
    assert audit.entries()[-1].record["decision"] == "bad_args"


async def test_secret_injected_but_absent_from_result_and_audit(tmp_path: Path) -> None:
    tool = FakeTool("send", side_effecting=True, idempotent=False)
    secret_value = "sk-LIVE-DEADBEEF"
    broker, audit, provider = _broker(
        tmp_path, {"send": tool}, secrets={"OPENAI_KEY": secret_value}
    )

    result = await broker.dispatch(
        "send",
        {"text": "{{secret:OPENAI_KEY}}"},
        allowed_tools={"send"},
        confirmed=True,
    )

    # The resolved secret reached the tool...
    assert tool.called_with is not None
    assert tool.called_with.text == secret_value
    assert provider is not None and provider.requested == ["OPENAI_KEY"]

    # ...but never appears in the ToolResult.
    assert secret_value not in repr(result.model_dump())

    # ...and never appears in the audit ledger (record or raw file bytes).
    entries = audit.entries()
    for entry in entries:
        assert secret_value not in repr(entry.record)
    raw = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert secret_value not in raw


async def test_audit_chain_detects_tampering_across_dispatches(tmp_path: Path) -> None:
    tool = FakeTool("read", side_effecting=False, idempotent=True)
    path = tmp_path / "audit.jsonl"
    audit = HashChainedAudit(path)
    broker = Broker(FakeRegistry({"read": tool}), audit)

    await broker.dispatch("read", {"text": "a"}, allowed_tools={"read"})
    await broker.dispatch("read", {"text": "b"}, allowed_tools={"read"})

    ok, broken_at = audit.verify()
    assert ok is True and broken_at is None

    # Tamper with the first record on disk.
    import json

    lines = path.read_text(encoding="utf-8").splitlines()
    obj = json.loads(lines[0])
    obj["record"]["ok"] = False
    lines[0] = json.dumps(obj)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok2, broken_at2 = HashChainedAudit(path).verify()
    assert ok2 is False
    assert broken_at2 == 0


async def test_secret_marker_without_provider_passes_through(tmp_path: Path) -> None:
    # No secret_provider wired: an unresolved marker is left as-is (not crashed).
    tool = FakeTool("read", side_effecting=False, idempotent=True)
    broker, _, _ = _broker(tmp_path, {"read": tool})

    result = await broker.dispatch(
        "read", {"text": "{{secret:NONE}}"}, allowed_tools={"read"}
    )

    assert result.ok is True
    assert tool.called_with is not None
    assert tool.called_with.text == "{{secret:NONE}}"


async def test_actor_and_channel_recorded(tmp_path: Path) -> None:
    tool = FakeTool("read", side_effecting=False, idempotent=True)
    broker, audit, _ = _broker(tmp_path, {"read": tool})

    await broker.dispatch(
        "read",
        {"text": "hi"},
        allowed_tools={"read"},
        actor="family",
        channel="voice",
    )

    record = audit.entries()[-1].record
    assert record["actor"] == "family"
    assert record["channel"] == "voice"
