"""The action :class:`Broker`: a fail-closed gate around tool execution.

Every tool call flows through :meth:`Broker.dispatch`, which runs a fixed,
auditable pipeline:

1. **VALIDATE** — coerce ``raw_args`` through the tool's ``args_model``. Invalid
   arguments are rejected (``code="bad_args"``) before any side effect.
2. **CLASSIFY** — derive reversibility from the tool's flags. A tool is
   *reversible* when it is not ``side_effecting``; *irreversible* when it is
   ``side_effecting`` **and not** ``idempotent`` (an idempotent side effect is
   treated as reversible — re-applying it is safe).
3. **GATE (fail-closed)** — deny-by-default. A tool absent from
   ``allowed_tools`` is denied (``code="denied"``). An irreversible tool without
   ``confirmed=True`` returns ``code="needs_confirmation"``. The tool never runs
   on either path.
4. **INJECT** — replace any argument value of the exact form
   ``"{{secret:NAME}}"`` with ``secret_provider.get("NAME")``. The resolved
   secret is passed to the tool but is **never** returned in the
   :class:`~friday.tools.base.ToolResult` and **never** written to the audit —
   the audit records the *marker*, not the secret.
5. **EXECUTE** — invoke the tool via the injected registry.
6. **AUDIT** — append exactly one hash-chained record (redacted args, tool, the
   gate decision, ``ok``, ``actor``, ``channel``) to the
   :class:`~friday.broker.audit.HashChainedAudit` ledger.

Every collaborator (registry, audit, secret provider) is injected, so this
module imports nothing from :mod:`friday.config` or :mod:`friday.app`.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from pydantic import ValidationError

from friday.broker.audit import REDACTED
from friday.tools.base import Tool, ToolError, ToolResult

# Matches an argument value that is *exactly* a secret placeholder, capturing the
# secret's name. Anchored so a placeholder embedded in a larger string is left
# untouched (a secret is injected wholesale, never string-spliced).
_SECRET_MARKER = re.compile(r"^\{\{secret:([^}]+)\}\}$")


class _Registry(Protocol):
    """The slice of the tool registry the broker depends on."""

    def get(self, name: str) -> Tool:
        """Return the registered tool named ``name``."""
        ...


class _Audit(Protocol):
    """The slice of the audit ledger the broker depends on."""

    def append(self, record: dict[str, Any]) -> Any:
        """Persist one audit record (hash-chained) and return the entry."""
        ...


class _SecretProvider(Protocol):
    """The slice of a secret store the broker depends on."""

    def get(self, name: str) -> str:
        """Resolve the secret named ``name``."""
        ...


class Broker:
    """Mediates tool execution with validation, gating, secrets, and auditing.

    Args:
        registry: Resolves tool names to :class:`~friday.tools.base.Tool`
            instances via ``get(name)``.
        audit: The hash-chained ledger; one record is appended per dispatch.
        secret_provider: Optional resolver for ``{{secret:NAME}}`` markers. When
            absent, markers are passed through to the tool unchanged.
    """

    def __init__(
        self,
        registry: _Registry,
        audit: _Audit,
        *,
        secret_provider: _SecretProvider | None = None,
    ) -> None:
        self._registry = registry
        self._audit = audit
        self._secrets = secret_provider

    async def dispatch(
        self,
        tool_name: str,
        raw_args: dict[str, Any],
        *,
        allowed_tools: frozenset[str] | set[str],
        confirmed: bool = False,
        actor: str = "owner",
        channel: str = "chat",
    ) -> ToolResult:
        """Run the full broker pipeline for one tool call and return its result.

        Exactly one audit record is written before returning, regardless of which
        gate (if any) short-circuits the call.
        """
        decision = "executed"
        result: ToolResult

        # --- GATE: deny-by-default. Resolve the tool only once permitted. ---
        if tool_name not in allowed_tools:
            decision = "denied"
            result = _fail(
                "denied",
                f"tool {tool_name!r} is not in the allowed set",
            )
            self._record(tool_name, raw_args, result, decision, actor, channel)
            return result

        try:
            tool = self._registry.get(tool_name)
        except KeyError:
            decision = "denied"
            result = _fail(
                "denied",
                f"tool {tool_name!r} is not registered",
            )
            self._record(tool_name, raw_args, result, decision, actor, channel)
            return result

        # --- VALIDATE: bad args never reach the tool. ---
        try:
            tool.args_model.model_validate(raw_args)
        except ValidationError as exc:
            decision = "bad_args"
            result = _fail(
                "bad_args",
                f"invalid arguments for tool {tool_name!r}: {exc.errors()}",
            )
            self._record(tool_name, raw_args, result, decision, actor, channel)
            return result

        # --- CLASSIFY: reversibility from the tool's flags. ---
        irreversible = bool(tool.side_effecting) and not bool(tool.idempotent)

        # --- GATE: irreversible actions require explicit confirmation. ---
        if irreversible and not confirmed:
            decision = "needs_confirmation"
            result = ToolResult(
                ok=False,
                data={"needs_confirmation": True, "tool": tool_name},
                error=ToolError(
                    code="needs_confirmation",
                    message=(
                        f"tool {tool_name!r} is irreversible (side-effecting and "
                        "not idempotent); explicit confirmation is required"
                    ),
                    retriable=False,
                ),
            )
            self._record(tool_name, raw_args, result, decision, actor, channel)
            return result

        # --- INJECT: resolve secret markers into the args the tool receives. ---
        # Audit uses the original ``raw_args`` (markers, not resolved secrets).
        injected_args, resolved_secrets = self._inject_secrets(raw_args)
        validated = tool.args_model.model_validate(injected_args)

        # --- EXECUTE ---
        result = await tool(validated)

        # Defence in depth: a resolved secret must never round-trip out of the
        # broker, so scrub any injected value that a tool echoed into its result.
        if resolved_secrets:
            result = _scrub_result(result, resolved_secrets)

        self._record(tool_name, raw_args, result, decision, actor, channel)
        return result

    def _inject_secrets(
        self, raw_args: dict[str, Any]
    ) -> tuple[dict[str, Any], set[str]]:
        """Resolve ``{{secret:NAME}}`` values; return ``(args, resolved_values)``.

        Only values that are *exactly* a marker string are replaced. With no
        secret provider wired, markers pass through untouched. The returned set
        holds every resolved secret value so the caller can guarantee none of
        them leak back through the tool's result.
        """
        if self._secrets is None:
            return dict(raw_args), set()
        out: dict[str, Any] = {}
        resolved: set[str] = set()
        for key, value in raw_args.items():
            if isinstance(value, str):
                match = _SECRET_MARKER.match(value)
                if match is not None:
                    secret = self._secrets.get(match.group(1))
                    out[key] = secret
                    resolved.add(secret)
                    continue
            out[key] = value
        return out, resolved

    def _record(
        self,
        tool_name: str,
        raw_args: dict[str, Any],
        result: ToolResult,
        decision: str,
        actor: str,
        channel: str,
    ) -> None:
        """Append one hash-chained audit row for this dispatch.

        ``raw_args`` (which still carries any ``{{secret:NAME}}`` markers, never a
        resolved secret) is passed to the audit, which itself redacts
        sensitive-keyed values before hashing/persisting.
        """
        self._audit.append(
            {
                "tool": tool_name,
                "actor": actor,
                "channel": channel,
                "decision": decision,
                "ok": result.ok,
                "error_code": result.error.code if result.error is not None else None,
                "args": dict(raw_args),
            }
        )


def _fail(code: str, message: str) -> ToolResult:
    """Build a failed :class:`ToolResult` carrying a typed error payload."""
    return ToolResult(
        ok=False,
        data={},
        error=ToolError(code=code, message=message, retriable=False),
    )


def _scrub_value(value: Any, secrets: set[str]) -> Any:
    """Recursively replace any occurrence of a resolved secret with the sentinel.

    Whole-value matches are redacted outright; secrets embedded inside a larger
    string are substring-replaced so a credential cannot survive concatenation.
    """
    if isinstance(value, str):
        if value in secrets:
            return REDACTED
        scrubbed = value
        for secret in secrets:
            if secret and secret in scrubbed:
                scrubbed = scrubbed.replace(secret, REDACTED)
        return scrubbed
    if isinstance(value, dict):
        return {key: _scrub_value(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub_value(item, secrets) for item in value]
    return value


def _scrub_result(result: ToolResult, secrets: set[str]) -> ToolResult:
    """Return ``result`` with any injected secret value masked in its ``data``."""
    return result.model_copy(update={"data": _scrub_value(result.data, secrets)})
