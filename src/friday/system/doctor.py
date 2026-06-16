# © Lakshya Badjatya — Author
"""``friday doctor`` — a one-shot health self-test over the runtime's config.

Answers "is this install wired up?" without starting the server: which LLM
providers have credentials, whether the build is offline-safe, which memory /
embedding backend is selected, and (when a verifier is supplied) whether the
tamper-evident audit ledger is intact. It is a **pure** function over an injected
:class:`~friday.config.Settings` plus an optional ledger-verify callable — no
network, no server, no wall clock — so the whole report is deterministic and
unit-testable. The CLI handler is the only thing that touches the real ledger.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from friday.config import Settings

#: A ledger verifier returns ``(ok, broken_at_index_or_None)`` (HashChainedAudit's shape).
LedgerVerify = Callable[[], "tuple[bool, int | None]"]


class Check(BaseModel):
    """One health check: a name, whether it passed, and a human detail line."""

    name: str
    ok: bool
    detail: str


class DoctorReport(BaseModel):
    """The full set of checks; :attr:`ok` is true only when every check passed."""

    checks: list[Check] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether all checks passed."""
        return all(check.ok for check in self.checks)

    def render(self) -> str:
        """A terminal-friendly rendering of the report."""
        lines = [
            f"[{'ok  ' if c.ok else 'FAIL'}] {c.name}: {c.detail}" for c in self.checks
        ]
        lines.append("")
        lines.append(
            "doctor: all checks passed" if self.ok else "doctor: ISSUES FOUND"
        )
        return "\n".join(lines)


def run_doctor(settings: Settings, *, audit_verify: LedgerVerify | None = None) -> DoctorReport:
    """Run the health checks over ``settings``; include the ledger when verifiable.

    The provider check passes when at least one provider key is present *or* the
    build is the offline ``fake`` LLM (which needs none). All other config checks
    are informational (always ``ok``). When ``audit_verify`` is supplied its
    ``(ok, broken_at)`` result becomes a real pass/fail check, so ``friday doctor``
    exits non-zero on a tampered ledger.
    """
    keys = {
        "nvidia": settings.nvidia_api_key is not None,
        "gemini": settings.gemini_api_key is not None,
        "openrouter": settings.openrouter_api_key is not None,
        "opencode": settings.opencode_api_key is not None,
    }
    present = [name for name, has in keys.items() if has]
    is_fake = settings.llm_provider == "fake"
    checks = [
        Check(
            name="providers",
            ok=bool(present) or is_fake,
            detail=(
                f"{len(present)} key(s): {', '.join(present)}"
                if present
                else ("offline fake LLM (no keys needed)" if is_fake else "no provider keys set")
            ),
        ),
        Check(
            name="llm_provider",
            ok=True,
            detail=f"provider={settings.llm_provider}"
            + (" (offline mode)" if settings.enable_offline_mode else ""),
        ),
        Check(
            name="memory_store",
            ok=True,
            detail=f"{'postgres' if settings.enable_postgres else 'sqlite'} "
            f"({settings.memory_db_path})",
        ),
        Check(
            name="embeddings",
            ok=True,
            detail=f"provider={settings.embedding_provider}",
        ),
    ]
    if audit_verify is not None:
        ledger_ok, broken_at = audit_verify()
        checks.append(
            Check(
                name="audit_ledger",
                ok=ledger_ok,
                detail="intact" if ledger_ok else f"broken at entry {broken_at}",
            )
        )
    return DoctorReport(checks=checks)
