"""System-automation tools: argv-only command execution, file search, app open.

Three tools the Automation agent reaches when ``enable_system_automation`` is set.
All three are SECURITY-hardened and off by default:

* :class:`RunCommandTool` (``run_command``) — runs a program via an argv LIST
  through :func:`asyncio.create_subprocess_exec` (**never** a shell), with a hard
  per-command timeout and capped stdout/stderr capture. An optional allow-list of
  command basenames gates which programs may run. Side-effecting + non-idempotent,
  so the registry confirm-step (build-spec §12) gates it before execution.
* :class:`FindFilesTool` (``find_files``) — globs files **confined** to
  ``system_automation_root``: a pattern or root resolving OUTSIDE that root is
  rejected (``path_not_allowed``) so a ``../etc`` traversal cannot escape. Read-only
  (``side_effecting=False``, ``idempotent=True``), so it skips the confirm-step.
* :class:`OpenAppTool` (``open_app``) — opens a target with the OS opener
  (``xdg-open`` / ``open`` / ``explorer`` by platform) via an argv LIST. The target
  must not look like a CLI flag (a leading ``-`` is rejected). Side-effecting +
  non-idempotent, so the confirm-step gates it.

SECURITY invariants (enforced + tested): every external process is spawned with an
argv list via ``create_subprocess_exec`` — there is **no** ``shell=True`` / ``os.system``
/ ``create_subprocess_shell`` anywhere; captured output is truncated to
:data:`MAX_OUTPUT_CHARS`; every spawn is bounded by ``get_settings().system_exec_timeout``;
and file search can never resolve outside the allowlisted root.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from friday.config import get_settings
from friday.tools.base import ToolError, ToolResult

logger = logging.getLogger("friday.tools.system_exec")

#: Hard cap on captured stdout/stderr characters (each stream truncated). The
#: returned string — marker included — never exceeds this length.
MAX_OUTPUT_CHARS = 10_000
#: Hard cap on the number of paths ``find_files`` returns.
MAX_MATCHES = 500

#: Marker appended to a truncated stream; it is counted within MAX_OUTPUT_CHARS.
_TRUNC_MARKER = "\n...[truncated]"


def _truncate(text: str) -> str:
    """Cap ``text`` at :data:`MAX_OUTPUT_CHARS` (marker included on truncation)."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    keep = MAX_OUTPUT_CHARS - len(_TRUNC_MARKER)
    return text[:keep] + _TRUNC_MARKER


# --------------------------------------------------------------------------- #
# run_command
# --------------------------------------------------------------------------- #
class RunCommandArgs(BaseModel):
    """Arguments for :class:`RunCommandTool`.

    ``command`` is the program to run (must be non-empty); ``args`` are its
    positional arguments. The program and each arg are passed as a single argv
    LIST to ``create_subprocess_exec`` — there is no shell, so spaces/quotes in an
    arg are never re-parsed.
    """

    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)


class RunCommandTool:
    """Run a program via an argv list (no shell), with a timeout + capped output.

    Side-effecting and non-idempotent, so the registry confirm-step gates it
    before execution. When ``get_settings().system_exec_allowlist`` is non-empty,
    the command's basename must be in it (else ``command_not_allowed``). A
    spawn/timeout failure is surfaced as a typed ``ToolResult`` error; a non-zero
    exit is reported honestly (``ok=True`` with ``returncode != 0`` and stderr).
    """

    name = "run_command"
    description = (
        "Run a local program via an argv list (no shell) with a timeout and "
        "capped output; returns stdout, stderr and the exit code."
    )
    args_model = RunCommandArgs
    required_permission = "system"
    idempotent = False
    side_effecting = True

    async def __call__(self, args: Any) -> ToolResult:
        """Gate on the allow-list, then run the command argv-only with a timeout."""
        if not isinstance(args, RunCommandArgs):
            args = RunCommandArgs.model_validate(args)

        settings = get_settings()
        allowlist = settings.system_exec_allowlist
        if allowlist and os.path.basename(args.command) not in allowlist:
            logger.warning(
                "run_command refused: %r not in allow-list", args.command
            )
            return ToolResult(
                ok=False,
                error=ToolError(
                    code="command_not_allowed",
                    message=(
                        f"command {args.command!r} is not in the system_exec "
                        "allow-list"
                    ),
                    retriable=False,
                ),
            )

        try:
            process = await asyncio.create_subprocess_exec(
                args.command,
                *args.args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError) as exc:
            logger.warning("run_command failed to spawn %r: %s", args.command, exc)
            return ToolResult(
                ok=False,
                error=ToolError(
                    code="spawn_failed",
                    message=f"failed to start {args.command!r}: {exc}",
                    retriable=False,
                ),
            )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=settings.system_exec_timeout
            )
        except TimeoutError:
            logger.warning(
                "run_command timed out after %ss: %r",
                settings.system_exec_timeout,
                args.command,
            )
            process.kill()
            await process.wait()
            return ToolResult(
                ok=False,
                error=ToolError(
                    code="timeout",
                    message=(
                        f"command {args.command!r} exceeded the "
                        f"{settings.system_exec_timeout}s timeout"
                    ),
                    retriable=True,
                ),
            )

        returncode = process.returncode if process.returncode is not None else -1
        logger.info(
            "run_command ran %r exit=%s", args.command, returncode
        )
        return ToolResult(
            ok=True,
            data={
                "stdout": _truncate(stdout.decode("utf-8", errors="replace")),
                "stderr": _truncate(stderr.decode("utf-8", errors="replace")),
                "returncode": returncode,
            },
            error=None,
        )


# --------------------------------------------------------------------------- #
# find_files
# --------------------------------------------------------------------------- #
class FindFilesArgs(BaseModel):
    """Arguments for :class:`FindFilesTool`.

    ``pattern`` is a glob (e.g. ``*.txt`` or ``**/*.py``) evaluated relative to
    ``root`` (defaulting to ``get_settings().system_automation_root``). Both are
    confined to the allowlisted root by the tool.
    """

    pattern: str = Field(min_length=1)
    root: str | None = None


class FindFilesTool:
    """Glob files confined to the allowlisted ``system_automation_root``.

    Read-only (``side_effecting=False``, ``idempotent=True``), so it skips the
    confirm-step. A ``pattern`` or ``root`` that resolves OUTSIDE the allowlisted
    root (a ``../`` traversal or an absolute path elsewhere) is rejected with
    ``path_not_allowed`` and no globbing happens. Returns up to :data:`MAX_MATCHES`
    matching paths (as strings).
    """

    name = "find_files"
    description = (
        "Find files by glob pattern, confined to the configured automation root "
        "(path traversal outside the root is rejected)."
    )
    args_model = FindFilesArgs
    required_permission = "system"
    idempotent = True
    side_effecting = False

    async def __call__(self, args: Any) -> ToolResult:
        """Resolve + confine the root, then glob ``pattern`` under it."""
        if not isinstance(args, FindFilesArgs):
            args = FindFilesArgs.model_validate(args)

        settings = get_settings()
        allowed_root = Path(settings.system_automation_root).resolve()

        # Resolve the search root (default: the allowlisted root) and confine it.
        requested_root = (
            Path(args.root) if args.root is not None else allowed_root
        )
        if not requested_root.is_absolute():
            requested_root = allowed_root / requested_root
        search_root = requested_root.resolve()
        if not _is_within(search_root, allowed_root):
            return self._path_refused(str(args.root))

        # A pattern can itself encode a traversal (``../etc/*``); resolve the
        # pattern's directory part against the search root and confine it too.
        pattern_path = Path(args.pattern)
        if pattern_path.is_absolute():
            return self._path_refused(args.pattern)
        # The literal (non-wildcard) leading part of the pattern must not climb
        # out of the root: ``../etc/*`` resolves its prefix outside it.
        literal_prefix = _literal_prefix(search_root, args.pattern)
        if not _is_within(literal_prefix, allowed_root):
            return self._path_refused(args.pattern)

        matches: list[str] = []
        for path in search_root.glob(args.pattern):
            resolved = path.resolve()
            if not _is_within(resolved, allowed_root):
                # A symlink/glob result that escaped the root is silently skipped.
                continue
            matches.append(str(path))
            if len(matches) >= MAX_MATCHES:
                break

        logger.info(
            "find_files matched %d path(s) for %r", len(matches), args.pattern
        )
        return ToolResult(ok=True, data={"matches": matches}, error=None)

    @staticmethod
    def _path_refused(target: str | None) -> ToolResult:
        logger.warning("find_files refused out-of-root target: %r", target)
        return ToolResult(
            ok=False,
            error=ToolError(
                code="path_not_allowed",
                message=(
                    f"path {target!r} resolves outside the allowed automation root"
                ),
                retriable=False,
            ),
        )


def _is_within(path: Path, root: Path) -> bool:
    """True if ``path`` is ``root`` or lives beneath it (both already resolved)."""
    return path == root or root in path.parents


def _literal_prefix(search_root: Path, pattern: str) -> Path:
    """Resolve the non-wildcard leading segments of ``pattern`` under ``search_root``.

    Used to reject a traversal that lives in the literal part of a glob (e.g.
    ``../etc/*``): we walk the pattern's path parts until the first one containing a
    glob metacharacter, join the literal prefix to ``search_root`` and resolve it so
    any ``..`` segments collapse. A pattern that climbs out of the root yields a
    resolved prefix outside it.
    """
    prefix = search_root
    for part in Path(pattern).parts:
        if any(ch in part for ch in "*?[]"):
            break
        prefix = prefix / part
    return prefix.resolve()


# --------------------------------------------------------------------------- #
# open_app
# --------------------------------------------------------------------------- #
def _opener_for_platform() -> str:
    """Return the OS opener program name for the current platform (argv[0])."""
    if sys.platform == "darwin":
        return "open"
    if sys.platform.startswith("win"):
        return "explorer"
    return "xdg-open"


class OpenAppArgs(BaseModel):
    """Arguments for :class:`OpenAppTool`.

    ``target`` is the file/URL/app to open; it must be non-empty and must not
    start with ``-`` (which the OS opener could parse as a flag).
    """

    target: str = Field(min_length=1)

    @field_validator("target")
    @classmethod
    def _reject_option_like_target(cls, value: str) -> str:
        """Block argv flag smuggling: a ``target`` must not look like a CLI option."""
        if value.startswith("-"):
            raise ValueError("target must not start with '-' (looks like a CLI flag)")
        return value


class OpenAppTool:
    """Open a target with the OS opener via an argv list (never a shell).

    Side-effecting and non-idempotent, so the registry confirm-step gates it
    before execution. The opener is chosen by platform (``xdg-open`` / ``open`` /
    ``explorer``) and invoked as ``[opener, "--", target]`` so a leading-dash
    target can never be parsed as a flag (and the model already rejects one).
    """

    name = "open_app"
    description = (
        "Open a file, URL or application with the OS opener (xdg-open/open/"
        "explorer) via an argv list — never a shell."
    )
    args_model = OpenAppArgs
    required_permission = "system"
    idempotent = False
    side_effecting = True

    async def __call__(self, args: Any) -> ToolResult:
        """Validate the target, then invoke the platform opener argv-only."""
        if not isinstance(args, OpenAppArgs):
            args = OpenAppArgs.model_validate(args)

        # Defence in depth: a target reaching here (e.g. via model_construct) that
        # looks like a flag is refused before any subprocess runs.
        if args.target.startswith("-"):
            logger.warning("open_app refused option-like target: %r", args.target)
            return ToolResult(
                ok=False,
                error=ToolError(
                    code="bad_target",
                    message="target must not start with '-' (looks like a CLI flag)",
                    retriable=False,
                ),
            )

        opener = _opener_for_platform()
        settings = get_settings()
        try:
            process = await asyncio.create_subprocess_exec(
                opener,
                "--",  # end-of-options separator (defence in depth)
                args.target,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=settings.system_exec_timeout
            )
        except (OSError, ValueError) as exc:
            logger.warning("open_app failed to spawn %r: %s", opener, exc)
            return ToolResult(
                ok=False,
                error=ToolError(
                    code="open_failed",
                    message=f"failed to open {args.target!r} via {opener}: {exc}",
                    retriable=False,
                ),
            )
        except TimeoutError:
            logger.warning("open_app timed out opening %r", args.target)
            process.kill()
            await process.wait()
            return ToolResult(
                ok=False,
                error=ToolError(
                    code="timeout",
                    message=(
                        f"opening {args.target!r} exceeded the "
                        f"{settings.system_exec_timeout}s timeout"
                    ),
                    retriable=True,
                ),
            )

        returncode = process.returncode if process.returncode is not None else -1
        logger.info("open_app opened %r via %s exit=%s", args.target, opener, returncode)
        return ToolResult(
            ok=True,
            data={
                "target": args.target,
                "opener": opener,
                "returncode": returncode,
                "stderr": _truncate(stderr.decode("utf-8", errors="replace")),
            },
            error=None,
        )
