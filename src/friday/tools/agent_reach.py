"""Keyless full-page reader + media transcription via the ``agent-reach`` reach.

:class:`AgentReachTool` gives the Research/Knowledge agents two read-only reach
extensions, both off by default (gated in :mod:`friday.app` behind
``FRIDAY_ENABLE_AGENT_REACH``):

* ``read_url`` — fetches a FULL page as clean markdown through **Jina Reader**
  (``GET https://r.jina.ai/<url>``). This is the headline keyless channel of the
  installed `agent-reach <https://github.com/Panniantong/agent-reach>`_ project,
  but it needs no binary: it is a pure :mod:`httpx` GET, so it works even when the
  CLI is absent. The contract mirrors :class:`~friday.tools.web_search.WebSearchTool`:
  one bounded retry on a retriable network error, then a retriable
  ``ToolResult(ok=False, error=ToolError(code="read_failed"))``.
* ``transcribe`` — shells out to the installed ``agent-reach transcribe <url>``
  CLI (URL/audio -> text via Whisper). When the binary is not on ``PATH`` the tool
  returns ``ToolResult(ok=False, error=ToolError(code="agent_reach_cli_missing"))``
  with a clear install hint and runs **no** subprocess; a non-zero exit yields
  ``ToolResult(ok=False, error=ToolError(code="transcribe_failed"))``.

The tool is READ-ONLY (``side_effecting=False``, ``idempotent=True``) and NEVER
fabricates content: on any failure it returns the error payload only, never an
invented page/transcript. The ``agent-reach`` CLI is installed in isolation (a
``uv tool`` on ``PATH``) and is deliberately NOT a FRIDAY dependency, so the tool
must degrade cleanly when it is missing.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, field_validator

from friday.tools.base import ToolError, ToolResult

logger = logging.getLogger("friday.tools.agent_reach")

#: Default keyless Jina Reader base; ``GET {base}{url}`` returns clean markdown.
DEFAULT_JINA_BASE = "https://r.jina.ai/"
#: Default name/path of the isolated ``agent-reach`` CLI looked up on ``PATH``.
DEFAULT_CLI_PATH = "agent-reach"
#: Default per-request timeout (seconds) for both the Jina GET and the CLI run.
DEFAULT_TIMEOUT = 60.0

# A browser-like UA reduces the chance the upstream serves a challenge page; it
# carries no secrets and is safe to hardcode (mirrors web_search).
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# httpx exceptions we treat as transient and therefore worth exactly one retry.
_RETRIABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.WriteError,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)
# HTTP statuses we treat as transient (worth surfacing as ``retriable=True``).
_TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})

# Shown when the CLI is absent so the owner can install the isolated tool.
_INSTALL_HINT = (
    "install agent-reach: uv tool install "
    "git+https://github.com/Panniantong/agent-reach"
)


class AgentReachArgs(BaseModel):
    """Arguments for :class:`AgentReachTool`.

    ``action`` selects the reach: ``read_url`` (keyless full-page markdown via
    Jina Reader) or ``transcribe`` (media -> text via the installed CLI).
    ``target`` is the URL (or, for ``transcribe``, the URL/audio reference) and
    must be non-empty.
    """

    action: Literal["read_url", "transcribe"] = "read_url"
    target: str = Field(min_length=1)

    @field_validator("target")
    @classmethod
    def _reject_option_like_target(cls, value: str) -> str:
        """Block argv flag smuggling: a ``target`` must not look like a CLI option.

        ``transcribe`` passes ``target`` to ``agent-reach``; a value such as
        ``--output=/etc/x`` could otherwise be parsed as a flag rather than a URL.
        We reject any leading ``-`` here (and also pass ``--`` before the URL in
        the subprocess call as defence in depth).
        """
        if value.startswith("-"):
            raise ValueError("target must not start with '-' (looks like a CLI flag)")
        return value


class AgentReachTool:
    """Read-only full-page reader (Jina) + media transcription (agent-reach CLI).

    Args:
        jina_base: Base of the keyless Jina Reader endpoint; ``read_url`` issues
            ``GET {jina_base}{target}`` and expects clean markdown back.
        timeout: Per-request wall-clock budget (seconds) shared by the Jina GET
            and the CLI subprocess.
        cli_path: Name/path of the ``agent-reach`` CLI looked up on ``PATH`` for
            ``transcribe``; never required for ``read_url``.
    """

    name = "agent_reach"
    description = (
        "Read a full web page as clean markdown (keyless, via Jina Reader) or "
        "transcribe media to text (via the installed agent-reach CLI)."
    )
    args_model = AgentReachArgs
    required_permission = "web"
    idempotent = True
    side_effecting = False

    def __init__(
        self,
        *,
        jina_base: str = DEFAULT_JINA_BASE,
        timeout: float = DEFAULT_TIMEOUT,
        cli_path: str = DEFAULT_CLI_PATH,
    ) -> None:
        self._jina_base = jina_base
        self._timeout = timeout
        self._cli_path = cli_path

    async def __call__(self, args: Any) -> ToolResult:
        """Dispatch to ``read_url`` / ``transcribe`` per the validated ``action``.

        ``args`` arrives validated from the registry, but coerce defensively so
        the tool is also safe to call directly.
        """
        if not isinstance(args, AgentReachArgs):
            args = AgentReachArgs.model_validate(args)

        if args.action == "transcribe":
            return await self._transcribe(args.target)
        return await self._read_url(args.target)

    # -- read_url (keyless, binary-independent) ---------------------------- #
    async def _fetch(self, target: str) -> httpx.Response:
        """Issue the Jina Reader GET and return the raw response (may raise)."""
        url = f"{self._jina_base}{target}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await client.get(url, headers={"User-Agent": _USER_AGENT})

    async def _read_url(self, target: str) -> ToolResult:
        """Fetch ``target`` as markdown with one bounded retry on a network blip.

        Never fabricates: any failure returns the error payload only.
        """
        last_exc: Exception | None = None
        # Two attempts total: initial call + one bounded retry on a retriable error.
        for attempt in range(2):
            try:
                response = await self._fetch(target)
            except _RETRIABLE_EXCEPTIONS as exc:
                last_exc = exc
                logger.warning(
                    "agent_reach read_url network error (attempt %d/2): %s",
                    attempt + 1,
                    exc,
                )
                continue

            if response.status_code != httpx.codes.OK:
                retriable = response.status_code in _TRANSIENT_STATUSES
                logger.warning(
                    "agent_reach read_url non-OK status %d for %r",
                    response.status_code,
                    target,
                )
                return ToolResult(
                    ok=False,
                    error=ToolError(
                        code="read_failed",
                        message=f"jina reader returned HTTP {response.status_code}",
                        retriable=retriable,
                    ),
                )

            return ToolResult(
                ok=True,
                data={
                    "content": response.text,
                    "source": "jina-reader",
                    "url": target,
                },
                error=None,
            )

        # Both attempts hit a retriable network error.
        return ToolResult(
            ok=False,
            error=ToolError(
                code="read_failed",
                message=f"jina reader request failed after retry: {last_exc}",
                retriable=True,
            ),
        )

    # -- transcribe (installed CLI; clean missing-binary degradation) ------ #
    async def _transcribe(self, target: str) -> ToolResult:
        """Run ``agent-reach transcribe <target>``; never fabricate on failure.

        When the CLI is not on ``PATH`` this returns the ``agent_reach_cli_missing``
        error *without* spawning any subprocess. Otherwise it runs the CLI with the
        configured timeout and returns the captured stdout as the transcript; a
        non-zero exit yields a ``transcribe_failed`` error carrying the captured
        stderr.
        """
        if shutil.which(self._cli_path) is None:
            logger.warning("agent_reach transcribe: CLI %r not on PATH", self._cli_path)
            return ToolResult(
                ok=False,
                error=ToolError(
                    code="agent_reach_cli_missing",
                    message=_INSTALL_HINT,
                    retriable=False,
                ),
            )

        try:
            process = await asyncio.create_subprocess_exec(
                self._cli_path,
                "transcribe",
                "--",  # end-of-options: a leading-dash target can't smuggle a flag
                target,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self._timeout
            )
        except (OSError, TimeoutError) as exc:
            logger.warning("agent_reach transcribe failed to run CLI: %s", exc)
            return ToolResult(
                ok=False,
                error=ToolError(
                    code="transcribe_failed",
                    message=f"agent-reach transcribe failed to run: {exc}",
                    retriable=True,
                ),
            )

        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            logger.warning(
                "agent_reach transcribe exited %s: %s", process.returncode, detail
            )
            return ToolResult(
                ok=False,
                error=ToolError(
                    code="transcribe_failed",
                    message=(
                        f"agent-reach transcribe exited {process.returncode}: {detail}"
                    ),
                    retriable=False,
                ),
            )

        transcript = stdout.decode("utf-8", errors="replace")
        return ToolResult(
            ok=True,
            data={"transcript": transcript, "source": "agent-reach"},
            error=None,
        )
