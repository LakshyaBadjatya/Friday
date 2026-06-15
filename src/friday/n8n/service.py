"""The n8n service: confirm-gated docker auto-start, draft, best-effort import.

:class:`N8nService` is the orchestration seam tying together the
:class:`~friday.n8n.client.N8nClient`, the
:class:`~friday.n8n.drafter.WorkflowDrafter`, and the optional docker auto-start.

:meth:`N8nService.make_workflow` is the one entrypoint:

1. **Liveness + confirm-gated start.** If n8n is not up:

   * ``confirmed=False`` -> return ``{"kind": "needs_confirmation",
     "action": "start_n8n", ...}`` and do NOTHING else (no subprocess, no draft).
   * ``confirmed=True`` -> run ``start_cmd`` via
     :func:`asyncio.create_subprocess_exec` (an argv LIST — NEVER a shell, the
     same hardening as :mod:`friday.tools.system_exec`), then continue.

2. **Draft.** Call the drafter (NON-FATAL — it never raises) for the workflow +
   setup notes.

3. **Best-effort import.** When the client has an API key, try to import the
   workflow. An import error is captured into the result (``"import_error"``) but
   the drafted JSON is STILL returned — so a failed import never loses the work.

The docker start is the only side-effecting step, and it is gated behind the
``confirmed`` flag exactly like the registry confirm-step gates a side-effecting
tool (build-spec §12).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from friday.n8n.client import N8nClient, N8nError
from friday.n8n.drafter import WorkflowDrafter

logger = logging.getLogger("friday.n8n.service")


class N8nService:
    """Coordinate the n8n liveness/start, drafting, and import for one request.

    Args:
        client: The n8n REST client (liveness + import).
        drafter: The LLM-backed workflow drafter (NON-FATAL).
        start_cmd: The argv LIST run to start n8n via docker — e.g.
            ``["docker", "compose", "-f", "docker-compose.yml", "up", "-d",
            "n8n"]``. Passed to :func:`asyncio.create_subprocess_exec` (no shell).
    """

    def __init__(
        self,
        client: N8nClient,
        drafter: WorkflowDrafter,
        *,
        start_cmd: list[str],
    ) -> None:
        self._client = client
        self._drafter = drafter
        self._start_cmd = list(start_cmd)

    @property
    def client(self) -> N8nClient:
        """The underlying n8n REST client (for liveness probes from the route)."""
        return self._client

    async def start(self, *, confirmed: bool = False) -> bool:
        """Start n8n via docker behind the confirm-gate; return whether it ran.

        Mirrors the confirm-step: ``confirmed=False`` is a no-op (returns
        ``False``, runs NO subprocess); ``confirmed=True`` runs ``start_cmd``
        argv-only and returns ``True``. The ``/n8n/start`` route uses this so a
        start is never issued without an explicit confirmation.
        """
        if not confirmed:
            return False
        await self._start_n8n()
        return True

    async def make_workflow(
        self, description: str, *, confirmed: bool = False
    ) -> dict[str, Any]:
        """Draft (and optionally import) a workflow; confirm-gate the docker start.

        Returns one of:

        * ``{"kind": "needs_confirmation", "action": "start_n8n", "message": ...}``
          when n8n is down and the turn is not confirmed — NOTHING is started or
          drafted.
        * ``{"kind": "workflow", "imported": bool, "workflow": <dict>,
          "setup_notes": [...], "started": bool, "import_error"?: str}`` once a
          workflow is drafted (and best-effort imported).
        """
        up = await self._client.is_up()
        started = False
        if not up:
            if not confirmed:
                return {
                    "kind": "needs_confirmation",
                    "action": "start_n8n",
                    "message": "n8n isn't running; start it with docker?",
                }
            await self._start_n8n()
            started = True

        draft = await self._drafter.draft(description)
        workflow = draft["workflow"]
        setup_notes = draft["setup_notes"]

        result: dict[str, Any] = {
            "kind": "workflow",
            "imported": False,
            "workflow": workflow,
            "setup_notes": setup_notes,
            "started": started,
        }

        if self._client.has_api_key:
            try:
                imported = await self._client.import_workflow(workflow)
            except N8nError as exc:
                # Best-effort: a failed import never loses the drafted JSON; the
                # error is surfaced in the result so the caller can report it.
                logger.warning("n8n import failed (best-effort): %s", exc)
                result["import_error"] = str(exc)
            else:
                result["imported"] = True
                result["workflow"] = imported

        return result

    async def _start_n8n(self) -> None:
        """Run ``start_cmd`` via ``create_subprocess_exec`` (argv list, no shell).

        Mirrors :class:`~friday.tools.system_exec.RunCommandTool`: the command is
        spawned as an argv LIST through :func:`asyncio.create_subprocess_exec`
        (there is NO ``shell=True`` / ``create_subprocess_shell`` anywhere), so a
        space/quote in an arg is never re-parsed by a shell. Output is drained and
        discarded; ``docker compose up -d`` detaches, so this returns once the
        compose command itself completes (its exit code is logged, not raised —
        starting is best-effort and the subsequent ``is_up`` on the caller's next
        turn is the real signal).
        """
        logger.info("starting n8n via docker", extra={"argv0": self._start_cmd[0]})
        process = await asyncio.create_subprocess_exec(
            *self._start_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()
        logger.info("n8n docker start finished", extra={"returncode": process.returncode})
