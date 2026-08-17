"""``/pc`` — running commands on the machine FRIDAY's agent is attached to.

Three endpoints, two audiences. The agent on the PC long-polls ``GET /pc/poll``
and posts back to ``POST /pc/jobs/{id}/result``; everything else — the phone, a
voice turn, a shortcut — uses ``POST /pc/run`` and waits for the answer.

The whole surface is behind ``FRIDAY_ENABLE_PC`` and, like the rest, is a 404
when the flag is off rather than a 403: an endpoint that can run shell commands
should not exist at all on a deployment that was never meant to have one.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from friday.logging import get_logger
from friday.pc.jobs import Job, JobQueue, JobResult
from friday.pc.safety import destructive_reason

logger = get_logger("friday.api.routes_pc")

router = APIRouter()

#: How long the agent is allowed to hold a poll open.
POLL_SECONDS = 25.0

#: How long a caller waits for the machine to answer before giving up. Longer
#: than most commands and shorter than a phone's patience.
RESULT_WAIT_SECONDS = 90.0


class RunRequest(BaseModel):
    """A command to run on the PC."""

    command: str = Field(min_length=1, max_length=4000)
    cwd: str = Field(default="", max_length=4000)
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    #: Say yes to something the agent would otherwise refuse.
    confirm: bool = False


def _enabled(request: Request) -> bool:
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "enable_pc", False))


def _queue(request: Request) -> JobQueue | None:
    queue = getattr(request.app.state, "pc_jobs", None)
    return queue if isinstance(queue, JobQueue) else None


def _disabled() -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": "pc bridge disabled"})


@router.post("/pc/run", response_model=None)
async def run(request: Request, body: RunRequest) -> JSONResponse:
    """Hand a command to the PC and wait for what it says back."""
    if not _enabled(request):
        return _disabled()
    queue = _queue(request)
    if queue is None:  # pragma: no cover - startup guard
        return _disabled()

    # Say no here as well as in the agent. The agent is the authority — it is
    # the only thing that can actually refuse to run — but answering in one
    # round trip is much better than sending a command across the world to be
    # told what could have been said immediately.
    reason = destructive_reason(body.command)
    if reason is not None and not body.confirm:
        # Logged here because it is refused here: the command never reaches the
        # PC, so the agent's audit file — which records what happened to the
        # disk — will have nothing to say about it. An attempt is still worth
        # knowing about, especially one that arrived from a transcript.
        logger.warning(
            "refused a destructive PC command",
            extra={"reason": reason, "command": body.command[:200]},
        )
        return JSONResponse(
            status_code=409,
            content={
                "status": "refused",
                "detail": (
                    f"That would mean {reason}. Send it again with confirm set "
                    "if you meant it."
                ),
            },
        )

    job = Job(
        command=body.command,
        cwd=body.cwd,
        timeout_seconds=body.timeout_seconds,
        confirmed=body.confirm,
    )
    try:
        future = queue.submit(job)
    except asyncio.QueueFull:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "the PC has too much queued already; is the agent running?"
            },
        )

    try:
        result = await asyncio.wait_for(future, timeout=RESULT_WAIT_SECONDS)
    except TimeoutError:
        queue.abandon(job.id)
        return JSONResponse(
            status_code=504,
            content={
                "detail": (
                    "the PC never answered — the agent may not be running "
                    "(`friday pc-agent` on the machine)"
                )
            },
        )
    return JSONResponse(status_code=200, content=result.model_dump())


@router.get("/pc/poll", response_model=None)
async def poll(request: Request) -> Response:
    """The agent asking for its next command; 204 when there is nothing."""
    if not _enabled(request):
        return _disabled()
    queue = _queue(request)
    if queue is None:  # pragma: no cover - startup guard
        return _disabled()

    job = await queue.take(timeout=POLL_SECONDS)
    if job is None:
        return Response(status_code=204)
    return JSONResponse(status_code=200, content=job.model_dump())


@router.post("/pc/jobs/{job_id}/result", response_model=None)
async def report(request: Request, job_id: str, body: JobResult) -> JSONResponse:
    """The agent reporting what happened."""
    if not _enabled(request):
        return _disabled()
    queue = _queue(request)
    if queue is None:  # pragma: no cover - startup guard
        return _disabled()

    # Trust the path, not the body: a result whose id disagreed with the URL
    # would otherwise be delivered to a different waiter than the one the agent
    # was answering.
    delivered = queue.complete(body.model_copy(update={"id": job_id}))
    if not delivered:
        logger.info("result arrived for %s with nobody waiting", job_id)
    return JSONResponse(status_code=200, content={"delivered": delivered})
