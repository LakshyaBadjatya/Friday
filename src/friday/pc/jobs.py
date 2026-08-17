"""The job queue between FRIDAY and the machine she runs commands on.

The agent lives on the PC and reaches *out* to FRIDAY; nothing reaches in. That
is the whole reason this queue exists rather than an endpoint on the PC: a
laptop behind a home router has no stable address and should not have an open
port, so the side with the public address holds the work and the side with the
shell comes and asks for it.

Jobs are held in memory. They are seconds old by construction — a command is
handed over, run, and answered — and a restart that loses one is better than a
restart that replays it. Anything that must outlive a deploy is a reminder, not
a job.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Literal

from pydantic import BaseModel, Field

#: Ceiling on queued work. A queue that grows without bound is a memory leak
#: with extra steps; an agent that is offline should fail loudly, not silently
#: accumulate a day of commands to run all at once when it returns.
MAX_PENDING = 32


class Job(BaseModel):
    """One command to run on the PC, and where its answer goes."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    command: str
    #: Working directory for the command; the agent's own default when empty.
    cwd: str = ""
    timeout_seconds: float = 60.0
    #: Set when the command matched a destructive pattern and was allowed anyway.
    confirmed: bool = False


class JobResult(BaseModel):
    """What the machine said back."""

    id: str
    status: Literal["ok", "error", "timeout", "refused"]
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    #: Present when the agent refused to run it at all, with the reason.
    detail: str = ""


class JobQueue:
    """A single-consumer queue of commands awaiting an agent.

    One consumer by design: there is one PC. Two agents polling the same queue
    would race for jobs and each would see half the conversation, which is worse
    than the second agent simply not being supported.
    """

    def __init__(self, max_pending: int = MAX_PENDING) -> None:
        self._pending: asyncio.Queue[Job] = asyncio.Queue(maxsize=max_pending)
        self._results: dict[str, asyncio.Future[JobResult]] = {}

    def submit(self, job: Job) -> asyncio.Future[JobResult]:
        """Queue ``job`` and hand back the future its result will land in.

        Raises :class:`asyncio.QueueFull` when the agent is not draining, which
        the route turns into a plain "the PC is not listening" rather than a
        request that hangs until it times out.
        """
        self._pending.put_nowait(job)
        future: asyncio.Future[JobResult] = asyncio.get_running_loop().create_future()
        self._results[job.id] = future
        return future

    async def take(self, timeout: float) -> Job | None:
        """Block up to ``timeout`` for the next job; None when nothing arrives.

        The agent long-polls with this, so an idle machine costs one open
        request rather than a request per second.
        """
        try:
            return await asyncio.wait_for(self._pending.get(), timeout=timeout)
        except TimeoutError:
            return None

    def complete(self, result: JobResult) -> bool:
        """Deliver ``result`` to whoever is waiting. False if nobody is."""
        future = self._results.pop(result.id, None)
        if future is None or future.done():
            return False
        future.set_result(result)
        return True

    def abandon(self, job_id: str) -> None:
        """Drop a waiter that gave up, so its future is not held forever."""
        self._results.pop(job_id, None)

    @property
    def pending(self) -> int:
        return self._pending.qsize()
