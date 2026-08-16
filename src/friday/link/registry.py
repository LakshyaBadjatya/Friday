"""Which machines are currently reachable, and how to ask them things.

The deployed instance holds one of these. A relay dials in, says hello, and is
listed here until its socket dies; anything wanting to reach a real machine goes
through :meth:`LinkRegistry.run`.

Two decisions worth stating.

**Jobs are request/response over one socket.** A relay may be asked several
things at once — a security scan while a file search is still running — so
replies carry the job id and are matched to a waiting future rather than being
read in order. Reading in order would mean a slow scan blocking a quick question
behind it, on a link that is already the slowest part of any answer.

**Nothing is queued for a machine that is not there.** A laptop is off more than
it is on, and a job accepted now and delivered in six hours is worse than a
refusal: the owner has moved on, and the answer arrives attached to a question
nobody remembers asking. If the machine is gone the caller is told immediately,
and says so.
"""

from __future__ import annotations

import asyncio
import secrets
from typing import Any

from friday.link.protocol import Job, JobKind, Result
from friday.logging import get_logger

logger = get_logger("friday.link.registry")


class Link:
    """One connected machine."""

    def __init__(self, machine: str, socket: Any, commands: list[str],
                 platform: str = "") -> None:
        self.machine = machine
        self.platform = platform
        self.commands = list(commands)
        self._socket = socket
        #: ``{job_id: future}`` for replies still outstanding.
        self._waiting: dict[str, asyncio.Future[Result]] = {}

    async def run(self, kind: JobKind, args: dict[str, Any] | None = None,
                  timeout: float = 30.0) -> Result:
        """Send a job and wait for its reply."""
        job = Job(id=secrets.token_hex(8), kind=kind, args=args or {},
                  timeout=timeout)
        loop = asyncio.get_running_loop()
        pending: asyncio.Future[Result] = loop.create_future()
        self._waiting[job.id] = pending
        try:
            await self._socket.send_text(job.model_dump_json())
            # A little longer than the relay's own limit, so a relay that times
            # out cleanly gets to say so rather than being cut off and reported
            # as unreachable — those are different problems and the owner should
            # be told which one happened.
            return await asyncio.wait_for(pending, timeout=timeout + 5.0)
        except TimeoutError:
            return Result(
                id=job.id, ok=False,
                error=f"{self.machine} did not answer within {timeout:.0f}s",
            )
        except Exception as exc:  # noqa: BLE001 - a dead socket is not a crash
            logger.warning("link: send to %s failed: %s", self.machine, exc)
            return Result(id=job.id, ok=False, error=f"lost the link to {self.machine}")
        finally:
            self._waiting.pop(job.id, None)

    def deliver(self, result: Result) -> None:
        """Hand a reply to whoever is waiting for it."""
        pending = self._waiting.pop(result.id, None)
        if pending is not None and not pending.done():
            pending.set_result(result)

    def abandon(self) -> None:
        """Fail everything outstanding, because the socket has gone."""
        for job_id, pending in list(self._waiting.items()):
            if not pending.done():
                pending.set_result(
                    Result(id=job_id, ok=False,
                           error=f"{self.machine} disconnected mid-job")
                )
        self._waiting.clear()


class LinkRegistry:
    """Every machine currently dialled in."""

    def __init__(self) -> None:
        self._links: dict[str, Link] = {}

    def attach(self, link: Link) -> None:
        """Register a machine, replacing any stale entry under the same name.

        A relay that reconnects after a suspend arrives while the old socket is
        still nominally open. The new one wins: it is the one demonstrably
        alive, and leaving the corpse in place would send every job to a socket
        that will never answer.
        """
        previous = self._links.get(link.machine)
        if previous is not None:
            previous.abandon()
        self._links[link.machine] = link
        logger.info("link: %s connected (%s)", link.machine, link.platform)

    def detach(self, machine: str, link: Link | None = None) -> None:
        """Drop a machine, if the link given is still the current one."""
        current = self._links.get(machine)
        if current is None or (link is not None and current is not link):
            return
        current.abandon()
        self._links.pop(machine, None)
        logger.info("link: %s disconnected", machine)

    def machines(self) -> list[str]:
        """Names of everything reachable right now."""
        return sorted(self._links)

    def get(self, machine: str = "") -> Link | None:
        """A named machine, or the only one connected when no name is given."""
        if machine:
            return self._links.get(machine)
        if len(self._links) == 1:
            return next(iter(self._links.values()))
        return None

    async def run(self, kind: JobKind, args: dict[str, Any] | None = None,
                  machine: str = "", timeout: float = 30.0) -> Result:
        """Run a job on a machine, or explain why that could not happen."""
        link = self.get(machine)
        if link is None:
            known = ", ".join(self.machines()) or "nothing"
            wanted = machine or "a machine"
            return Result(
                id="", ok=False,
                error=(
                    f"{wanted} isn't linked right now — connected: {known}. "
                    f"Start the FRIDAY relay on it and I'll be able to look."
                ),
            )
        return await link.run(kind, args, timeout)


def registry_of(app: Any) -> LinkRegistry:
    """The app's registry, created on first use."""
    existing = getattr(app.state, "link_registry", None)
    if existing is None:
        existing = LinkRegistry()
        app.state.link_registry = existing
    return existing
