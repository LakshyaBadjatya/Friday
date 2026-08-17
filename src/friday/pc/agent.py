"""The half of the PC bridge that lives on the PC.

It holds no port open and accepts no connection. It asks FRIDAY whether there
is anything to run, runs it, and posts back what happened — so the machine with
the shell is only ever a client, and a home network needs no forwarded port, no
dynamic DNS and no firewall hole for any of this to work from a phone on mobile
data.

The consequence, stated plainly because it is the price of that convenience:
whoever holds the FRIDAY token can run commands here. The token is now as
sensitive as an SSH key, and this process is the thing that makes it so —
which is why it runs only when started, and writes down everything it did.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import httpx

from friday.logging import get_logger
from friday.pc.jobs import Job, JobResult
from friday.pc.safety import destructive_reason

logger = get_logger("friday.pc.agent")

#: Long-poll length. Long enough that an idle machine is nearly free, short
#: enough that a dropped connection is noticed rather than waited out.
POLL_SECONDS = 25.0

#: Output kept per stream. A `find /` can print megabytes, and none of it helps
#: once it has to be read aloud or carried in an LLM context.
MAX_OUTPUT = 16_000


def _audit_path() -> Path:
    return Path(
        os.environ.get("FRIDAY_PC_AUDIT", Path.home() / ".friday" / "pc-audit.log")
    )


def _audit(command: str, result: JobResult) -> None:
    """Write down every command, before anyone asks what she did.

    Appended rather than rotated: this file is the only record that a voice in
    a room caused something to happen on a disk, and losing the start of it is
    losing exactly the part someone would be looking for.
    """
    path = _audit_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp}\t{result.status}\t{result.exit_code}\t{command}\n")
    except OSError as exc:  # pragma: no cover - never fail a job over its log
        logger.warning("could not write the PC audit log: %s", exc)


def run_job(job: Job, default_cwd: Path) -> JobResult:
    """Run one command and describe what happened.

    ``shell=True`` is deliberate and is the feature: the useful sentences are
    shell sentences — pipes, globs, redirections — and a command parsed out of
    them would not do what was asked. It is also exactly why
    :func:`destructive_reason` guards the way in.
    """
    reason = destructive_reason(job.command)
    if reason is not None and not job.confirmed:
        result = JobResult(
            id=job.id,
            status="refused",
            detail=(
                f"That would mean {reason}, so I need you to confirm it. "
                "Say it again with 'confirm' if you meant it."
            ),
        )
        _audit(job.command, result)
        return result

    cwd = Path(job.cwd) if job.cwd else default_cwd
    if not cwd.is_dir():
        return JobResult(id=job.id, status="error", detail=f"No such directory: {cwd}")

    try:
        completed = subprocess.run(  # noqa: S602 - a shell is the point of this
            job.command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=job.timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        result = JobResult(
            id=job.id,
            status="timeout",
            detail=f"Still running after {job.timeout_seconds:.0f}s, so I stopped it.",
        )
        _audit(job.command, result)
        return result
    except OSError as exc:
        result = JobResult(id=job.id, status="error", detail=str(exc))
        _audit(job.command, result)
        return result

    result = JobResult(
        id=job.id,
        status="ok" if completed.returncode == 0 else "error",
        stdout=completed.stdout[:MAX_OUTPUT],
        stderr=completed.stderr[:MAX_OUTPUT],
        exit_code=completed.returncode,
    )
    _audit(job.command, result)
    return result


def serve(base_url: str, token: str, default_cwd: Path | None = None) -> None:
    """Ask for work, do it, report back. Forever, until interrupted.

    Every network failure is treated as "try again shortly" rather than fatal,
    because the common one is FRIDAY's free-tier host having gone to sleep, and
    an agent that exits on that would need starting by hand exactly when nobody
    is at the machine.
    """
    root = default_cwd or Path.home()
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}"}
    logger.info("PC agent attached to %s, running commands in %s", base, root)

    with httpx.Client(timeout=POLL_SECONDS + 15.0, headers=headers) as client:
        while True:
            try:
                response = client.get(f"{base}/pc/poll")
                if response.status_code == 204:
                    continue          # nothing to do; ask again
                response.raise_for_status()
                job = Job.model_validate(response.json())
            except httpx.HTTPError as exc:
                logger.warning("could not reach FRIDAY (%s); retrying", exc)
                time.sleep(3.0)
                continue

            logger.info("running: %s", job.command)
            result = run_job(job, root)
            try:
                client.post(f"{base}/pc/jobs/{job.id}/result", json=result.model_dump())
            except httpx.HTTPError as exc:
                logger.warning("ran the command but could not report it: %s", exc)
