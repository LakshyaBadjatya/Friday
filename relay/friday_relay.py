#!/usr/bin/env python3
"""The half of FRIDAY that runs on your machine.

She lives in a container on Render and cannot see this laptop — no route to it,
no address for it, nothing listening on it. So this process dials *out* to her
and holds the line open. Nothing here accepts connections; if this script is not
running, the door does not exist.

Run it:

    FRIDAY_LINK_TOKEN=... python relay/friday_relay.py --machine thinkpad

It reconnects on its own, so a suspended laptop or a dropped wifi link comes
back without anyone doing anything.

**What it will and will not do.** The jobs it accepts are a fixed list, written
here, on the machine that runs them. The deployed side can ask for a security
scan, a file listing, or a command *by name* from ``COMMANDS`` below — it cannot
send a command to run, because there is no code path that would execute one.
That is deliberate and it is the whole reason this is a separate process: the
deployed FRIDAY builds her prompts partly from Discord messages and scraped web
pages, and she must never be one persuasive sentence away from a shell here.

Widening what she can do means editing this file, on this machine, on purpose.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import platform
import shutil
import socket
import subprocess  # noqa: S404 - fixed argv from COMMANDS, never a shell string
import sys
from pathlib import Path
from typing import Any

# The relay imports from the FRIDAY package so the checks and the wire format
# cannot drift from the deployed side. Running from a checkout is the expected
# case; this makes `python relay/friday_relay.py` work without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import websockets  # noqa: E402

from friday.link.checks import security_scan  # noqa: E402
from friday.link.protocol import (  # noqa: E402
    HEARTBEAT_SECONDS,
    Hello,
    Job,
    JobKind,
    Result,
)

#: Commands FRIDAY may ask for by name. The name is all that crosses the wire;
#: what it maps to lives here. Read-only by intention — nothing in this list
#: installs, deletes, restarts or writes.
COMMANDS: dict[str, list[str]] = {
    "uptime": ["uptime"],
    "disk_free": ["df", "-h"],
    "memory": ["free", "-h"],
    "who_is_logged_in": ["who"],
    "top_processes": ["ps", "-eo", "pid,pcpu,pmem,comm", "--sort=-pcpu"],
    "docker_containers": ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}"],
    "git_status": ["git", "status", "--short", "--branch"],
    "listening_ports": ["ss", "-tulnH"],
    "pending_updates": ["apt", "list", "--upgradable"],
    "kernel": ["uname", "-a"],
}

#: Where ``find_files`` may look. A rooted search, so a request cannot walk out
#: into /etc or someone else's home by way of a clever pattern.
SEARCH_ROOTS = [Path.home()]

_MAX_OUTPUT = 4000
_MAX_MATCHES = 60


def _run_named(name: str, timeout: float) -> Result:
    """Run one allowlisted command."""
    argv = COMMANDS.get(name)
    if argv is None:
        return Result(
            id="", ok=False,
            error=f"'{name}' isn't in this machine's allowlist. Available: "
                  f"{', '.join(sorted(COMMANDS))}",
        )
    if shutil.which(argv[0]) is None:
        return Result(id="", ok=False, error=f"{argv[0]} isn't installed here")
    try:
        done = subprocess.run(  # noqa: S603 - argv from COMMANDS, no shell
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return Result(id="", ok=False, error=f"{name} took longer than {timeout:.0f}s")
    except OSError as exc:
        return Result(id="", ok=False, error=f"{name} could not run: {exc}")
    return Result(
        id="", ok=True,
        data={
            "command": name,
            "exit_code": done.returncode,
            "output": (done.stdout or done.stderr or "")[:_MAX_OUTPUT],
        },
    )


def _find_files(pattern: str) -> Result:
    """Names and sizes of files matching a glob, under the rooted directories.

    Names and sizes only — never contents. A file listing answers "where did I
    put that project"; shipping the contents of arbitrary files to a chat bot is
    a different and much worse thing.
    """
    cleaned = (pattern or "").strip()
    if not cleaned or cleaned.startswith("/") or ".." in cleaned:
        return Result(
            id="", ok=False,
            error="pattern must be relative and must not contain '..'",
        )
    matches: list[dict[str, Any]] = []
    for root in SEARCH_ROOTS:
        for found in root.glob(cleaned):
            if len(matches) >= _MAX_MATCHES:
                break
            try:
                stat = found.stat()
            except OSError:
                continue
            matches.append({
                "path": str(found),
                "bytes": stat.st_size,
                "directory": found.is_dir(),
            })
    return Result(id="", ok=True, data={"pattern": cleaned, "matches": matches})


def _identify(machine: str) -> Result:
    return Result(
        id="", ok=True,
        data={
            "machine": machine,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "commands": sorted(COMMANDS),
        },
    )


async def _handle(job: Job, machine: str) -> Result:
    """Do one job. Never raises — a failure is a Result the owner can read."""
    try:
        if job.kind is JobKind.SECURITY_SCAN:
            scan = await asyncio.to_thread(security_scan)
            return Result(id=job.id, ok=True, data=scan)
        if job.kind is JobKind.IDENTIFY:
            return _identify(machine).model_copy(update={"id": job.id})
        if job.kind is JobKind.NAMED_COMMAND:
            name = str(job.args.get("name") or "")
            got = await asyncio.to_thread(_run_named, name, job.timeout)
            return got.model_copy(update={"id": job.id})
        if job.kind is JobKind.FIND_FILES:
            got = await asyncio.to_thread(
                _find_files, str(job.args.get("pattern") or "")
            )
            return got.model_copy(update={"id": job.id})
    except Exception as exc:  # noqa: BLE001 - report, never die
        return Result(id=job.id, ok=False, error=f"{job.kind} failed here: {exc}")
    return Result(id=job.id, ok=False, error=f"this relay does not do {job.kind}")


async def _heartbeat(socket_: Any) -> None:
    """Keep the connection warm through NAT timeouts."""
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        await socket_.send(json.dumps({"beat": 1}))


async def _serve(url: str, machine: str) -> None:
    """One connection, from hello until the socket dies."""
    async with websockets.connect(url, ping_interval=20, max_size=2**20) as socket_:
        await socket_.send(
            Hello(
                machine=machine,
                platform=platform.platform(),
                commands=sorted(COMMANDS),
            ).model_dump_json()
        )
        print(f"[relay] linked as {machine!r}", flush=True)
        beat = asyncio.create_task(_heartbeat(socket_))
        try:
            async for frame in socket_:
                try:
                    job = Job.model_validate_json(frame)
                except Exception:  # noqa: BLE001 - ignore anything unrecognised
                    continue
                print(f"[relay] {job.kind}", flush=True)
                result = await _handle(job, machine)
                await socket_.send(result.model_dump_json())
        finally:
            beat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await beat


async def _forever(url: str, machine: str) -> None:
    """Reconnect for as long as this process lives.

    Backs off to half a minute rather than hammering. A laptop wakes from
    suspend to find the socket long dead, and the first few retries will fail
    while the network comes back — that is normal and not worth shouting about.
    """
    delay = 2.0
    while True:
        try:
            await _serve(url, machine)
            delay = 2.0
        except Exception as exc:  # noqa: BLE001 - reconnecting is the job
            print(f"[relay] disconnected ({exc}); retrying in {delay:.0f}s", flush=True)
        await asyncio.sleep(delay)
        delay = min(delay * 2, 30.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="FRIDAY's relay on this machine")
    parser.add_argument(
        "--url", default=os.environ.get(
            "FRIDAY_LINK_URL", "wss://friday-backend-v2.onrender.com/link/connect"
        ),
    )
    parser.add_argument("--machine", default=os.environ.get(
        "FRIDAY_MACHINE", socket.gethostname()))
    args = parser.parse_args()

    token = os.environ.get("FRIDAY_LINK_TOKEN", "").strip()
    if not token:
        print("FRIDAY_LINK_TOKEN is not set — refusing to connect.", file=sys.stderr)
        return 2

    url = f"{args.url}?token={token}"
    print(f"[relay] connecting to {args.url} as {args.machine!r}", flush=True)
    try:
        asyncio.run(_forever(url, args.machine))
    except KeyboardInterrupt:
        print("\n[relay] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
