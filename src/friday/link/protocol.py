"""The contract between the deployed FRIDAY and a machine she can reach.

Asked to check the security of his devices she made a joke, and the joke was
closer to honest than it looked: she runs in a container on Render and has no
route to a laptop sitting behind a home router. There was nothing to audit from
where she stands.

So the machine calls her. A small relay process runs on the owner's side, opens
an outbound WebSocket to the deployed instance, and waits. Outbound because
inbound is not available in any useful sense — no static address, no port
forwarding, no certificate — and because a machine that dials out is a far
smaller thing to secure than one that listens.

Everything crossing that socket is one of two shapes: a :class:`Job` going down
and a :class:`Result` coming back. Both are small, boring, and explicitly
enumerated. The relay refuses anything it does not recognise, so widening what
FRIDAY may do on a personal machine means editing the relay on that machine —
not persuading a model to emit a different string.

That last point is the whole design. The deployed side is exposed to Discord,
Telegram, Siri and the open web, and its prompts are assembled partly from text
other people wrote. It must never be one convincing message away from running a
command on the owner's laptop. So the job *kinds* are a closed set, their
arguments are typed, and the dangerous one — running a command — carries no
free-form string at all.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class JobKind(StrEnum):
    """Everything a linked machine will do, and nothing else.

    A closed set on purpose. The alternative — a generic "run this" — would put
    the deployed instance's prompt in charge of a personal machine, and that
    prompt is assembled from Discord messages and scraped web pages.
    """

    #: Read-only posture check: listening ports, failed logins, firewall,
    #: pending updates, secrets readable by everyone.
    SECURITY_SCAN = "security_scan"
    #: Which relay is this, what is it running, how long has it been up.
    IDENTIFY = "identify"
    #: A named, pre-approved command from the relay's own allowlist. The name is
    #: chosen from a list the relay publishes; the command it maps to lives on
    #: the machine, never in the request.
    NAMED_COMMAND = "named_command"
    #: Files matching a glob under a rooted directory, names and sizes only.
    FIND_FILES = "find_files"


class Job(BaseModel):
    """One piece of work sent down to a linked machine."""

    id: str = Field(description="Correlates the result; unique per job.")
    kind: JobKind
    #: Arguments, validated by the relay against the kind before anything runs.
    args: dict[str, Any] = Field(default_factory=dict)
    #: Seconds the relay may spend. Nothing on a personal machine should take
    #: minutes, and a hung job must not hold the socket open indefinitely.
    timeout: float = Field(default=30.0, ge=1.0, le=120.0)


class Result(BaseModel):
    """What came back."""

    id: str
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    #: Present when ``ok`` is false. Written for a person to act on, because the
    #: usual reader is the owner being told why his own machine said no.
    error: str = ""


class Hello(BaseModel):
    """The relay's opening message, identifying itself and its capabilities."""

    type: Literal["hello"] = "hello"
    #: A name the owner recognises: "thinkpad", "desk", "phone".
    machine: str
    platform: str = ""
    #: The command names this relay will run, so the deployed side can say what
    #: is possible instead of guessing and being refused.
    commands: list[str] = Field(default_factory=list)
    version: str = "1"


#: Longest a relay may go silent before the deployed side drops it. A laptop
#: that suspends does not close its socket politely; without this the registry
#: would keep offering a machine that stopped listening hours ago.
HEARTBEAT_SECONDS = 30.0
STALE_AFTER = HEARTBEAT_SECONDS * 3
