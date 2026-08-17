"""Which commands run unasked, and which have to be confirmed first.

FRIDAY has a full shell on this machine because that is what makes her useful
on it — "find the file", "make me a folder", "what's taking up the disk" are
all one command each, and a curated list of allowed verbs would miss most of
them. The cost of that reach is that the sentence reaching the shell was
*transcribed*, and transcription is wrong sometimes.

So the line drawn here is not between allowed and forbidden. It is between
commands that undo and commands that do not: anything that deletes, overwrites
in place, formats, or hands root a pipe from the internet has to come with
``confirmed`` set. Everything else — listing, searching, reading, creating —
just runs, because being asked "are you sure?" before `ls` is how a safety
feature teaches people to say yes without reading.
"""

from __future__ import annotations

import re

#: Patterns that must not run on a transcript alone.
#:
#: Written against the whole command line rather than the parsed program,
#: because the danger is usually in the arguments: ``rm`` is fine, ``rm -rf`` on
#: a path that was misheard is not.
_DESTRUCTIVE: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\brm\b.*\s-[a-zA-Z]*[rf]", re.IGNORECASE), "recursive or forced delete"),
    (re.compile(r"\brmdir\b", re.IGNORECASE), "directory removal"),
    (re.compile(r"\bmkfs(\.\w+)?\b", re.IGNORECASE), "filesystem format"),
    (re.compile(r"\bdd\b[^|]*\bof=", re.IGNORECASE), "raw write to a device"),
    (re.compile(r"\bshred\b", re.IGNORECASE), "secure erase"),
    (re.compile(r">\s*/dev/(sd|nvme|hd)", re.IGNORECASE), "write to a raw disk"),
    (re.compile(r"\bchmod\b\s+-R\b", re.IGNORECASE), "recursive permission change"),
    (re.compile(r"\bchown\b\s+-R\b", re.IGNORECASE), "recursive ownership change"),
    (re.compile(r"\b(shutdown|reboot|poweroff|halt)\b", re.IGNORECASE), "power off or reboot"),
    (re.compile(r"\bkill(all)?\b\s+-9", re.IGNORECASE), "forced process kill"),
    (
        re.compile(
            r"\bgit\b.*\b(reset\s+--hard|clean\s+-[a-z]*f|push\s+--force)", re.IGNORECASE
        ),
        "destructive git operation",
    ),
    (
        re.compile(
            r"\b(apt|apt-get|dnf|pacman|yum)\b.*\b(remove|purge|erase)\b", re.IGNORECASE
        ),
        "package removal",
    ),
    # A download piped into a shell is the classic way a machine is lost, and it
    # is never what someone meant to say out loud.
    (
        re.compile(r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba|z|k)?sh\b", re.IGNORECASE),
        "piping a download into a shell",
    ),
    (re.compile(r"\bsudo\b", re.IGNORECASE), "running as root"),
    (re.compile(r":\(\)\s*\{.*\};\s*:", re.DOTALL), "fork bomb"),
)


def destructive_reason(command: str) -> str | None:
    """Why ``command`` needs confirming, or None if it may just run.

    Returns the human-readable reason rather than a boolean so the refusal can
    say which part of the sentence it objected to — "are you sure?" with no
    subject is a question nobody can answer well.
    """
    for pattern, reason in _DESTRUCTIVE:
        if pattern.search(command):
            return reason
    return None
