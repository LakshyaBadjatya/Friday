"""Lecture-board capture: many frames of one slowly-growing blackboard.

Burst-capturing a board produces frames that are mostly the same board with a
little more written on it. Keeping all of them makes a useless note. The rule
here: a frame that contains what the previous frame said replaces it; a frame
that does not is a new board and starts a new section.

Comparison is on extracted text, not pixels — the camera moves, the writing
does not.
"""

from __future__ import annotations

#: Fraction of the previous frame's lines that must survive for a frame to be
#: considered the same board, grown.
_CONTINUATION = 0.6


def _lines(text: str) -> set[str]:
    return {line.strip() for line in text.splitlines() if line.strip()}


def dedupe_frames(frames: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Collapse a burst into the frames worth keeping, in capture order."""
    kept: list[tuple[str, str]] = []
    for item_id, text in frames:
        current = _lines(text)
        if not kept:
            kept.append((item_id, text))
            continue
        previous = _lines(kept[-1][1])
        if not previous:
            kept[-1] = (item_id, text)
            continue
        overlap = len(previous & current) / len(previous)
        if overlap >= _CONTINUATION:
            # Same board: keep whichever frame says more (ties go to the later
            # frame, which is the one with the clearest photo of the same text).
            if len(current) >= len(previous):
                kept[-1] = (item_id, text)
        else:
            kept.append((item_id, text))
    return kept


def stitch(frames: list[tuple[str, str]]) -> str:
    """Join deduped frames into one continuous body of text."""
    return "\n\n".join(text for _item_id, text in frames)
