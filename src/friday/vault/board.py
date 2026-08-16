"""Lecture-board capture: many frames of one slowly-growing blackboard.

Burst-capturing a board produces frames that are mostly the same board with a
little more written on it. Keeping all of them makes a useless note. The rule
here: a frame that reads as an edit-tolerant match for what the previous kept
frame said replaces it (the board grew); a frame that reads as substantially
different text is a new board and starts a new section (the board was
erased).

Comparison is on extracted OCR text via difflib.SequenceMatcher, not exact
line equality. OCR output for the very same photographed board varies from
shot to shot — a misread character, a re-wrapped line, a merged word — so
treating one differing character as "an entirely different line" fragments a
single growing board into many spurious sections. A character-level
similarity ratio tolerates that noise while still recognising a genuinely
erased board as unrelated text. The camera moves, the writing does not; OCR
of the writing still wobbles a little, and the comparison has to survive that.
"""

from __future__ import annotations

from difflib import SequenceMatcher

#: Similarity ratio (0..1, from difflib.SequenceMatcher.ratio() on normalized
#: text) at or above which a frame is considered a continuation of the
#: previous kept frame's board, rather than a new (erased) board.
#:
#: Chosen empirically against the module's stress-test fixture: the highest
#: observed ratio between two genuinely different boards was 0.529 (an
#: erasure that happens to leave a shared lecture title behind), and the
#: lowest observed ratio between consecutive frames of one genuinely growing
#: board was 0.625. That leaves a window of (0.529, 0.625]; 0.58 sits near
#: the middle of it. The margin is real but not huge — about 0.10 wide — so
#: this is not a threshold to nudge casually; re-run the stress tests in
#: tests/unit/test_vault_board.py against any change.
_CONTINUATION = 0.58


def _normalize(text: str) -> str:
    """Normalize OCR text for comparison.

    Per line: collapse internal whitespace runs to a single space, strip,
    lowercase. Drop blank lines. Drop exact-duplicate lines, keeping the
    first occurrence.

    The duplicate-line removal is still exact-match, not fuzzy — it doesn't
    reintroduce the fragility of comparing whole lines for equality. It only
    stops a board that happens to repeat a line (a rewritten header, a table
    row written twice) from padding out its own length and dragging down the
    similarity ratio against a frame that says the same thing without the
    repeat.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for line in text.splitlines():
        normalized_line = " ".join(line.split()).lower()
        if normalized_line and normalized_line not in seen_set:
            seen_set.add(normalized_line)
            seen.append(normalized_line)
    return " ".join(seen)


def _similarity(previous_norm: str, current_norm: str) -> float:
    return SequenceMatcher(None, previous_norm, current_norm).ratio()


def dedupe_frames(frames: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Collapse a burst into the frames worth keeping, in capture order."""
    kept: list[tuple[str, str]] = []
    for item_id, text in frames:
        current_norm = _normalize(text)
        if not kept:
            kept.append((item_id, text))
            continue
        previous_norm = _normalize(kept[-1][1])
        if not previous_norm:
            # Previous kept frame was blank/unreadable: any real content
            # supersedes it without needing to look like the same board.
            kept[-1] = (item_id, text)
            continue
        if not current_norm:
            # Blank/unreadable photo: treat as noise. Don't start a new,
            # empty section, and don't discard the board already kept.
            continue
        if _similarity(previous_norm, current_norm) >= _CONTINUATION:
            # Same board: keep whichever frame says more, measured on
            # normalized length rather than raw line count — line counts are
            # exactly what unstable OCR line-wrapping gets wrong. Ties go to
            # the later frame, which is the one more likely to be the
            # clearest photo of the same text.
            if len(current_norm) >= len(previous_norm):
                kept[-1] = (item_id, text)
        else:
            kept.append((item_id, text))
    return kept


def stitch(frames: list[tuple[str, str]]) -> str:
    """Join deduped frames into one continuous body of text."""
    return "\n\n".join(text for _item_id, text in frames)
