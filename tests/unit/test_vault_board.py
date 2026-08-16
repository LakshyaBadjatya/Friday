"""Unit tests for lecture-board frame stitching."""

from __future__ import annotations

from friday.vault.board import dedupe_frames, stitch


def test_identical_frames_collapse_to_one() -> None:
    frames = [("a", "x = 1"), ("b", "x = 1"), ("c", "x = 1")]
    assert [i for i, _ in dedupe_frames(frames)] == ["c"]


def test_a_growing_board_keeps_only_the_fullest_frame() -> None:
    frames = [("a", "x = 1"), ("b", "x = 1\ny = 2"), ("c", "x = 1\ny = 2\nz = 3")]
    assert [i for i, _ in dedupe_frames(frames)] == ["c"]


def test_an_erased_board_starts_a_new_frame() -> None:
    frames = [("a", "x = 1\ny = 2"), ("b", "completely different content here")]
    assert [i for i, _ in dedupe_frames(frames)] == ["a", "b"]


def test_stitch_joins_frames_in_order() -> None:
    assert stitch([("a", "first"), ("b", "second")]) == "first\n\nsecond"


# --- Stress tests -----------------------------------------------------------
#
# The heuristic (exact-line-set overlap against a fixed 0.6 threshold) was
# designed on paper, never run against realistic OCR output. These tests
# probe it deliberately, including cases that expose real weaknesses.


def test_realistic_ocr_noise_fragments_a_genuinely_growing_board() -> None:
    """CRITICAL FINDING: independent per-shot OCR errors on different lines
    compound across frames and repeatedly push overlap under the 0.6
    threshold, even though the board never stopped growing.

    Real OCR does not misread the same character in the same place on every
    shot -- a shaky hand, glare, or a slightly different crop moves the error
    around. Here the board grows 4 -> 5 -> 6 -> 6 lines (last shot is just a
    clearer re-photo of the same six lines), and each shot has 0-2 lines with
    a single character corrupted, but never the SAME line twice. That alone
    is enough to fragment one lecture into three "boards".
    """
    base = [f"line {i} = {i}" for i in range(1, 7)]

    def noisy(lines: list[str], bad_idx: list[int]) -> str:
        out = list(lines)
        for i in bad_idx:
            out[i] = out[i].replace("=", "_")  # simulates a misread character
        return "\n".join(out)

    frames = [
        ("a", noisy(base[:4], [])),
        ("b", noisy(base[:4], [1]) + "\n" + base[4]),  # +1 real line, noise on line 2
        ("c", noisy(base[:6], [0, 3])),  # +1 real line, noise moves to lines 1 & 4
        ("d", noisy(base[:6], [2])),  # same 6 lines, noise moves to line 3
    ]
    kept_ids = [item_id for item_id, _ in dedupe_frames(frames)]

    # Documents the CURRENT (undesirable) behaviour: a single continuously
    # growing board is split into three sections purely because OCR noise
    # landed on different lines from one shot to the next. A human reviewing
    # this lecture note would see three "boards" that were actually one.
    assert kept_ids == ["b", "c", "d"]


def test_short_board_single_character_ocr_error_causes_false_split() -> None:
    """A one-line (or two-line) board is the worst case: a single misread
    character drops overlap straight from 1.0 to 0.0, since there is no
    partial credit within a single line. Early in a lesson, before much is
    on the board, frames are exactly this short.
    """
    frames = [("a", "y = 2"), ("b", "y = Z")]  # OCR reads '2' as 'Z'
    kept_ids = [item_id for item_id, _ in dedupe_frames(frames)]
    # Wrongly treated as an erased board + a new one, instead of one frame.
    assert kept_ids == ["a", "b"]


def test_line_rewrap_causes_false_split_even_with_real_growth() -> None:
    """OCR line segmentation is not stable across shots -- a different crop
    or angle can merge/split lines that describe the same words. Exact-line
    comparison has no tolerance for this at all, even when the underlying
    content clearly grew.
    """
    frames = [
        ("a", "The quick brown fox\njumps over"),
        # Same two clauses, rewrapped into different lines, plus new text.
        ("b", "The quick brown fox jumps over\nthe lazy dog"),
    ]
    kept_ids = [item_id for item_id, _ in dedupe_frames(frames)]
    assert kept_ids == ["a", "b"]  # wrongly split; this was one growing board


def test_strict_subset_frame_is_treated_as_continuation() -> None:
    """A photo taken mid-erasure (subset of the previous frame) should not
    start a new section -- it's noise within the same board's lifecycle.
    The fuller (previous) frame is correctly retained.
    """
    frames = [("a", "x = 1\ny = 2\nz = 3"), ("b", "x = 1\ny = 2")]
    kept = dedupe_frames(frames)
    assert [i for i, _ in kept] == ["a"]
    assert kept[0][1] == "x = 1\ny = 2\nz = 3"


def test_overlap_exactly_on_the_threshold_counts_as_continuation() -> None:
    """0.6 is documented as inclusive (>=). 3 of 5 previous lines survive,
    which is exactly 0.6.
    """
    frames = [
        ("a", "l1\nl2\nl3\nl4\nl5"),
        ("b", "l1\nl2\nl3\nNEW1\nNEW2"),
    ]
    assert [i for i, _ in dedupe_frames(frames)] == ["b"]


def test_duplicate_lines_within_a_frame_do_not_distort_the_ratio() -> None:
    """`_lines` is a set, so repeated lines (e.g. a repeated header, or a
    table row written twice) collapse before the ratio is computed. A
    genuinely growing board should still be recognised as such.
    """
    frames = [
        ("a", "x = 1\nx = 1\nx = 1\ny = 2"),  # collapses to {x=1, y=2}
        ("b", "x = 1\ny = 2\nz = 3"),
    ]
    assert [i for i, _ in dedupe_frames(frames)] == ["b"]


def test_empty_and_whitespace_frames_do_not_crash_and_get_superseded() -> None:
    """Empty/whitespace-only OCR output (a blank or unreadable photo) must
    not raise a ZeroDivisionError, and should not block later real content
    from being kept.
    """
    frames = [("a", ""), ("b", "   \n  \n"), ("c", "real content")]
    assert [i for i, _ in dedupe_frames(frames)] == ["c"]


def test_single_frame_is_kept_as_is() -> None:
    assert dedupe_frames([("only", "hello")]) == [("only", "hello")]


def test_empty_frame_list_returns_empty_list() -> None:
    assert dedupe_frames([]) == []


def test_long_lesson_grow_erase_grow_erase_yields_three_sections() -> None:
    """A full lesson: board grows, gets erased, grows again, gets erased and
    a third topic starts. With clean (noise-free) OCR text, this should
    collapse to exactly three sections -- one per board lifecycle.
    """
    frames = [
        ("a", "A\nB"),
        ("b", "A\nB\nC"),  # grows -> still board 1
        ("c", "totally different topic here"),  # erased -> board 2 starts
        ("d", "totally different topic here\nmore stuff"),  # grows -> still board 2
        ("e", "yet another unrelated board"),  # erased -> board 3 starts
    ]
    assert [i for i, _ in dedupe_frames(frames)] == ["b", "d", "e"]
