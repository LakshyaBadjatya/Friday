"""Unit tests for lecture-board frame stitching."""

from __future__ import annotations

from fractions import Fraction

from friday.vault.board import _CONTINUATION, dedupe_frames, stitch


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
# The original heuristic (exact-line-set overlap against a fixed threshold)
# did not survive contact with realistic OCR noise -- see git history for the
# evidence. The module now compares normalized text with
# difflib.SequenceMatcher, which is tolerant of small per-shot differences.
# These tests confirm the fix actually fixes the cases that broke it, without
# becoming so tolerant that genuine erasures are missed.


def test_realistic_ocr_noise_no_longer_fragments_a_growing_board() -> None:
    """A board grows 4 -> 5 -> 6 -> 6 lines (the last shot is just a clearer
    re-photo of the same six lines). Each shot has 0-2 lines with a single
    character corrupted (simulating a misread character), and critically the
    noise lands on a DIFFERENT line each time -- real OCR doesn't misread the
    same spot twice. This used to fragment into three sections; it must now
    collapse into one.
    """
    base = [f"line {i} = {i}" for i in range(1, 7)]

    def noisy(lines: list[str], bad_idx: list[int]) -> str:
        out = list(lines)
        for i in bad_idx:
            out[i] = out[i].replace("=", "_")
        return "\n".join(out)

    frames = [
        ("a", noisy(base[:4], [])),
        ("b", noisy(base[:4], [1]) + "\n" + base[4]),
        ("c", noisy(base[:6], [0, 3])),
        ("d", noisy(base[:6], [2])),
    ]
    kept = dedupe_frames(frames)
    assert [item_id for item_id, _ in kept] == ["d"]
    assert kept[0][1] == noisy(base[:6], [2])


def test_short_board_single_character_ocr_error_no_longer_splits() -> None:
    """A one-line board with a single misread character used to drop overlap
    straight from 1.0 to 0.0 under exact-line comparison. Character-level
    similarity gives partial credit instead.
    """
    frames = [("a", "y = 2"), ("b", "y = Z")]  # OCR reads '2' as 'Z'
    assert [item_id for item_id, _ in dedupe_frames(frames)] == ["b"]


def test_line_rewrap_no_longer_splits_a_growing_board() -> None:
    """OCR line segmentation is not stable across shots -- a different crop
    or angle can merge/split lines describing the same words. Character-level
    similarity does not depend on line boundaries lining up between shots.
    """
    frames = [
        ("a", "The quick brown fox\njumps over"),
        # Same two clauses, rewrapped into different lines, plus new text.
        ("b", "The quick brown fox jumps over\nthe lazy dog"),
    ]
    assert [item_id for item_id, _ in dedupe_frames(frames)] == ["b"]


def test_erasure_is_still_detected_even_with_a_shared_lecture_title() -> None:
    """The failure mode traded against: becoming so tolerant that a real
    erasure is missed. A board is erased and rewritten under the same
    lecture title -- title text is shared, but the body is unrelated. This
    must still split into two sections. (This is also the tightest case in
    the fixture: it sits closest to the continuation threshold, 0.529 vs the
    0.58 cutoff, and is what the threshold was chosen against.)
    """
    frames = [
        (
            "a",
            "Lecture 4: Graph Algorithms\n"
            "BFS visits nodes level by level\n"
            "Queue-based implementation",
        ),
        (
            "b",
            "Lecture 4: Graph Algorithms\n"
            "Dijkstra requires non-negative weights\n"
            "Uses a priority queue",
        ),
    ]
    assert [item_id for item_id, _ in dedupe_frames(frames)] == ["a", "b"]


def test_similarity_threshold_is_inclusive_at_the_boundary() -> None:
    """The docstring says "at or above" -- confirm >= is really used, right
    at the boundary, not > by accident.

    Constructed so the similarity ratio lands on exactly `_CONTINUATION`:
    SequenceMatcher.ratio() for a string P that is a strict prefix of string
    C is 2*len(P) / (len(P) + len(C)). Solving 2n / (2n + k) == _CONTINUATION
    for integer n, k gives the counts below.
    """
    threshold = Fraction(_CONTINUATION).limit_denominator(1000)
    p, q = threshold.numerator, threshold.denominator
    # 2n / (2n + k) == p/q  =>  n / k == p / (2*(q - p))
    n = p
    k = 2 * (q - p)

    previous_text = "a" * n
    at_boundary = previous_text + "b" * k
    just_below_boundary = previous_text + "b" * (k + 1)

    kept_at = dedupe_frames([("a", previous_text), ("b", at_boundary)])
    assert [item_id for item_id, _ in kept_at] == ["b"]  # continuation

    kept_below = dedupe_frames([("a", previous_text), ("b", just_below_boundary)])
    assert [item_id for item_id, _ in kept_below] == ["a", "b"]  # new section


def test_strict_subset_frame_is_treated_as_continuation() -> None:
    """A photo taken mid-erasure (subset of the previous frame) should not
    start a new section -- it's noise within the same board's lifecycle.
    The fuller (previous) frame is correctly retained.
    """
    frames = [("a", "x = 1\ny = 2\nz = 3"), ("b", "x = 1\ny = 2")]
    kept = dedupe_frames(frames)
    assert [i for i, _ in kept] == ["a"]
    assert kept[0][1] == "x = 1\ny = 2\nz = 3"


def test_duplicate_lines_within_a_frame_do_not_distort_the_ratio() -> None:
    """Repeated lines (e.g. a repeated header, or a table row written twice)
    are deduplicated during normalization so they don't pad out a frame's
    length and unfairly drag down its similarity to a frame that says the
    same thing without the repeat. A genuinely growing board should still be
    recognised as such.
    """
    frames = [
        ("a", "x = 1\nx = 1\nx = 1\ny = 2"),
        ("b", "x = 1\ny = 2\nz = 3"),
    ]
    assert [i for i, _ in dedupe_frames(frames)] == ["b"]


def test_empty_and_whitespace_frames_do_not_crash_and_get_superseded() -> None:
    """Empty/whitespace-only OCR output (a blank or unreadable photo) must
    not raise, and should not block later real content from being kept.
    """
    frames = [("a", ""), ("b", "   \n  \n"), ("c", "real content")]
    assert [i for i, _ in dedupe_frames(frames)] == ["c"]


def test_blank_frame_in_the_middle_of_real_content_is_skipped() -> None:
    """A single unreadable/blank photo taken mid-burst (someone walked in
    front of the board) must not be treated as an erasure, and must not stop
    the frame after it from being recognised as a continuation of the frame
    before it.
    """
    frames = [
        ("a", "real board content here"),
        ("b", "   "),
        ("c", "real board content here plus more"),
    ]
    assert [i for i, _ in dedupe_frames(frames)] == ["c"]


def test_single_frame_is_kept_as_is() -> None:
    assert dedupe_frames([("only", "hello")]) == [("only", "hello")]


def test_empty_frame_list_returns_empty_list() -> None:
    assert dedupe_frames([]) == []


def test_long_lesson_grow_erase_grow_erase_yields_three_sections() -> None:
    """A full lesson: board grows, gets erased, grows again, gets erased and
    a third topic starts. This should collapse to exactly three sections --
    one per board lifecycle.
    """
    frames = [
        ("a", "A\nB"),
        ("b", "A\nB\nC"),  # grows -> still board 1
        ("c", "totally different topic here"),  # erased -> board 2 starts
        ("d", "totally different topic here\nmore stuff"),  # grows -> still board 2
        ("e", "yet another unrelated board"),  # erased -> board 3 starts
    ]
    assert [i for i, _ in dedupe_frames(frames)] == ["b", "d", "e"]


def test_long_incremental_growth_does_not_drift_into_a_spurious_split() -> None:
    """Comparison is always against the immediately preceding KEPT frame,
    which itself keeps getting replaced by a fuller one as the board grows.
    That must prevent drift: even after many replacements, a steadily
    growing board should stay one section right to the end, because each
    step is only ever compared against its immediate predecessor (small,
    high-similarity step), never against the board's very first frame (which
    would look nothing alike after ten rounds of growth).
    """
    topics = [f"topic point {i}: some explanatory text goes here" for i in range(1, 11)]
    frames = [(str(i), "\n".join(topics[:i])) for i in range(1, 11)]

    kept = dedupe_frames(frames)

    assert [item_id for item_id, _ in kept] == ["10"]
    assert kept[0][1] == "\n".join(topics)
