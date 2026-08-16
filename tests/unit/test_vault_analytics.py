"""Unit tests for the mastery rollup."""

from __future__ import annotations

from friday.vault.analytics import Analytics
from friday.vault.index import SQLiteVaultIndex
from friday.vault.models import (
    CaptureSource,
    Classification,
    Consensus,
    Item,
    Privacy,
    Solve,
    Verification,
)


def _stock(
    index: SQLiteVaultIndex,
    item_id: str,
    chapter: str,
    *,
    verified: bool,
    subject: str = "physics",
    privacy: Privacy = Privacy.PRIVATE,
    created_at: str = "2026-08-16T10:00:00+00:00",
    with_solve: bool = True,
) -> None:
    solve_id = f"s-{item_id}" if with_solve else "ghost-solve-id"
    if with_solve:
        solve = Solve(
            id=solve_id,
            item_ids=[item_id],
            verification=Verification(ok=verified),
            consensus=Consensus(final_answer="x", agreement="3/3"),
            created_at=created_at,
        )
        index.put_solve(solve)
    index.put_item(
        Item(
            id=item_id,
            owner_uid="u1",
            privacy=privacy,
            source=CaptureSource.CAMERA,
            created_at=created_at,
            solve_id=solve_id,
            classification=Classification(kind="problem", subject=subject, chapter=chapter),
        )
    )


def test_rollup_counts_attempts_per_chapter() -> None:
    index = SQLiteVaultIndex(":memory:")
    _stock(index, "a", "rotational motion", verified=False)
    _stock(index, "b", "rotational motion", verified=False)
    _stock(index, "c", "optics", verified=True)

    weakest = Analytics(index).rollup("u1")[0]
    assert weakest.chapter == "rotational motion"
    assert weakest.attempts == 2
    assert weakest.verified == 0
    assert weakest.mastery == 0.0


def test_rollup_sorts_weakest_first() -> None:
    index = SQLiteVaultIndex(":memory:")
    _stock(index, "a", "optics", verified=True)
    _stock(index, "b", "thermo", verified=False)
    assert [r.chapter for r in Analytics(index).rollup("u1")] == ["thermo", "optics"]


def test_items_without_a_solve_are_ignored() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(
        Item(
            id="z",
            owner_uid="u1",
            source=CaptureSource.CAMERA,
            created_at="2026-08-16T10:00:00+00:00",
            classification=Classification(kind="photo"),
        )
    )
    assert Analytics(index).rollup("u1") == []


def test_dangling_solve_id_is_not_counted_as_an_attempt() -> None:
    """An item's solve_id can point at a solve that no longer exists (deleted,
    or dropped by the index's own malformed-row skip). With no verification
    verdict to attach, it must not silently count as either a pass or a
    failure — it should simply not appear."""
    index = SQLiteVaultIndex(":memory:")
    _stock(index, "a", "optics", verified=True, with_solve=False)
    assert Analytics(index).rollup("u1") == []


def test_locked_items_are_excluded_from_the_rollup() -> None:
    """Locked items never appear in list_items(include_locked=False), and this
    rollup keeps that default: a mastery rollup would leak a locked chapter's
    existence, classification, and correctness through aggregate stats, which
    is exactly what marking something LOCKED is meant to prevent."""
    index = SQLiteVaultIndex(":memory:")
    _stock(index, "a", "optics", verified=True, privacy=Privacy.LOCKED)
    assert Analytics(index).rollup("u1") == []


def test_locked_items_do_not_leak_into_an_otherwise_visible_chapter() -> None:
    index = SQLiteVaultIndex(":memory:")
    _stock(index, "a", "optics", verified=True)
    _stock(index, "b", "optics", verified=True, privacy=Privacy.LOCKED)
    rollup = Analytics(index).rollup("u1")
    assert len(rollup) == 1
    assert rollup[0].attempts == 1


def test_unclassified_items_bucket_under_sentinel_labels() -> None:
    index = SQLiteVaultIndex(":memory:")
    _stock(index, "a", "", subject="", verified=False)
    row = Analytics(index).rollup("u1")[0]
    assert row.subject == "unclassified"
    assert row.chapter == "unfiled"


def test_ties_break_deterministically_by_subject_then_chapter() -> None:
    """Two chapters with identical mastery and attempts must not depend on
    dict insertion order for their relative position."""
    index = SQLiteVaultIndex(":memory:")
    _stock(index, "a", "waves", subject="physics", verified=False)
    _stock(index, "b", "acids", subject="chemistry", verified=False)
    rollup = Analytics(index).rollup("u1")
    assert [(r.subject, r.chapter) for r in rollup] == [
        ("chemistry", "acids"),
        ("physics", "waves"),
    ]


def test_tie_break_prefers_more_attempts_among_equally_weak_chapters() -> None:
    index = SQLiteVaultIndex(":memory:")
    _stock(index, "a", "optics", verified=False)
    _stock(index, "b", "optics", verified=False)
    _stock(index, "c", "thermo", verified=False)
    rollup = Analytics(index).rollup("u1")
    assert rollup[0].chapter == "optics"
    assert rollup[0].attempts == 2
    assert rollup[1].chapter == "thermo"
    assert rollup[1].attempts == 1


def test_unverifiable_but_correct_answers_still_score_zero_mastery() -> None:
    """Documents a real limitation rather than hiding it: SymPy only attempts
    a bare single-variable equation, so a chapter of entirely correct
    chemistry (or any word-problem-shaped) answers reports ok=False across the
    board and rolls up to mastery == 0.0, identical to a chapter that is
    genuinely being gotten wrong. `mastery` here means "fraction whose
    arithmetic was independently checked," not "fraction correct.\""""
    index = SQLiteVaultIndex(":memory:")
    _stock(index, "a", "stoichiometry", subject="chemistry", verified=False)
    _stock(index, "b", "stoichiometry", subject="chemistry", verified=False)
    row = Analytics(index).rollup("u1")[0]
    assert row.chapter == "stoichiometry"
    assert row.mastery == 0.0
