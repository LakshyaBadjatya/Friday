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
    VerificationStatus,
)


def _stock(
    index: SQLiteVaultIndex,
    item_id: str,
    chapter: str,
    *,
    status: VerificationStatus = VerificationStatus.NOT_VERIFIABLE,
    agreement: str = "3/3",
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
            verification=Verification(status=status),
            consensus=Consensus(final_answer="x", agreement=agreement),
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
    _stock(index, "a", "rotational motion", status=VerificationStatus.REFUTED)
    _stock(index, "b", "rotational motion", status=VerificationStatus.REFUTED)
    _stock(index, "c", "optics", status=VerificationStatus.VERIFIED)

    weakest = Analytics(index).rollup("u1")[0]
    assert weakest.chapter == "rotational motion"
    assert weakest.attempts == 2
    assert weakest.verified == 0
    assert weakest.mastery == 0.0
    assert weakest.basis == "sympy"


def test_rollup_sorts_weakest_first() -> None:
    index = SQLiteVaultIndex(":memory:")
    _stock(index, "a", "optics", status=VerificationStatus.VERIFIED)
    _stock(index, "b", "thermo", status=VerificationStatus.REFUTED)
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
    _stock(index, "a", "optics", status=VerificationStatus.VERIFIED, with_solve=False)
    assert Analytics(index).rollup("u1") == []


def test_locked_items_are_excluded_from_the_rollup() -> None:
    """Locked items never appear in list_items(include_locked=False), and this
    rollup keeps that default: a mastery rollup would leak a locked chapter's
    existence, classification, and correctness through aggregate stats, which
    is exactly what marking something LOCKED is meant to prevent."""
    index = SQLiteVaultIndex(":memory:")
    _stock(index, "a", "optics", status=VerificationStatus.VERIFIED, privacy=Privacy.LOCKED)
    assert Analytics(index).rollup("u1") == []


def test_locked_items_do_not_leak_into_an_otherwise_visible_chapter() -> None:
    index = SQLiteVaultIndex(":memory:")
    _stock(index, "a", "optics", status=VerificationStatus.VERIFIED)
    _stock(index, "b", "optics", status=VerificationStatus.VERIFIED, privacy=Privacy.LOCKED)
    rollup = Analytics(index).rollup("u1")
    assert len(rollup) == 1
    assert rollup[0].attempts == 1


def test_unclassified_items_bucket_under_sentinel_labels() -> None:
    index = SQLiteVaultIndex(":memory:")
    _stock(index, "a", "", subject="", status=VerificationStatus.REFUTED)
    row = Analytics(index).rollup("u1")[0]
    assert row.subject == "unclassified"
    assert row.chapter == "unfiled"


def test_ties_break_deterministically_by_subject_then_chapter() -> None:
    """Two chapters with identical mastery and attempts must not depend on
    dict insertion order for their relative position."""
    index = SQLiteVaultIndex(":memory:")
    _stock(index, "a", "waves", subject="physics", status=VerificationStatus.REFUTED)
    _stock(index, "b", "acids", subject="chemistry", status=VerificationStatus.REFUTED)
    rollup = Analytics(index).rollup("u1")
    assert [(r.subject, r.chapter) for r in rollup] == [
        ("chemistry", "acids"),
        ("physics", "waves"),
    ]


def test_tie_break_prefers_more_attempts_among_equally_weak_chapters() -> None:
    index = SQLiteVaultIndex(":memory:")
    _stock(index, "a", "optics", status=VerificationStatus.REFUTED)
    _stock(index, "b", "optics", status=VerificationStatus.REFUTED)
    _stock(index, "c", "thermo", status=VerificationStatus.REFUTED)
    rollup = Analytics(index).rollup("u1")
    assert rollup[0].chapter == "optics"
    assert rollup[0].attempts == 2
    assert rollup[1].chapter == "thermo"
    assert rollup[1].attempts == 1


def test_unverifiable_but_correct_answers_do_not_score_zero_mastery() -> None:
    """The bug this whole change set exists to fix: a chapter that is never
    checkable by SymPy (chemistry naming, prose, word problems) used to
    conflate "never checked" with "checked and wrong" and roll up to
    mastery == 0.0 — indistinguishable from a chapter genuinely being gotten
    wrong. Now NOT_VERIFIABLE solves are excluded from the sympy ratio
    entirely, and the chapter falls back to ensemble agreement instead — an
    always-unanimous panel must not score 0.0."""
    index = SQLiteVaultIndex(":memory:")
    _stock(
        index,
        "a",
        "stoichiometry",
        subject="chemistry",
        status=VerificationStatus.NOT_VERIFIABLE,
        agreement="3/3",
    )
    _stock(
        index,
        "b",
        "stoichiometry",
        subject="chemistry",
        status=VerificationStatus.NOT_VERIFIABLE,
        agreement="3/3",
    )
    row = Analytics(index).rollup("u1")[0]
    assert row.chapter == "stoichiometry"
    assert row.mastery != 0.0
    assert row.mastery == 1.0
    assert row.basis == "agreement"
    assert row.verifiable_attempts == 0


def test_a_mixed_chapter_reports_sympy_basis_and_ignores_unverifiable_solves() -> None:
    """One verified, one refuted, and one not-verifiable in the same chapter:
    the not-verifiable solve must be excluded from the ratio's denominator
    entirely, not folded in as a success or a failure."""
    index = SQLiteVaultIndex(":memory:")
    _stock(index, "a", "kinematics", status=VerificationStatus.VERIFIED)
    _stock(index, "b", "kinematics", status=VerificationStatus.REFUTED)
    _stock(index, "c", "kinematics", status=VerificationStatus.NOT_VERIFIABLE, agreement="3/3")
    row = Analytics(index).rollup("u1")[0]
    assert row.chapter == "kinematics"
    assert row.attempts == 3
    assert row.verifiable_attempts == 2
    assert row.verified == 1
    assert row.mastery == 0.5
    assert row.basis == "sympy"


def test_a_chapter_with_none_verifiable_reports_agreement_basis() -> None:
    index = SQLiteVaultIndex(":memory:")
    _stock(index, "a", "organic naming", status=VerificationStatus.NOT_VERIFIABLE, agreement="2/3")
    row = Analytics(index).rollup("u1")[0]
    assert row.basis == "agreement"
    assert row.verifiable_attempts == 0


def test_a_split_panel_scores_lower_than_a_unanimous_one_under_agreement_basis() -> None:
    """Neither chapter has anything SymPy can check, so both fall back to the
    agreement signal — but a chapter where the panel always split is a
    weaker confidence signal than one where it was always unanimous, and the
    rollup must reflect that ordering."""
    index = SQLiteVaultIndex(":memory:")
    _stock(index, "a", "always split", status=VerificationStatus.NOT_VERIFIABLE, agreement="1/3")
    _stock(index, "b", "always split", status=VerificationStatus.NOT_VERIFIABLE, agreement="1/3")
    _stock(
        index, "c", "always unanimous", status=VerificationStatus.NOT_VERIFIABLE, agreement="3/3"
    )
    _stock(
        index, "d", "always unanimous", status=VerificationStatus.NOT_VERIFIABLE, agreement="3/3"
    )
    rollup = Analytics(index).rollup("u1")
    assert [r.chapter for r in rollup] == ["always split", "always unanimous"]
    assert rollup[0].mastery == 0.0
    assert rollup[1].mastery == 1.0
    assert rollup[0].basis == rollup[1].basis == "agreement"


def test_malformed_agreement_strings_do_not_crash_and_count_as_non_unanimous() -> None:
    index = SQLiteVaultIndex(":memory:")
    _stock(index, "a", "garbled", status=VerificationStatus.NOT_VERIFIABLE, agreement="not-a-ratio")
    _stock(index, "b", "garbled", status=VerificationStatus.NOT_VERIFIABLE, agreement="0/0")
    _stock(index, "c", "garbled", status=VerificationStatus.NOT_VERIFIABLE, agreement="")
    row = Analytics(index).rollup("u1")[0]
    assert row.chapter == "garbled"
    assert row.basis == "agreement"
    assert row.mastery == 0.0
