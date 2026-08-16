"""Unit tests for the vault domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from friday.vault.models import (
    CaptureSource,
    Classification,
    CloudinaryAsset,
    Draft,
    ExamSession,
    ExamSessionStatus,
    Item,
    ItemStatus,
    Note,
    Privacy,
    Solve,
    Verification,
    VerificationStatus,
)


def _asset() -> CloudinaryAsset:
    return CloudinaryAsset(
        public_id="vault/u1/abc",
        version=1,
        format="jpg",
        bytes=1024,
        width=800,
        height=600,
        resource_type="image",
    )


def test_item_defaults_to_private_and_pending() -> None:
    item = Item(
        id="i1",
        owner_uid="u1",
        source=CaptureSource.CAMERA,
        created_at="2026-08-16T10:00:00+00:00",
    )
    assert item.privacy is Privacy.PRIVATE
    assert item.status is ItemStatus.PENDING
    assert item.space == "private"
    assert item.cloudinary is None
    assert item.note_ids == []


def test_locked_items_refuse_model_calls() -> None:
    item = Item(
        id="i2",
        owner_uid="u1",
        source=CaptureSource.SCREEN,
        privacy=Privacy.LOCKED,
        created_at="2026-08-16T10:00:00+00:00",
    )
    assert item.may_leave_for_model() is False


def test_private_items_allow_model_calls() -> None:
    item = Item(
        id="i3",
        owner_uid="u1",
        source=CaptureSource.CAMERA,
        created_at="2026-08-16T10:00:00+00:00",
    )
    assert item.may_leave_for_model() is True


def test_shared_items_also_allow_model_calls() -> None:
    """Only LOCKED blocks a provider call — SHARED must not be mistaken for it."""
    item = Item(
        id="i3b",
        owner_uid="u1",
        source=CaptureSource.CAMERA,
        privacy=Privacy.SHARED,
        created_at="2026-08-16T10:00:00+00:00",
    )
    assert item.may_leave_for_model() is True


def test_item_round_trips_a_committed_asset() -> None:
    item = Item(
        id="i4",
        owner_uid="u1",
        source=CaptureSource.CAMERA,
        status=ItemStatus.READY,
        cloudinary=_asset(),
        classification=Classification(kind="problem", subject="physics"),
        created_at="2026-08-16T10:00:00+00:00",
    )
    restored = Item.model_validate(item.model_dump())
    assert restored.cloudinary is not None
    assert restored.cloudinary.public_id == "vault/u1/abc"
    assert restored.classification.subject == "physics"


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Item(
            id="i5",
            owner_uid="u1",
            source="telepathy",  # type: ignore[arg-type]
            created_at="2026-08-16T10:00:00+00:00",
        )


def test_item_mutable_defaults_are_not_shared_between_instances() -> None:
    """Field(default_factory=list) must give each Item its own note_ids list."""
    first = Item(
        id="i6",
        owner_uid="u1",
        source=CaptureSource.CAMERA,
        created_at="2026-08-16T10:00:00+00:00",
    )
    second = Item(
        id="i7",
        owner_uid="u1",
        source=CaptureSource.CAMERA,
        created_at="2026-08-16T10:00:00+00:00",
    )
    first.note_ids.append("n1")
    assert first.note_ids == ["n1"]
    assert second.note_ids == []


def test_solve_round_trips_drafts_and_consensus() -> None:
    solve = Solve(
        id="s1",
        item_ids=["i4"],
        subject="physics",
        statement="Find the acceleration.",
        drafts=[
            {
                "operator": "VISION",
                "model_id": "claude",
                "steps": ["step 1", "step 2"],
                "final_answer": "9.8 m/s^2",
                "equation": "10*v = 98",
                "confidence": 0.8,
                "latency_ms": 1200,
            }
        ],
        verification={"engine": "sympy", "status": "verified", "detail": "checks out"},
        consensus={"final_answer": "9.8 m/s^2", "agreement": "3/3", "judged": True},
        dissent=["ORACLE disagreed initially"],
        created_at="2026-08-16T10:00:00+00:00",
    )
    restored = Solve.model_validate(solve.model_dump())
    assert restored.drafts[0].final_answer == "9.8 m/s^2"
    assert restored.drafts[0].steps == ["step 1", "step 2"]
    assert restored.drafts[0].equation == "10*v = 98"
    assert restored.verification.ok is True
    assert restored.consensus.agreement == "3/3"
    assert restored.dissent == ["ORACLE disagreed initially"]


def test_draft_equation_defaults_to_empty_string() -> None:
    draft = Draft(operator="VISION", final_answer="9 V")
    assert draft.equation == ""


def test_note_round_trips_item_links() -> None:
    note = Note(
        id="n1",
        owner_uid="u1",
        title="Kinematics recap",
        subject="physics",
        item_ids=["i4"],
        flashcard_ids=[1, 2, 3],
        created_at="2026-08-16T10:00:00+00:00",
    )
    restored = Note.model_validate(note.model_dump())
    assert restored.item_ids == ["i4"]
    assert restored.flashcard_ids == [1, 2, 3]
    assert restored.title == "Kinematics recap"


def test_exam_session_round_trips_grading() -> None:
    exam = ExamSession(
        id="e1",
        owner_uid="u1",
        paper_item_ids=["i1", "i2"],
        answer_item_ids=["i3"],
        started_at="2026-08-16T10:00:00+00:00",
        duration_s=3600,
        status=ExamSessionStatus.GRADED,
        grading=[
            {"q": "1a", "marks_awarded": 4.0, "marks_total": 5.0, "feedback": "close"},
        ],
        total=4.0,
    )
    restored = ExamSession.model_validate(exam.model_dump())
    assert restored.paper_item_ids == ["i1", "i2"]
    assert restored.status is ExamSessionStatus.GRADED
    assert restored.grading[0].q == "1a"
    assert restored.grading[0].marks_awarded == 4.0
    assert restored.total == 4.0


def test_exam_session_defaults_to_open() -> None:
    exam = ExamSession(
        id="e2",
        owner_uid="u1",
        started_at="2026-08-16T10:00:00+00:00",
    )
    assert exam.status is ExamSessionStatus.OPEN


def test_exam_session_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        ExamSession(
            id="e3",
            owner_uid="u1",
            started_at="2026-08-16T10:00:00+00:00",
            status="archived",  # type: ignore[arg-type]
        )


def test_solve_requires_created_at() -> None:
    with pytest.raises(ValidationError):
        Solve(id="s2", item_ids=["i1"])  # type: ignore[call-arg]


def test_note_requires_created_at() -> None:
    with pytest.raises(ValidationError):
        Note(id="n2", owner_uid="u1")  # type: ignore[call-arg]


def test_exam_session_requires_started_at() -> None:
    with pytest.raises(ValidationError):
        ExamSession(id="e4", owner_uid="u1")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("status", "expected_ok"),
    [
        (VerificationStatus.VERIFIED, True),
        (VerificationStatus.REFUTED, False),
        (VerificationStatus.NOT_VERIFIABLE, False),
    ],
)
def test_verification_status_round_trips_and_ok_reflects_only_verified(
    status: VerificationStatus, expected_ok: bool
) -> None:
    verification = Verification(status=status, detail="d")
    restored = Verification.model_validate(verification.model_dump())
    assert restored.status is status
    assert restored.ok is expected_ok


def test_verification_defaults_to_not_verifiable() -> None:
    assert Verification().status is VerificationStatus.NOT_VERIFIABLE
    assert Verification().ok is False


def test_verification_ok_is_not_a_settable_field() -> None:
    """``ok`` is a read-only view onto ``status`` now — passing it as a
    constructor kwarg must not silently flip the outcome (pydantic's default
    ``extra="ignore"`` drops it, so status keeps its own default)."""
    verification = Verification(status=VerificationStatus.REFUTED, ok=True)  # type: ignore[call-arg]
    assert verification.status is VerificationStatus.REFUTED
    assert verification.ok is False
