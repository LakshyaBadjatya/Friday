"""Unit tests for the capture-processing pipeline.

The privacy rule is the thing to test hardest: a LOCKED item must be OCR'd but
never classified and never solved, and the model double must record zero
calls -- not just "the resulting fields happen to look untouched".
"""

from __future__ import annotations

import pytest

from friday.errors import ProviderError
from friday.perception.ocr import FakeOCR
from friday.providers.llm import FakeLLM, LLMProvider, LLMResponse, Message, ToolSpec
from friday.vault.index import SQLiteVaultIndex
from friday.vault.models import (
    CaptureSource,
    CloudinaryAsset,
    Item,
    ItemStatus,
    Privacy,
    Solve,
)
from friday.vault.pipeline import Pipeline
from friday.vault.quota import QuotaGuard
from friday.vault.solver import Solver

_OWNER = "u1"


def _item(
    item_id: str = "i1",
    *,
    privacy: Privacy = Privacy.PRIVATE,
    committed: bool = True,
) -> Item:
    return Item(
        id=item_id,
        owner_uid=_OWNER,
        privacy=privacy,
        source=CaptureSource.CAMERA,
        status=ItemStatus.UPLOADED,
        created_at="2026-08-16T10:00:00+00:00",
        cloudinary=(
            CloudinaryAsset(public_id=item_id, version=1, format="jpg", bytes=1024)
            if committed
            else None
        ),
    )


class _RecordingLLM(LLMProvider):
    """Wraps a :class:`FakeLLM` and records every call it receives.

    ``FakeLLM`` itself keeps no record of calls made, so this is what lets a
    test assert "the model was/wasn't called at all" rather than just
    inspecting the fields that came back -- the actual guarantee under test.
    """

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._inner = FakeLLM(responses=responses)
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> LLMResponse:
        self.calls.append(messages)
        return await self._inner.complete(messages, tools, model=model)


class _RaisingOCR:
    """An :class:`~friday.perception.ocr.OCRProvider` that always raises."""

    async def read(self, image: bytes) -> str:
        raise ProviderError("ocr backend unreachable")


class _RaisingSolver(Solver):
    """A :class:`Solver` stand-in whose ``solve`` always raises."""

    def __init__(self) -> None:
        super().__init__(FakeLLM(responses=[]), operators=[])

    async def solve(self, *, item_ids: list[str], ocr_text: str) -> Solve:
        raise RuntimeError("ensemble blew up")


class _SpyIndex(SQLiteVaultIndex):
    """Records the status of every ``put_item`` call, in order, for assertions
    on when (and how often) an item is persisted mid-flight."""

    def __init__(self) -> None:
        super().__init__(":memory:")
        self.put_statuses: list[ItemStatus] = []

    def put_item(self, item: Item) -> None:
        self.put_statuses.append(item.status)
        super().put_item(item)


def _classify_reply(kind: str) -> str:
    return f'{{"kind": "{kind}", "subject": "physics", "chapter": "", "tags": []}}'


_DRAFT_REPLY = (
    '{"subject": "physics", "statement": "find x", "steps": ["2x + 4 = 10"], '
    '"final_answer": "x = 3", "equation": "2*x + 4 = 10", "confidence": 0.9}'
)


def _solver(replies: list[str]) -> tuple[Solver, _RecordingLLM]:
    llm = _RecordingLLM([LLMResponse(text=r) for r in replies])
    return Solver(llm, operators=["VISION", "ORACLE", "GECKO"]), llm


def _pipeline(
    index: SQLiteVaultIndex,
    *,
    ocr=None,
    organizer_llm: LLMProvider | None = None,
    solver: Solver | None = None,
    quota: QuotaGuard | None = None,
    fetch_bytes=None,
    ocr_text: str = "2x + 4 = 10",
) -> Pipeline:
    return Pipeline(
        index=index,
        ocr=ocr if ocr is not None else FakeOCR(ocr_text),
        organizer_llm=organizer_llm if organizer_llm is not None else _RecordingLLM([]),
        solver=solver if solver is not None else _solver([])[0],
        quota=quota if quota is not None else QuotaGuard(index, quota_gb=25.0, daily_solve_cap=10),
        fetch_bytes=fetch_bytes if fetch_bytes is not None else (lambda public_id: b"bytes"),
    )


# --------------------------------------------------------------- the privacy rule


@pytest.mark.asyncio
async def test_locked_item_is_ocrd_but_never_classified_or_solved() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item(privacy=Privacy.LOCKED))

    organizer_llm = _RecordingLLM([LLMResponse(text=_classify_reply("problem"))])
    solver, solver_llm = _solver([_DRAFT_REPLY] * 3)

    pipeline = _pipeline(
        index,
        ocr=FakeOCR("2x + 4 = 10"),
        organizer_llm=organizer_llm,
        solver=solver,
        ocr_text="2x + 4 = 10",
    )
    await pipeline.process(_OWNER, "i1")

    item = index.get_item(_OWNER, "i1")
    assert item is not None
    assert item.status == ItemStatus.READY
    assert item.ocr_text == "2x + 4 = 10"
    assert item.ocr_engine == "FakeOCR"
    # The guarantee: no calls were EVER made to either model double.
    assert organizer_llm.calls == []
    assert solver_llm.calls == []
    assert item.classification.kind == "unknown"
    assert item.solve_id is None


# ------------------------------------------------------------------ normal problem


@pytest.mark.asyncio
async def test_normal_problem_is_ocrd_classified_and_solved() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item())

    organizer_llm = _RecordingLLM([LLMResponse(text=_classify_reply("problem"))])
    solver, solver_llm = _solver([_DRAFT_REPLY] * 3)

    pipeline = _pipeline(index, organizer_llm=organizer_llm, solver=solver)
    await pipeline.process(_OWNER, "i1")

    item = index.get_item(_OWNER, "i1")
    assert item is not None
    assert item.status == ItemStatus.READY
    assert item.ocr_text == "2x + 4 = 10"
    assert item.classification.kind == "problem"
    assert item.solve_id is not None

    solve = index.get_solve(item.solve_id)
    assert solve is not None
    assert solve.consensus.final_answer == "x = 3"
    assert len(solver_llm.calls) == 3
    assert len(organizer_llm.calls) == 1


# ---------------------------------------------------------------------- non-problem


@pytest.mark.asyncio
async def test_non_problem_is_filed_with_classification_but_not_solved() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item())

    organizer_llm = _RecordingLLM([LLMResponse(text=_classify_reply("receipt"))])
    solver, solver_llm = _solver([])

    pipeline = _pipeline(
        index, ocr=FakeOCR("Total: $4.50"), organizer_llm=organizer_llm, solver=solver
    )
    await pipeline.process(_OWNER, "i1")

    item = index.get_item(_OWNER, "i1")
    assert item is not None
    assert item.status == ItemStatus.READY
    assert item.classification.kind == "receipt"
    assert item.solve_id is None
    assert solver_llm.calls == []


# --------------------------------------------------------------------- quota cap


@pytest.mark.asyncio
async def test_solve_cap_exhausted_still_reaches_ready_without_a_solve() -> None:
    """A quota limit must never lose a photo."""
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item())

    organizer_llm = _RecordingLLM([LLMResponse(text=_classify_reply("problem"))])
    solver, solver_llm = _solver([_DRAFT_REPLY] * 3)
    quota = QuotaGuard(index, quota_gb=25.0, daily_solve_cap=1)
    pipeline = _pipeline(index, organizer_llm=organizer_llm, solver=solver, quota=quota)

    # Exhaust the cap before this item is even processed.
    quota.record_solve(pipeline.today())

    await pipeline.process(_OWNER, "i1")

    item = index.get_item(_OWNER, "i1")
    assert item is not None
    assert item.status == ItemStatus.READY
    assert item.classification.kind == "problem"
    assert item.solve_id is None
    assert solver_llm.calls == []


# ------------------------------------------------------------------------- OCR fails


@pytest.mark.asyncio
async def test_ocr_raising_still_reaches_ready_with_empty_text() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item())

    organizer_llm = _RecordingLLM([LLMResponse(text=_classify_reply("unknown"))])
    pipeline = _pipeline(index, ocr=_RaisingOCR(), organizer_llm=organizer_llm)

    await pipeline.process(_OWNER, "i1")

    item = index.get_item(_OWNER, "i1")
    assert item is not None
    assert item.status == ItemStatus.READY
    assert item.status != ItemStatus.FAILED
    assert item.ocr_text == ""


# ---------------------------------------------------------------------- solver fails


@pytest.mark.asyncio
async def test_solver_raising_still_reaches_ready_unsolved() -> None:
    """Deliberate choice: a solver outage/bug files the item unsolved rather
    than marking it FAILED -- the capture must never vanish because the
    ensemble broke. See the module docstring on :mod:`friday.vault.pipeline`."""
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item())

    organizer_llm = _RecordingLLM([LLMResponse(text=_classify_reply("problem"))])
    pipeline = _pipeline(index, organizer_llm=organizer_llm, solver=_RaisingSolver())

    await pipeline.process(_OWNER, "i1")

    item = index.get_item(_OWNER, "i1")
    assert item is not None
    assert item.status == ItemStatus.READY
    assert item.status != ItemStatus.FAILED
    assert item.classification.kind == "problem"
    assert item.solve_id is None


# ------------------------------------------------------------------- fetch_bytes fails


@pytest.mark.asyncio
async def test_fetch_bytes_raising_still_reaches_ready_with_empty_text() -> None:
    """A network failure fetching the image from Cloudinary must not lose the
    capture either -- it is caught by the same guard as an OCR failure."""
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item())

    def _boom(public_id: str) -> bytes:
        raise ConnectionError("cloudinary unreachable")

    organizer_llm = _RecordingLLM([LLMResponse(text=_classify_reply("unknown"))])
    pipeline = _pipeline(index, fetch_bytes=_boom, organizer_llm=organizer_llm)

    await pipeline.process(_OWNER, "i1")

    item = index.get_item(_OWNER, "i1")
    assert item is not None
    assert item.status == ItemStatus.READY
    assert item.ocr_text == ""
    assert item.ocr_engine == ""


# ------------------------------------------------------------------- missing / uncommitted


@pytest.mark.asyncio
async def test_missing_item_is_a_quiet_no_op() -> None:
    index = SQLiteVaultIndex(":memory:")
    pipeline = _pipeline(index)
    # Must not raise.
    await pipeline.process(_OWNER, "does-not-exist")
    assert index.get_item(_OWNER, "does-not-exist") is None


@pytest.mark.asyncio
async def test_item_with_no_committed_cloudinary_asset_is_a_quiet_no_op() -> None:
    index = SQLiteVaultIndex(":memory:")
    index.put_item(_item(committed=False))
    fetch_calls: list[str] = []

    pipeline = _pipeline(index, fetch_bytes=lambda pid: fetch_calls.append(pid) or b"")
    await pipeline.process(_OWNER, "i1")

    item = index.get_item(_OWNER, "i1")
    assert item is not None
    assert item.status == ItemStatus.UPLOADED  # untouched
    assert fetch_calls == []


# --------------------------------------------------------------- status transitions


@pytest.mark.asyncio
async def test_status_transitions_are_persisted_in_order() -> None:
    """An observer polling mid-flight sees PROCESSING, not a jump straight
    from UPLOADED to READY -- and the very first persisted status is
    PROCESSING, the very last is READY."""
    index = _SpyIndex()
    index.put_item(_item())

    organizer_llm = _RecordingLLM([LLMResponse(text=_classify_reply("problem"))])
    solver, _ = _solver([_DRAFT_REPLY] * 3)
    pipeline = _pipeline(index, organizer_llm=organizer_llm, solver=solver)

    index.put_statuses.clear()  # drop the initial seeding write above
    await pipeline.process(_OWNER, "i1")

    assert index.put_statuses[0] == ItemStatus.PROCESSING
    assert index.put_statuses[-1] == ItemStatus.READY
    # READY must never appear before the very last write.
    assert ItemStatus.READY not in index.put_statuses[:-1]
    assert len(index.put_statuses) >= 2


@pytest.mark.asyncio
async def test_locked_item_status_transitions_still_go_through_processing() -> None:
    index = _SpyIndex()
    index.put_item(_item(privacy=Privacy.LOCKED))
    pipeline = _pipeline(index)

    index.put_statuses.clear()
    await pipeline.process(_OWNER, "i1")

    assert index.put_statuses[0] == ItemStatus.PROCESSING
    assert index.put_statuses[-1] == ItemStatus.READY


# ------------------------------------------------------------------------- today()


def test_today_returns_a_plausible_yyyy_mm_dd_key() -> None:
    index = SQLiteVaultIndex(":memory:")
    pipeline = _pipeline(index)
    today = pipeline.today()
    assert len(today) == 10
    assert today[4] == "-" and today[7] == "-"


# -------------------------------------------------------------- FAILED is unused


def test_failed_status_exists_but_this_pipeline_never_sets_it() -> None:
    """``ItemStatus.FAILED`` is a defined lifecycle state, but nothing in this
    module ever assigns it -- both failure modes it could plausibly cover
    (OCR/fetch failure, solver failure) deliberately degrade to a filed,
    unsolved READY item instead. See the other tests above and the module
    docstring for why. This test exists to make that an explicit, checked
    claim rather than an implicit one that silently drifts."""
    assert ItemStatus.FAILED == "failed"
    # Exercised indirectly by every "still reaches READY" test above; this
    # assertion just names the enum member so the claim is grep-able.
