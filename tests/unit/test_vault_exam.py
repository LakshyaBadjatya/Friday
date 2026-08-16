"""Unit tests for :mod:`friday.vault.exam`.

``_ScriptedLLM`` implements the real :class:`~friday.providers.llm.LLMProvider`
contract (``complete(messages: list[Message], tools=None, *, model=None) ->
LLMResponse``) and records the last prompt sent, so tests can assert on what
text actually reached the model — the load-bearing check for the locked-item
privacy tests below.
"""

from __future__ import annotations

import pytest

from friday.errors import ProviderError
from friday.providers.llm import FakeLLM, LLMProvider, LLMResponse, Message, ToolSpec
from friday.vault.exam import ExamRunner
from friday.vault.index import SQLiteVaultIndex
from friday.vault.models import ExamSessionStatus, Item, Privacy


class _ScriptedLLM(LLMProvider):
    """Returns a queued reply per call and records every prompt sent."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> LLMResponse:
        self.prompts.append(messages[-1].content or "" if messages else "")
        text = self._replies.pop(0) if self._replies else ""
        return LLMResponse(text=text)


def _make_item(
    item_id: str,
    owner_uid: str,
    ocr_text: str,
    *,
    privacy: Privacy = Privacy.PRIVATE,
) -> Item:
    return Item(
        id=item_id,
        owner_uid=owner_uid,
        privacy=privacy,
        source="camera",
        ocr_text=ocr_text,
        created_at="2026-08-16T10:00:00+00:00",
    )


def _index_with(*items: Item) -> SQLiteVaultIndex:
    index = SQLiteVaultIndex()
    for item in items:
        index.put_item(item)
    return index


_ONE_QUESTION_REPLY = (
    '{"questions": [{"q": "1", "marks_awarded": 4, "marks_total": 5, '
    '"feedback": "dropped the unit on the final line"}], "total": 4}'
)


# --------------------------------------------------------------------------- #
# start / get
# --------------------------------------------------------------------------- #
def test_start_records_paper_ids_clock_and_open_status() -> None:
    index = _index_with(_make_item("p1", "u1", "Q1. Find x."))
    runner = ExamRunner(index=index, llm=FakeLLM([]), clock=lambda: "2099-01-01T00:00:00+00:00")

    session = runner.start("u1", ["p1"], duration_s=1800)

    assert session.owner_uid == "u1"
    assert session.paper_item_ids == ["p1"]
    assert session.started_at == "2099-01-01T00:00:00+00:00"
    assert session.duration_s == 1800
    assert session.status is ExamSessionStatus.OPEN
    assert session.grading == []
    assert session.total == 0.0


def test_get_returns_the_started_session_and_none_for_unknown() -> None:
    index = _index_with(_make_item("p1", "u1", "Q1. Find x."))
    runner = ExamRunner(index=index, llm=FakeLLM([]))

    session = runner.start("u1", ["p1"], duration_s=600)

    assert runner.get(session.id) is session
    assert runner.get("does-not-exist") is None


# --------------------------------------------------------------------------- #
# grade: happy path
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_grade_produces_per_question_marks_feedback_total_and_graded_status() -> None:
    index = _index_with(
        _make_item("p1", "u1", "Q1. Find the terminal voltage."),
        _make_item("a1", "u1", "V = 4 - (1)(1) = 3 V"),
    )
    llm = _ScriptedLLM([_ONE_QUESTION_REPLY])
    runner = ExamRunner(index=index, llm=llm)
    session = runner.start("u1", ["p1"], duration_s=1800)

    graded = await runner.grade("u1", session.id, ["a1"])

    assert graded.status is ExamSessionStatus.GRADED
    assert graded.answer_item_ids == ["a1"]
    assert len(graded.grading) == 1
    assert graded.grading[0].q == "1"
    assert graded.grading[0].marks_awarded == 4.0
    assert graded.grading[0].marks_total == 5.0
    assert "unit" in graded.grading[0].feedback
    assert graded.total == 4.0
    # The same object is stored, so a later get() reflects the grading.
    assert runner.get(session.id) is graded


# --------------------------------------------------------------------------- #
# grade: access control
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_grading_an_unknown_session_id_raises_keyerror() -> None:
    index = _index_with()
    runner = ExamRunner(index=index, llm=FakeLLM([]))

    with pytest.raises(KeyError):
        await runner.grade("u1", "no-such-session", ["a1"])


@pytest.mark.asyncio
async def test_grading_another_owners_session_raises_keyerror_not_leaking_it() -> None:
    """A real access-control boundary: owner B must not be able to grade (or
    learn anything about) owner A's session, even though the id is known."""
    index = _index_with(_make_item("p1", "u1", "Q1. Find x."), _make_item("a1", "u2", "x = 3"))
    runner = ExamRunner(index=index, llm=FakeLLM([]))
    session = runner.start("u1", ["p1"], duration_s=600)

    with pytest.raises(KeyError):
        await runner.grade("u2", session.id, ["a1"])

    # And the session itself must be untouched by the attempt.
    unchanged = runner.get(session.id)
    assert unchanged is not None
    assert unchanged.status is ExamSessionStatus.OPEN
    assert unchanged.grading == []


# --------------------------------------------------------------------------- #
# grade: total is recomputed, never trusted from the model
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_total_is_recomputed_from_marks_awarded_not_trusted_from_the_model() -> None:
    """The model claims total=50, but its own per-question marks only sum to
    7. The per-question breakdown is the ground truth for this feature — a
    headline total that contradicts it would be actively misleading — so the
    stored total must be the recomputed 7, not the model's 50."""
    index = _index_with(_make_item("p1", "u1", "Q1. Q2."), _make_item("a1", "u1", "ans1. ans2."))
    reply = (
        '{"questions": ['
        '{"q": "1", "marks_awarded": 3, "marks_total": 5, "feedback": "ok"}, '
        '{"q": "2", "marks_awarded": 4, "marks_total": 5, "feedback": "ok"}'
        '], "total": 50}'
    )
    llm = _ScriptedLLM([reply])
    runner = ExamRunner(index=index, llm=llm)
    session = runner.start("u1", ["p1"], duration_s=600)

    graded = await runner.grade("u1", session.id, ["a1"])

    assert graded.total == 7.0


# --------------------------------------------------------------------------- #
# grade: malformed model replies never raise
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_prose_reply_grades_to_an_honest_empty_result() -> None:
    index = _index_with(_make_item("p1", "u1", "Q1."), _make_item("a1", "u1", "ans"))
    llm = _ScriptedLLM(["Sure, I graded it! Looks about 80% to me."])
    runner = ExamRunner(index=index, llm=llm)
    session = runner.start("u1", ["p1"], duration_s=600)

    graded = await runner.grade("u1", session.id, ["a1"])

    assert graded.status is ExamSessionStatus.GRADED
    assert graded.grading == []
    assert graded.total == 0.0


@pytest.mark.asyncio
async def test_reply_missing_questions_key_grades_to_an_honest_empty_result() -> None:
    index = _index_with(_make_item("p1", "u1", "Q1."), _make_item("a1", "u1", "ans"))
    llm = _ScriptedLLM(['{"total": 10}'])
    runner = ExamRunner(index=index, llm=llm)
    session = runner.start("u1", ["p1"], duration_s=600)

    graded = await runner.grade("u1", session.id, ["a1"])

    assert graded.status is ExamSessionStatus.GRADED
    assert graded.grading == []
    assert graded.total == 0.0


@pytest.mark.asyncio
async def test_malformed_question_entries_are_tolerated_individually() -> None:
    """A non-dict entry is dropped; a dict entry with missing/non-numeric
    marks keeps its place with those fields defaulted, instead of the whole
    reply being thrown away."""
    index = _index_with(_make_item("p1", "u1", "Q1. Q2. Q3."), _make_item("a1", "u1", "ans"))
    reply = (
        '{"questions": ['
        '"not even an object", '
        '{"q": "2", "marks_awarded": "a lot", "feedback": "no numeric marks given"}, '
        '{"q": "3", "marks_awarded": 2, "marks_total": 2, "feedback": "full marks"}'
        '], "total": 999}'
    )
    llm = _ScriptedLLM([reply])
    runner = ExamRunner(index=index, llm=llm)
    session = runner.start("u1", ["p1"], duration_s=600)

    graded = await runner.grade("u1", session.id, ["a1"])

    assert len(graded.grading) == 2
    assert graded.grading[0].q == "2"
    assert graded.grading[0].marks_awarded == 0.0
    assert graded.grading[0].marks_total == 0.0
    assert graded.grading[1].q == "3"
    assert graded.grading[1].marks_awarded == 2.0
    assert graded.total == 2.0


# --------------------------------------------------------------------------- #
# grade: provider failure must not corrupt the session
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_provider_failure_propagates_and_leaves_the_session_open() -> None:
    index = _index_with(_make_item("p1", "u1", "Q1."), _make_item("a1", "u1", "ans"))
    # FakeLLM raises ProviderError once its scripted responses are exhausted.
    runner = ExamRunner(index=index, llm=FakeLLM([]))
    session = runner.start("u1", ["p1"], duration_s=600)

    with pytest.raises(ProviderError):
        await runner.grade("u1", session.id, ["a1"])

    unchanged = runner.get(session.id)
    assert unchanged is not None
    assert unchanged.status is ExamSessionStatus.OPEN
    assert unchanged.grading == []
    assert unchanged.total == 0.0
    assert unchanged.answer_item_ids == []


# --------------------------------------------------------------------------- #
# grade: locked items never reach the model
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_fully_locked_paper_never_reaches_the_model() -> None:
    """If every paper item is locked, there is no gradable paper text at all
    — grading must refuse before ever calling the provider, not send an
    empty prompt and improvise a grade over it."""
    index = _index_with(
        _make_item("p1", "u1", "SECRET paper text", privacy=Privacy.LOCKED),
        _make_item("a1", "u1", "ans"),
    )
    llm = _ScriptedLLM([_ONE_QUESTION_REPLY])
    runner = ExamRunner(index=index, llm=llm)
    session = runner.start("u1", ["p1"], duration_s=600)

    with pytest.raises(ValueError):
        await runner.grade("u1", session.id, ["a1"])

    assert llm.prompts == []
    unchanged = runner.get(session.id)
    assert unchanged is not None
    assert unchanged.status is ExamSessionStatus.OPEN


@pytest.mark.asyncio
async def test_fully_locked_answer_script_never_reaches_the_model() -> None:
    index = _index_with(
        _make_item("p1", "u1", "Q1."),
        _make_item("a1", "u1", "SECRET answer text", privacy=Privacy.LOCKED),
    )
    llm = _ScriptedLLM([_ONE_QUESTION_REPLY])
    runner = ExamRunner(index=index, llm=llm)
    session = runner.start("u1", ["p1"], duration_s=600)

    with pytest.raises(ValueError):
        await runner.grade("u1", session.id, ["a1"])

    assert llm.prompts == []


@pytest.mark.asyncio
async def test_a_partially_locked_paper_excludes_only_the_locked_pages_text() -> None:
    """One page of a multi-page paper is locked, one is not. Grading must
    still proceed on the unlocked page's text, and the locked page's text
    must never appear in what was actually sent to the model."""
    index = _index_with(
        _make_item("p1", "u1", "PUBLIC page one text", privacy=Privacy.PRIVATE),
        _make_item("p2", "u1", "SECRET locked page text", privacy=Privacy.LOCKED),
        _make_item("a1", "u1", "answer text"),
    )
    llm = _ScriptedLLM([_ONE_QUESTION_REPLY])
    runner = ExamRunner(index=index, llm=llm)
    session = runner.start("u1", ["p1", "p2"], duration_s=600)

    graded = await runner.grade("u1", session.id, ["a1"])

    assert graded.status is ExamSessionStatus.GRADED
    assert len(llm.prompts) == 1
    assert "PUBLIC page one text" in llm.prompts[0]
    assert "SECRET locked page text" not in llm.prompts[0]


@pytest.mark.asyncio
async def test_a_partially_locked_answer_script_excludes_only_the_locked_pages_text() -> None:
    index = _index_with(
        _make_item("p1", "u1", "Q1."),
        _make_item("a1", "u1", "PUBLIC answer text"),
        _make_item("a2", "u1", "SECRET locked answer text", privacy=Privacy.LOCKED),
    )
    llm = _ScriptedLLM([_ONE_QUESTION_REPLY])
    runner = ExamRunner(index=index, llm=llm)
    session = runner.start("u1", ["p1"], duration_s=600)

    graded = await runner.grade("u1", session.id, ["a1", "a2"])

    assert graded.status is ExamSessionStatus.GRADED
    assert "PUBLIC answer text" in llm.prompts[0]
    assert "SECRET locked answer text" not in llm.prompts[0]


# --------------------------------------------------------------------------- #
# grade: missing items and an empty answer list
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_grading_with_only_nonexistent_paper_items_raises_without_calling_model() -> None:
    index = _index_with(_make_item("a1", "u1", "ans"))
    llm = _ScriptedLLM([_ONE_QUESTION_REPLY])
    runner = ExamRunner(index=index, llm=llm)
    session = runner.start("u1", ["does-not-exist"], duration_s=600)

    with pytest.raises(ValueError):
        await runner.grade("u1", session.id, ["a1"])

    assert llm.prompts == []


@pytest.mark.asyncio
async def test_grading_with_an_empty_answer_list_raises_without_calling_model() -> None:
    index = _index_with(_make_item("p1", "u1", "Q1."))
    llm = _ScriptedLLM([_ONE_QUESTION_REPLY])
    runner = ExamRunner(index=index, llm=llm)
    session = runner.start("u1", ["p1"], duration_s=600)

    with pytest.raises(ValueError):
        await runner.grade("u1", session.id, [])

    assert llm.prompts == []
