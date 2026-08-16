"""Exam mode: photograph a paper, sit it under a timer, get marked honestly.

The workflow is: photograph a question paper (:meth:`ExamRunner.start` opens a
timed session over it), sit the timer, photograph the completed answer script,
then :meth:`ExamRunner.grade` marks the script against the paper.

The useful output of grading is *not* a score. A single percentage tells you
nothing you can act on; "you lost two marks on Q3 for dropping the unit" does.
So grading always returns marks broken down per question, each with specific
feedback about where it was lost — that breakdown is the feature, and the
``total`` is a derived summary of it, never an independent number (see the
note on :func:`_parse_grading` below for why it is *recomputed* rather than
trusted from the model's reply).

Sessions live in a plain in-process dict, not the vault index. That is a
deliberate scope decision for a personal, single-user surface: losing an
in-flight exam timer on a process restart costs the user re-photographing a
paper, which is cheap, while giving every timed session a database row would
not be. Nothing here is durable across a restart, and nothing should be until
that trade-off is revisited.

Depends only on the :class:`~friday.providers.llm.LLMProvider` contract and
the :class:`~friday.vault.index.VaultIndex` protocol — it imports no LLM SDK
and no concrete index backend.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from friday.providers.llm import LLMProvider, Message
from friday.vault.index import VaultIndex
from friday.vault.models import ExamSession, ExamSessionStatus, GradedQuestion

logger = logging.getLogger("friday.vault.exam")

_GRADE_PROMPT = """You are FRIDAY, marking a photographed exam against its
photographed answer script. Behave like a real examiner: award method marks
for working that is right even when the final figure is off, and say exactly
where each mark was lost — a dropped unit, an arithmetic slip, a missing step
— never just "wrong".

Reply with ONLY this JSON object, no prose, no code fence:
{"questions": [{"q": "...", "marks_awarded": 0, "marks_total": 0, "feedback": "..."}],
 "total": 0}

q: the question number/label as written on the paper.
marks_awarded / marks_total: numbers.
feedback: specific to that question — which step or unit cost the mark.

Question paper (extracted text):
---
%s
---

Candidate's answer script (extracted text):
---
%s
---
"""


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_str(value: object) -> str:
    """Coerce ``value`` to ``str``, treating ``None`` (and absence) as ``""``."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _safe_float(value: object) -> float:
    """Coerce ``value`` to ``float``, defaulting to ``0.0`` on anything unusable."""
    if value is None:
        return 0.0
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _parse_grading(reply: str) -> list[GradedQuestion]:
    """Decode the model's reply into per-question marks.

    Never raises: a reply that is prose, is JSON but missing ``questions``, or
    has malformed question entries all degrade to as much as can honestly be
    salvaged (down to an empty list) rather than sinking the grading pass. An
    individual entry that isn't a JSON object is dropped; a dict entry with
    missing or non-numeric ``marks_awarded``/``marks_total`` keeps its place
    with those fields defaulted to ``0.0``, so a single sloppy question does
    not erase its siblings.

    The reply's own top-level ``"total"`` key is read here only implicitly (it
    is ignored) — the caller derives the real total by summing
    ``marks_awarded`` across the parsed questions. See the caller for why.
    """
    text = reply.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```")
        text = text.removesuffix("```").strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return []
    text = text[start : end + 1]

    try:
        raw = json.loads(text)
    except (ValueError, TypeError):
        return []
    if not isinstance(raw, dict):
        return []

    questions_raw = raw.get("questions")
    if not isinstance(questions_raw, list):
        return []

    questions: list[GradedQuestion] = []
    for entry in questions_raw:
        if not isinstance(entry, dict):
            continue
        questions.append(
            GradedQuestion(
                q=_safe_str(entry.get("q")),
                marks_awarded=_safe_float(entry.get("marks_awarded")),
                marks_total=_safe_float(entry.get("marks_total")),
                feedback=_safe_str(entry.get("feedback")),
            )
        )
    return questions


class ExamRunner:
    """Opens timed exam sessions and marks them.

    Args:
        index: Where photographed papers and answer scripts live as
            :class:`~friday.vault.models.Item` rows. Only ``get_item`` is used.
        llm: The provider used for the single grading pass. Only the abstract
            ``complete`` contract is depended upon.
        clock: Returns the ISO-8601 UTC timestamp stamped on a session at
            :meth:`start`. Overridable for tests; defaults to the real wall
            clock.

    Sessions are held in a plain ``dict`` in this instance — see the module
    docstring for why that is deliberate rather than an oversight.
    """

    def __init__(
        self,
        *,
        index: VaultIndex,
        llm: LLMProvider,
        clock: Callable[[], str] = _utc_now_iso,
    ) -> None:
        self._index = index
        self._llm = llm
        self._clock = clock
        self._sessions: dict[str, ExamSession] = {}

    def start(self, owner_uid: str, paper_item_ids: list[str], *, duration_s: int) -> ExamSession:
        """Open a timed session over an already-photographed paper.

        ``duration_s`` is recorded on the session for the client to show a
        countdown against, but nothing in this module enforces it — grading a
        session after its nominal duration has elapsed still succeeds. Timer
        enforcement, if wanted, belongs at the call site (or a later task);
        flagging it here rather than silently pretending it is covered.
        """
        session = ExamSession(
            id=uuid.uuid4().hex[:16],
            owner_uid=owner_uid,
            paper_item_ids=list(paper_item_ids),
            started_at=self._clock(),
            duration_s=duration_s,
        )
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> ExamSession | None:
        return self._sessions.get(session_id)

    def _gather_text(self, owner_uid: str, item_ids: list[str], *, label: str) -> str:
        """Join the extracted text of every leaveable item in ``item_ids``.

        An item that does not exist is skipped (logged, not raised) — a stale
        or mistyped id must not sink the whole grading pass. An item that
        exists but is :attr:`~friday.vault.models.Privacy.LOCKED` is *also*
        skipped, and skipped the same way: :meth:`Item.may_leave_for_model`
        is the one guarantee the vault makes about locked material, and
        grading sends text to an LLM provider, so a locked paper page or
        answer page is simply never read into that text — not redacted, not
        summarized, just absent, exactly as if it had never been captured.
        """
        parts: list[str] = []
        for item_id in item_ids:
            item = self._index.get_item(owner_uid, item_id)
            if item is None:
                logger.warning("exam %s item missing, skipping: %s", label, item_id)
                continue
            if not item.may_leave_for_model():
                logger.warning("exam %s item is locked, excluding from grading: %s", label, item_id)
                continue
            if item.ocr_text.strip():
                parts.append(item.ocr_text)
        return "\n\n".join(parts)

    async def grade(
        self, owner_uid: str, session_id: str, answer_item_ids: list[str]
    ) -> ExamSession:
        """Mark the answer script against the paper and return the session.

        Raises :class:`KeyError` if ``session_id`` is unknown, or if it belongs
        to a different owner — the two cases share one message so a caller
        cannot use the error to probe for the existence of someone else's
        session.

        Raises :class:`ValueError` if there is nothing gradable: every paper
        or every answer item is missing, locked, or has no extracted text (an
        empty ``answer_item_ids`` list falls into this too). This is checked
        *before* any model call, so a fully-locked exam never reaches the
        provider at all.

        If the provider call itself fails (outage, timeout — anything
        ``complete`` raises), that exception propagates unchanged and the
        session is left exactly as it was: still ``OPEN``, ``grading`` still
        empty, ``answer_item_ids`` not yet overwritten. Nothing on the session
        is mutated until a reply has actually been received, so a provider
        failure can never leave a session claiming ``GRADED`` with no marks —
        it fails loudly instead of corrupting the session into a silent
        half-graded state.

        A reply that *is* received but doesn't parse (prose, missing
        ``questions``, malformed entries) is a different situation: the model
        answered, it was just unusable. That does not raise (see
        :func:`_parse_grading`) — the session is still marked ``GRADED``,
        honestly reporting an empty ``grading`` list and a ``total`` of 0.0,
        because the grading pass genuinely ran and genuinely produced nothing
        usable; re-running it is a fresh grade, not owed automatically here.

        ``total`` is always the sum of the parsed questions'
        ``marks_awarded`` — never the model's own top-level ``"total"``
        field, which is discarded. A marking system whose headline number
        disagrees with the per-question breakdown printed right below it is
        worse than no total at all, and the per-question marks are the part a
        real examiner (and this feature) actually trusts; recomputing is the
        only way the two can never contradict each other.
        """
        session = self._sessions.get(session_id)
        if session is None or session.owner_uid != owner_uid:
            raise KeyError(f"unknown exam session for this owner: {session_id}")

        paper_text = self._gather_text(owner_uid, session.paper_item_ids, label="paper")
        answer_text = self._gather_text(owner_uid, answer_item_ids, label="answer")
        if not paper_text.strip() or not answer_text.strip():
            raise ValueError(
                "nothing gradable: paper and answer items must resolve to at "
                "least some non-locked extracted text"
            )

        prompt = _GRADE_PROMPT % (paper_text[:6000], answer_text[:6000])
        response = await self._llm.complete([Message(role="user", content=prompt)])

        questions = _parse_grading(response.text or "")
        total = sum(q.marks_awarded for q in questions)

        session.answer_item_ids = list(answer_item_ids)
        session.grading = questions
        session.total = total
        session.status = ExamSessionStatus.GRADED
        return session
