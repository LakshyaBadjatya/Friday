"""The meeting-capture pipeline: audio -> transcript -> summary + action items.

:class:`MeetingCapture` is the single orchestration seam for turning a recorded
meeting into structured notes. It depends only on the typed provider boundaries
(:class:`~friday.providers.stt.STTProvider`,
:class:`~friday.providers.llm.LLMProvider`) and the optional RAG
:class:`~friday.rag.ingest.DocumentIngestor`, so it imports no SDK and runs fully
offline against :class:`~friday.providers.stt.FakeSTT` + a scripted
:class:`~friday.providers.llm.FakeLLM` in tests.

Two binding rules:

* **LLM summarization is NON-FATAL.** Exactly one LLM pass asks for a
  ``{summary, action_items[]}`` JSON object derived from the transcript. *Any*
  failure — a provider error/timeout, empty text, or a payload that does not
  parse into the expected shape — degrades to **transcript-only** notes: a short
  first-lines fallback summary and an empty action-item list. ``process`` never
  raises because the LLM (or its output) misbehaved.
* **Ingestion is best-effort.** When a :class:`DocumentIngestor` is supplied the
  transcript is also ingested under ``source_id=f"meeting:{title}"`` so the
  meeting becomes answerable via the existing Knowledge path. That ingest is
  wrapped so a failure there never fails the capture either.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from friday.providers.llm import LLMProvider, Message
from friday.providers.stt import STTProvider
from friday.rag.ingest import DocumentIngestor

logger = logging.getLogger("friday.meetings.capture")

# How many leading non-empty transcript lines the fallback summary keeps when the
# LLM pass fails — enough to be useful, short enough not to echo the whole note.
_FALLBACK_SUMMARY_LINES = 3

# Instruction handed to the LLM. It asks for a single strict-JSON object so the
# parse is deterministic; any deviation simply trips the non-fatal fallback.
_SUMMARY_INSTRUCTION = (
    "You are summarizing a meeting transcript. Reply with a single JSON object "
    'and nothing else, of the exact shape {"summary": str, "action_items": '
    "[str, ...]}. The summary is two or three concise sentences capturing the "
    "decisions and outcomes. action_items is a list of short, imperative "
    "follow-up tasks mentioned in the meeting (an empty list if there are none). "
    "Do not invent details that are not in the transcript.\n\nTranscript:\n"
)


class MeetingNotes(BaseModel):
    """Structured notes produced from one captured meeting.

    ``id`` is ``None`` until the notes are persisted (the store assigns it on
    :meth:`~friday.meetings.store.SQLiteMeetingStore.add`). ``transcript`` is the
    STT output verbatim; ``summary`` and ``action_items`` come from the LLM pass
    (or the transcript-only fallback when that pass fails). ``created_at`` is an
    ISO-8601 UTC timestamp.
    """

    id: int | None = None
    title: str
    transcript: str
    summary: str
    action_items: list[str] = Field(default_factory=list)
    created_at: str


class MeetingCapture:
    """Turn meeting audio into :class:`MeetingNotes` (transcribe -> summarize).

    Args:
        stt: The speech-to-text provider; its ``transcribe(audio, lang)`` yields
            the transcript (real Whisper in production, ``FakeSTT`` in tests).
        llm: The live LLM provider used for the single, non-fatal summary +
            action-item extraction pass.
        ingestor: Optional RAG ingestor. When supplied, the transcript is also
            ingested under ``meeting:{title}`` (best-effort, non-fatal) so the
            meeting is answerable via the existing Knowledge path with citations.
    """

    def __init__(
        self,
        stt: STTProvider,
        llm: LLMProvider,
        *,
        ingestor: DocumentIngestor | None = None,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._ingestor = ingestor

    async def process(
        self, title: str, audio: bytes, *, lang: str | None = None
    ) -> MeetingNotes:
        """Capture a meeting end-to-end and return its :class:`MeetingNotes`.

        Pipeline: ``stt.transcribe(audio, lang)`` -> transcript; one non-fatal LLM
        pass for ``{summary, action_items[]}`` (any error -> transcript-only
        fallback summary + empty action items); then, if an ingestor is wired, a
        best-effort ingest of the transcript under ``meeting:{title}``. The
        returned notes have ``id is None`` (the store assigns the id on save).
        """
        transcript = (await self._stt.transcribe(audio, lang)).text
        summary, action_items = await self._summarize(transcript)
        await self._ingest(title, transcript)
        return MeetingNotes(
            id=None,
            title=title,
            transcript=transcript,
            summary=summary,
            action_items=action_items,
            created_at=datetime.now(UTC).isoformat(),
        )

    # -- summary + action items (non-fatal) -------------------------------- #
    async def _summarize(self, transcript: str) -> tuple[str, list[str]]:
        """One LLM pass for ``(summary, action_items)``; never raise.

        The LLM completion and its JSON parse are wrapped in a broad ``except``:
        any provider error/timeout, empty text, or a payload that does not parse
        into the expected ``{summary, action_items[]}`` shape degrades to the
        transcript-only fallback (a short first-lines summary, no action items).
        """
        prompt = _SUMMARY_INSTRUCTION + transcript
        try:
            response = await self._llm.complete([Message(role="user", content=prompt)])
            text = (response.text or "").strip()
            if not text:
                raise ValueError("empty LLM summary")
            parsed = _SummaryPayload.model_validate_json(_extract_json(text))
        except Exception:  # noqa: BLE001 - summarization is optional + non-fatal
            logger.warning(
                "meeting LLM summarization failed; using transcript-only notes"
            )
            return _fallback_summary(transcript), []
        return parsed.summary.strip(), [item for item in parsed.action_items if item]

    # -- best-effort ingestion --------------------------------------------- #
    async def _ingest(self, title: str, transcript: str) -> None:
        """Ingest the transcript under ``meeting:{title}`` when wired; never raise.

        No-op when no ingestor was supplied. The ingest is wrapped so a failure
        (store error, etc.) is logged and swallowed — capture must never fail
        because the meeting could not be indexed for retrieval.
        """
        if self._ingestor is None:
            return
        try:
            await self._ingestor.ingest(f"meeting:{title}", transcript)
        except Exception:  # noqa: BLE001 - ingestion is best-effort + non-fatal
            logger.warning(
                "meeting transcript ingestion failed; notes still captured",
                extra={"title": title},
            )


class _SummaryPayload(BaseModel):
    """The strict shape the LLM summary pass is parsed into.

    ``extra`` keys are ignored so a chatty model that adds fields still parses;
    the two fields we need are required and typed, so a missing/mistyped one
    raises :class:`ValidationError` and trips the non-fatal fallback.
    """

    model_config = {"extra": "ignore"}

    summary: str
    action_items: list[str] = Field(default_factory=list)


def _extract_json(text: str) -> str:
    """Return the first ``{...}`` JSON object substring of ``text``.

    Tolerates a model that wraps the JSON in prose or a ``` ```json ``` ``` fence
    by slicing from the first ``{`` to the last ``}``. When no braces are present
    the original text is returned so the downstream parse fails loudly into the
    non-fatal fallback rather than silently succeeding on garbage.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start : end + 1]


def _fallback_summary(transcript: str) -> str:
    """A transcript-only summary: the first few non-empty lines, joined.

    Used when the LLM pass fails so the notes still carry a human-readable gist
    without inventing content. An empty transcript yields an empty summary.
    """
    lines = [line.strip() for line in transcript.splitlines() if line.strip()]
    if not lines:
        return transcript.strip()
    return " ".join(lines[:_FALLBACK_SUMMARY_LINES])
