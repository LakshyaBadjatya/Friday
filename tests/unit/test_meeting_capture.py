"""Unit tests for meeting capture (the pipeline) and the SQLite notes store.

Fully offline against :class:`~friday.providers.stt.FakeSTT` + a scripted
:class:`~friday.providers.llm.FakeLLM`; no models, no network, no real audio.

Covered:
* ``MeetingCapture.process`` returns notes whose transcript is the FakeSTT output
  and whose summary/action_items come from the scripted LLM JSON (``id`` unset).
* The requested ``lang`` is propagated through to STT.
* An LLM error (and a malformed/non-JSON payload) is NON-FATAL: the notes are
  transcript-only (a fallback summary, empty action items), never raising.
* A provided RAG ingestor receives the transcript under ``meeting:<title>`` (the
  shared vector store gains a chunk) — and an ingestor that raises is non-fatal.
* :class:`SQLiteMeetingStore` round-trips a note (add assigns id, get returns it),
  lists most-recent first, and deletes (idempotently).
"""

from __future__ import annotations

import json
from pathlib import Path

from friday.errors import ProviderError
from friday.meetings.capture import MeetingCapture, MeetingNotes
from friday.meetings.store import MeetingStore, SQLiteMeetingStore
from friday.memory.long_term import SQLiteLongTermStore
from friday.memory.vector import SQLiteVectorStore
from friday.providers.embeddings import FakeEmbeddings
from friday.providers.llm import FakeLLM, LLMProvider, LLMResponse, Message, ToolSpec
from friday.providers.stt import FakeSTT, Transcript
from friday.rag.ingest import DocumentIngestor


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _llm_with_json(summary: str, action_items: list[str]) -> FakeLLM:
    """A scripted LLM that returns one JSON ``{summary, action_items}`` object."""
    payload = json.dumps({"summary": summary, "action_items": action_items})
    return FakeLLM(responses=[LLMResponse(text=payload)])


class _RaisingLLM(LLMProvider):
    """An LLM whose ``complete`` always raises, to exercise the non-fatal path."""

    async def complete(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> LLMResponse:
        raise ProviderError("boom")


class _RaisingIngestor:
    """A stand-in ingestor whose ``ingest`` raises, to prove ingest is non-fatal."""

    async def ingest(self, source_id: str, text: str) -> None:
        raise RuntimeError("ingest exploded")


def _vector_ingestor() -> tuple[DocumentIngestor, SQLiteVectorStore]:
    """A real ingestor over a fresh in-memory vector + long-term store pair."""
    vector = SQLiteVectorStore(":memory:", embedder=FakeEmbeddings(dim=64), dim=64)
    long_term = SQLiteLongTermStore(":memory:")
    return DocumentIngestor(vector, long_term), vector


# --------------------------------------------------------------------------- #
# MeetingCapture.process — happy path
# --------------------------------------------------------------------------- #
async def test_process_returns_transcript_and_scripted_summary() -> None:
    capture = MeetingCapture(
        FakeSTT(), _llm_with_json("We shipped the slice.", ["File the report"])
    )
    notes = await capture.process("Standup", b"audio-bytes")

    assert isinstance(notes, MeetingNotes)
    assert notes.id is None
    assert notes.title == "Standup"
    # Transcript is the FakeSTT output verbatim.
    assert notes.transcript == "fake transcript"
    # Summary + action items come from the scripted LLM JSON.
    assert notes.summary == "We shipped the slice."
    assert notes.action_items == ["File the report"]
    assert notes.created_at  # an ISO-8601 timestamp is stamped


async def test_process_propagates_lang_to_stt() -> None:
    class _RecordingSTT:
        def __init__(self) -> None:
            self.seen_lang: str | None = "unset"

        async def transcribe(self, audio: bytes, lang: str | None) -> Transcript:
            self.seen_lang = lang
            return Transcript(text="hola", lang=lang)

    stt = _RecordingSTT()
    capture = MeetingCapture(stt, _llm_with_json("resumen", []))
    notes = await capture.process("Reunion", b"x", lang="es")

    assert stt.seen_lang == "es"
    assert notes.transcript == "hola"


# --------------------------------------------------------------------------- #
# MeetingCapture.process — non-fatal summarization
# --------------------------------------------------------------------------- #
async def test_llm_error_yields_transcript_only_notes() -> None:
    capture = MeetingCapture(FakeSTT(), _RaisingLLM())
    notes = await capture.process("Sync", b"audio")

    # Never raised; degraded to transcript-only notes.
    assert notes.transcript == "fake transcript"
    assert notes.action_items == []
    # The fallback summary is derived from the transcript (no fabrication).
    assert notes.summary
    assert "fake transcript" in notes.summary


async def test_malformed_llm_json_is_non_fatal() -> None:
    capture = MeetingCapture(
        FakeSTT(), FakeLLM(responses=[LLMResponse(text="not json at all")])
    )
    notes = await capture.process("Sync", b"audio")
    assert notes.action_items == []
    assert notes.summary  # fallback, not the garbage text parsed as a summary


async def test_empty_llm_text_is_non_fatal() -> None:
    capture = MeetingCapture(FakeSTT(), FakeLLM(responses=[LLMResponse(text="")]))
    notes = await capture.process("Sync", b"audio")
    assert notes.action_items == []
    assert notes.summary


async def test_llm_json_wrapped_in_prose_is_parsed() -> None:
    """A model that wraps the JSON in prose/fences still parses (robust extract)."""
    wrapped = 'Here you go:\n```json\n{"summary": "ok", "action_items": ["do x"]}\n```'
    capture = MeetingCapture(FakeSTT(), FakeLLM(responses=[LLMResponse(text=wrapped)]))
    notes = await capture.process("Sync", b"audio")
    assert notes.summary == "ok"
    assert notes.action_items == ["do x"]


# --------------------------------------------------------------------------- #
# MeetingCapture.process — best-effort ingestion
# --------------------------------------------------------------------------- #
async def test_provided_ingestor_receives_transcript() -> None:
    ingestor, vector = _vector_ingestor()
    capture = MeetingCapture(
        FakeSTT(), _llm_with_json("s", []), ingestor=ingestor
    )
    await capture.process("Weekly Review", b"audio")

    # The shared vector store gains a ``meeting:<title>`` chunk for the transcript.
    hits = vector.query("fake transcript", k=4)
    assert any(hit.source_id.startswith("meeting:Weekly Review#") for hit in hits)


async def test_failing_ingestor_is_non_fatal() -> None:
    capture = MeetingCapture(
        FakeSTT(), _llm_with_json("s", ["a"]), ingestor=_RaisingIngestor()  # type: ignore[arg-type]
    )
    # The ingest raises internally but capture still returns complete notes.
    notes = await capture.process("Sync", b"audio")
    assert notes.summary == "s"
    assert notes.action_items == ["a"]


# --------------------------------------------------------------------------- #
# SQLiteMeetingStore — round-trip / list / delete
# --------------------------------------------------------------------------- #
def _note(title: str = "Standup") -> MeetingNotes:
    return MeetingNotes(
        id=None,
        title=title,
        transcript="t",
        summary="s",
        action_items=["a1", "a2"],
        created_at="2026-06-15T10:00:00+00:00",
    )


def test_store_is_meeting_store_protocol() -> None:
    assert isinstance(SQLiteMeetingStore(":memory:"), MeetingStore)


def test_store_add_assigns_id_and_get_round_trips() -> None:
    store = SQLiteMeetingStore(":memory:")
    stored = store.add(_note())
    assert isinstance(stored.id, int) and stored.id > 0

    fetched = store.get(stored.id)
    assert fetched is not None
    assert fetched.id == stored.id
    assert fetched.title == "Standup"
    assert fetched.transcript == "t"
    assert fetched.summary == "s"
    # Action items round-trip through the JSON column with order preserved.
    assert fetched.action_items == ["a1", "a2"]
    assert fetched.created_at == "2026-06-15T10:00:00+00:00"


def test_store_get_missing_returns_none() -> None:
    store = SQLiteMeetingStore(":memory:")
    assert store.get(999) is None


def test_store_list_is_most_recent_first() -> None:
    store = SQLiteMeetingStore(":memory:")
    first = store.add(_note("First"))
    second = store.add(_note("Second"))

    listed = store.list_meetings()
    assert [m.id for m in listed] == [second.id, first.id]
    assert [m.title for m in listed] == ["Second", "First"]


def test_store_list_honors_limit() -> None:
    store = SQLiteMeetingStore(":memory:")
    for i in range(5):
        store.add(_note(f"M{i}"))
    assert len(store.list_meetings(limit=2)) == 2


def test_store_delete_removes_and_is_idempotent() -> None:
    store = SQLiteMeetingStore(":memory:")
    stored = store.add(_note())
    assert store.delete(stored.id) == 1  # type: ignore[arg-type]
    assert store.get(stored.id) is None  # type: ignore[arg-type]
    # Deleting again removes nothing (idempotent).
    assert store.delete(stored.id) == 0  # type: ignore[arg-type]


def test_store_round_trips_empty_action_items() -> None:
    store = SQLiteMeetingStore(":memory:")
    note = MeetingNotes(
        id=None,
        title="Empty",
        transcript="t",
        summary="s",
        action_items=[],
        created_at="2026-06-15T10:00:00+00:00",
    )
    stored = store.add(note)
    fetched = store.get(stored.id)  # type: ignore[arg-type]
    assert fetched is not None
    assert fetched.action_items == []


def test_store_persists_across_instances(tmp_path: Path) -> None:
    """A file-backed store round-trips across separate instances (durable)."""
    db = str(tmp_path / "meetings.db")
    SQLiteMeetingStore(db).add(_note("Durable"))
    reopened = SQLiteMeetingStore(db)
    listed = reopened.list_meetings()
    assert [m.title for m in listed] == ["Durable"]
