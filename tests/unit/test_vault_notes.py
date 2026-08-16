"""Unit tests for note generation and its fan-out.

``_ScriptedLLM`` implements the *real* :class:`~friday.providers.llm.LLMProvider`
contract — ``complete(messages: list[Message], tools=None, *, model=None) ->
LLMResponse`` — and records every prompt it was sent, so privacy and
prompt-content assertions can be made directly rather than trusted. It can also
be scripted to raise, to exercise the "the note must still save" path when the
model itself is unreachable.

``_RecordingIngestor`` and ``_RecordingStudyStore`` are the "simple recorders"
the module docstring promises the narrow ``Protocol``s enable: neither imports
anything from :mod:`friday.rag.ingest` or :mod:`friday.study.store`.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from friday.providers.llm import LLMProvider, LLMResponse, Message, ToolSpec
from friday.vault.index import SQLiteVaultIndex
from friday.vault.models import CaptureSource, Item, Privacy
from friday.vault.notes import NoteWriter

_OWNER = "u1"

_NOTE_REPLY = (
    '{"title": "EMF and internal resistance", "subject": "physics", '
    '"chapter": "current electricity", '
    '"markdown": "**V = 8 V**\\n\\nWorking: I = emf/(R+r).\\n\\n'
    'Trap: forgetting the internal resistance drop.", '
    '"cards": [{"front": "What drops V below emf?", '
    '"back": "Internal resistance times current"}]}'
)


def _make_item(
    item_id: str,
    *,
    owner_uid: str = _OWNER,
    privacy: Privacy = Privacy.PRIVATE,
    ocr_text: str = "",
    caption: str = "",
) -> Item:
    return Item(
        id=item_id,
        owner_uid=owner_uid,
        privacy=privacy,
        source=CaptureSource.CAMERA,
        ocr_text=ocr_text,
        caption=caption,
        created_at="2024-01-01T00:00:00+00:00",
    )


class _ScriptedLLM(LLMProvider):
    """Pops a queued reply (or raises a queued exception) per call; records prompts."""

    def __init__(self, replies: list[str | Exception]) -> None:
        self._replies: list[str | Exception] = list(replies)
        self.prompts: list[str] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> LLMResponse:
        self.prompts.append(messages[-1].content or "" if messages else "")
        reply = self._replies.pop(0) if self._replies else ""
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(text=reply)


class _RecordingIngestor:
    """Records ``(source_id, text)`` pairs; can be told to raise instead."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self._fail = fail

    async def ingest(self, source_id: str, text: str) -> None:
        self.calls.append((source_id, text))
        if self._fail:
            raise RuntimeError("ingestor exploded")


class _RecordingStudyStore:
    """Records ``(deck, front, back)`` calls and hands back incrementing ids."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self._next_id = 1
        self._fail = fail

    def add_card(self, deck: str, front: str, back: str) -> SimpleNamespace:
        self.calls.append((deck, front, back))
        if self._fail:
            raise RuntimeError("study store exploded")
        card = SimpleNamespace(id=self._next_id)
        self._next_id += 1
        return card


def _writer(
    index: SQLiteVaultIndex,
    llm: LLMProvider,
    *,
    ingestor: _RecordingIngestor | None = None,
    study_store: _RecordingStudyStore | None = None,
) -> NoteWriter:
    return NoteWriter(index=index, llm=llm, ingestor=ingestor, study_store=study_store)


# --------------------------------------------------------------------------- #
# Privacy: locked items never reach the model, never appear as sources
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_locked_item_excluded_from_sources_and_never_in_the_prompt() -> None:
    index = SQLiteVaultIndex()
    index.put_item(_make_item("open1", ocr_text="a cell of emf 4 V and 1 ohm internal"))
    index.put_item(_make_item("locked1", privacy=Privacy.LOCKED, ocr_text="TOP SECRET EXAM ANSWER"))
    llm = _ScriptedLLM([_NOTE_REPLY])

    note = await _writer(index, llm).write(_OWNER, ["open1", "locked1"])

    assert note.item_ids == ["open1"]
    assert len(llm.prompts) == 1
    assert "TOP SECRET EXAM ANSWER" not in llm.prompts[0]
    assert "a cell of emf 4 V" in llm.prompts[0]


# --------------------------------------------------------------------------- #
# The ingest argument order that the plan originally had backwards
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_ingest_receives_source_id_first_and_markdown_second() -> None:
    index = SQLiteVaultIndex()
    index.put_item(_make_item("i1", ocr_text="a cell of emf 4 V"))
    llm = _ScriptedLLM([_NOTE_REPLY])
    ingestor = _RecordingIngestor()

    note = await _writer(index, llm, ingestor=ingestor).write(_OWNER, ["i1"])

    assert len(ingestor.calls) == 1
    source_id, text = ingestor.calls[0]
    assert source_id == f"note:{note.id}"
    assert text == note.markdown
    # Pin the order explicitly: the id must never end up where the markdown goes.
    assert source_id != note.markdown
    assert note.rag_source_id == source_id


# --------------------------------------------------------------------------- #
# Fan-out failures must not lose the note
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_ingestor_raising_still_saves_and_returns_the_note() -> None:
    index = SQLiteVaultIndex()
    index.put_item(_make_item("i1", ocr_text="a cell of emf 4 V"))
    llm = _ScriptedLLM([_NOTE_REPLY])
    ingestor = _RecordingIngestor(fail=True)

    note = await _writer(index, llm, ingestor=ingestor).write(_OWNER, ["i1"])

    assert note is not None
    assert note.rag_source_id == ""
    assert index.get_note(_OWNER, note.id) is not None


@pytest.mark.asyncio
async def test_add_card_raising_still_saves_and_returns_the_note() -> None:
    index = SQLiteVaultIndex()
    index.put_item(_make_item("i1", ocr_text="a cell of emf 4 V"))
    llm = _ScriptedLLM([_NOTE_REPLY])
    study_store = _RecordingStudyStore(fail=True)

    note = await _writer(index, llm, study_store=study_store).write(_OWNER, ["i1"])

    assert note is not None
    assert note.flashcard_ids == []
    assert index.get_note(_OWNER, note.id) is not None
    assert len(study_store.calls) == 1  # it was tried


@pytest.mark.asyncio
async def test_llm_raising_still_produces_and_saves_a_usable_note() -> None:
    """Not just fan-out: the model call itself failing must not lose the note."""
    index = SQLiteVaultIndex()
    index.put_item(_make_item("i1", ocr_text="a cell of emf 4 V"))
    llm = _ScriptedLLM([RuntimeError("provider exploded")])

    note = await _writer(index, llm).write(_OWNER, ["i1"])

    assert note is not None
    assert "a cell of emf 4 V" in note.markdown
    assert index.get_note(_OWNER, note.id) is not None


# --------------------------------------------------------------------------- #
# Both fan-out targets absent
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_no_ingestor_and_no_study_store_still_works() -> None:
    index = SQLiteVaultIndex()
    index.put_item(_make_item("i1", ocr_text="a cell of emf 4 V"))
    llm = _ScriptedLLM([_NOTE_REPLY])

    note = await _writer(index, llm).write(_OWNER, ["i1"])

    assert note.rag_source_id == ""
    assert note.flashcard_ids == []
    assert index.get_note(_OWNER, note.id) is not None


# --------------------------------------------------------------------------- #
# Graceful degradation on a non-JSON reply
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_prose_reply_still_produces_a_usable_note() -> None:
    index = SQLiteVaultIndex()
    index.put_item(_make_item("i1", ocr_text="a cell of emf 4 V"))
    llm = _ScriptedLLM(["The terminal voltage is 8 V because of the internal drop."])

    note = await _writer(index, llm).write(_OWNER, ["i1"])

    assert note.markdown == "The terminal voltage is 8 V because of the internal drop."
    assert note.title == "The terminal voltage is 8 V because of the internal drop."[:80]
    assert note.flashcard_ids == []


# --------------------------------------------------------------------------- #
# Malformed / partial `cards`
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_missing_cards_key_yields_no_flashcards() -> None:
    index = SQLiteVaultIndex()
    index.put_item(_make_item("i1", ocr_text="x"))
    llm = _ScriptedLLM(['{"title": "t", "subject": "s", "markdown": "m"}'])
    study_store = _RecordingStudyStore()

    note = await _writer(index, llm, study_store=study_store).write(_OWNER, ["i1"])

    assert note.flashcard_ids == []
    assert study_store.calls == []


@pytest.mark.asyncio
async def test_empty_cards_list_yields_no_flashcards() -> None:
    index = SQLiteVaultIndex()
    index.put_item(_make_item("i1", ocr_text="x"))
    llm = _ScriptedLLM(['{"title": "t", "subject": "s", "markdown": "m", "cards": []}'])
    study_store = _RecordingStudyStore()

    note = await _writer(index, llm, study_store=study_store).write(_OWNER, ["i1"])

    assert note.flashcard_ids == []
    assert study_store.calls == []


@pytest.mark.asyncio
async def test_malformed_card_entries_are_skipped_not_fatal() -> None:
    index = SQLiteVaultIndex()
    index.put_item(_make_item("i1", ocr_text="x"))
    reply = (
        '{"title": "t", "subject": "physics", "markdown": "m", "cards": ['
        '"not a dict", '
        '{"front": "only a front"}, '
        '{"back": "only a back"}, '
        '{"front": "", "back": ""}, '
        '{"front": "good front", "back": "good back"}'
        "]}"
    )
    llm = _ScriptedLLM([reply])
    study_store = _RecordingStudyStore()

    note = await _writer(index, llm, study_store=study_store).write(_OWNER, ["i1"])

    assert study_store.calls == [("physics", "good front", "good back")]
    assert note.flashcard_ids == [1]


@pytest.mark.asyncio
async def test_cards_are_capped_and_deck_falls_back_when_subject_is_blank() -> None:
    cards = ", ".join(f'{{"front": "f{i}", "back": "b{i}"}}' for i in range(12))
    reply = f'{{"title": "t", "subject": "", "markdown": "m", "cards": [{cards}]}}'
    index = SQLiteVaultIndex()
    index.put_item(_make_item("i1", ocr_text="x"))
    llm = _ScriptedLLM([reply])
    study_store = _RecordingStudyStore()

    note = await _writer(index, llm, study_store=study_store).write(_OWNER, ["i1"])

    assert len(study_store.calls) == 8
    assert all(call[0] == "general" for call in study_store.calls)
    assert note.flashcard_ids == list(range(1, 9))


# --------------------------------------------------------------------------- #
# Missing items are skipped, not an error
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_nonexistent_item_ids_are_skipped_without_error() -> None:
    index = SQLiteVaultIndex()
    index.put_item(_make_item("real1", ocr_text="a cell of emf 4 V"))
    llm = _ScriptedLLM([_NOTE_REPLY])

    note = await _writer(index, llm).write(_OWNER, ["real1", "ghost1", "ghost2"])

    assert note.item_ids == ["real1"]
    assert len(llm.prompts) == 1


# --------------------------------------------------------------------------- #
# note_ids link back onto source items, without duplication
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_every_source_item_gets_the_note_id_appended() -> None:
    index = SQLiteVaultIndex()
    index.put_item(_make_item("i1", ocr_text="a"))
    index.put_item(_make_item("i2", ocr_text="b"))
    llm = _ScriptedLLM([_NOTE_REPLY])

    note = await _writer(index, llm).write(_OWNER, ["i1", "i2"])

    for item_id in ("i1", "i2"):
        stored = index.get_item(_OWNER, item_id)
        assert stored is not None
        assert stored.note_ids == [note.id]


@pytest.mark.asyncio
async def test_relinking_the_same_note_id_does_not_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive: if the same note id were ever linked twice, it must not repeat."""
    import friday.vault.notes as notes_mod

    fixed_uuid = uuid.UUID(int=42)
    monkeypatch.setattr(notes_mod.uuid, "uuid4", lambda: fixed_uuid)

    index = SQLiteVaultIndex()
    index.put_item(_make_item("i1", ocr_text="a"))
    llm = _ScriptedLLM([_NOTE_REPLY, _NOTE_REPLY])
    writer = _writer(index, llm)

    first = await writer.write(_OWNER, ["i1"])
    second = await writer.write(_OWNER, ["i1"])

    assert first.id == second.id  # both wrote under the same (patched) id
    stored = index.get_item(_OWNER, "i1")
    assert stored is not None
    assert stored.note_ids == [first.id]


# --------------------------------------------------------------------------- #
# Empty item_ids: no model call, but a real (minimal) note is still saved
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_empty_item_ids_skips_the_model_but_still_saves_a_note() -> None:
    index = SQLiteVaultIndex()
    llm = _ScriptedLLM([_NOTE_REPLY])

    note = await _writer(index, llm).write(_OWNER, [])

    assert llm.prompts == []
    assert note.item_ids == []
    assert note.markdown == ""
    assert note.flashcard_ids == []
    assert index.get_note(_OWNER, note.id) is not None


@pytest.mark.asyncio
async def test_all_locked_item_ids_also_skips_the_model() -> None:
    index = SQLiteVaultIndex()
    index.put_item(_make_item("locked1", privacy=Privacy.LOCKED, ocr_text="secret"))
    llm = _ScriptedLLM([_NOTE_REPLY])

    note = await _writer(index, llm).write(_OWNER, ["locked1"])

    assert llm.prompts == []
    assert note.item_ids == []


# --------------------------------------------------------------------------- #
# The optional extra prompt reaches the model
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_extra_prompt_reaches_the_model() -> None:
    index = SQLiteVaultIndex()
    index.put_item(_make_item("i1", ocr_text="a cell of emf 4 V"))
    llm = _ScriptedLLM([_NOTE_REPLY])

    await _writer(index, llm).write(_OWNER, ["i1"], prompt="Focus on the internal resistance trap.")

    assert len(llm.prompts) == 1
    assert "Focus on the internal resistance trap." in llm.prompts[0]


@pytest.mark.asyncio
async def test_no_extra_prompt_omits_the_instruction_line() -> None:
    index = SQLiteVaultIndex()
    index.put_item(_make_item("i1", ocr_text="a cell of emf 4 V"))
    llm = _ScriptedLLM([_NOTE_REPLY])

    await _writer(index, llm).write(_OWNER, ["i1"])

    assert "Additional instruction" not in llm.prompts[0]


# --------------------------------------------------------------------------- #
# Falls back to caption when OCR text is empty
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_falls_back_to_caption_when_ocr_text_is_empty() -> None:
    index = SQLiteVaultIndex()
    index.put_item(_make_item("i1", ocr_text="", caption="a hand-drawn circuit diagram"))
    llm = _ScriptedLLM([_NOTE_REPLY])

    await _writer(index, llm).write(_OWNER, ["i1"])

    assert "a hand-drawn circuit diagram" in llm.prompts[0]
