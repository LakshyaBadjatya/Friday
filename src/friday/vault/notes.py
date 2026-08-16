"""Note generation and its fan-out — the join that makes the vault worth building.

A capture, on its own, is a photo with some OCR text sitting in a database. A
:class:`~friday.vault.models.Note` is what turns a set of captures into
something a student can actually use again: a written explanation, on
request only (never automatically — a note is a deliberate act, not a side
effect of uploading), that gets stitched into every place FRIDAY already
knows how to reach.

:class:`NoteWriter` is that stitching. Writing a note fans out to three
places at once:

* **RAG memory** (via the injected :class:`Ingestor`), so "what was that emf
  problem?" three weeks later actually finds this note and can cite it —
  reusing the exact write path :mod:`friday.rag.ingest` already owns, not a
  second one.
* **the study store** (via the injected :class:`StudyStore`), as SM-2
  flashcards, so a photographed doubt comes back for spaced review instead of
  being solved once and forgotten.
* **the source items themselves**, whose ``note_ids`` grow by one so a
  capture knows which notes discuss it — the reverse edge of ``item_ids``.

Both fan-out targets are optional and structurally typed (:class:`Ingestor`,
:class:`StudyStore` — narrow ``Protocol``s covering only the one method each
uses), so callers that haven't wired RAG or study yet can pass ``None`` for
either, and tests can pass a two-line recorder instead of the real thing.
Either one failing — a slow vector store, a full disk under the study
database — is logged and swallowed, never raised: the note is the record
that the student did the work and got an explanation, and losing *that*
because a downstream index hiccuped would be a strictly worse outcome than a
note that exists but hasn't been indexed or turned into cards yet. Contrast
this with :meth:`~friday.vault.index.VaultIndex.put_note` itself, which is
not optional and is allowed to raise — persisting the note is the one step
that must actually succeed for the call to have done anything at all.

Before any of that, :meth:`NoteWriter.write` loads the source items and drops
two kinds: ones that no longer exist, and — the privacy-critical case — ones
whose :meth:`~friday.vault.models.Item.may_leave_for_model` is ``False``. A
locked capture must never reach an LLM provider and must never be listed as
one of a note's sources; that guarantee is enforced once, here, before the
prompt is even built, the same way :mod:`friday.vault.solver` never sends a
locked item's OCR text to the panel.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from friday.logging import get_logger
from friday.providers.llm import LLMProvider, Message
from friday.vault.index import VaultIndex
from friday.vault.models import Item, Note

logger = get_logger("friday.vault.notes")

#: Upper bound on how many flashcards a note is asked to produce. Enough to
#: cover a genuinely dense problem without turning every note into a deck.
_MAX_CARDS = 8

#: Deck a note's cards land in when the model didn't give (or agree on) a
#: subject — cards must go *somewhere* rather than being silently dropped.
_FALLBACK_DECK = "general"

_NOTE_PROMPT = """You are FRIDAY, writing a study note over one or more of a \
student's photographed homework captures. Read the extracted text below and \
write a note a student would actually want to reread — not a transcript of \
the capture.

Reply with ONLY this JSON object:

{"title": "...", "subject": "...", "chapter": "...", "markdown": "...", \
"cards": [{"front": "...", "back": "..."}]}

markdown: lead with what matters (the result), then show the working, then \
name the one trap that loses marks on this kind of problem. Plain markdown,
no surrounding prose, no code fences.
cards: up to %d spaced-repetition flashcards distilling the note — front is
the recall cue, back is the answer. Fewer good cards beats %d padded ones;
omit anything trivial.

Extracted text of the capture(s):
---
%s
---
"""


@runtime_checkable
class Ingestor(Protocol):
    """The one method note-writing uses from :class:`~friday.rag.ingest.DocumentIngestor`."""

    async def ingest(self, source_id: str, text: str) -> object: ...


@runtime_checkable
class _CardResult(Protocol):
    """The one attribute read off whatever :class:`StudyStore.add_card` returns."""

    id: int


@runtime_checkable
class StudyStore(Protocol):
    """The one method note-writing uses from :class:`~friday.study.store.SQLiteStudyStore`."""

    def add_card(self, deck: str, front: str, back: str) -> _CardResult: ...


@dataclass
class _ParsedNote:
    """The model's reply, normalized to what a :class:`Note` needs."""

    title: str = ""
    subject: str = ""
    chapter: str = ""
    markdown: str = ""
    cards: list[tuple[str, str]] = field(default_factory=list)


def _item_text(item: Item) -> str:
    """The best available extracted text for one item — OCR, else the caption."""
    text = item.ocr_text.strip()
    return text if text else item.caption.strip()


def _build_source_text(items: list[Item]) -> str:
    """Concatenate the surviving items' extracted text into one labeled block."""
    blocks = [
        f"[Capture {i} — {item.id}]\n{text}"
        for i, item in enumerate(items, start=1)
        if (text := _item_text(item))
    ]
    return "\n\n".join(blocks)


def _parse_reply(reply: str) -> _ParsedNote:
    """Parse the model's reply into a :class:`_ParsedNote`, never raising.

    The happy path is one JSON object matching the schema in ``_NOTE_PROMPT``;
    missing fields default empty and malformed card entries (not a dict,
    empty/missing ``front``/``back``) are dropped rather than rejecting the
    whole reply. When the model doesn't return JSON at all — prose, a code
    fence around plain text, broken JSON, a JSON value that isn't an object —
    the entire reply becomes the note's markdown body, with a title lifted
    from its first non-empty line. A model that ignores the format must still
    produce a *usable* note, not an exception.
    """
    text = reply.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    raw: object = None
    if text:
        try:
            raw = json.loads(text)
        except (ValueError, TypeError):
            raw = None

    if isinstance(raw, dict):
        cards: list[tuple[str, str]] = []
        raw_cards = raw.get("cards")
        if isinstance(raw_cards, list):
            for entry in raw_cards[:_MAX_CARDS]:
                if not isinstance(entry, dict):
                    continue
                front = str(entry.get("front") or "").strip()
                back = str(entry.get("back") or "").strip()
                if front and back:
                    cards.append((front, back))
        return _ParsedNote(
            title=str(raw.get("title") or "").strip(),
            subject=str(raw.get("subject") or "").strip(),
            chapter=str(raw.get("chapter") or "").strip(),
            markdown=str(raw.get("markdown") or "").strip(),
            cards=cards,
        )

    fallback_markdown = text or reply.strip()
    first_line = next((line.strip() for line in fallback_markdown.splitlines() if line.strip()), "")
    return _ParsedNote(title=first_line[:80], markdown=fallback_markdown)


class NoteWriter:
    """Writes a :class:`~friday.vault.models.Note` over a set of captures, and fans it out.

    Args:
        index: The vault index a note (and its updated source items) is
            persisted through. Not optional — this is the one write in
            :meth:`write` that is allowed to raise.
        llm: The provider asked to turn extracted capture text into a note.
            Only the abstract :class:`~friday.providers.llm.LLMProvider`
            contract is used, so any real adapter or
            :class:`~friday.providers.llm.FakeLLM` works.
        ingestor: Optional RAG write path (``None`` skips this fan-out).
        study_store: Optional flashcard store (``None`` skips this fan-out).
    """

    def __init__(
        self,
        *,
        index: VaultIndex,
        llm: LLMProvider,
        ingestor: Ingestor | None,
        study_store: StudyStore | None,
    ) -> None:
        self._index = index
        self._llm = llm
        self._ingestor = ingestor
        self._study_store = study_store

    async def write(self, owner_uid: str, item_ids: list[str], *, prompt: str = "") -> Note:
        """Write one note over ``item_ids``, fan it out, and return it.

        Missing items and locked items (``may_leave_for_model() is False``)
        are silently dropped from the source set before the model is ever
        called — a locked capture's text never appears in the prompt, and
        never appears in the resulting note's ``item_ids``.

        When no items survive — an empty ``item_ids``, ids that don't exist,
        or captures that are all locked — the model is not called at all:
        there is no material to write about, and a model asked to produce a
        note from nothing would either hallucinate content it was never
        shown or need special-casing to refuse, neither of which is better
        than an honest, empty note. A minimal :class:`Note` (no title, no
        markdown, no sources, no cards) is still created and persisted, so a
        caller always gets back a real note id rather than the call silently
        doing nothing.

        Fan-out (RAG ingest, flashcard creation) never prevents the note from
        being returned — see the module docstring for why. Persisting the
        note itself, and re-persisting its source items with the note id
        appended to ``note_ids``, are not treated as fan-out and can raise.
        """
        items = self._load_sources(owner_uid, item_ids)
        parsed = await self._ask_model(items, prompt) if items else _ParsedNote()

        note = Note(
            id=uuid.uuid4().hex[:16],
            owner_uid=owner_uid,
            title=parsed.title,
            subject=parsed.subject,
            chapter=parsed.chapter,
            markdown=parsed.markdown,
            item_ids=[item.id for item in items],
            created_at=datetime.now(UTC).isoformat(),
        )
        note.flashcard_ids = self._write_cards(note, parsed.cards)
        note.rag_source_id = await self._write_to_memory(note)

        self._index.put_note(note)
        self._link_items(items, note.id)
        return note

    def _load_sources(self, owner_uid: str, item_ids: list[str]) -> list[Item]:
        """Load ``item_ids``, dropping missing ids and any item locked against models."""
        items: list[Item] = []
        for item_id in item_ids:
            item = self._index.get_item(owner_uid, item_id)
            if item is None or not item.may_leave_for_model():
                continue
            items.append(item)
        return items

    async def _ask_model(self, items: list[Item], extra_prompt: str) -> _ParsedNote:
        """Build the prompt from ``items``' text and ask the model for a note.

        A provider failure (timeout, ``ProviderError``, anything) degrades to
        a bare-bones note built from the raw source text rather than
        propagating — the same "must always produce something usable" rule
        :func:`_parse_reply` applies to a malformed reply also applies to no
        reply at all.
        """
        source_text = _build_source_text(items)
        filled = _NOTE_PROMPT % (_MAX_CARDS, _MAX_CARDS, source_text[:8000])
        if extra_prompt:
            filled += f"\n\nAdditional instruction from the student: {extra_prompt}\n"

        messages = [Message(role="user", content=filled)]
        try:
            response = await self._llm.complete(messages)
        except Exception as exc:  # noqa: BLE001 - a dead model must not lose the note
            logger.warning("note generation failed", extra={"error": str(exc)})
            return _ParsedNote(markdown=source_text[:4000])
        return _parse_reply(response.text or "")

    def _write_cards(self, note: Note, cards: list[tuple[str, str]]) -> list[int]:
        """Write ``cards`` into the study store; collect the ids that succeeded."""
        if self._study_store is None or not cards:
            return []
        deck = note.subject or _FALLBACK_DECK
        ids: list[int] = []
        for front, back in cards:
            try:
                result = self._study_store.add_card(deck, front, back)
            except Exception as exc:  # noqa: BLE001 - fan-out must not lose the note
                logger.warning("note flashcard fan-out failed", extra={"error": str(exc)})
                continue
            ids.append(result.id)
        return ids

    async def _write_to_memory(self, note: Note) -> str:
        """Ingest the note's markdown under ``f"note:{note.id}"``; return the source id."""
        if self._ingestor is None:
            return ""
        source_id = f"note:{note.id}"
        try:
            await self._ingestor.ingest(source_id, note.markdown)
        except Exception as exc:  # noqa: BLE001 - fan-out must not lose the note
            logger.warning("note rag fan-out failed", extra={"error": str(exc)})
            return ""
        return source_id

    def _link_items(self, items: list[Item], note_id: str) -> None:
        """Append ``note_id`` to each source item's ``note_ids`` and re-persist it."""
        for item in items:
            if note_id not in item.note_ids:
                item.note_ids.append(note_id)
            self._index.put_item(item)
