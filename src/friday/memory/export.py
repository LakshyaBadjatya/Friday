# © Lakshya Badjatya — Author
"""Second-brain export: render notes/facts/journal into Obsidian-style Markdown.

A local-first knowledge base is only as useful as your ability to get it back
out. This module renders structured items — notes, facts, journal entries — into
plain Markdown with optional YAML frontmatter (``tags`` + ``date``), the shape
Obsidian and most Markdown vaults expect. It is pure and deterministic: it takes
already-loaded data and returns a string, importing no LLM SDK, reading no
configuration, and performing no I/O (writing the file is the caller's job).
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field


class Note(BaseModel):
    """One exportable note.

    Attributes:
        title: The note's heading.
        body: The note's Markdown body.
        tags: Optional tags, rendered into the frontmatter (and dropped if empty).
        date: Optional ISO date string for the frontmatter.
    """

    title: str
    body: str = ""
    tags: list[str] = Field(default_factory=list)
    date: str = ""


def _frontmatter(note: Note) -> str:
    """Render a YAML frontmatter block for ``note``, or ``""`` when it has none."""
    lines: list[str] = []
    if note.tags:
        lines.append("tags: [" + ", ".join(note.tags) + "]")
    if note.date:
        lines.append(f"date: {note.date}")
    if not lines:
        return ""
    return "---\n" + "\n".join(lines) + "\n---\n"


def to_markdown(note: Note) -> str:
    """Render a single :class:`Note` as Markdown (frontmatter + heading + body)."""
    front = _frontmatter(note)
    parts = [front] if front else []
    parts.append(f"# {note.title}")
    if note.body.strip():
        parts.append(note.body.rstrip())
    return "\n".join(parts).rstrip() + "\n"


def export(notes: Iterable[Note]) -> str:
    """Render many notes into one document, separated by a horizontal rule."""
    docs = [to_markdown(note) for note in notes]
    return "\n\n---\n\n".join(docs)


def facts_to_note(
    facts: Iterable[tuple[str, str]], *, title: str = "Facts", date: str = ""
) -> Note:
    """Bundle ``(text, source_id)`` facts into one bulleted :class:`Note`.

    Each fact becomes a bullet annotated with its source id, so the exported note
    keeps provenance. An empty iterable yields a note with an empty body.
    """
    bullets = [f"- {text} ({source_id})" for text, source_id in facts]
    return Note(title=title, body="\n".join(bullets), date=date)
