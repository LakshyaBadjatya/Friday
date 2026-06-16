# © Lakshya Badjatya — Author
"""Unit tests for second-brain Markdown export."""

from __future__ import annotations

from friday.memory.export import Note, export, facts_to_note, to_markdown


def test_to_markdown_with_frontmatter() -> None:
    note = Note(title="Ideas", body="some body", tags=["ai", "notes"], date="2026-06-16")
    md = to_markdown(note)
    assert md.startswith("---\ntags: [ai, notes]\ndate: 2026-06-16\n---\n")
    assert "# Ideas" in md
    assert "some body" in md


def test_to_markdown_without_frontmatter() -> None:
    md = to_markdown(Note(title="Bare", body="x"))
    assert md == "# Bare\nx\n"


def test_export_joins_with_rule() -> None:
    out = export([Note(title="A"), Note(title="B")])
    assert "# A" in out and "# B" in out
    assert "\n\n---\n\n" in out


def test_facts_to_note_keeps_provenance() -> None:
    note = facts_to_note([("the sky is blue", "src1"), ("water is wet", "src2")])
    assert note.title == "Facts"
    assert "- the sky is blue (src1)" in note.body
    assert "- water is wet (src2)" in note.body


def test_facts_to_note_empty() -> None:
    note = facts_to_note([])
    assert note.body == ""
