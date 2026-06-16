# © Lakshya Badjatya — Author
"""Unit tests for citations & provenance formatting."""

from __future__ import annotations

import pytest

from friday.memory.citations import CitationFormatter, Source


def test_references_dedup_first_seen_order() -> None:
    fmt = CitationFormatter()
    sources = [Source(source_id="b"), Source(source_id="a"), Source(source_id="b")]
    cites = fmt.references(sources)
    assert [(c.marker, c.source_id) for c in cites] == [("[1]", "b"), ("[2]", "a")]


def test_format_block_lists_sources() -> None:
    fmt = CitationFormatter()
    block = fmt.format_block([Source(source_id="doc1"), Source(source_id="doc2")])
    assert block == "Sources:\n[1] doc1\n[2] doc2"


def test_format_block_empty_when_no_sources() -> None:
    assert CitationFormatter().format_block([]) == ""


def test_snippet_chars_includes_text() -> None:
    fmt = CitationFormatter(snippet_chars=10)
    block = fmt.format_block([Source(source_id="d", text="hello world this is long")])
    assert block == "Sources:\n[1] d — hello worl"


def test_attach_appends_block_and_passthrough_when_empty() -> None:
    fmt = CitationFormatter()
    answer = "The sky is blue."
    out = fmt.attach(answer, [Source(source_id="sky")])
    assert out == "The sky is blue.\n\nSources:\n[1] sky"
    assert fmt.attach(answer, []) == answer  # no sources -> unchanged


def test_negative_snippet_rejected() -> None:
    with pytest.raises(ValueError):
        CitationFormatter(snippet_chars=-1)
