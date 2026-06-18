"""Unit tests for :func:`friday.siri.speech.for_speech`.

``for_speech`` shapes the orchestrator's (often markdown-y, arbitrarily long)
reply into a short, plain string that Siri's "Speak Text" can read aloud
cleanly. These pin the behaviour: markdown is stripped, whitespace collapses,
and over-long replies are truncated on a sentence boundary with an ellipsis.
"""

from __future__ import annotations

from friday.siri.speech import for_speech


def test_strips_markdown_emphasis_and_headings() -> None:
    out = for_speech("# Title\n\n**Hello** _world_")
    assert "*" not in out
    assert "#" not in out
    assert out == "Title Hello world"


def test_converts_links_to_their_label() -> None:
    assert for_speech("see [the docs](https://x.test/y)") == "see the docs"


def test_collapses_whitespace_and_newlines() -> None:
    assert for_speech("a\n\n  b\tc ") == "a b c"


def test_strips_bullets_and_numbering() -> None:
    assert for_speech("- one\n- two\n1. three") == "one two three"


def test_removes_code_backticks() -> None:
    assert for_speech("run `friday serve` now") == "run friday serve now"


def test_truncates_long_text_with_ellipsis() -> None:
    out = for_speech("word " * 200, max_chars=50)
    assert len(out) <= 51
    assert out.endswith("…")


def test_truncates_on_sentence_boundary_when_possible() -> None:
    out = for_speech("First sentence. " + "x" * 500, max_chars=40)
    assert "First sentence." in out
    assert out.endswith("…")


def test_short_text_is_returned_unchanged() -> None:
    assert for_speech("All good, boss.") == "All good, boss."


def test_empty_or_blank_input_returns_empty_string() -> None:
    assert for_speech("") == ""
    assert for_speech("   \n  ") == ""
