"""Unit tests for Instagram DM intent parsing.

Count/read/reply phrases classify correctly (with reply most specific and read
winning over count); non-Instagram phrases — "tell mom I'll be late" (no token),
"what's the weather", "any messages for me" — must return ``None`` so the Siri
pipeline falls through to the circle / orchestrator unchanged. Pure regex, no I/O.
"""

from __future__ import annotations

import pytest

from friday.instagram.intents import (
    CountDMs,
    ReadAloud,
    ReplyDM,
    is_bare_read,
    parse_intent,
)


@pytest.mark.parametrize(
    "phrase",
    [
        "any instagram dms",
        "any new instagram messages for me",
        "do i have any instagram messages",
        "check my instagram",
        "any dms on instagram",
        "instagram inbox",
        "any insta messages",
        "check my ig",
    ],
)
def test_count_phrases(phrase: str) -> None:
    assert isinstance(parse_intent(phrase), CountDMs)


@pytest.mark.parametrize(
    "phrase",
    [
        "read my instagram messages",
        "read out my instagram dms",
        "read my insta dms",
    ],
)
def test_read_phrases(phrase: str) -> None:
    assert isinstance(parse_intent(phrase), ReadAloud)


def test_read_wins_over_count() -> None:
    # "read my instagram messages" matches both the read and count regexes; read wins.
    assert isinstance(parse_intent("read my instagram messages"), ReadAloud)


@pytest.mark.parametrize(
    ("phrase", "name", "text"),
    [
        ("reply to rahul on instagram saying i'll call later", "rahul", "i'll call later"),
        ("dm rahul on instagram i'm on my way", "rahul", "i'm on my way"),
        ("message priya on instagram that dinner is at 8", "priya", "dinner is at 8"),
        ("reply to alex on ig: see you soon", "alex", "see you soon"),
    ],
)
def test_reply_phrases(phrase: str, name: str, text: str) -> None:
    intent = parse_intent(phrase)
    assert isinstance(intent, ReplyDM)
    assert intent.name == name
    assert intent.text == text


@pytest.mark.parametrize(
    "phrase",
    [
        "tell mom I'll be late",  # no instagram token -> not a reply
        "what's the weather",
        "any messages for me",  # no instagram token
        "do i have any messages",
        "read me a story",
        "tell me a joke",
    ],
)
def test_non_matches_return_none(phrase: str) -> None:
    assert parse_intent(phrase) is None


@pytest.mark.parametrize(
    "phrase",
    ["read them aloud", "read those out", "read it aloud", "read these"],
)
def test_bare_read_detected(phrase: str) -> None:
    # Bare reads aren't intents on their own, but the Siri layer honours them inside
    # the just-asked-about-Instagram window.
    assert parse_intent(phrase) is None
    assert is_bare_read(phrase) is True


def test_bare_read_negative() -> None:
    assert is_bare_read("read my instagram messages") is False
