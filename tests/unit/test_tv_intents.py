from __future__ import annotations

import pytest

from friday.tv.intents import parse_tv_command, strip_tv_suffix
from friday.tv.models import TVAction, TVActionType


def test_tvaction_round_trips_through_model_dump() -> None:
    action = TVAction(type=TVActionType.OPEN_APP, app="youtube", speak="Opening YouTube.")
    dumped = action.model_dump()
    assert dumped["type"] == "open_app"
    assert dumped["app"] == "youtube"
    assert dumped["query"] is None
    assert dumped["speak"] == "Opening YouTube."


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("pause", "play_pause"),
        ("resume", "play_pause"),
        ("stop", "stop"),
        ("next", "next"),
        ("skip", "next"),
        ("previous", "previous"),
        ("rewind", "rewind"),
        ("fast forward", "fast_forward"),
    ],
)
def test_parse_media_keys(text: str, key: str) -> None:
    action = parse_tv_command(text)
    assert action is not None
    assert action.type.value == "media"
    assert action.key == key


@pytest.mark.parametrize(
    ("text", "key"),
    [("home", "home"), ("go home", "home"), ("back", "back"), ("go back", "back")],
)
def test_parse_navigation_keys(text: str, key: str) -> None:
    action = parse_tv_command(text)
    assert action is not None
    assert action.type.value == "navigate"
    assert action.key == key


def test_parse_returns_none_for_chitchat() -> None:
    assert parse_tv_command("what's the weather tomorrow") is None


def test_parse_open_app() -> None:
    action = parse_tv_command("open YouTube")
    assert action is not None
    assert action.type.value == "open_app"
    assert action.app == "youtube"
    assert action.speak == "Opening youtube."


def test_parse_open_strips_the_and_app_suffix() -> None:
    action = parse_tv_command("launch the Netflix app")
    assert action is not None
    assert action.type.value == "open_app"
    assert action.app == "netflix"


def test_parse_play_defaults_to_youtube() -> None:
    action = parse_tv_command("play lofi beats")
    assert action is not None
    assert action.type.value == "play"
    assert action.app == "youtube"
    assert action.query == "lofi beats"
    assert "lofi beats" in action.speak


def test_parse_play_on_named_app() -> None:
    action = parse_tv_command("play Stranger Things on Netflix")
    assert action is not None
    assert action.type.value == "play"
    assert action.app == "netflix"
    assert action.query == "stranger things"


def test_parse_search() -> None:
    action = parse_tv_command("search cooking videos on YouTube")
    assert action is not None
    assert action.type.value == "search"
    assert action.app == "youtube"
    assert action.query == "cooking videos"


def test_strip_tv_suffix_present() -> None:
    assert strip_tv_suffix("play lofi beats on the TV") == "play lofi beats"
    assert strip_tv_suffix("open Netflix on tv") == "open Netflix"


def test_strip_tv_suffix_absent_returns_none() -> None:
    assert strip_tv_suffix("play lofi beats") is None
    assert strip_tv_suffix("what's on TV tonight") is None
