"""Unit tests for the grounded distance helper (parse + formatting, offline)."""

from __future__ import annotations

import pytest

import friday.maps.distance as d


def test_parses_two_places() -> None:
    assert d._places("what's the distance between kota and bombay") == ("kota", "bombay")
    assert d._places("distance kota to jaipur") == ("kota", "jaipur")
    assert d._places("how far is delhi from agra") == ("delhi", "agra")
    assert d._places("distance from pune to mumbai") == ("pune", "mumbai")


def test_non_distance_returns_none() -> None:
    assert d._places("what's the weather") is None
    assert d._places("who made you") is None


def test_reply_formats_from_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(d, "_geocode", lambda _p: (1.0, 2.0))
    monkeypatch.setattr(
        d, "_route", lambda _a, _b: (893000.0, 39600.0, "NH 52", [[1, 2], [3, 4], [5, 6]])
    )
    monkeypatch.setattr(d, "_via_cities", lambda _c: ["Indore", "Ujjain"])
    reply = d.distance_reply("distance between kota and bombay")
    assert reply is not None
    assert "893 kilometres" in reply and "11 hours" in reply
    assert "via Indore, Ujjain" in reply


def test_reply_falls_back_to_road_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(d, "_geocode", lambda _p: (1.0, 2.0))
    monkeypatch.setattr(
        d, "_route", lambda _a, _b: (251000.0, 10800.0, "NH 52", [[1, 2], [3, 4], [5, 6]])
    )
    monkeypatch.setattr(d, "_via_cities", lambda _c: [])  # reverse-geocode unavailable
    reply = d.distance_reply("distance between kota and jaipur")
    assert reply is not None and "via NH 52" in reply


def test_reply_graceful_when_geocode_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    # A distance query that can't be grounded must NOT return None (which would
    # let the LLM hallucinate a wrong number) — it returns a graceful message.
    monkeypatch.setattr(d, "_geocode", lambda _p: None)
    reply = d.distance_reply("distance between nowhere and elsewhere")
    assert reply is not None
    assert "couldn't pull up" in reply


def test_non_distance_still_returns_none() -> None:
    # Non-distance text yields None so other handlers/LLM take it.
    assert d.distance_reply("what's the weather like") is None
