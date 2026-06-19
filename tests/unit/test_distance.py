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
    monkeypatch.setattr(d, "_route", lambda _a, _b: (893000.0, 39600.0))  # 893km / 11h
    reply = d.distance_reply("distance between kota and bombay")
    assert reply is not None
    assert "893 kilometres" in reply and "11 hours" in reply


def test_reply_none_when_geocode_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(d, "_geocode", lambda _p: None)
    assert d.distance_reply("distance between nowhere and elsewhere") is None
