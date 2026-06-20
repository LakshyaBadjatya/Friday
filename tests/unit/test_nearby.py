"""Unit tests for ``friday.maps.nearby`` — heuristic + AI nearby detection.

Importers/callers under test: ``nearby_reply`` / ``nearby_from_filter`` /
``classify_nearby`` (all used by the ``POST /siri/ask`` near-me branch). Overpass
is stubbed via ``monkeypatch`` on ``friday.maps.nearby._overpass`` so the tests
are offline. The LLM is a fixed-text stub. No persisted schema. Verbatim
instruction: "make the recognize phrases dynamic and ai should auto guess when i
am asking for location or near by places."
"""

from __future__ import annotations

import asyncio

import friday.maps.nearby as nearby
from friday.providers.llm import LLMResponse


class _StubLLM:
    def __init__(self, text: str) -> None:
        self._text = text

    async def complete(self, messages, tools=None, *, model=None):  # noqa: ANN001, ANN002
        return LLMResponse(text=self._text)


_ELEMENTS = [
    {"lat": 25.18, "lon": 75.84, "tags": {"name": "City Palace"}},
    {"lat": 25.20, "lon": 75.86, "tags": {"name": "Seven Wonders Park"}},
]


def test_intent_hint_maps_hungry_to_food() -> None:
    cat = nearby._category("i'm really hungry")
    assert cat is not None
    assert cat[1] == "places to eat"


def test_explicit_attractions_category() -> None:
    cat = nearby._category("tourist attractions")
    assert cat is not None
    assert cat[1] == "attractions"


def test_nearby_reply_formats(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(nearby, "_overpass", lambda f, lat, lon: _ELEMENTS)
    out = nearby.nearby_reply("good restaurants near me", 25.17, 75.84)
    assert out is not None
    spoken, share = out
    assert "City Palace" in spoken
    assert "Map:" in share


def test_nearby_reply_skips_non_place_query(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(nearby, "_overpass", lambda f, lat, lon: _ELEMENTS)
    # No place category and no near-word -> heuristic declines (AI path handles it).
    assert nearby.nearby_reply("what is the capital of France", 25.17, 75.84) is None


def test_classify_nearby_true(monkeypatch) -> None:  # noqa: ANN001
    llm = _StubLLM('{"nearby": true, "category": "cafe"}')
    inferred = asyncio.run(nearby.classify_nearby(llm, "I could really go for a latte"))
    assert inferred is not None
    filt, label = inferred
    assert label == "cafes"
    assert "cafe" in filt


def test_classify_nearby_false() -> None:
    llm = _StubLLM('{"nearby": false}')
    assert asyncio.run(nearby.classify_nearby(llm, "tell me a joke")) is None


def test_classify_nearby_handles_garbage() -> None:
    llm = _StubLLM("I am not sure what you mean")
    assert asyncio.run(nearby.classify_nearby(llm, "blah")) is None


def test_classify_nearby_no_llm() -> None:
    assert asyncio.run(nearby.classify_nearby(None, "food")) is None


def test_looks_like_junk() -> None:
    # OSM vandalism/test entries (no lowercase, short) are dropped...
    assert nearby._looks_like_junk("MKLJFAL")
    assert nearby._looks_like_junk("PA 1")
    assert nearby._looks_like_junk("POI")
    # ...real names are kept.
    assert not nearby._looks_like_junk("Kota zoo")
    assert not nearby._looks_like_junk("Seven Wonders Park")


def test_attractions_lead_with_famous_places(monkeypatch) -> None:  # noqa: ANN001
    # Wikipedia supplies famous, notable places; OSM adds junk + a colony park.
    monkeypatch.setattr(
        nearby, "_wikipedia_nearby", lambda lat, lon: [(0.4, "Umed Bhawan Palace")]
    )
    monkeypatch.setattr(
        nearby,
        "_overpass",
        lambda f, lat, lon: [
            {"lat": 25.2, "lon": 75.86, "tags": {"name": "MKLJFAL"}},
            {"lat": 25.21, "lon": 75.87, "tags": {"name": "Chambal Garden"}},
        ],
    )
    out = nearby.nearby_from_filter("[whatever]", "attractions", 25.21, 75.86)
    assert out is not None
    spoken, _share = out
    assert "Umed Bhawan Palace" in spoken  # famous, nearest -> first
    assert "Chambal Garden" in spoken
    assert "MKLJFAL" not in spoken  # vandalism dropped


def test_wikipedia_filters_admin_and_keeps_attractions(monkeypatch) -> None:  # noqa: ANN001
    import json as _json

    payload = _json.dumps(
        {
            "query": {
                "geosearch": [
                    {"title": "Garh Palace", "dist": 500.0},
                    {"title": "Kota North Assembly constituency", "dist": 800.0},
                    {"title": "Kota Junction railway station", "dist": 1900.0},
                    {"title": "Some Residential Colony", "dist": 300.0},
                ]
            }
        }
    ).encode()

    class _Resp:
        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *a):  # noqa: ANN002, ANN204
            return False

        def read(self):  # noqa: ANN201
            return payload

    monkeypatch.setattr(nearby.urllib.request, "urlopen", lambda *a, **k: _Resp())
    out = nearby._wikipedia_nearby(25.21, 75.86)
    names = [n for _d, n in out]
    assert "Garh Palace" in names  # attraction kept
    assert all("constituency" not in n and "railway" not in n for n in names)
    assert "Some Residential Colony" not in names  # no attraction keyword -> dropped
