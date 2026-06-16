# © Lakshya Badjatya — Author
"""Unit tests for :class:`friday.tools.weather.WeatherTool`.

Fully offline: HTTP is mocked with ``respx`` (no live network — wttr.in is never
hit). The tool is read-only and must NEVER fabricate weather on failure (no
``temp_c``/``summary`` keys in the failure payload). Covers the happy-path j1
parse, the tool attributes/flags, argument validation (empty / control-char
location), HTTP failures (retriable 500 vs non-retriable 404), and the bounded
single retry on a transient network error.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from friday.tools.base import ToolResult
from friday.tools.weather import WeatherArgs, WeatherTool

# wttr.in serves j1 JSON at ``https://wttr.in/<location>?format=j1``; the route is
# matched on the path (the ``format`` query is mocked through).
KOTA_URL = "https://wttr.in/Kota"

# A trimmed but structurally faithful wttr.in ``j1`` payload: the
# ``current_condition`` block carries the fields the tool reads, and
# ``nearest_area`` carries the resolved area name.
KOTA_J1 = {
    "current_condition": [
        {
            "temp_C": "38",
            "FeelsLikeC": "41",
            "humidity": "20",
            "windspeedKmph": "12",
            "weatherDesc": [{"value": "Sunny"}],
            "observation_time": "12:00 PM",
        }
    ],
    "nearest_area": [
        {"areaName": [{"value": "Kota"}]},
    ],
}


# -- attributes / args --------------------------------------------------- #


def test_weather_tool_attrs() -> None:
    tool = WeatherTool()
    assert tool.name == "weather"
    assert tool.side_effecting is False
    assert tool.idempotent is True
    assert tool.required_permission == "web"
    assert tool.args_model is WeatherArgs
    # The description must make the tool obvious to an LLM tool-caller.
    assert "weather" in tool.description.lower()


def test_weather_args_rejects_empty_location() -> None:
    with pytest.raises(ValueError):
        WeatherArgs(location="")


def test_weather_args_rejects_control_char_location() -> None:
    with pytest.raises(ValueError):
        WeatherArgs(location="Kota\nHost: evil")


# -- happy path ---------------------------------------------------------- #


@respx.mock
async def test_weather_parses_j1_into_summary_and_fields() -> None:
    respx.get(KOTA_URL).mock(return_value=httpx.Response(200, json=KOTA_J1))
    tool = WeatherTool()
    result = await tool(WeatherArgs(location="Kota"))

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.error is None
    data = result.data
    assert data["location"] == "Kota"
    assert data["temp_c"] == 38
    assert data["feels_like_c"] == 41
    assert data["description"] == "Sunny"
    assert data["humidity_pct"] == 20
    assert data["wind_kph"] == 12
    assert data["summary"] == (
        "Kota: Sunny, 38°C (feels 41°C), humidity 20%"
    )


@respx.mock
async def test_weather_uses_resolved_area_name() -> None:
    # When wttr.in resolves the query to a fuller area name, the summary uses it.
    payload = {
        "current_condition": KOTA_J1["current_condition"],
        "nearest_area": [{"areaName": [{"value": "Kota, Rajasthan"}]}],
    }
    respx.get(KOTA_URL).mock(return_value=httpx.Response(200, json=payload))
    tool = WeatherTool()
    result = await tool(WeatherArgs(location="Kota"))
    assert result.ok is True
    assert result.data["location"] == "Kota, Rajasthan"
    assert result.data["summary"].startswith("Kota, Rajasthan: ")


# -- HTTP failures ------------------------------------------------------- #


@respx.mock
async def test_weather_http_500_is_retriable_failure() -> None:
    respx.get(KOTA_URL).mock(return_value=httpx.Response(500, text="boom"))
    tool = WeatherTool()
    result = await tool(WeatherArgs(location="Kota"))
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "weather_failed"
    assert result.error.retriable is True
    # Never fabricate on failure.
    assert "temp_c" not in result.data
    assert "summary" not in result.data


@respx.mock
async def test_weather_http_404_is_non_retriable_failure() -> None:
    respx.get(KOTA_URL).mock(return_value=httpx.Response(404, text="not found"))
    tool = WeatherTool()
    result = await tool(WeatherArgs(location="Kota"))
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "weather_failed"
    assert result.error.retriable is False
    assert "temp_c" not in result.data


# -- malformed body ------------------------------------------------------ #


@respx.mock
async def test_weather_malformed_json_is_parse_failed() -> None:
    # A 200 with a body that is valid JSON but lacks the expected j1 shape.
    respx.get(KOTA_URL).mock(
        return_value=httpx.Response(200, json={"unexpected": "shape"})
    )
    tool = WeatherTool()
    result = await tool(WeatherArgs(location="Kota"))
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "weather_parse_failed"
    assert result.error.retriable is False
    assert "temp_c" not in result.data


@respx.mock
async def test_weather_non_json_body_is_parse_failed() -> None:
    # wttr.in's plain ASCII-art fallback (not JSON) -> parse failure, not a crash.
    respx.get(KOTA_URL).mock(
        return_value=httpx.Response(200, text="Kota: ☀️ +38°C")
    )
    tool = WeatherTool()
    result = await tool(WeatherArgs(location="Kota"))
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "weather_parse_failed"
    assert "temp_c" not in result.data


# -- network retry ------------------------------------------------------- #


@respx.mock
async def test_weather_connect_error_retried_then_fails() -> None:
    route = respx.get(KOTA_URL).mock(
        side_effect=httpx.ConnectError("no route to host")
    )
    tool = WeatherTool()
    result = await tool(WeatherArgs(location="Kota"))

    # Exactly two attempts: the initial call plus one bounded retry.
    assert route.call_count == 2
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "weather_failed"
    assert result.error.retriable is True
    # No fabricated data on failure.
    assert "temp_c" not in result.data


@respx.mock
async def test_weather_succeeds_on_retry() -> None:
    route = respx.get(KOTA_URL).mock(
        side_effect=[
            httpx.ConnectError("transient blip"),
            httpx.Response(200, json=KOTA_J1),
        ]
    )
    tool = WeatherTool()
    result = await tool(WeatherArgs(location="Kota"))

    assert route.call_count == 2
    assert result.ok is True
    assert result.data["temp_c"] == 38
