"""Unit tests for looking at what gets posted in Discord.

Nothing here reaches the network: the one function that opens a socket is
replaced, and what is asserted is the *request* that would have been sent —
because the three bugs these cover were all in the request rather than in the
handling of the reply. A photographed exam question came back as "No problem to
solve" because the ceiling truncated it, the timeout expired before the model
answered, and a transient 503 threw the whole picture away.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from friday.discord import vision


class _Secret:
    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


class _Settings:
    """The three attributes ``describe`` reads off app settings."""

    def __init__(self, key: str = "k") -> None:
        self.gemini_api_key = _Secret(key) if key else None
        self.gemini_base_url = "https://example.invalid/v1"
        self.gemini_model = "gemini-2.5-flash"


# --- which way to look -------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Friday solve the question in blue pen",
        "friday what does this say",
        "read this for me",
        "solve question 3",
        "friday calculate this one",
        "help with this homework",
    ],
)
def test_a_picture_of_a_question_is_transcribed(text: str) -> None:
    assert vision.wants_transcription(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "friday look at this",
        "friday lol",
        "what do you think of this",
        "friday rate my setup",
    ],
)
def test_a_picture_posted_to_be_looked_at_is_only_described(text: str) -> None:
    assert vision.wants_transcription(text) is False


@pytest.mark.asyncio
async def test_transcribing_asks_for_the_page_not_a_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Summarising a physics question throws away the numbers that ARE it."""
    sent: list[list[dict[str, Any]]] = []

    def _fake_ask(base: str, key: str, model: str, parts: list[dict[str, Any]]) -> str:
        sent.append(parts)
        return "transcribed"

    monkeypatch.setattr(vision, "_fetch", lambda url: "data:image/png;base64,AA==")
    monkeypatch.setattr(vision, "_ask", _fake_ask)

    images = [{"url": "https://cdn.discordapp.com/a.png", "name": "a"}]
    await vision.describe(_Settings(), images, verbatim=True)
    assert "Transcribe everything written" in sent[0][0]["text"]

    await vision.describe(_Settings(), images)
    assert "one or two plain sentences" in sent[1][0]["text"]


# --- the request that goes out -----------------------------------------------


def _capture(monkeypatch: pytest.MonkeyPatch, outcomes: list[Any]) -> list[dict]:
    """Answer ``_ask_once``'s HTTP call from ``outcomes``, recording each body."""
    bodies: list[dict] = []
    calls = {"n": 0}

    class _Resp:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload
            self.headers: dict[str, str] = {}

        def read(self) -> bytes:
            return self._payload

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_: object) -> None:
            return None

    def _fake_urlopen(request: Any, timeout: float = 0) -> Any:
        bodies.append(json.loads(request.data.decode()))
        bodies[-1]["_timeout"] = timeout
        outcome = outcomes[min(calls["n"], len(outcomes) - 1)]
        calls["n"] += 1
        if isinstance(outcome, Exception):
            raise outcome
        return _Resp(json.dumps({"choices": [{"message": {"content": outcome}}]}).encode())

    monkeypatch.setattr(vision.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(vision.time, "sleep", lambda _s: None)
    return bodies


def _http_error(code: int) -> Exception:
    return vision.urllib.error.HTTPError(
        "u", code, "err", {}, None  # type: ignore[arg-type]
    )


def test_the_ceiling_leaves_room_for_a_whole_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At 300 a six-line question came back cut off mid-word at "charge to ma"."""
    bodies = _capture(monkeypatch, ["ok"])
    vision._ask("https://x.invalid", "k", "m", [{"type": "text", "text": "t"}])
    assert bodies[0]["max_tokens"] >= 1500
    # Thinking tokens are charged against that same ceiling, and reading a page
    # is not a reasoning task.
    assert bodies[0]["reasoning_effort"] == "none"


def test_the_model_is_given_longer_than_it_actually_takes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured at ~50s for one page; the old 30s ceiling expired every time."""
    bodies = _capture(monkeypatch, ["ok"])
    vision._ask("https://x.invalid", "k", "m", [{"type": "text", "text": "t"}])
    assert bodies[0]["_timeout"] >= 60


def test_a_busy_model_is_asked_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """"High demand" is temporary, and it used to cost the whole picture."""
    bodies = _capture(monkeypatch, [_http_error(503), "the page"])
    assert vision._ask("https://x.invalid", "k", "m", [{"type": "text"}]) == "the page"
    assert len(bodies) == 2


def test_a_rejected_request_is_not_repeated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 400 will be a 400 again; retrying it just spends the time twice."""
    bodies = _capture(monkeypatch, [_http_error(400)])
    assert vision._ask("https://x.invalid", "k", "m", [{"type": "text"}]) is None
    assert len(bodies) == 1


def test_it_gives_up_rather_than_retrying_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bodies = _capture(monkeypatch, [_http_error(503)])
    assert vision._ask("https://x.invalid", "k", "m", [{"type": "text"}]) is None
    assert len(bodies) == vision._ATTEMPTS


# --- handing the picture to the solver ---------------------------------------
#
# A transcription carries a written question perfectly and a drawn one not at
# all: a circuit or a free-body diagram IS the problem statement, and the most
# careful paragraph about it still loses which node joins which. The solver only
# ever received words.


@pytest.mark.asyncio
async def test_the_image_is_downloaded_once_and_used_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fetching again for the solver would double the slowest step in the turn."""
    downloads: list[str] = []

    def _fake_fetch(url: str) -> str:
        downloads.append(url)
        return "data:image/png;base64,AA=="

    monkeypatch.setattr(vision, "_fetch", _fake_fetch)
    monkeypatch.setattr(vision, "_ask", lambda *a: "read")

    images = [{"url": "https://cdn.discordapp.com/a.png", "name": "a"}]
    fetched = await vision.fetch_all(images)
    assert fetched == ["data:image/png;base64,AA=="]

    await vision.describe(_Settings(), images, verbatim=True, fetched=fetched)
    assert len(downloads) == 1  # describe reused them rather than re-downloading


def test_a_data_uri_becomes_an_inline_image() -> None:
    from friday.discord import tutor

    part = tutor._inline("data:image/jpeg;base64,QUJD")
    assert part == {"inline_data": {"mime_type": "image/jpeg", "data": "QUJD"}}
    assert tutor._inline("https://example.invalid/a.png") is None


@pytest.mark.asyncio
async def test_the_solver_is_given_the_picture_as_well_as_the_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from friday.discord import tutor

    seen: dict[str, Any] = {}

    def _fake_ask(
        key: str, question: str, system: str = "", images: list[str] | None = None
    ) -> str:
        seen["question"] = question
        seen["images"] = images
        return "worked"

    monkeypatch.setattr(tutor, "_ask", _fake_ask)
    await tutor.solve(
        _Settings(), "solve this", ["data:image/png;base64,AA=="]
    )
    assert seen["images"] == ["data:image/png;base64,AA=="]
    assert seen["question"] == "solve this"


@pytest.mark.asyncio
async def test_a_picture_with_no_words_is_still_a_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only an empty message AND no picture means there is nothing to work on."""
    from friday.discord import tutor

    monkeypatch.setattr(tutor, "_ask", lambda *a, **k: "worked")
    assert await tutor.solve(_Settings(), "", ["data:image/png;base64,AA=="])
    assert await tutor.solve(_Settings(), "", []) is None


def test_the_instructions_are_read_before_the_picture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A picture met with no idea what to do about it is just a picture."""
    from friday.discord import tutor

    bodies: list[dict] = []

    class _Resp:
        def read(self) -> bytes:
            return json.dumps(
                {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
            ).encode()

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_: object) -> None:
            return None

    def _fake_urlopen(request: Any, timeout: float = 0) -> Any:
        bodies.append(json.loads(request.data.decode()))
        return _Resp()

    monkeypatch.setattr(tutor.urllib.request, "urlopen", _fake_urlopen)
    tutor._ask("k", "a problem", "", ["data:image/png;base64,AA=="])

    parts = bodies[0]["contents"][0]["parts"]
    assert "text" in parts[0]
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
