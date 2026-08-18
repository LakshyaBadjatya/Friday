"""The wiring that made her look healthy while saying nothing.

Every failure in this file used to be silent. She sat in the member list with a
status line, answered on Telegram and the HUD, and ignored Discord entirely —
because each of these things fails by returning quietly rather than by raising.
The tests exist so a future edit cannot quietly drop them again.
"""

from __future__ import annotations

import urllib.error
from types import SimpleNamespace
from typing import Any

import anyio

from friday.api import routes_discord
from friday.discord import gateway


class _App:
    """The slice of ``FastAPI`` these functions actually touch."""

    def __init__(self, **settings: Any) -> None:
        self.state = SimpleNamespace(settings=SimpleNamespace(**settings))


# --- intents --------------------------------------------------------------- #
def test_she_can_hear_a_dm() -> None:
    """DIRECT_MESSAGES, or a DM is never delivered and nothing says so."""
    assert gateway._INTENTS & (1 << 12)


def test_she_can_still_hear_a_channel() -> None:
    """GUILD_MESSAGES and MESSAGE_CONTENT, the pair that carry a typed message."""
    assert gateway._INTENTS & (1 << 9)
    assert gateway._INTENTS & (1 << 15)


# --- who she is ------------------------------------------------------------ #
def test_ready_tells_her_who_she_is() -> None:
    """The authoritative source, and one that cannot be left unset."""
    app = _App(discord_application_id="")
    gateway._remember_self(app, {"user": {"id": "111"}})
    assert gateway._self_id(app) == "111"


def test_an_unset_application_id_no_longer_makes_her_deaf_to_mentions() -> None:
    """The original bug: an empty id matched no mention, ever."""
    app = _App(discord_application_id="")
    assert gateway._self_id(app) == ""          # before READY
    gateway._remember_self(app, {"user": {"id": "222"}})
    assert gateway._self_id(app) == "222"       # after READY, whatever the config


def test_the_configured_id_still_covers_the_gap_before_ready() -> None:
    app = _App(discord_application_id="333")
    assert gateway._self_id(app) == "333"


def test_a_ready_without_a_user_id_does_not_overwrite_anything() -> None:
    app = _App(discord_application_id="444")
    gateway._remember_self(app, {})
    assert gateway._self_id(app) == "444"


# --- the follow-up address ------------------------------------------------- #
def test_the_interaction_names_the_application() -> None:
    """Discord sends it every time, so the placeholder can always be replaced."""
    settings = SimpleNamespace(discord_application_id="")
    assert routes_discord._app_id({"application_id": "999"}, settings) == "999"


def test_configuration_is_only_the_fallback() -> None:
    settings = SimpleNamespace(discord_application_id="555")
    assert routes_discord._app_id({}, settings) == "555"


# --- why a send failed ----------------------------------------------------- #
def test_a_rejection_reports_its_status_and_discord_s_complaint() -> None:
    """"discord send failed" on its own could not tell a 403 from a 429."""
    exc = urllib.error.HTTPError(
        "https://discord.com/api/v10/channels/1/messages",
        403,
        "Forbidden",
        {},  # type: ignore[arg-type]
        None,
    )
    why = gateway._why(exc)
    assert "403" in why
    assert "Forbidden" in why


def test_an_unreachable_api_says_so() -> None:
    why = gateway._why(urllib.error.URLError("name resolution failed"))
    assert "unreachable" in why
    assert "name resolution failed" in why


# --- retrying a throttled send --------------------------------------------- #
class _Headers(dict):
    """``HTTPError.headers`` only needs ``.get`` here."""

    def get(self, key: str, default: Any = None) -> Any:
        return dict.get(self, key, default)


def _throttled(retry_after: str | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://discord.com/api/v10/channels/1/messages",
        429,
        "Too Many Requests",
        _Headers({} if retry_after is None else {"Retry-After": retry_after}),  # type: ignore[arg-type]
        None,
    )


def test_a_throttled_send_is_worth_another_try() -> None:
    """429 is the Cloudflare 1015 case: the IP, not the request. It clears."""
    assert 429 in gateway._RETRY_STATUSES
    assert 503 in gateway._RETRY_STATUSES
    assert 403 not in gateway._RETRY_STATUSES     # a permission problem never clears
    assert 401 not in gateway._RETRY_STATUSES


def test_discord_s_own_retry_after_is_obeyed() -> None:
    assert gateway._retry_after(_throttled("2.5"), 0) == 2.5


def test_cloudflare_names_no_delay_so_we_back_off() -> None:
    """1015 carries no Retry-After, so doubling is the fallback."""
    assert [gateway._retry_after(_throttled(), i) for i in range(4)] == [1.0, 2.0, 4.0, 8.0]


def test_a_nonsense_retry_after_does_not_raise() -> None:
    assert gateway._retry_after(_throttled("soon"), 1) == 2.0


def test_the_real_wait_is_reported_not_a_clamped_one() -> None:
    """Clamping the number before logging it hid an hour-long block behind "30.0s"."""
    assert gateway._retry_after(_throttled("3600"), 0) == 3600.0


def test_a_wait_longer_than_we_will_sit_through_is_recognisable() -> None:
    """The caller gives up on these rather than burning four doomed attempts."""
    assert gateway._retry_after(_throttled("3600"), 0) > gateway._MAX_WAIT
    assert gateway._retry_after(_throttled("2"), 0) <= gateway._MAX_WAIT
    assert routes_discord._MAX_WAIT == gateway._MAX_WAIT


def test_the_follow_up_retries_on_the_same_statuses() -> None:
    """The frozen "thinking…" placeholder came from this path giving up at once."""
    assert routes_discord._RETRY_STATUSES == gateway._RETRY_STATUSES
    assert routes_discord._EDIT_ATTEMPTS >= 2


# --- looking alive while she thinks ----------------------------------------- #
def _message(text: str, *, dm: bool = False) -> dict[str, Any]:
    m: dict[str, Any] = {
        "id": "1", "content": text, "channel_id": "999",
        "author": {"id": "42", "username": "someone"}, "mentions": [],
    }
    if not dm:
        m["guild_id"] = "1538168912225636466"
    return m


def _run_on_message(monkeypatch: Any, message: dict[str, Any]) -> list[str]:
    """Drive ``_on_message`` with the model and the network stubbed out."""
    seen: list[str] = []

    async def fake_compose(*_a: Any, **_k: Any) -> str:
        return "a reply"

    async def fake_typing(_token: str, _channel: str) -> None:
        seen.append("typing")

    async def fake_send(*_a: Any, **_k: Any) -> str:
        seen.append("send")
        return "id"

    monkeypatch.setattr(gateway, "_compose", fake_compose)
    monkeypatch.setattr(gateway, "_typing", fake_typing)
    monkeypatch.setattr(gateway, "_send", fake_send)

    app = _App(discord_application_id="", discord_owner_id="")
    anyio.run(gateway._on_message, app, "tok", message)
    return seen


def test_she_shows_she_is_working_on_it(monkeypatch: Any) -> None:
    """Three to six seconds of silence is most of what "slow" meant."""
    assert "typing" in _run_on_message(monkeypatch, _message("friday hello"))


def test_a_dm_gets_the_same_courtesy(monkeypatch: Any) -> None:
    assert "typing" in _run_on_message(monkeypatch, _message("hello", dm=True))


def test_she_does_not_type_at_a_room_she_is_not_answering(monkeypatch: Any) -> None:
    """Typing dots followed by nothing reads worse than staying quiet."""
    assert "typing" not in _run_on_message(monkeypatch, _message("just chatting"))


# --- reaching the machine --------------------------------------------------- #
def test_the_owner_s_name_means_the_same_machine_as_my() -> None:
    """"lakshya's pc" fell through to a chat answer about a Mac."""
    from friday.core.orchestrator import _is_pc_request

    assert _is_pc_request("friday what is running on lakshya's pc")
    assert _is_pc_request("what is running on my pc")


def test_asking_whether_she_can_reach_it_counts() -> None:
    """She could, and said she could not, because nothing matched."""
    from friday.core.orchestrator import _is_pc_request

    assert _is_pc_request("Friday can you access my pc")
    assert _is_pc_request("friday check my pc")


def test_ordinary_talk_about_a_pc_is_still_just_talk() -> None:
    from friday.core.orchestrator import _is_pc_request

    assert not _is_pc_request("friday hello")
    assert not _is_pc_request("my pc game is fun")


def test_a_machine_turn_is_given_room_to_finish() -> None:
    """Two model calls around a 45s job cannot fit a spoken one-liner's budget."""
    from friday.api import routes_siri

    assert routes_siri._PC_TURN_SECONDS > 45.0


def test_a_guest_may_ask_her_anything_but_not_the_pc() -> None:
    """A Discord channel has other people in it, and the prompt quotes them."""
    import anyio

    app = _App(discord_owner_id="111,222", discord_application_id="")
    out = anyio.run(
        lambda: gateway._compose(app, "what is running on lakshya's pc", "chan")
    )
    assert out is None or "owner-only" in out


def test_the_gate_only_turns_away_guests() -> None:
    """The owner's side is asserted here rather than by driving ``_compose``.

    Reaching the machine through ``_compose`` means the real brain, and stubbing
    it layer by layer tests the stubs. What matters and can be pinned down is
    that the gate keys on being a guest and on nothing else.
    """
    owned = _App(discord_owner_id="111,222")
    assert gateway._who_is(owned, "111") == "owner"
    assert gateway._who_is(owned, "222") == "queen"
    assert gateway._who_is(owned, "999") == "guest"
    # Unconfigured stays trusting, as it was before any of this.
    assert gateway._who_is(_App(discord_owner_id=""), "999") == "owner"
