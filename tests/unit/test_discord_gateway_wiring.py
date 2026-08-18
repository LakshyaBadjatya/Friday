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


def test_the_wait_is_capped() -> None:
    """A reply nobody is still waiting for is not worth sending."""
    assert gateway._retry_after(_throttled("9999"), 0) == 30.0
    assert gateway._retry_after(_throttled(), 20) == 30.0


def test_the_follow_up_retries_on_the_same_statuses() -> None:
    """The frozen "thinking…" placeholder came from this path giving up at once."""
    assert routes_discord._RETRY_STATUSES == gateway._RETRY_STATUSES
    assert routes_discord._EDIT_ATTEMPTS >= 2
