"""Integration tests for the E2EE chat + listing endpoints on /circle.

Offline (in-memory stores, the ``siri_identities`` seam stands in for Firebase
ID-token verification). Drives: create group -> invite -> accept, then post an
(opaque) ciphertext message, read it back as the other member, confirm a
non-member is refused (403), and check the my-groups + invite-preview listings.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from friday.app import create_app
from friday.config import get_settings

ADMIN = {"Authorization": "Bearer tok-a"}
FRIEND = {"Authorization": "Bearer tok-b"}
STRANGER = {"Authorization": "Bearer tok-c"}


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "fake")
    monkeypatch.setenv("FRIDAY_LLM_FALLBACK_PROVIDER", "none")
    monkeypatch.setenv("FRIDAY_RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("FRIDAY_ENABLE_CIRCLE", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _client() -> TestClient:
    client = TestClient(create_app())
    client.__enter__()
    client.app.state.siri_identities = {
        "tok-a": "u-a",
        "tok-b": "u-b",
        "tok-c": "u-c",
    }
    return client


def _make_group_of_two(client: TestClient) -> str:
    gid = client.post(
        "/circle/groups", json={"name": "Us", "display_name": "Me"}, headers=ADMIN
    ).json()["id"]
    code = client.post(f"/circle/groups/{gid}/invites", json={}, headers=ADMIN).json()[
        "code"
    ]
    client.post(
        f"/circle/invites/{code}/accept",
        json={"display_name": "Bestie", "tz": "America/New_York"},
        headers=FRIEND,
    )
    return gid


def test_e2ee_chat_roundtrip_and_listings() -> None:
    client = _client()
    try:
        gid = _make_group_of_two(client)

        # Admin posts ciphertext; the server stores only ciphertext + nonce.
        posted = client.post(
            f"/circle/groups/{gid}/messages",
            json={"ciphertext": "BASE64CIPHER", "nonce": "BASE64NONCE"},
            headers=ADMIN,
        )
        assert posted.status_code == 200
        assert posted.json()["ciphertext"] == "BASE64CIPHER"
        assert posted.json()["sender_uid"] == "u-a"

        # The friend (a member) reads it back.
        history = client.get(f"/circle/groups/{gid}/messages", headers=FRIEND)
        assert history.status_code == 200
        msgs = history.json()["messages"]
        assert [m["ciphertext"] for m in msgs] == ["BASE64CIPHER"]

        # A non-member cannot post or read.
        assert (
            client.post(
                f"/circle/groups/{gid}/messages",
                json={"ciphertext": "x", "nonce": "y"},
                headers=STRANGER,
            ).status_code
            == 403
        )
        assert (
            client.get(f"/circle/groups/{gid}/messages", headers=STRANGER).status_code
            == 403
        )

        # my-groups lists the group for a member; invite-preview shows its name.
        mine = client.get("/circle/groups", headers=FRIEND).json()["groups"]
        assert {g["id"] for g in mine} == {gid}

        code = client.post(
            f"/circle/groups/{gid}/invites", json={}, headers=ADMIN
        ).json()["code"]
        preview = client.get(f"/circle/invites/{code}", headers=FRIEND)
        assert preview.status_code == 200
        assert preview.json()["group"]["name"] == "Us"
    finally:
        client.__exit__(None, None, None)


def test_stream_ticket_is_member_gated() -> None:
    client = _client()
    try:
        gid = _make_group_of_two(client)
        # A member mints a single-use stream ticket (no token in any URL).
        minted = client.post(f"/circle/groups/{gid}/stream/ticket", headers=ADMIN)
        assert minted.status_code == 200
        assert minted.json()["ticket"]
        # A non-member can't; an anonymous caller can't.
        assert (
            client.post(
                f"/circle/groups/{gid}/stream/ticket", headers=STRANGER
            ).status_code
            == 403
        )
        assert (
            client.post(f"/circle/groups/{gid}/stream/ticket").status_code == 401
        )
        # The stream rejects a missing/garbage ticket (no bearer fallback in the URL).
        assert client.get(f"/circle/groups/{gid}/stream").status_code == 401
        assert (
            client.get(f"/circle/groups/{gid}/stream?ticket=nope").status_code == 401
        )
    finally:
        client.__exit__(None, None, None)


def test_chat_requires_auth_and_flag() -> None:
    client = _client()
    try:
        gid = _make_group_of_two(client)
        # No bearer token -> 401.
        assert (
            client.post(
                f"/circle/groups/{gid}/messages",
                json={"ciphertext": "x", "nonce": "y"},
            ).status_code
            == 401
        )
    finally:
        client.__exit__(None, None, None)
