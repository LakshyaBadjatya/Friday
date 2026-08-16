"""Unit tests for the Cloudinary signer, fake, and verifier."""

from __future__ import annotations

import hashlib

import httpx
import pytest
import respx

from friday.errors import ProviderError
from friday.vault.cloudinary import CloudinarySigner, FakeCloudinary, sign_params


def test_sign_params_matches_cloudinary_algorithm() -> None:
    # Cloudinary: sort params by key, join as k=v&k=v, append the secret, sha1.
    params = {"timestamp": 1700000000, "public_id": "vault/u1/abc", "folder": "vault/u1"}
    expected = hashlib.sha1(
        b"folder=vault/u1&public_id=vault/u1/abc&timestamp=1700000000secret",
        usedforsecurity=False,
    ).hexdigest()
    assert sign_params(params, "secret") == expected


def test_sign_params_ignores_empty_and_unsigned_keys() -> None:
    signed = sign_params({"timestamp": 1, "public_id": "", "signature": "x", "file": "y"}, "s")
    bare = sign_params({"timestamp": 1}, "s")
    assert signed == bare


def test_sign_params_is_deterministic_and_order_independent() -> None:
    # Cloudinary sorts by key before joining, so caller insertion order must not
    # matter. A dict-order-dependent implementation would pass the algorithm
    # test above by luck alone, so this pins that down explicitly.
    a = sign_params({"timestamp": 1, "public_id": "x", "folder": "vault/u1"}, "s")
    b = sign_params({"folder": "vault/u1", "public_id": "x", "timestamp": 1}, "s")
    assert a == b
    # Same inputs, called twice, must agree with themselves.
    assert a == sign_params({"timestamp": 1, "public_id": "x", "folder": "vault/u1"}, "s")


def test_sign_params_is_sensitive_to_every_input() -> None:
    # A signer that silently dropped a param would still "work" until an
    # attacker exploited exactly that gap, so pin sensitivity to each field.
    base = sign_params({"timestamp": 1, "public_id": "vault/u1/i1"}, "secret")
    assert sign_params({"timestamp": 2, "public_id": "vault/u1/i1"}, "secret") != base
    assert sign_params({"timestamp": 1, "public_id": "vault/u1/i2"}, "secret") != base
    assert sign_params({"timestamp": 1, "public_id": "vault/u1/i1"}, "different") != base


def test_signer_builds_a_complete_upload_payload() -> None:
    signer = CloudinarySigner(
        cloud_name="dsvxqsvs1",
        api_key="key",
        api_secret="secret",
        clock=lambda: 1700000000,
    )
    payload = signer.upload_params(owner_uid="u1", item_id="i1")
    assert payload.url == "https://api.cloudinary.com/v1_1/dsvxqsvs1/image/upload"
    assert payload.params["api_key"] == "key"
    assert payload.params["timestamp"] == 1700000000
    assert payload.params["public_id"] == "vault/u1/i1"
    assert payload.params["type"] == "authenticated"
    assert payload.params["signature"] == sign_params(
        {
            "timestamp": 1700000000,
            "public_id": "vault/u1/i1",
            "folder": "vault/u1",
            "type": "authenticated",
        },
        "secret",
    )


def test_signer_scopes_the_public_id_to_owner_and_item() -> None:
    # The public_id is the only thing standing between one user's signed
    # upload and writing into another user's folder — a signature that didn't
    # actually bind owner_uid/item_id would let a client replay it anywhere.
    signer = CloudinarySigner(cloud_name="c", api_key="k", api_secret="s", clock=lambda: 1)
    payload = signer.upload_params(owner_uid="alice", item_id="i1")
    assert payload.params["public_id"] == "vault/alice/i1"
    assert payload.params["folder"] == "vault/alice"
    other = signer.upload_params(owner_uid="mallory", item_id="i1")
    assert other.params["public_id"] != payload.params["public_id"]
    assert other.params["signature"] != payload.params["signature"]


def test_signer_refuses_without_credentials() -> None:
    with pytest.raises(ProviderError, match="cloudinary"):
        CloudinarySigner(cloud_name="", api_key="", api_secret="")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cloud_name": "c", "api_key": "", "api_secret": "s"},
        {"cloud_name": "", "api_key": "k", "api_secret": "s"},
        {"cloud_name": "c", "api_key": "k", "api_secret": ""},
    ],
)
def test_signer_refuses_with_partial_credentials(kwargs: dict[str, str]) -> None:
    # All-empty is the easy case; a real misconfiguration is more likely to
    # leave one field blank (e.g. secret missing from the environment) while
    # the others are set, so that must fail loudly too.
    with pytest.raises(ProviderError, match="cloudinary"):
        CloudinarySigner(**kwargs)


def test_delivery_url_changes_with_the_clock_and_embeds_ttl() -> None:
    # A signed delivery URL that didn't actually expire would defeat the
    # point of using type=authenticated in the first place.
    clock = {"now": 1700000000}
    signer = CloudinarySigner(
        cloud_name="c", api_key="k", api_secret="s", clock=lambda: clock["now"]
    )
    first = signer.delivery_url("vault/u1/i1", ttl_s=60)
    clock["now"] += 1
    second = signer.delivery_url("vault/u1/i1", ttl_s=60)
    assert first != second
    assert "ttl_s=60" in first


def test_fake_verifies_only_what_it_was_given() -> None:
    fake = FakeCloudinary()
    fake.assets["vault/u1/i1"] = {"version": 3, "format": "jpg", "bytes": 10}
    assert fake.verify("vault/u1/i1") is not None
    assert fake.verify("vault/u1/nope") is None


@pytest.mark.asyncio
async def test_fake_delete_removes_the_asset() -> None:
    fake = FakeCloudinary()
    fake.assets["vault/u1/i1"] = {"version": 1, "format": "jpg", "bytes": 10}
    assert await fake.delete("vault/u1/i1") is True
    assert fake.verify("vault/u1/i1") is None


# --------------------------------------------------------------------------- #
# CloudinarySigner.verify / delete — the trust boundary itself.
#
# This is the one piece of logic that decides whether FRIDAY believes a phone's
# "I uploaded it" claim, so its branching (found / not found / network error)
# is worth pinning down with respx rather than only exercising via the fakes.
# --------------------------------------------------------------------------- #


@respx.mock
def test_verify_returns_metadata_when_cloudinary_confirms_the_asset() -> None:
    signer = CloudinarySigner(cloud_name="c", api_key="k", api_secret="s", clock=lambda: 1)
    route = respx.get(
        "https://api.cloudinary.com/v1_1/c/resources/image/authenticated/vault/u1/i1"
    ).mock(return_value=httpx.Response(200, json={"bytes": 1234, "format": "jpg", "version": 7}))

    body = signer.verify("vault/u1/i1")

    assert body == {"bytes": 1234, "format": "jpg", "version": 7}
    # The Admin API call must itself be authenticated with key/secret, not left open.
    assert route.calls.last.request.headers["authorization"].startswith("Basic ")


@respx.mock
def test_verify_returns_none_when_cloudinary_has_no_such_asset() -> None:
    # A phone claiming an upload Cloudinary never received must not be trusted —
    # this is the exact lie the verify step exists to catch.
    signer = CloudinarySigner(cloud_name="c", api_key="k", api_secret="s", clock=lambda: 1)
    respx.get("https://api.cloudinary.com/v1_1/c/resources/image/authenticated/vault/u1/nope").mock(
        return_value=httpx.Response(404, json={"error": {"message": "not found"}})
    )

    assert signer.verify("vault/u1/nope") is None


@respx.mock
def test_verify_returns_none_on_network_failure() -> None:
    signer = CloudinarySigner(cloud_name="c", api_key="k", api_secret="s", clock=lambda: 1)
    respx.get("https://api.cloudinary.com/v1_1/c/resources/image/authenticated/vault/u1/i1").mock(
        side_effect=httpx.ConnectError("boom")
    )

    assert signer.verify("vault/u1/i1") is None


@respx.mock
async def test_delete_returns_true_only_on_cloudinarys_ok_result() -> None:
    signer = CloudinarySigner(cloud_name="c", api_key="k", api_secret="s", clock=lambda: 1)
    route = respx.post("https://api.cloudinary.com/v1_1/c/image/destroy").mock(
        return_value=httpx.Response(200, json={"result": "ok"})
    )

    assert await signer.delete("vault/u1/i1") is True
    sent = dict(httpx.QueryParams(route.calls.last.request.content.decode()))
    assert sent["public_id"] == "vault/u1/i1"
    assert sent["signature"] == sign_params(
        {"public_id": "vault/u1/i1", "timestamp": 1, "type": "authenticated"}, "s"
    )


@respx.mock
async def test_delete_returns_false_when_cloudinary_reports_not_found() -> None:
    # "not found" from Cloudinary must not be reported as a successful delete.
    signer = CloudinarySigner(cloud_name="c", api_key="k", api_secret="s", clock=lambda: 1)
    respx.post("https://api.cloudinary.com/v1_1/c/image/destroy").mock(
        return_value=httpx.Response(200, json={"result": "not found"})
    )

    assert await signer.delete("vault/u1/i1") is False


@respx.mock
async def test_delete_returns_false_on_network_failure() -> None:
    signer = CloudinarySigner(cloud_name="c", api_key="k", api_secret="s", clock=lambda: 1)
    respx.post("https://api.cloudinary.com/v1_1/c/image/destroy").mock(
        side_effect=httpx.ConnectError("boom")
    )

    assert await signer.delete("vault/u1/i1") is False
