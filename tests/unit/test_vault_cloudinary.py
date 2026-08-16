"""Unit tests for the Cloudinary signer, fake, and verifier."""

from __future__ import annotations

import hashlib

import httpx
import pytest
import respx

from friday.errors import ProviderError
from friday.vault.cloudinary import CloudinarySigner, FakeCloudinary, sign_params
from friday.vault.models import CloudinaryAsset


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
    # No `folder` param: Cloudinary prepends folder as a prefix to public_id,
    # so sending both would double the vault/{owner} segment in storage.
    assert "folder" not in payload.params
    assert payload.params["signature"] == sign_params(
        {
            "timestamp": 1700000000,
            "public_id": "vault/u1/i1",
            "type": "authenticated",
        },
        "secret",
    )


def test_signer_upload_public_id_has_no_doubled_folder_prefix() -> None:
    # Regression: a live Cloudinary probe found that sending both `folder` and
    # a fully-qualified `public_id` makes Cloudinary glue the folder prefix on
    # a second time, storing the asset at vault/u1/vault/u1/i1 instead of
    # vault/u1/i1 — which then makes every later verify() miss it.
    signer = CloudinarySigner(cloud_name="c", api_key="k", api_secret="s", clock=lambda: 1)
    payload = signer.upload_params(owner_uid="u1", item_id="i1")
    assert payload.params["public_id"] == "vault/u1/i1"
    assert payload.params["public_id"].count("vault/u1") == 1


def test_signer_scopes_the_public_id_to_owner_and_item() -> None:
    # The public_id is the only thing standing between one user's signed
    # upload and writing into another user's folder — a signature that didn't
    # actually bind owner_uid/item_id would let a client replay it anywhere.
    signer = CloudinarySigner(cloud_name="c", api_key="k", api_secret="s", clock=lambda: 1)
    payload = signer.upload_params(owner_uid="alice", item_id="i1")
    assert payload.params["public_id"] == "vault/alice/i1"
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


def test_fake_upload_params_carries_a_real_signature() -> None:
    # FakeCloudinary used to hand back a hard-coded "fake-signature" — a place
    # a real behavioural divergence from CloudinarySigner could hide, which is
    # exactly how the doubled-folder bug went unnoticed. Pin the fake down to
    # actually computing its signature the same way the real adapter does.
    fake = FakeCloudinary(cloud_name="fake", api_secret="fake-secret")
    payload = fake.upload_params(owner_uid="u1", item_id="i1")
    assert payload.params["signature"] == sign_params(
        {"timestamp": 0, "public_id": "vault/u1/i1", "type": "authenticated"},
        "fake-secret",
    )
    assert "folder" not in payload.params


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


def test_cloudinary_asset_round_trips_secure_url() -> None:
    # secure_url is what the backend now serves directly at read time (no
    # extra API call, no hand-rolled URL) — it must survive model round-trips
    # untouched, including the default for assets recorded before this field
    # existed.
    asset = CloudinaryAsset(
        public_id="vault/u1/i1",
        version=7,
        format="png",
        bytes=1234,
        secure_url="https://res.cloudinary.com/c/image/authenticated/s--sig--/v7/vault/u1/i1.png",
    )
    restored = CloudinaryAsset.model_validate(asset.model_dump())
    assert restored.secure_url == asset.secure_url
    assert CloudinaryAsset(public_id="x", version=1, format="jpg", bytes=1).secure_url == ""
