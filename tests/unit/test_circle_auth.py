"""Unit tests for the circle auth seam (Firebase token verification).

``resolve_caller`` turns an ``Authorization: Bearer <token>`` header into a uid by
trying a :class:`TokenVerifier` (Firebase in production) first, then a static
token->uid map (dev/Siri device tokens). Everything is offline; the real
``FirebaseTokenVerifier`` degrades to ``None`` without the SDK/credentials.
"""

from __future__ import annotations

from friday.circle.auth import (
    FakeTokenVerifier,
    FirebaseTokenVerifier,
    resolve_caller,
)


def test_fake_verifier_maps_tokens() -> None:
    verifier = FakeTokenVerifier({"tok": "u-1"})
    assert verifier.verify("tok") == "u-1"
    assert verifier.verify("nope") is None


def test_resolve_caller_uses_the_verifier_first() -> None:
    verifier = FakeTokenVerifier({"id-token-abc": "u-firebase"})
    assert (
        resolve_caller("Bearer id-token-abc", verifier=verifier, identities=None)
        == "u-firebase"
    )


def test_resolve_caller_falls_back_to_the_identity_map() -> None:
    # No verifier -> use the dev/device token map.
    assert (
        resolve_caller("Bearer dev-tok", verifier=None, identities={"dev-tok": "u-dev"})
        == "u-dev"
    )
    # Verifier present but doesn't recognise the token -> still fall back.
    assert (
        resolve_caller(
            "Bearer dev-tok",
            verifier=FakeTokenVerifier({}),
            identities={"dev-tok": "u-dev"},
        )
        == "u-dev"
    )


def test_resolve_caller_rejects_bad_headers() -> None:
    assert resolve_caller(None, verifier=None, identities={}) is None
    assert resolve_caller("dev-tok", verifier=None, identities={"dev-tok": "u"}) is None
    assert resolve_caller("Bearer    ", verifier=None, identities={}) is None
    assert resolve_caller("Bearer unknown", verifier=None, identities={}) is None


def test_firebase_verifier_is_graceful_without_the_sdk() -> None:
    # firebase-admin isn't part of the offline build, and even if present a bogus
    # token fails verification -> None (never raises), so the caller is "unknown".
    assert FirebaseTokenVerifier().verify("not-a-real-token") is None
