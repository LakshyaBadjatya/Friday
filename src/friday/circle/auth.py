"""The auth seam: turn a bearer token into a uid.

In production a Firebase ID token is verified via ``firebase-admin``
(:class:`FirebaseTokenVerifier`); in tests/dev a static token->uid map stands in.
:func:`resolve_caller` tries the verifier first, then the map, so a real Google
login and a dev/Siri device token both resolve through one code path. The
``firebase-admin`` import is lazy and failures degrade to ``None`` (unknown
caller) so the offline build needs neither the SDK nor credentials.
"""

from __future__ import annotations

from typing import Protocol


class TokenVerifier(Protocol):
    """Verifies an opaque token, returning the caller's uid or ``None``."""

    def verify(self, id_token: str) -> str | None: ...


class FakeTokenVerifier:
    """A static token->uid verifier for tests and local development."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = dict(mapping)

    def verify(self, id_token: str) -> str | None:
        return self._mapping.get(id_token)


class FirebaseTokenVerifier:
    """Verify a Firebase ID token via ``firebase-admin`` (imported lazily).

    The SDK and a configured app are only present in a real deployment; without
    either — or for any invalid token — :meth:`verify` returns ``None`` rather
    than raising, so a misconfigured/offline build simply treats the caller as
    unknown instead of erroring.
    """

    def __init__(self, app: object | None = None) -> None:
        self._app = app

    def verify(self, id_token: str) -> str | None:
        try:
            from firebase_admin import auth as firebase_auth  # noqa: PLC0415
        except ImportError:
            return None
        try:
            decoded = firebase_auth.verify_id_token(id_token, app=self._app)
        except Exception:  # noqa: BLE001 - any verification failure -> unknown caller
            return None
        uid = decoded.get("uid")
        return uid if isinstance(uid, str) else None


def resolve_caller(
    auth_header: str | None,
    *,
    verifier: TokenVerifier | None,
    identities: dict[str, str] | None,
) -> str | None:
    """Resolve an ``Authorization`` header to a uid, or ``None`` if unknown."""
    if not auth_header or not auth_header.lower().startswith("bearer "):
        return None
    token = auth_header[7:].strip()
    if not token:
        return None
    if verifier is not None:
        uid = verifier.verify(token)
        if uid is not None:
            return uid
    if identities is not None:
        mapped = identities.get(token)
        if isinstance(mapped, str):
            return mapped
    return None
