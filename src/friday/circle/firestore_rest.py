"""Keyless Firestore access for the Siri path — act AS the caller via their token.

The web app + HUD talk to Firestore directly; this lets the Siri backend read/write
that SAME data without a service account, by using the caller's own Firebase
credential. A long-lived refresh token (stable enough to live in a Shortcut) is
exchanged for a short-lived ID token, which authorises Firestore REST calls under
the project's security rules. Everything is lazy + failure-tolerant: any error
returns ``None``/``False`` so the Siri path falls back to the orchestrator and the
live endpoint can never break.

Firestore REST encodes values with type tags (``stringValue``, ``doubleValue``,
``mapValue``…); :func:`_decode` / :func:`_encode` translate to/from plain Python.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from friday.config import get_settings

#: Public Firebase web config (NOT a secret — access is governed by security rules);
#: sourced from settings/.env so the literal is never committed to source.
_settings = get_settings()
_PROJECT = _settings.firebase_project_id or "lakufriday"
_API_KEY = _settings.firebase_web_api_key
_DOCS = (
    f"https://firestore.googleapis.com/v1/projects/{_PROJECT}"
    "/databases/(default)/documents"
)
_TOKEN_URL = f"https://securetoken.googleapis.com/v1/token?key={_API_KEY}"
_TIMEOUT = 8


def _http(url: str, *, method: str, headers: dict[str, str], body: bytes | None) -> Any:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, TimeoutError):
        return None


def _decode(value: dict[str, Any]) -> Any:
    """Turn one Firestore typed value into a plain Python value."""
    if "stringValue" in value:
        return value["stringValue"]
    if "integerValue" in value:
        return int(value["integerValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "booleanValue" in value:
        return bool(value["booleanValue"])
    if "timestampValue" in value:
        return value["timestampValue"]
    if "mapValue" in value:
        return {
            k: _decode(v)
            for k, v in value["mapValue"].get("fields", {}).items()
        }
    if "arrayValue" in value:
        return [_decode(v) for v in value["arrayValue"].get("values", [])]
    if "nullValue" in value:
        return None
    return None


def _encode(value: Any) -> dict[str, Any]:
    """Turn a plain Python value into a Firestore typed value."""
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int):
        return {"integerValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, dict):
        return {"mapValue": {"fields": {k: _encode(v) for k, v in value.items()}}}
    if isinstance(value, list):
        return {"arrayValue": {"values": [_encode(v) for v in value]}}
    return {"stringValue": str(value)}


def _fields(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: _decode(v) for k, v in doc.get("fields", {}).items()}


def _uid_from_jwt(token: str) -> str | None:
    """Decode (not verify) a Firebase ID token's payload for the uid."""
    import base64  # noqa: PLC0415

    try:
        part = token.split(".")[1]
        part += "=" * (-len(part) % 4)
        payload = json.loads(base64.urlsafe_b64decode(part).decode("utf-8"))
    except (ValueError, IndexError):
        return None
    uid = payload.get("user_id") or payload.get("sub") or payload.get("uid")
    return uid if isinstance(uid, str) else None


def resolve_token(token: str) -> tuple[str, str] | None:
    """Return ``(id_token, uid)`` for a bearer that's an ID token OR a refresh token.

    A Firebase ID token is a 3-part JWT (used as-is); anything else is treated as a
    refresh token and exchanged. ``None`` on any failure.
    """
    token = token.strip()
    if token.count(".") == 2:
        uid = _uid_from_jwt(token)
        return (token, uid) if uid else None
    body = urllib.parse.urlencode(
        {"grant_type": "refresh_token", "refresh_token": token}
    ).encode()
    data = _http(
        _TOKEN_URL,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        body=body,
    )
    if not isinstance(data, dict):
        return None
    id_token = data.get("id_token")
    uid = data.get("user_id")
    if isinstance(id_token, str) and isinstance(uid, str):
        return id_token, uid
    return None


class FirestoreRest:
    """A tiny Firestore REST client scoped to one caller's ID token."""

    def __init__(self, id_token: str) -> None:
        self._headers = {
            "Authorization": "Bearer " + id_token,
            "Content-Type": "application/json",
        }

    def get(self, path: str) -> dict[str, Any] | None:
        """Read one document's fields (``path`` like ``groups/g1/members/u1``)."""
        data = _http(f"{_DOCS}/{path}", method="GET", headers=self._headers, body=None)
        return _fields(data) if isinstance(data, dict) and "fields" in data else None

    def list(self, collection_path: str) -> list[dict[str, Any]]:
        """List a collection's documents as field dicts (best-effort, one page)."""
        data = _http(
            f"{_DOCS}/{collection_path}?pageSize=300",
            method="GET",
            headers=self._headers,
            body=None,
        )
        if not isinstance(data, dict):
            return []
        return [_fields(d) for d in data.get("documents", [])]

    def patch(self, path: str, fields: dict[str, Any]) -> bool:
        """Merge-update the given fields on a document."""
        mask = "&".join(
            f"updateMask.fieldPaths={urllib.parse.quote(k)}" for k in fields
        )
        body = json.dumps({"fields": {k: _encode(v) for k, v in fields.items()}})
        data = _http(
            f"{_DOCS}/{path}?{mask}",
            method="PATCH",
            headers=self._headers,
            body=body.encode(),
        )
        return isinstance(data, dict)

    def create(self, collection_path: str, fields: dict[str, Any]) -> bool:
        """Create a document with an auto id in the collection."""
        body = json.dumps({"fields": {k: _encode(v) for k, v in fields.items()}})
        data = _http(
            f"{_DOCS}/{collection_path}",
            method="POST",
            headers=self._headers,
            body=body.encode(),
        )
        return isinstance(data, dict)
