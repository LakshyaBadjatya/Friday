"""Lazy ``firebase-admin`` initialisation for the circle's persistent backend.

A single named ``firebase-admin`` app and its Firestore client are built on first
use from the service-account credential in settings (either the raw JSON string or
a filesystem path to it). Everything is lazy and failure-tolerant: with no
credential — or no SDK installed, or a bad credential — :func:`get_backend` returns
``None`` and callers fall back to the in-memory stores. So the offline build needs
neither ``firebase-admin`` nor any secret, and a misconfigured deployment degrades
to non-persistent rather than crashing on import.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

#: A fixed app name so we never clash with a ``[DEFAULT]`` app another module made.
_APP_NAME = "friday-circle"


@dataclass
class FirebaseBackend:
    """A configured ``firebase-admin`` app and its Firestore client."""

    app: Any
    firestore: Any


# Cache by a digest of the credential so a test swapping credentials rebuilds, but
# the normal single-credential process initialises exactly once.
_CACHE: dict[str, FirebaseBackend] = {}


def _credential(raw: str) -> Any:
    """Build a ``credentials.Certificate`` from raw JSON or a path."""
    from firebase_admin import credentials  # noqa: PLC0415

    text = raw.strip()
    if text.startswith("{"):
        return credentials.Certificate(json.loads(text))
    return credentials.Certificate(text)


def get_backend(
    service_account: str | None, project_id: str = ""
) -> FirebaseBackend | None:
    """Return a cached :class:`FirebaseBackend`, or ``None`` if unconfigured.

    ``service_account`` is the raw service-account JSON or a path to it. Any
    failure (missing SDK, unparseable credential, init error) yields ``None``.
    """
    if not service_account:
        return None
    key = hashlib.sha256(service_account.encode("utf-8")).hexdigest()
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    try:
        import firebase_admin  # noqa: PLC0415
        from firebase_admin import firestore  # noqa: PLC0415
    except ImportError:
        return None
    try:
        cred = _credential(service_account)
        options = {"projectId": project_id} if project_id else None
        try:
            app = firebase_admin.get_app(_APP_NAME)
        except ValueError:
            app = firebase_admin.initialize_app(cred, options, name=_APP_NAME)
        client = firestore.client(app)
    except Exception:  # noqa: BLE001 - any init failure -> fall back to in-memory
        return None
    backend = FirebaseBackend(app=app, firestore=client)
    _CACHE[key] = backend
    return backend
