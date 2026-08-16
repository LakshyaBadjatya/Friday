"""Cloudinary boundary: sign an upload, verify it landed, mint delivery URLs.

The phone never holds the API secret. It asks the backend for a signature scoped
to exactly one upload, uploads the bytes directly, and reports back; the backend
then independently confirms with Cloudinary's Admin API that the asset exists
before it will trust a single field of the report.

Everything uploads as ``type=authenticated``, so no vault asset is reachable by
guessing a URL — delivery URLs are signed per request and expire.

``httpx`` is imported lazily inside the networked methods, so importing this
module costs nothing and the offline test path never touches the network.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from friday.errors import ProviderError

#: Cloudinary drops these from the signature base string.
_UNSIGNED_KEYS = frozenset({"file", "cloud_name", "resource_type", "api_key", "signature"})

_MISSING_CREDS = (
    "cloudinary credentials are not configured; set FRIDAY_CLOUDINARY_CLOUD_NAME, "
    "FRIDAY_CLOUDINARY_API_KEY and FRIDAY_CLOUDINARY_API_SECRET"
)


def sign_params(params: dict[str, Any], api_secret: str) -> str:
    """Return Cloudinary's signature for ``params``.

    The documented algorithm: drop unsigned and empty keys, sort the remainder
    by key, join as ``k=v&k=v``, append the API secret, and SHA-1 the result.
    """
    signable = {k: v for k, v in sorted(params.items()) if k not in _UNSIGNED_KEYS and v != ""}
    base = "&".join(f"{k}={v}" for k, v in signable.items())
    return hashlib.sha1(f"{base}{api_secret}".encode(), usedforsecurity=False).hexdigest()


class UploadPayload(BaseModel):
    """Everything the phone needs to upload one asset, and nothing more."""

    url: str
    params: dict[str, Any]


@runtime_checkable
class CloudinaryProvider(Protocol):
    """The slice of Cloudinary the vault depends on."""

    def upload_params(self, *, owner_uid: str, item_id: str) -> UploadPayload: ...

    def verify(self, public_id: str) -> dict[str, Any] | None: ...

    async def delete(self, public_id: str) -> bool: ...

    def delivery_url(self, public_id: str, *, ttl_s: int = 300) -> str: ...


class FakeCloudinary:
    """A deterministic provider for tests — an in-memory asset table."""

    def __init__(self, cloud_name: str = "fake") -> None:
        self.cloud_name = cloud_name
        self.assets: dict[str, dict[str, Any]] = {}

    def upload_params(self, *, owner_uid: str, item_id: str) -> UploadPayload:
        public_id = f"vault/{owner_uid}/{item_id}"
        return UploadPayload(
            url=f"https://api.cloudinary.com/v1_1/{self.cloud_name}/image/upload",
            params={
                "api_key": "fake",
                "timestamp": 0,
                "public_id": public_id,
                "type": "authenticated",
                "signature": "fake-signature",
            },
        )

    def verify(self, public_id: str) -> dict[str, Any] | None:
        return self.assets.get(public_id)

    async def delete(self, public_id: str) -> bool:
        return self.assets.pop(public_id, None) is not None

    def delivery_url(self, public_id: str, *, ttl_s: int = 300) -> str:
        return f"https://fake.invalid/{public_id}?ttl={ttl_s}"


class CloudinarySigner:
    """The real provider. Signs locally; only verify/delete touch the network."""

    def __init__(
        self,
        *,
        cloud_name: str,
        api_key: str,
        api_secret: str,
        clock: Callable[[], int] = lambda: int(time.time()),
    ) -> None:
        if not (cloud_name and api_key and api_secret):
            raise ProviderError(_MISSING_CREDS)
        self.cloud_name = cloud_name
        self._key = api_key
        self._secret = api_secret
        self._clock = clock

    @property
    def _api_base(self) -> str:
        return f"https://api.cloudinary.com/v1_1/{self.cloud_name}"

    def upload_params(self, *, owner_uid: str, item_id: str) -> UploadPayload:
        """Signature and params for one authenticated upload.

        No ``folder`` param is sent: Cloudinary treats ``folder`` as a prefix
        it prepends to ``public_id``, so sending both would double the
        ``vault/{owner_uid}`` segment. The fully-qualified ``public_id`` alone
        already provides the folder scoping.
        """
        signable: dict[str, Any] = {
            "timestamp": self._clock(),
            "public_id": f"vault/{owner_uid}/{item_id}",
            "type": "authenticated",
        }
        params = dict(signable)
        params["api_key"] = self._key
        params["signature"] = sign_params(signable, self._secret)
        return UploadPayload(url=f"{self._api_base}/image/upload", params=params)

    def verify(self, public_id: str) -> dict[str, Any] | None:
        """Confirm the asset exists, returning its metadata or ``None``."""
        import httpx  # noqa: PLC0415

        url = f"{self._api_base}/resources/image/authenticated/{public_id}"
        try:
            resp = httpx.get(url, auth=(self._key, self._secret), timeout=10.0)
        except httpx.HTTPError:
            return None
        if resp.status_code != 200:
            return None
        body: dict[str, Any] = resp.json()
        return body

    async def delete(self, public_id: str) -> bool:
        """Destroy the asset; ``False`` when Cloudinary reports anything else."""
        import httpx  # noqa: PLC0415

        timestamp = self._clock()
        signable = {
            "public_id": public_id,
            "timestamp": timestamp,
            "type": "authenticated",
        }
        data = dict(signable)
        data["api_key"] = self._key
        data["signature"] = sign_params(signable, self._secret)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(f"{self._api_base}/image/destroy", data=data)
        except httpx.HTTPError:
            return False
        return resp.status_code == 200 and resp.json().get("result") == "ok"

    def delivery_url(self, public_id: str, *, ttl_s: int = 300) -> str:
        """A time-limited signed URL for an authenticated asset."""
        expires = self._clock() + ttl_s
        signature = sign_params({"public_id": public_id, "expires": expires}, self._secret)
        return (
            f"https://res.cloudinary.com/{self.cloud_name}/image/authenticated/"
            f"s--{signature[:8]}--/{public_id}?_a={expires}&ttl_s={ttl_s}"
        )
