"""A thin ``httpx`` adapter over the n8n REST API.

:class:`N8nClient` mirrors the keyless-tool style of
:mod:`friday.tools.web_search`: a small async surface built on ``httpx``, with
typed failures rather than bare exceptions leaking out.

Surface:

* :meth:`N8nClient.is_up` — a best-effort liveness probe. It issues a ``GET`` to
  ``{base}/healthz`` (falling back to ``/rest/login``) and returns ``True`` only
  on a 2xx; *any* error (connect/timeout/transport/non-2xx) returns ``False`` so
  the caller can decide whether to offer the docker auto-start. It never raises.
* :meth:`N8nClient.import_workflow` — ``POST {base}/api/v1/workflows`` with the
  ``X-N8N-API-KEY`` header; returns the created workflow JSON or raises a typed
  :class:`N8nError`. When no API key is configured it raises a clear
  "n8n api key not set" error before touching the network.
* :meth:`N8nClient.list_workflows` — ``GET {base}/api/v1/workflows`` (same auth),
  returning the parsed ``data`` list.

SECURITY: the API key is held as a plain ``str`` here but originates from a
:class:`~pydantic.SecretStr` in config; it is ONLY ever placed in the
``X-N8N-API-KEY`` request header and is never logged (no ``logger`` call includes
it, and the base URL — never the key — is what appears in error messages).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from friday.errors import FridayError

logger = logging.getLogger("friday.n8n.client")

#: Default per-request wall-clock budget (seconds) for every n8n REST call.
_DEFAULT_TIMEOUT = 15.0


class N8nError(FridayError):
    """An n8n REST call failed (missing key, transport error, or non-2xx)."""


class N8nClient:
    """Async ``httpx`` client for the subset of the n8n REST API FRIDAY uses.

    Args:
        base_url: Base URL of the n8n instance (e.g. ``http://localhost:5678``);
            a trailing slash is stripped so path joining is unambiguous.
        api_key: The n8n REST API key, or ``None`` when unset. Sent as
            ``X-N8N-API-KEY`` on authenticated calls; :meth:`import_workflow` /
            :meth:`list_workflows` raise a clear error when it is ``None``.
        timeout: Per-request wall-clock budget in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    @property
    def base_url(self) -> str:
        """The normalized base URL (no trailing slash)."""
        return self._base_url

    @property
    def has_api_key(self) -> bool:
        """Whether an API key is configured (without exposing it)."""
        return bool(self._api_key)

    def _auth_headers(self) -> dict[str, str]:
        """The authenticated-request headers; raises when no key is set.

        The key is placed ONLY here, in the ``X-N8N-API-KEY`` header — never in a
        log line or an error message.
        """
        if not self._api_key:
            raise N8nError(
                "n8n api key not set — configure FRIDAY_N8N_API_KEY to import "
                "workflows"
            )
        return {"X-N8N-API-KEY": self._api_key}

    async def is_up(self) -> bool:
        """Best-effort liveness probe; ``True`` only on a 2xx, never raises.

        Tries ``GET {base}/healthz`` first (n8n's unauthenticated health route);
        if that does not return a 2xx, falls back to ``GET {base}/rest/login``.
        Any connection/timeout/transport error — or a non-2xx from both — yields
        ``False`` so the service can offer to start n8n via docker.
        """
        for path in ("/healthz", "/rest/login"):
            url = f"{self._base_url}{path}"
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(url)
            except httpx.HTTPError as exc:
                logger.debug("n8n is_up probe %s failed: %s", path, exc)
                continue
            if response.is_success:
                return True
            logger.debug("n8n is_up probe %s returned HTTP %d", path, response.status_code)
        return False

    async def import_workflow(self, workflow: dict[str, Any]) -> dict[str, Any]:
        """``POST {base}/api/v1/workflows``; return the created workflow JSON.

        Raises :class:`N8nError` when no API key is configured (before any
        network I/O), on a transport error, or on a non-2xx response. On success
        the parsed JSON body of the created workflow is returned.
        """
        headers = self._auth_headers()
        url = f"{self._base_url}/api/v1/workflows"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=workflow, headers=headers)
        except httpx.HTTPError as exc:
            logger.warning("n8n import_workflow transport error: %s", exc)
            raise N8nError(f"n8n import request failed: {exc}") from exc

        if not response.is_success:
            logger.warning(
                "n8n import_workflow returned HTTP %d", response.status_code
            )
            raise N8nError(
                f"n8n import returned HTTP {response.status_code}: "
                f"{_safe_body(response)}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise N8nError(f"n8n import returned a non-JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise N8nError("n8n import returned an unexpected (non-object) body")
        return body

    async def list_workflows(self) -> list[dict[str, Any]]:
        """``GET {base}/api/v1/workflows``; return the parsed ``data`` list.

        Raises :class:`N8nError` when no API key is configured, on a transport
        error, or on a non-2xx response. n8n wraps the list in a ``{"data": [...]}``
        envelope; a bare list (older API) is also accepted.
        """
        headers = self._auth_headers()
        url = f"{self._base_url}/api/v1/workflows"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            logger.warning("n8n list_workflows transport error: %s", exc)
            raise N8nError(f"n8n list request failed: {exc}") from exc

        if not response.is_success:
            logger.warning(
                "n8n list_workflows returned HTTP %d", response.status_code
            )
            raise N8nError(
                f"n8n list returned HTTP {response.status_code}: "
                f"{_safe_body(response)}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise N8nError(f"n8n list returned a non-JSON body: {exc}") from exc
        if isinstance(body, dict):
            data = body.get("data", [])
            return [item for item in data if isinstance(item, dict)]
        if isinstance(body, list):
            return [item for item in body if isinstance(item, dict)]
        raise N8nError("n8n list returned an unexpected body shape")


def _safe_body(response: httpx.Response) -> str:
    """A short, key-free snippet of a response body for an error message.

    Truncated so a large/HTML error page does not bloat the raised error. Carries
    no FRIDAY secret (the API key is only ever in the request header).
    """
    text = response.text or ""
    snippet = text.strip().replace("\n", " ")
    return snippet[:200]
