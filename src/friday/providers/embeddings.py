"""Embedding provider abstraction, a deterministic fake, and the NVIDIA adapter.

This module owns the typed *embedding* boundary for FRIDAY, mirroring the LLM
boundary in :mod:`friday.providers.llm`:

* :class:`EmbeddingProvider` — the structural async ``embed`` contract that the
  persistent vector store depends on.
* :class:`FakeEmbeddings` — a deterministic, offline, dependency-free provider
  that hashes each input text into a fixed-dimension, unit-normalized vector. It
  needs no API key and touches no network, so every test embeds with it.
* :class:`NvidiaEmbeddings` — the real adapter over the OpenAI-compatible NVIDIA
  NIM ``/embeddings`` endpoint, lazily constructing an ``openai`` client and
  wrapping every transport error in :class:`~friday.errors.ProviderError`.

IMPORTANT: alongside :mod:`friday.providers.llm`, this is the **only** module in
the codebase permitted to import an LLM SDK (``openai``). All business logic
depends on the :class:`EmbeddingProvider` abstraction so the SDK stays isolated
behind ``providers/``.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from friday.errors import ProviderError

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# Default embedding dimension for the deterministic fake. Small enough to keep
# tests fast, large enough that distinct texts map to distinct vectors.
DEFAULT_FAKE_DIM = 64


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Structural contract for turning text into dense vectors.

    Implementations map a batch of ``texts`` to one float vector per text, in
    input order. The method is async so a real, network-backed provider can be
    awaited; the fake satisfies it trivially.
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in order."""
        ...


class FakeEmbeddings:
    """A deterministic, offline embedding provider for tests.

    Each text is hashed (SHA-256, expanded with a counter for longer vectors)
    into ``dim`` floats which are then L2-normalized to a unit vector. The
    mapping is pure: the same text always yields the same vector, across calls
    and across instances with the same ``dim``, and distinct texts yield
    distinct vectors with overwhelming probability. No key, no network.
    """

    def __init__(self, dim: int = DEFAULT_FAKE_DIM) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self._dim = dim

    @property
    def dim(self) -> int:
        """The fixed dimensionality of every vector this provider returns."""
        return self._dim

    def _vector(self, text: str) -> list[float]:
        """Hash ``text`` into a ``dim``-length, unit-normalized float vector."""
        # Produce enough deterministic bytes by hashing (text || counter) until
        # we have one 64-bit double's worth of entropy per dimension.
        raw = bytearray()
        counter = 0
        needed = self._dim * 8
        while len(raw) < needed:
            digest = hashlib.sha256(f"{text}\x00{counter}".encode()).digest()
            raw.extend(digest)
            counter += 1

        vector: list[float] = []
        for i in range(self._dim):
            (bits,) = struct.unpack_from("<Q", raw, i * 8)
            # Map the 64-bit unsigned int into a centered [-1.0, 1.0) float so
            # vectors span the space rather than living in one orthant.
            vector.append((bits / 2**64) * 2.0 - 1.0)

        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:  # pragma: no cover - astronomically unlikely
            return vector
        return [component / norm for component in vector]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one deterministic unit vector per text (empty -> empty)."""
        return [self._vector(text) for text in texts]


class NvidiaEmbeddings:
    """Real :class:`EmbeddingProvider` over the OpenAI-compatible NVIDIA NIM API.

    Lazily constructs an ``openai`` :class:`AsyncOpenAI` client pointed at
    ``base_url`` and calls ``embeddings.create``. Every client/transport error
    is wrapped in :class:`~friday.errors.ProviderError` so callers depend only on
    FRIDAY's error family, not the SDK's.

    The client is built with ``max_retries=0`` so a misbehaving endpoint surfaces
    promptly rather than being silently retried; retry/fallback policy lives
    above this adapter.

    ``dim`` records the expected output dimensionality for the configured model
    so the vector store can be sized consistently; it is not sent on the wire.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        dim: int,
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._dim = dim
        # Lazy import keeps the openai SDK out of import-time for callers that
        # only ever use the fake, and confines the dependency to providers/.
        from openai import AsyncOpenAI

        self._client: AsyncOpenAI = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )

    @property
    def dim(self) -> int:
        """Expected dimensionality of vectors produced by the configured model."""
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` via NIM, mapping transport errors to ProviderError."""
        if not texts:
            return []

        from openai import (
            APIConnectionError,
            APITimeoutError,
            OpenAIError,
        )

        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=texts,
            )
        except (APITimeoutError, APIConnectionError) as exc:
            raise ProviderError(
                f"NVIDIA NIM embeddings request timed out or could not connect: {exc}"
            ) from exc
        except OpenAIError as exc:
            raise ProviderError(f"NVIDIA NIM embeddings request failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive transport guard
            raise ProviderError(f"NVIDIA NIM embeddings request failed: {exc}") from exc

        try:
            # The API may reorder; sort by ``index`` to restore input order.
            ordered = sorted(response.data, key=lambda item: item.index)
            return [list(item.embedding) for item in ordered]
        except (AttributeError, TypeError) as exc:
            raise ProviderError(
                f"NVIDIA NIM embeddings returned an unexpected payload: {exc}"
            ) from exc
