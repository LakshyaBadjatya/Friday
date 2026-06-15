"""Speaker verification (voiceprint): owner recognition, advisory by default.

This module provides FRIDAY's *speaker verification* boundary — the ability to
recognize the owner's voice for personalization and to optionally gate the most
sensitive actions. It deliberately stays out of the way of everyday use:

**Advisory by default.** Anyone can wake FRIDAY and talk to it; owner
recognition is purely advisory. :meth:`OwnerIdentity.verify` never raises and
never blocks — it returns a typed :class:`OwnerVerification` and lets the caller
decide what to do. Only callers that *opt in* (by inspecting the result, or by
calling :meth:`OwnerIdentity.require_owner`) gate behavior on it. The intent is
that voice recognition enhances personalization and protects sensitive actions
without ever locking a guest out of the assistant.

The pieces:

* :class:`SpeakerVerifier` — the runtime-checkable protocol. ``enroll`` turns a
  handful of voice samples into an opaque *profile blob*; ``score`` compares a
  fresh sample against a profile and returns a ``[0, 1]`` similarity.
* :class:`FakeVoiceprint` — a deterministic verifier for tests: it scores
  ``1.0`` for the exact bytes it was enrolled on and ``~0`` for anything else.
* :class:`ResemblyzerVerifier` — the real adapter, lazily importing
  ``resemblyzer``/``numpy`` inside its methods so importing this module never
  needs the heavy backend. A missing backend raises a clear
  :class:`friday.errors.ProviderError` with an install hint.
* :class:`OwnerIdentity` — wraps a verifier + the owner's profile, exposing
  :meth:`is_owner` against a threshold while keeping verification advisory.
* :class:`EnrollmentStore` — persists the owner profile blob to a path.

No heavy voice library is imported at module top level, and this module takes
every dependency as a parameter (no application config/singleton import).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from friday.errors import ProviderError

# Default similarity threshold above which a sample is treated as the owner.
# Conservative enough to avoid false-accepts while staying advisory (callers
# decide whether to act on a positive match).
DEFAULT_OWNER_THRESHOLD = 0.75

# Install hint shared with the rest of the voice package: the resemblyzer
# speaker-embedding backend is optional and excluded from the uv lock.
_INSTALL_HINT = (
    "resemblyzer is not installed. Voice extras are optional and excluded from "
    "the uv lock; install them with `make install-voice` (or `pip install "
    "resemblyzer`)."
)


@runtime_checkable
class SpeakerVerifier(Protocol):
    """Contract for enrolling a voice and scoring samples against a profile.

    A *profile blob* is opaque bytes produced by :meth:`enroll`; its internal
    format is the verifier's business. :meth:`score` returns a ``[0, 1]``
    similarity where higher means "more likely the same speaker".
    """

    def enroll(self, samples: list[bytes]) -> bytes:
        """Build an opaque profile blob from one or more voice ``samples``.

        Args:
            samples: Raw audio samples (e.g. PCM/WAV bytes) of the speaker to
                enroll. At least one is required.

        Returns:
            An opaque profile blob to persist and later pass to :meth:`score`.
        """
        ...

    def score(self, sample: bytes, profile: bytes) -> float:
        """Return the ``[0, 1]`` similarity of ``sample`` to ``profile``."""
        ...


class OwnerVerification(BaseModel):
    """Typed, advisory result of comparing a sample against the owner profile.

    ``is_owner`` is ``score >= threshold``. This is *advisory*: producing it
    never blocks anything — callers inspect it and decide.
    """

    is_owner: bool
    score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)


class FakeVoiceprint:
    """A deterministic :class:`SpeakerVerifier` for tests.

    Enrollment fingerprints each sample with a stable hash; :meth:`score`
    returns ``1.0`` when the scored sample matches one of the enrolled
    fingerprints and ``0.0`` otherwise. This makes "the enrolled owner sample"
    score high and any other sample score ``~0`` with no randomness, no model,
    and no I/O.
    """

    # Stable separator so the joined profile blob round-trips unambiguously.
    _SEP = b"\n"

    def enroll(self, samples: list[bytes]) -> bytes:
        """Return a profile blob of the enrolled samples' fingerprints.

        Args:
            samples: Raw audio samples to enroll; at least one is required.

        Returns:
            A profile blob: newline-joined hex digests of each sample.

        Raises:
            ValueError: If ``samples`` is empty.
        """
        if not samples:
            raise ValueError("enroll requires at least one sample")
        digests = [self._fingerprint(sample) for sample in samples]
        return self._SEP.join(digests)

    def score(self, sample: bytes, profile: bytes) -> float:
        """Return ``1.0`` if ``sample`` matches an enrolled fingerprint, else ``0.0``."""
        enrolled = set(profile.split(self._SEP)) if profile else set()
        return 1.0 if self._fingerprint(sample) in enrolled else 0.0

    @staticmethod
    def _fingerprint(sample: bytes) -> bytes:
        """Return a stable hex digest of ``sample`` (deterministic, no salt)."""
        return hashlib.sha256(sample).hexdigest().encode("ascii")


class ResemblyzerVerifier:
    """Real :class:`SpeakerVerifier` backed by ``resemblyzer`` (lazy).

    The heavy ``resemblyzer``/``numpy`` imports happen *inside* the methods so
    importing this module never requires the backend and the ``uv`` lock stays
    unaffected (``resemblyzer`` is intentionally excluded). When the backend is
    missing, a :class:`friday.errors.ProviderError` is raised with an install
    hint.

    A profile blob is the mean speaker embedding serialized as raw
    little-endian ``float32`` bytes; :meth:`score` reconstructs it and returns
    the cosine similarity (clamped to ``[0, 1]``) against a fresh sample's
    embedding.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        """Construct the verifier.

        Args:
            sample_rate: Sample rate (Hz) of the raw PCM samples passed to
                :meth:`enroll`/:meth:`score`. The default matches the canonical
                voice-pipeline format.
        """
        self._sample_rate = sample_rate

    def enroll(self, samples: list[bytes]) -> bytes:
        """Embed each sample and return the mean embedding as ``float32`` bytes.

        Args:
            samples: Raw audio samples of the speaker to enroll; at least one.

        Returns:
            The mean speaker embedding serialized as little-endian ``float32``.

        Raises:
            ValueError: If ``samples`` is empty.
            ProviderError: If ``resemblyzer``/``numpy`` is not installed.
        """
        if not samples:
            raise ValueError("enroll requires at least one sample")
        np = self._numpy()
        encoder = self._encoder()
        embeddings = [encoder.embed_utterance(self._to_wav(np, sample)) for sample in samples]
        mean = np.mean(np.stack(embeddings), axis=0)
        return bytes(np.asarray(mean, dtype="<f4").tobytes())

    def score(self, sample: bytes, profile: bytes) -> float:
        """Return the cosine similarity of ``sample`` to ``profile`` in ``[0, 1]``.

        Raises:
            ProviderError: If ``resemblyzer``/``numpy`` is not installed.
        """
        np = self._numpy()
        encoder = self._encoder()
        ref = np.frombuffer(profile, dtype="<f4")
        emb = np.asarray(encoder.embed_utterance(self._to_wav(np, sample)), dtype="<f4")
        denom = float(np.linalg.norm(ref) * np.linalg.norm(emb))
        if denom == 0.0:
            return 0.0
        cosine = float(np.dot(ref, emb) / denom)
        # Cosine is in [-1, 1]; clamp to the [0, 1] similarity contract.
        return max(0.0, min(1.0, cosine))

    # -- lazy backend helpers --------------------------------------------- #
    @staticmethod
    def _numpy() -> Any:
        """Return the lazily-imported ``numpy`` module, or raise a typed error."""
        try:
            import numpy as np  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ProviderError(_INSTALL_HINT) from exc
        return np

    @staticmethod
    def _encoder() -> Any:
        """Return a fresh resemblyzer ``VoiceEncoder``, or raise a typed error."""
        try:
            from resemblyzer import VoiceEncoder  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ProviderError(_INSTALL_HINT) from exc
        return VoiceEncoder()

    @staticmethod
    def _to_wav(np: Any, sample: bytes) -> Any:
        """Decode raw int16 PCM ``sample`` to the float32 waveform the encoder wants."""
        pcm = np.frombuffer(sample, dtype="<i2").astype("float32")
        # Normalize int16 range to [-1, 1] as resemblyzer expects.
        return pcm / 32768.0


class OwnerIdentity:
    """Advisory owner recognition over a :class:`SpeakerVerifier` + profile.

    Wraps a verifier and the owner's enrolled profile blob, exposing
    :meth:`is_owner` (a bool against ``threshold``) and :meth:`verify` (a typed
    :class:`OwnerVerification`).

    **Advisory by default.** Anyone can wake/talk to FRIDAY; owner recognition
    is for personalization and only gates the *most sensitive* actions when a
    caller opts in. :meth:`verify` never raises and never blocks — it returns a
    result and lets the caller decide. Callers that want a hard gate call
    :meth:`require_owner`, which is the *only* method here that raises.
    """

    def __init__(
        self,
        verifier: SpeakerVerifier,
        profile: bytes,
        *,
        threshold: float = DEFAULT_OWNER_THRESHOLD,
    ) -> None:
        """Bind a verifier to the owner profile.

        Args:
            verifier: The injected :class:`SpeakerVerifier` implementation.
            profile: The owner's enrolled profile blob (from ``verifier.enroll``).
            threshold: Similarity at or above which a sample counts as the owner.
        """
        self._verifier = verifier
        self._profile = profile
        self.threshold = float(threshold)

    def score(self, sample: bytes) -> float:
        """Return the raw ``[0, 1]`` similarity of ``sample`` to the owner profile."""
        return self._verifier.score(sample, self._profile)

    def verify(self, sample: bytes) -> OwnerVerification:
        """Return an advisory :class:`OwnerVerification` for ``sample``.

        Never raises and never blocks — this is purely informational. Callers
        inspect ``is_owner`` and decide whether to personalize or gate.
        """
        score = self.score(sample)
        return OwnerVerification(
            is_owner=score >= self.threshold,
            score=score,
            threshold=self.threshold,
        )

    def is_owner(self, sample: bytes) -> bool:
        """Return whether ``sample`` scores at or above ``threshold`` (advisory)."""
        return self.verify(sample).is_owner

    def require_owner(self, sample: bytes) -> OwnerVerification:
        """Opt-in hard gate: raise :class:`PermissionError` when not the owner.

        Use this only for the most sensitive actions where a caller *wants* to
        block non-owners. The advisory default (:meth:`verify`/:meth:`is_owner`)
        never blocks; this method is the explicit opt-in.

        Raises:
            friday.errors.PermissionError: If ``sample`` is not the owner.
        """
        # Imported here (function-local) to keep the advisory default path free
        # of any hard-gate machinery and to avoid shadowing the builtin above.
        from friday.errors import PermissionError as FridayPermissionError  # noqa: PLC0415

        result = self.verify(sample)
        if not result.is_owner:
            raise FridayPermissionError(
                f"speaker not recognized as owner "
                f"(score {result.score:.3f} < threshold {result.threshold:.3f})"
            )
        return result


class EnrollmentStore:
    """Persist the owner's profile blob to a single filesystem path.

    A tiny, local-first store: the profile blob is written verbatim (binary) to
    ``path`` and read back the same way. Parent directories are created on save.
    """

    def __init__(self, path: str | Path) -> None:
        """Bind the store to ``path`` (the file holding the profile blob)."""
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """The filesystem path this store reads/writes the profile blob at."""
        return self._path

    def exists(self) -> bool:
        """Return whether a saved profile blob is present at :attr:`path`."""
        return self._path.is_file()

    def save(self, profile: bytes) -> None:
        """Write ``profile`` to :attr:`path`, creating parent directories."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(profile)

    def load(self) -> bytes:
        """Read and return the saved profile blob.

        Raises:
            FileNotFoundError: If no profile has been saved at :attr:`path`.
        """
        return self._path.read_bytes()
