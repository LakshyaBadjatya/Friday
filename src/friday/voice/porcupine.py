"""Porcupine wake-word adapter (lazy, optional backend).

:class:`PorcupineWakeWord` is a second :class:`friday.voice.wake_word.WakeWordDetector`
backend alongside :class:`friday.voice.wake_word.OpenWakeWordDetector`, wrapping
Picovoice's `pvporcupine <https://github.com/Picovoice/porcupine>`_ engine.

Like the ``openwakeword`` adapter, the heavy ``pvporcupine`` import happens
*inside* ``__init__`` so importing this module never requires the backend and the
``uv`` lock stays unaffected (``pvporcupine`` is intentionally excluded). When the
backend is absent, a :class:`friday.errors.ProviderError` is raised with a clear
``make install-voice`` hint. Porcupine additionally needs a Picovoice access key;
a missing key surfaces the same typed error pointing at the key requirement. The
key is treated as a secret and is never logged.

The default detection threshold comes from application settings
(``FRIDAY_WAKE_WORD_THRESHOLD`` when present) and falls back to
:data:`friday.voice.wake_word.DEFAULT_WAKE_THRESHOLD`.
"""

from __future__ import annotations

from friday.config import get_settings
from friday.errors import ProviderError
from friday.voice.wake_word import DEFAULT_WAKE_THRESHOLD, WakeResult

# Install hint shared with the rest of the voice package: the Porcupine engine is
# optional, excluded from the uv lock, and additionally needs a Picovoice key.
_INSTALL_HINT = (
    "pvporcupine is not installed. Voice extras are optional and excluded from "
    "the uv lock; install them with `make install-voice` (or `pip install "
    "pvporcupine`) and set a Picovoice access key (FRIDAY_PICOVOICE_ACCESS_KEY)."
)

# Raised when the backend is present but no Picovoice access key was provided.
_MISSING_KEY_HINT = (
    "A Picovoice access key is required for the Porcupine wake-word engine. Set "
    "FRIDAY_PICOVOICE_ACCESS_KEY (get a free key at https://console.picovoice.ai)."
)


def _settings_threshold() -> float:
    """Return the configured wake threshold, or :data:`DEFAULT_WAKE_THRESHOLD`.

    Mirrors :func:`friday.voice.wake_word._settings_threshold`: reads
    ``wake_word_threshold`` off :func:`friday.config.get_settings` when the field
    exists, keeping this adapter independent of whether the config field has
    landed yet.
    """
    raw = getattr(get_settings(), "wake_word_threshold", DEFAULT_WAKE_THRESHOLD)
    return float(raw)


def _resolve_access_key(explicit: str | None) -> str:
    """Return the Picovoice access key: explicit arg, else settings, else empty.

    Reads ``picovoice_access_key`` off :func:`friday.config.get_settings`
    defensively (via ``getattr``) so this slice does not require the config field
    to have landed; the value may be a :class:`pydantic.SecretStr` (unwrapped via
    ``get_secret_value``) or a plain string. The key itself is never logged.
    """
    if explicit:
        return explicit
    configured = getattr(get_settings(), "picovoice_access_key", None)
    if configured is None:
        return ""
    secret_value = getattr(configured, "get_secret_value", None)
    if callable(secret_value):
        return str(secret_value())
    return str(configured)


class PorcupineWakeWord:
    """Real :class:`WakeWordDetector` backed by ``pvporcupine`` (lazy).

    The heavy ``pvporcupine`` import happens inside ``__init__`` so importing this
    module never requires the backend. When the backend is missing — or no
    Picovoice access key is available — a :class:`friday.errors.ProviderError` is
    raised with an actionable hint.
    """

    def __init__(
        self,
        keyword: str = "porcupine",
        access_key: str | None = None,
        threshold: float | None = None,
    ) -> None:
        """Construct the detector, creating the Porcupine engine handle.

        Args:
            keyword: Built-in Porcupine keyword to listen for (e.g. ``"porcupine"``,
                ``"jarvis"``); passed through as ``keywords=[keyword]``.
            access_key: Picovoice access key; ``None``/empty falls back to the
                ``picovoice_access_key`` setting. The key is never logged.
            threshold: Detection threshold; defaults to the configured wake
                threshold (or :data:`DEFAULT_WAKE_THRESHOLD`).

        Raises:
            ProviderError: If ``pvporcupine`` is not installed, or no Picovoice
                access key is available.
        """
        # Record the threshold first so it is set regardless of which guard fires.
        self.threshold = _settings_threshold() if threshold is None else float(threshold)
        self._keyword = keyword

        key = _resolve_access_key(access_key)
        if not key:
            raise ProviderError(_MISSING_KEY_HINT)

        try:
            # Optional voice backend: excluded from the uv lock, so mypy has no
            # stub for it; lazily imported here and guarded by the ImportError.
            import pvporcupine  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ProviderError(_INSTALL_HINT) from exc

        self._pvporcupine = pvporcupine
        self._engine = pvporcupine.create(access_key=key, keywords=[keyword])

    def detect(self, frame: bytes) -> WakeResult:
        """Evaluate ``frame`` with the Porcupine engine and return the result.

        Porcupine's ``process`` consumes int16 PCM samples and returns the index
        of the detected keyword (``-1`` for no detection). A non-negative index is
        a confident hit; we report a binary score saturated against the threshold.
        """
        import numpy as np  # type: ignore[import-not-found]  # noqa: PLC0415

        pcm = np.frombuffer(frame, dtype=np.int16)
        index = int(self._engine.process(pcm))
        detected = index >= 0
        score = 1.0 if detected else 0.0
        return WakeResult(detected=detected, score=score)
