"""Clipboard boundary, in-memory fake, and lazy real adapter.

* :class:`ClipboardProvider` — the runtime-checkable ``read``/``write`` protocol.
* :class:`FakeClipboard` — an in-memory provider, so tests run with zero heavy
  libraries and no OS clipboard access.
* :class:`SystemClipboard` — the real adapter that lazy-imports ``pyperclip``
  inside its methods and raises a clear error (with a ``make install-perception``
  hint) when the backend is absent.

No heavy perception library is imported at module top level, so importing this
module never requires ``pyperclip`` and ``uv sync`` stays unaffected.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from friday.errors import ProviderError

_INSTALL_HINT = (
    "pyperclip is not installed. Perception extras are optional and excluded "
    "from the uv lock; install them with `make install-perception`."
)


@runtime_checkable
class ClipboardProvider(Protocol):
    """Contract reading from / writing to the system clipboard."""

    def read(self) -> str:
        """Return the current clipboard text (possibly empty)."""
        ...

    def write(self, text: str) -> None:
        """Replace the clipboard contents with ``text``."""
        ...


class FakeClipboard:
    """An in-memory :class:`ClipboardProvider` for tests.

    Holds a single ``str`` buffer so a :meth:`write` followed by :meth:`read`
    round-trips deterministically with no OS clipboard access.
    """

    def __init__(self, initial: str = "") -> None:
        """Create the fake clipboard.

        Args:
            initial: The starting clipboard contents.
        """
        self._buffer = initial

    def read(self) -> str:
        return self._buffer

    def write(self, text: str) -> None:
        self._buffer = text


class SystemClipboard:
    """Real :class:`ClipboardProvider` backed by ``pyperclip`` (lazy).

    The ``pyperclip`` import happens inside the methods, so importing this module
    never requires the backend. When the backend is missing, a
    :class:`friday.errors.ProviderError` is raised with a
    ``make install-perception`` hint.
    """

    def _pyperclip(self) -> object:
        """Lazy-import ``pyperclip``, raising a helpful error when absent."""
        try:
            # Optional perception backend: excluded from the uv lock; lazily
            # imported here and guarded by the ImportError. The import is
            # type-ignored (it may resolve as untyped or be wholly absent) and
            # narrowed to ``object`` so callers go through ``# type: ignore``.
            import pyperclip  # type: ignore[import-untyped, import-not-found, unused-ignore]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ProviderError(_INSTALL_HINT) from exc
        return pyperclip

    def read(self) -> str:
        """Return the current OS clipboard text."""
        pyperclip = self._pyperclip()
        return str(pyperclip.paste())  # type: ignore[attr-defined]

    def write(self, text: str) -> None:
        """Replace the OS clipboard contents with ``text``."""
        pyperclip = self._pyperclip()
        pyperclip.copy(text)  # type: ignore[attr-defined]
