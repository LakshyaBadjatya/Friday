"""Secret storage backends + a plaintext-secret scanner for FRIDAY.

This module owns FRIDAY's small, dependency-injected secret boundary. Nothing
here imports :mod:`friday.config` or :mod:`friday.app`; every backend is
constructed with explicit parameters so it can be wired by a later integration
pass without coupling to the global settings object.

Backends
--------
* :class:`SecretVault` — the runtime-checkable structural protocol every
  backend satisfies: ``get(name) -> str | None`` / ``set(name, value) -> None``.
* :class:`MemoryVault` — an in-process dict; the default for tests and a safe
  dev fallback. Never persists anything to disk.
* :class:`EnvVault` — reads/writes secrets in the process environment under a
  configurable prefix (default ``FRIDAY_``); the looked-up variable is the
  uppercased ``<PREFIX><NAME>`` (e.g. ``FRIDAY_NVIDIA_API_KEY``).
* :class:`FileVault` — a JSON file written with ``0600`` permissions. This is a
  *developer* fallback only; production should prefer the OS keyring.
* :class:`KeyringVault` — wraps the optional ``keyring`` package under a
  ``"friday"`` service. ``keyring`` is **lazy-imported** inside ``__init__`` so
  merely importing this module never requires the optional dependency; if it is
  missing the constructor raises a clear, actionable :class:`SecretVaultError`.

Scanner
-------
:func:`scan_for_plaintext_secrets` walks a directory tree and flags
secret-looking string literals (NVIDIA ``nvapi-…`` keys, Google ``AIza…`` keys,
OpenAI ``sk-…`` keys, AWS ``AKIA…`` access keys, and long base64 blobs) in
tracked source files. A startup self-check uses it to nudge against booting a
repo that has a credential committed in source. It deliberately ignores ``.env``
(git-ignored and a legitimate local secret store) while still scanning
``.env.example`` and other committed ``.env``-family files.

To avoid boot-time false positives it scans only *production* sources: the
entire ``tests/`` tree is skipped (its fixtures legitimately carry
oauth/secret-shaped strings), as is any file whose name starts with ``test_``.
It also skips obvious non-secret contexts — the broad base64 catch-all does not
fire on URL paths (a ``…/v3/calendars/primary/events`` run is base64-valid only
because ``/`` is in the alphabet, not because it is a credential).

No heavy/optional dependency is imported at module top level.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from friday.errors import FridayError

__all__ = [
    "EnvVault",
    "FileVault",
    "Finding",
    "KeyringVault",
    "MemoryVault",
    "SecretVault",
    "SecretVaultError",
    "scan_for_plaintext_secrets",
]

#: Service namespace used for every key stored via the OS keyring.
_KEYRING_SERVICE = "friday"

_KEYRING_INSTALL_HINT = (
    "The `keyring` package is required for KeyringVault but is not importable. "
    "It is an optional dependency kept out of the default lock; install it with "
    "`uv pip install keyring` (or add `keyring` to your environment) to use the "
    "OS keychain backend. Until then, use EnvVault or FileVault instead."
)


class SecretVaultError(FridayError):
    """A secret backend could not be constructed or used."""


@runtime_checkable
class SecretVault(Protocol):
    """Structural contract for a secret store.

    Implementations resolve a logical secret ``name`` (e.g.
    ``"nvidia_api_key"``) to its value. ``get`` returns ``None`` when the
    secret is absent rather than raising, so callers can fall back across
    layered vaults.
    """

    def get(self, name: str) -> str | None:
        """Return the secret value for ``name``, or ``None`` if unset."""
        ...

    def set(self, name: str, value: str) -> None:
        """Store ``value`` under ``name``."""
        ...


class MemoryVault:
    """In-process secret store backed by a plain dict.

    The default backend for tests and a safe dev fallback: nothing is ever
    written to disk. An optional ``seed`` mapping pre-populates the vault; it is
    copied (never aliased) so later ``set`` calls do not mutate the caller's
    dict.
    """

    def __init__(self, seed: dict[str, str] | None = None) -> None:
        self._store: dict[str, str] = dict(seed) if seed else {}

    def get(self, name: str) -> str | None:
        return self._store.get(name)

    def set(self, name: str, value: str) -> None:
        self._store[name] = value


class EnvVault:
    """Secret store backed by the process environment.

    The logical ``name`` is mapped to an environment variable named
    ``<PREFIX><NAME>`` with the whole thing uppercased — e.g. with the default
    ``FRIDAY_`` prefix, ``get("nvidia_api_key")`` reads
    ``FRIDAY_NVIDIA_API_KEY``. ``set`` mutates :data:`os.environ` for the
    current process (useful in dev/tests; it does not persist).
    """

    def __init__(self, prefix: str = "FRIDAY_") -> None:
        self._prefix = prefix

    def _key(self, name: str) -> str:
        return f"{self._prefix}{name}".upper()

    def get(self, name: str) -> str | None:
        return os.environ.get(self._key(name))

    def set(self, name: str, value: str) -> None:
        os.environ[self._key(name)] = value


class FileVault:
    """Developer-fallback secret store: a ``0600`` JSON file.

    The file is created lazily on first ``set`` (its parent directory is created
    if needed) and is always (re)written with ``0600`` permissions so it is not
    world/group readable. Intended for local development only; production should
    prefer :class:`KeyringVault`.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)

    def _load(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items()}

    def get(self, name: str) -> str | None:
        return self._load().get(name)

    def set(self, name: str, value: str) -> None:
        data = self._load()
        data[name] = value
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a sibling temp file, then atomically rename it into place. This
        # means a concurrent reader never observes a truncated/half-written file
        # (the rename is atomic on the same filesystem), and the file is created
        # with restrictive perms *before* any bytes hit disk: os.open's mode is
        # umask-masked, but masking only ever removes bits from 0o600 — it can
        # never add group/world bits — so the secrets never exist looser than 0o600.
        tmp = self._path.with_name(self._path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(data))
        os.replace(tmp, self._path)  # atomic swap (same dir → same filesystem)
        # Tighten any pre-existing file that an earlier looser write may have left.
        self._path.chmod(0o600)


class KeyringVault:
    """Secret store backed by the OS keychain via the optional ``keyring`` pkg.

    ``keyring`` is **lazy-imported** in ``__init__`` so importing this module (or
    constructing any other backend) never requires the optional dependency. If
    ``keyring`` is not importable the constructor raises
    :class:`SecretVaultError` with an actionable install hint. Every secret is
    namespaced under the ``"friday"`` service by default.
    """

    def __init__(self, service: str = _KEYRING_SERVICE) -> None:
        try:
            import keyring  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise SecretVaultError(_KEYRING_INSTALL_HINT) from exc
        self._keyring = keyring
        self._service = service

    def get(self, name: str) -> str | None:
        result = self._keyring.get_password(self._service, name)
        return None if result is None else str(result)

    def set(self, name: str, value: str) -> None:
        self._keyring.set_password(self._service, name, value)


class Finding(BaseModel):
    """One plaintext-secret hit: the ``file``, 1-based ``line``, and ``kind``."""

    file: str
    line: int
    kind: str


# Ordered most-specific first so a single line attributes to the tightest
# matching kind. ``base64`` is last as the broad catch-all.
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("nvapi", re.compile(r"nvapi-[A-Za-z0-9_\-]{20,}")),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("base64", re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")),
)

#: The catch-all kind whose matches are URL-path noise far more often than real
#: credentials; it gets the extra non-secret-context guard below.
_BASE64_KIND = "base64"

#: File suffixes treated as scannable source.
_SCANNED_SUFFIXES: frozenset[str] = frozenset({".py", ".env"})

#: Filenames that legitimately hold local secrets and are git-ignored; skipped.
_IGNORED_FILENAMES: frozenset[str] = frozenset({".env"})

#: Filename prefix marking a test module; its body carries fixture/oauth-shaped
#: strings that are false positives, so such files are skipped wholesale.
_TEST_FILE_PREFIX = "test_"

#: Directory names never worth scanning (vendored/build/VCS noise).
_SKIP_DIRS: frozenset[str] = frozenset(
    {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache"}
)

#: Source trees deliberately excluded from the scan: test fixtures legitimately
#: carry secret-shaped literals, so scanning them only yields false positives.
_SKIP_SOURCE_DIRS: frozenset[str] = frozenset({"tests"})

#: A base64 match that overlaps a URL (``scheme://…``) is a path fragment, not a
#: secret — ``/`` is base64-valid, so REST paths trip the broad catch-all.
_URL_RE: re.Pattern[str] = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://\S+")


def _is_scannable(path: Path) -> bool:
    """Return True if ``path`` is a committed *production* source file to scan.

    Excludes the git-ignored ``.env`` and any ``test_*`` module (whose fixture
    bodies legitimately hold secret-shaped strings). Test *directories* are
    pruned earlier, during the walk, via :data:`_SKIP_SOURCE_DIRS`.
    """
    name = path.name
    if name in _IGNORED_FILENAMES:
        return False
    if name.startswith(_TEST_FILE_PREFIX):
        return False
    if path.suffix in _SCANNED_SUFFIXES:
        return True
    # ``.env.example`` / ``.env.sample`` etc. are committed and must be scanned.
    return name.startswith(".env.")


def _is_url_match(line: str, start: int, end: int) -> bool:
    """Return True if the ``[start, end)`` span overlaps a URL in ``line``."""
    return any(m.start() < end and start < m.end() for m in _URL_RE.finditer(line))


def _scan_line(line: str) -> str | None:
    """Return the matched secret ``kind`` for ``line``, or ``None``.

    The broad ``base64`` catch-all is suppressed when its match is part of a URL
    (an obvious non-secret context), but the specific provider patterns
    (``nvapi-``/``AIza``/``sk-``/``AKIA``) always fire — a real key embedded in a
    URL is still a leak.
    """
    for kind, pattern in _SECRET_PATTERNS:
        match = pattern.search(line)
        if match is None:
            continue
        if kind == _BASE64_KIND and _is_url_match(line, match.start(), match.end()):
            continue
        return kind
    return None


def scan_for_plaintext_secrets(root: str) -> list[Finding]:
    """Walk ``root`` and flag secret-looking literals in tracked source files.

    Returns a list of :class:`Finding` (one per offending line) for committed
    ``.py`` and ``.env``-family files (excluding the git-ignored ``.env``
    itself). The whole ``tests/`` tree and any ``test_*`` file are skipped to
    avoid fixture-shaped false positives, as are base64-looking URL paths.
    Binary/undecodable files are skipped silently. The result is deterministic:
    files are visited in sorted path order and lines in order.
    """
    base = Path(root)
    findings: list[Finding] = []
    pruned_dirs = _SKIP_DIRS | _SKIP_SOURCE_DIRS

    for path in sorted(base.rglob("*")):
        if any(part in pruned_dirs for part in path.parts):
            continue
        if not path.is_file() or not _is_scannable(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            kind = _scan_line(line)
            if kind is not None:
                findings.append(Finding(file=str(path), line=lineno, kind=kind))

    return findings
