"""Unit tests for the secret vault + plaintext-secret scanner.

The vault layer (``friday.secrets.vault``) defines a tiny structural
:class:`SecretVault` protocol (``get`` / ``set``) and four concrete backends:

* :class:`MemoryVault` — an in-process dict, used by tests and as a dev default.
* :class:`EnvVault` — reads secrets from the process environment under a
  ``FRIDAY_`` prefix (uppercased name), e.g. ``FRIDAY_NVIDIA_API_KEY``.
* :class:`FileVault` — a JSON file written ``0600`` (dev fallback only).
* :class:`KeyringVault` — wraps the optional ``keyring`` package under a
  ``"friday"`` service; if ``keyring`` is missing the constructor raises a
  clear, actionable install error rather than failing obscurely later.

The scanner (:func:`scan_for_plaintext_secrets`) walks a directory tree and
flags secret-looking string literals (``nvapi-…``, ``AIza…``, ``sk-…``,
``AKIA…``, long base64 blobs) in tracked source files so a startup self-check
can refuse to boot a repo that has committed a credential. It must IGNORE
``.env`` (which is git-ignored and legitimately holds secrets locally).

Everything here is fully offline: no real keyring, no network, no environment
mutation that leaks across tests (``monkeypatch`` is scoped per-test).
"""

from __future__ import annotations

import builtins
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from friday.errors import FridayError
from friday.secrets import (
    EnvVault,
    KeyringVault,
    MemoryVault,
    scan_for_plaintext_secrets,
)
from friday.secrets.vault import (
    FileVault,
    Finding,
    SecretVault,
)

# A planted FAKE secret — not a real credential, only shaped like one so the
# scanner's regexes fire. Assembled from parts so the file itself does not
# contain a contiguous literal that would trip the scanner on the test tree.
_FAKE_NVAPI = "nvapi-" + "A" * 40


# -- protocol conformance ------------------------------------------------ #


def test_memory_vault_satisfies_protocol() -> None:
    assert isinstance(MemoryVault(), SecretVault)


def test_env_vault_satisfies_protocol() -> None:
    assert isinstance(EnvVault(), SecretVault)


# -- MemoryVault round-trip --------------------------------------------- #


def test_memory_vault_round_trip() -> None:
    vault = MemoryVault()
    assert vault.get("nvidia_api_key") is None

    vault.set("nvidia_api_key", "secret-value")
    assert vault.get("nvidia_api_key") == "secret-value"


def test_memory_vault_overwrite() -> None:
    vault = MemoryVault()
    vault.set("k", "v1")
    vault.set("k", "v2")
    assert vault.get("k") == "v2"


def test_memory_vault_seeded_from_mapping() -> None:
    vault = MemoryVault({"k": "v"})
    assert vault.get("k") == "v"


def test_memory_vault_does_not_alias_seed_mapping() -> None:
    seed = {"k": "v"}
    vault = MemoryVault(seed)
    vault.set("k", "changed")
    assert seed["k"] == "v"  # the caller's dict is not mutated


# -- EnvVault ------------------------------------------------------------ #


def test_env_vault_reads_prefixed_uppercase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRIDAY_NVIDIA_API_KEY", "from-env")
    vault = EnvVault()
    assert vault.get("nvidia_api_key") == "from-env"


def test_env_vault_missing_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRIDAY_NVIDIA_API_KEY", raising=False)
    assert EnvVault().get("nvidia_api_key") is None


def test_env_vault_custom_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X_FOO", "bar")
    assert EnvVault(prefix="X_").get("foo") == "bar"


def test_env_vault_set_mutates_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRIDAY_TOKEN", raising=False)
    vault = EnvVault()
    vault.set("token", "abc")
    assert os.environ["FRIDAY_TOKEN"] == "abc"
    assert vault.get("token") == "abc"


# -- FileVault ----------------------------------------------------------- #


def test_file_vault_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "secrets.json"
    vault = FileVault(str(path))
    assert vault.get("k") is None
    vault.set("k", "v")
    assert vault.get("k") == "v"


def test_file_vault_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "secrets.json"
    FileVault(str(path)).set("k", "v")
    assert FileVault(str(path)).get("k") == "v"


def test_file_vault_file_is_0600(tmp_path: Path) -> None:
    path = tmp_path / "secrets.json"
    FileVault(str(path)).set("k", "v")
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600


def test_file_vault_creates_file_with_restrictive_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The secrets file is created 0600 *before* any bytes are written.

    The post-write ``chmod`` masks an over-permissive *creation* mode from a plain
    ``stat`` check, so spy on ``os.open`` to pin the mode requested at creation —
    guarding against a regression to ``write_text`` (which opens 0666 & umask,
    leaving a world/group-readable window).
    """
    recorded: list[int] = []
    real_open = os.open

    def _spy_open(path: Any, flags: int, mode: int = 0o777, *args: Any) -> int:
        recorded.append(mode)
        return real_open(path, flags, mode, *args)

    monkeypatch.setattr(os, "open", _spy_open)

    path = tmp_path / "secrets.json"
    FileVault(str(path)).set("k", "v")

    assert recorded and recorded[-1] == 0o600
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_file_vault_writes_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "secrets.json"
    FileVault(str(path)).set("k", "v")
    assert json.loads(path.read_text()) == {"k": "v"}


def test_file_vault_tolerates_missing_dir(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "secrets.json"
    vault = FileVault(str(path))
    vault.set("k", "v")
    assert vault.get("k") == "v"


# -- KeyringVault -------------------------------------------------------- #


def test_keyring_vault_raises_clear_error_without_keyring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing KeyringVault without the optional package fails loudly."""
    real_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "keyring":
            raise ImportError("No module named 'keyring'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(FridayError) as excinfo:
        KeyringVault()

    msg = str(excinfo.value)
    assert "keyring" in msg.lower()
    assert "install" in msg.lower()  # actionable: tells the user what to do


def test_keyring_vault_uses_friday_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """When keyring is importable, get/set delegate under the 'friday' service."""
    calls: dict[str, Any] = {}

    class _FakeKeyring:
        @staticmethod
        def get_password(service: str, name: str) -> str | None:
            calls["get"] = (service, name)
            return "stored" if name == "present" else None

        @staticmethod
        def set_password(service: str, name: str, value: str) -> None:
            calls["set"] = (service, name, value)

    real_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "keyring":
            return _FakeKeyring
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    vault = KeyringVault()
    assert vault.get("present") == "stored"
    assert calls["get"] == ("friday", "present")
    assert vault.get("absent") is None

    vault.set("nvidia_api_key", "v")
    assert calls["set"] == ("friday", "nvidia_api_key", "v")


def test_keyring_vault_custom_service(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class _FakeKeyring:
        @staticmethod
        def get_password(service: str, name: str) -> str | None:
            seen["service"] = service
            return None

        @staticmethod
        def set_password(service: str, name: str, value: str) -> None: ...

    real_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "keyring":
            return _FakeKeyring
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    KeyringVault(service="other").get("k")
    assert seen["service"] == "other"


# -- scan_for_plaintext_secrets ----------------------------------------- #


def test_scan_flags_planted_secret(tmp_path: Path) -> None:
    src = tmp_path / "leaky.py"
    src.write_text(f'API_KEY = "{_FAKE_NVAPI}"\n')

    findings = scan_for_plaintext_secrets(str(tmp_path))

    assert len(findings) == 1
    finding = findings[0]
    assert isinstance(finding, Finding)
    assert finding.file == str(src)
    assert finding.line == 1
    assert finding.kind == "nvapi"


def test_scan_ignores_dotenv(tmp_path: Path) -> None:
    """.env is git-ignored and legitimately holds secrets locally."""
    (tmp_path / ".env").write_text(f"FRIDAY_NVIDIA_API_KEY={_FAKE_NVAPI}\n")
    assert scan_for_plaintext_secrets(str(tmp_path)) == []


def test_scan_clean_tree_returns_empty(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("x = 1\nname = 'friday'\n")
    assert scan_for_plaintext_secrets(str(tmp_path)) == []


def test_scan_detects_google_api_key(tmp_path: Path) -> None:
    secret = "AIza" + "B" * 35
    (tmp_path / "g.py").write_text(f'KEY = "{secret}"\n')
    findings = scan_for_plaintext_secrets(str(tmp_path))
    assert [f.kind for f in findings] == ["google_api_key"]


def test_scan_detects_openai_key(tmp_path: Path) -> None:
    secret = "sk-" + "c" * 40
    (tmp_path / "o.py").write_text(f'KEY = "{secret}"\n')
    findings = scan_for_plaintext_secrets(str(tmp_path))
    assert [f.kind for f in findings] == ["openai_key"]


def test_scan_detects_aws_access_key(tmp_path: Path) -> None:
    secret = "AKIA" + "D" * 16
    (tmp_path / "a.py").write_text(f'KEY = "{secret}"\n')
    findings = scan_for_plaintext_secrets(str(tmp_path))
    assert [f.kind for f in findings] == ["aws_access_key"]


def test_scan_reports_correct_line_number(tmp_path: Path) -> None:
    src = tmp_path / "multi.py"
    src.write_text("\n".join(["x = 1", "y = 2", f'k = "{_FAKE_NVAPI}"', "z = 3"]) + "\n")
    findings = scan_for_plaintext_secrets(str(tmp_path))
    assert len(findings) == 1
    assert findings[0].line == 3


def test_scan_skips_non_tracked_extensions(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text(f'token: "{_FAKE_NVAPI}"\n')
    (tmp_path / "data.bin").write_bytes(_FAKE_NVAPI.encode())
    assert scan_for_plaintext_secrets(str(tmp_path)) == []


def test_scan_finds_secret_in_committed_env_example(tmp_path: Path) -> None:
    """A committed .env.example WITH a real-looking secret is still flagged."""
    (tmp_path / ".env.example").write_text(f"FRIDAY_KEY={_FAKE_NVAPI}\n")
    findings = scan_for_plaintext_secrets(str(tmp_path))
    assert [f.kind for f in findings] == ["nvapi"]


def test_scan_handles_undecodable_file_gracefully(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_bytes(b"\xff\xfe not utf8")
    # Must not raise; binary garbage simply yields no findings.
    assert scan_for_plaintext_secrets(str(tmp_path)) == []


def test_scan_recurses_into_subdirs(tmp_path: Path) -> None:
    nested = tmp_path / "pkg" / "sub"
    nested.mkdir(parents=True)
    src = nested / "deep.py"
    src.write_text(f'K = "{_FAKE_NVAPI}"\n')
    findings = scan_for_plaintext_secrets(str(tmp_path))
    assert len(findings) == 1
    assert findings[0].file == str(src)


def test_finding_is_clean_value_object() -> None:
    f = Finding(file="/x.py", line=3, kind="nvapi")
    assert f.model_dump() == {"file": "/x.py", "line": 3, "kind": "nvapi"}


# -- scanner false-positive tightening ---------------------------------- #


def test_scan_skips_tests_directory_entirely(tmp_path: Path) -> None:
    """A planted secret under a ``tests/`` directory is NOT flagged.

    Test trees legitimately carry fixture/oauth-shaped strings; scanning them
    produces only false positives at boot.
    """
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "fixtures.py").write_text(f'TOKEN = "{_FAKE_NVAPI}"\n')

    assert scan_for_plaintext_secrets(str(tmp_path)) == []


def test_scan_skips_nested_tests_directory(tmp_path: Path) -> None:
    """``tests/`` is skipped at any depth, not only at the scan root."""
    nested = tmp_path / "pkg" / "tests" / "data"
    nested.mkdir(parents=True)
    (nested / "leaky.py").write_text(f'K = "{_FAKE_NVAPI}"\n')

    assert scan_for_plaintext_secrets(str(tmp_path)) == []


def test_scan_flags_planted_secret_in_src(tmp_path: Path) -> None:
    """The mirror of the tests/ case: a secret under ``src/`` IS flagged."""
    src_dir = tmp_path / "src" / "friday"
    src_dir.mkdir(parents=True)
    leaky = src_dir / "leaky.py"
    leaky.write_text(f'API_KEY = "{_FAKE_NVAPI}"\n')

    findings = scan_for_plaintext_secrets(str(tmp_path))

    assert len(findings) == 1
    assert findings[0].file == str(leaky)
    assert findings[0].kind == "nvapi"


def test_scan_skips_test_prefixed_files(tmp_path: Path) -> None:
    """A file named ``test_*`` is skipped even outside a ``tests/`` directory."""
    (tmp_path / "test_helpers.py").write_text(f'K = "{_FAKE_NVAPI}"\n')

    assert scan_for_plaintext_secrets(str(tmp_path)) == []


def test_scan_still_flags_non_test_prefixed_file(tmp_path: Path) -> None:
    """A ``test_``-containing-but-not-prefixed name is NOT exempted."""
    (tmp_path / "latest_keys.py").write_text(f'K = "{_FAKE_NVAPI}"\n')

    findings = scan_for_plaintext_secrets(str(tmp_path))
    assert [f.kind for f in findings] == ["nvapi"]


def test_scan_skips_url_non_secret_context(tmp_path: Path) -> None:
    """A URL whose path trips the broad base64 catch-all is NOT flagged.

    e.g. ``https://www.googleapis.com/calendar/v3/calendars/primary/events``
    contains a 40+ ``[A-Za-z0-9/]`` run only because ``/`` is base64-valid.
    """
    (tmp_path / "client.py").write_text(
        '_EVENTS_URL = "https://www.googleapis.com/calendar/v3/calendars/primary/events"\n'
    )

    assert scan_for_plaintext_secrets(str(tmp_path)) == []


def test_scan_still_flags_real_base64_blob(tmp_path: Path) -> None:
    """A genuine long base64 token (no URL scheme) is still flagged."""
    blob = "QUJD" * 12  # 48 contiguous base64 chars, no URL context
    (tmp_path / "blob.py").write_text(f'SECRET = "{blob}"\n')

    findings = scan_for_plaintext_secrets(str(tmp_path))
    assert [f.kind for f in findings] == ["base64"]
