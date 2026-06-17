# © Lakshya Badjatya — Author
"""Encrypted, authenticated backup of FRIDAY's durable state — stdlib only.

Bundles a set of files (the memory DB, the tamper-evident audit ledger) into one
gzip tar, encrypts it under a passphrase, and authenticates the whole blob so a
wrong key or any tampering is detected on restore. The format is:

    MAGIC (9B) ‖ salt (16B) ‖ ciphertext ‖ HMAC-SHA256 tag (32B)

* **Key derivation:** PBKDF2-HMAC-SHA256 (200k iters) over the passphrase + salt
  yields 64 bytes split into an encryption key and a *separate* MAC key.
* **Confidentiality:** a keystream — HMAC-SHA256(enc_key, counter) blocks — is
  XORed with the plaintext (a counter-mode stream cipher over a PRF).
* **Integrity / authenticity:** encrypt-then-MAC. The tag covers MAGIC‖salt‖
  ciphertext and is checked with :func:`hmac.compare_digest`, so a flipped bit or
  a wrong passphrase fails closed before anything is written.

Dependency-free by design (no ``cryptography`` package, matching the offline-first
posture). If you later add ``cryptography``, swapping the body for AES-GCM/Fernet
is the natural hardening upgrade — the public functions can keep their shapes.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import os
import tarfile
from pathlib import Path

from friday.errors import FridayError

_MAGIC = b"FRIDAYBK1"
_PBKDF2_ITERS = 200_000
_SALT_LEN = 16
_TAG_LEN = 32
_BLOCK = 32  # HMAC-SHA256 digest size = keystream block size


class BackupError(FridayError):
    """A backup could not be created or restored (bad key, tampering, bad input)."""


def _derive_keys(passphrase: str, salt: bytes) -> tuple[bytes, bytes]:
    """Derive (enc_key, mac_key) from the passphrase + salt via PBKDF2-SHA256."""
    material = hashlib.pbkdf2_hmac(
        "sha256", passphrase.encode("utf-8"), salt, _PBKDF2_ITERS, dklen=64
    )
    return material[:32], material[32:]


def _keystream(enc_key: bytes, nbytes: int) -> bytes:
    """Counter-mode keystream: HMAC-SHA256(enc_key, counter) blocks, truncated."""
    out = bytearray()
    counter = 0
    while len(out) < nbytes:
        out.extend(
            hmac.new(enc_key, counter.to_bytes(8, "big"), hashlib.sha256).digest()
        )
        counter += 1
    return bytes(out[:nbytes])


def _xor(data: bytes, stream: bytes) -> bytes:
    """XOR ``data`` with an equal-length keystream slice."""
    return bytes(a ^ b for a, b in zip(data, stream, strict=True))


def _bundle(paths: list[Path]) -> bytes:
    """Pack the existing files among ``paths`` into a gzip tar (by basename)."""
    buf = io.BytesIO()
    added = 0
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in paths:
            if path.exists() and path.is_file():
                tar.add(str(path), arcname=path.name)
                added += 1
    if added == 0:
        raise BackupError("nothing to back up: none of the given paths exist")
    return buf.getvalue()


def create_backup(paths: list[Path], passphrase: str) -> bytes:
    """Bundle ``paths`` and return one encrypted, authenticated backup blob.

    Raises :class:`BackupError` if the passphrase is empty or no input file
    exists (an empty backup is almost always an operator mistake, so it fails
    loudly rather than writing a useless archive).
    """
    if not passphrase:
        raise BackupError("a non-empty passphrase is required")
    plaintext = _bundle(paths)
    salt = os.urandom(_SALT_LEN)
    enc_key, mac_key = _derive_keys(passphrase, salt)
    ciphertext = _xor(plaintext, _keystream(enc_key, len(plaintext)))
    body = _MAGIC + salt + ciphertext
    tag = hmac.new(mac_key, body, hashlib.sha256).digest()
    return body + tag


def restore_backup(blob: bytes, passphrase: str, dest: Path) -> list[str]:
    """Verify + decrypt ``blob`` and extract its files into ``dest``.

    Returns the extracted file names. Raises :class:`BackupError` on a malformed
    blob, a wrong passphrase, or any tampering (the HMAC check fails closed before
    a single byte is decrypted or written). Extraction uses the ``data`` tar
    filter, so path-traversal entries cannot escape ``dest``.
    """
    if len(blob) < len(_MAGIC) + _SALT_LEN + _TAG_LEN:
        raise BackupError("backup blob is too short to be valid")
    body, tag = blob[:-_TAG_LEN], blob[-_TAG_LEN:]
    if not body.startswith(_MAGIC):
        raise BackupError("not a FRIDAY backup (bad magic header)")
    salt = body[len(_MAGIC) : len(_MAGIC) + _SALT_LEN]
    ciphertext = body[len(_MAGIC) + _SALT_LEN :]
    enc_key, mac_key = _derive_keys(passphrase, salt)
    expected = hmac.new(mac_key, body, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, tag):
        raise BackupError("authentication failed: wrong passphrase or tampered backup")
    plaintext = _xor(ciphertext, _keystream(enc_key, len(ciphertext)))
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(plaintext), mode="r:gz") as tar:
            names = [member.name for member in tar.getmembers()]
            tar.extractall(dest, filter="data")
    except (tarfile.TarError, OSError) as exc:  # corrupt archive after a valid MAC
        raise BackupError(f"backup decrypted but could not be extracted: {exc}") from exc
    return names
