"""FRIDAY secret-management package.

Re-exports the public secret-vault surface so callers can import the common
backends and the plaintext-secret scanner directly from :mod:`friday.secrets`.
See :mod:`friday.secrets.vault` for the full backend documentation.
"""

from __future__ import annotations

from friday.secrets.vault import (
    EnvVault,
    FileVault,
    Finding,
    KeyringVault,
    MemoryVault,
    SecretVault,
    SecretVaultError,
    scan_for_plaintext_secrets,
)

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
