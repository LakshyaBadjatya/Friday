"""Unit tests for the speaker-verification (voiceprint) boundary.

Pins the voiceprint contract with the deterministic :class:`FakeVoiceprint`:
enrollment then high score for the owner sample, low for another; the
:class:`OwnerIdentity` threshold; and the advisory default (verifying a
non-owner never raises). Also covers the :class:`EnrollmentStore` round-trip and
that the real :class:`ResemblyzerVerifier` raises a clear error when its backend
is absent. No real audio, no model, no network.
"""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import pytest

from friday.errors import PermissionError as FridayPermissionError
from friday.errors import ProviderError
from friday.voice.voiceprint import (
    DEFAULT_OWNER_THRESHOLD,
    EnrollmentStore,
    FakeVoiceprint,
    OwnerIdentity,
    OwnerVerification,
    ResemblyzerVerifier,
    SpeakerVerifier,
)

# Deterministic stand-in samples (any distinct bytes work for the fake).
OWNER_SAMPLE = b"owner-voice-sample-bytes"
OWNER_SAMPLE_2 = b"owner-voice-sample-bytes-take-2"
OTHER_SAMPLE = b"some-other-speaker-bytes"


# --------------------------------------------------------------------------- #
# Protocol membership
# --------------------------------------------------------------------------- #
def test_fake_is_speaker_verifier() -> None:
    assert isinstance(FakeVoiceprint(), SpeakerVerifier)


def test_resemblyzer_is_speaker_verifier() -> None:
    assert isinstance(ResemblyzerVerifier(), SpeakerVerifier)


# --------------------------------------------------------------------------- #
# FakeVoiceprint enroll/score
# --------------------------------------------------------------------------- #
def test_enroll_then_score_high_for_owner_low_for_other() -> None:
    verifier = FakeVoiceprint()
    profile = verifier.enroll([OWNER_SAMPLE])

    assert verifier.score(OWNER_SAMPLE, profile) == 1.0
    assert verifier.score(OTHER_SAMPLE, profile) == 0.0


def test_enroll_accepts_multiple_samples() -> None:
    verifier = FakeVoiceprint()
    profile = verifier.enroll([OWNER_SAMPLE, OWNER_SAMPLE_2])

    assert verifier.score(OWNER_SAMPLE, profile) == 1.0
    assert verifier.score(OWNER_SAMPLE_2, profile) == 1.0
    assert verifier.score(OTHER_SAMPLE, profile) == 0.0


def test_enroll_is_deterministic() -> None:
    verifier = FakeVoiceprint()
    assert verifier.enroll([OWNER_SAMPLE]) == verifier.enroll([OWNER_SAMPLE])


def test_enroll_empty_samples_raises() -> None:
    with pytest.raises(ValueError):
        FakeVoiceprint().enroll([])


def test_score_against_empty_profile_is_zero() -> None:
    assert FakeVoiceprint().score(OWNER_SAMPLE, b"") == 0.0


def test_score_is_in_unit_interval() -> None:
    verifier = FakeVoiceprint()
    profile = verifier.enroll([OWNER_SAMPLE])
    for sample in (OWNER_SAMPLE, OTHER_SAMPLE, b""):
        score = verifier.score(sample, profile)
        assert 0.0 <= score <= 1.0


# --------------------------------------------------------------------------- #
# OwnerIdentity: is_owner threshold
# --------------------------------------------------------------------------- #
def _owner_identity(threshold: float = DEFAULT_OWNER_THRESHOLD) -> OwnerIdentity:
    verifier = FakeVoiceprint()
    profile = verifier.enroll([OWNER_SAMPLE])
    return OwnerIdentity(verifier, profile, threshold=threshold)


def test_is_owner_true_for_owner_false_for_other() -> None:
    identity = _owner_identity()
    assert identity.is_owner(OWNER_SAMPLE) is True
    assert identity.is_owner(OTHER_SAMPLE) is False


def test_default_threshold_value() -> None:
    identity = _owner_identity()
    assert identity.threshold == DEFAULT_OWNER_THRESHOLD


def test_threshold_boundary_inclusive() -> None:
    # FakeVoiceprint scores exactly 1.0 for owner; a threshold of 1.0 still
    # admits the owner (score >= threshold is inclusive) but rejects others.
    identity = _owner_identity(threshold=1.0)
    assert identity.is_owner(OWNER_SAMPLE) is True
    assert identity.is_owner(OTHER_SAMPLE) is False


def test_verify_returns_typed_result() -> None:
    identity = _owner_identity()
    result = identity.verify(OWNER_SAMPLE)
    assert isinstance(result, OwnerVerification)
    assert result.is_owner is True
    assert result.score == 1.0
    assert result.threshold == DEFAULT_OWNER_THRESHOLD


def test_score_passthrough() -> None:
    identity = _owner_identity()
    assert identity.score(OWNER_SAMPLE) == 1.0
    assert identity.score(OTHER_SAMPLE) == 0.0


# --------------------------------------------------------------------------- #
# Advisory mode: verifying a non-owner never raises / never blocks
# --------------------------------------------------------------------------- #
def test_verify_non_owner_does_not_raise() -> None:
    identity = _owner_identity()
    # Advisory by default: a non-owner sample yields a result, not an exception.
    result = identity.verify(OTHER_SAMPLE)
    assert result.is_owner is False


def test_is_owner_non_owner_does_not_raise() -> None:
    identity = _owner_identity()
    assert identity.is_owner(OTHER_SAMPLE) is False  # no exception


# --------------------------------------------------------------------------- #
# Opt-in hard gate (require_owner) is the only blocking path
# --------------------------------------------------------------------------- #
def test_require_owner_allows_owner() -> None:
    identity = _owner_identity()
    result = identity.require_owner(OWNER_SAMPLE)
    assert result.is_owner is True


def test_require_owner_blocks_non_owner() -> None:
    identity = _owner_identity()
    with pytest.raises(FridayPermissionError):
        identity.require_owner(OTHER_SAMPLE)


# --------------------------------------------------------------------------- #
# EnrollmentStore round-trip
# --------------------------------------------------------------------------- #
def test_store_save_load_roundtrip(tmp_path: Path) -> None:
    verifier = FakeVoiceprint()
    profile = verifier.enroll([OWNER_SAMPLE])
    store = EnrollmentStore(tmp_path / "owner.profile")

    assert store.exists() is False
    store.save(profile)
    assert store.exists() is True
    assert store.load() == profile


def test_store_creates_parent_directories(tmp_path: Path) -> None:
    store = EnrollmentStore(tmp_path / "nested" / "dir" / "owner.profile")
    store.save(b"blob")
    assert store.load() == b"blob"


def test_store_load_missing_raises(tmp_path: Path) -> None:
    store = EnrollmentStore(tmp_path / "absent.profile")
    with pytest.raises(FileNotFoundError):
        store.load()


def test_store_path_accepts_str(tmp_path: Path) -> None:
    store = EnrollmentStore(str(tmp_path / "owner.profile"))
    store.save(b"blob")
    assert store.path == tmp_path / "owner.profile"
    assert store.load() == b"blob"


# --------------------------------------------------------------------------- #
# End-to-end: enroll -> persist -> reload -> recognize owner advisorily
# --------------------------------------------------------------------------- #
def test_enroll_persist_reload_and_recognize(tmp_path: Path) -> None:
    verifier = FakeVoiceprint()
    profile = verifier.enroll([OWNER_SAMPLE, OWNER_SAMPLE_2])
    store = EnrollmentStore(tmp_path / "owner.profile")
    store.save(profile)

    identity = OwnerIdentity(verifier, store.load())
    assert identity.is_owner(OWNER_SAMPLE) is True
    assert identity.is_owner(OTHER_SAMPLE) is False


# --------------------------------------------------------------------------- #
# Real adapter: lazy import, helpful error when backend missing
# --------------------------------------------------------------------------- #
def test_resemblyzer_missing_backend_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "resemblyzer" or name.startswith("resemblyzer."):
            raise ImportError("No module named 'resemblyzer'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ProviderError) as exc:
        ResemblyzerVerifier().enroll([OWNER_SAMPLE])
    assert "install-voice" in str(exc.value)


def test_resemblyzer_missing_numpy_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "numpy" or name.startswith("numpy."):
            raise ImportError("No module named 'numpy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ProviderError) as exc:
        ResemblyzerVerifier().score(OWNER_SAMPLE, b"\x00\x00\x00\x00")
    assert "install-voice" in str(exc.value)


def test_module_import_does_not_require_resemblyzer() -> None:
    import importlib
    import sys

    assert "resemblyzer" not in sys.modules
    importlib.import_module("friday.voice.voiceprint")
    assert "resemblyzer" not in sys.modules
