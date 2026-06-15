"""Contract tests for ``persona/friday.md`` (Task 1.8).

The persona spec is a load-bearing prompt asset: the orchestrator injects it to
shape FRIDAY's voice and to enforce honesty/safety guardrails. These tests pin
the required sections and markers so the contract cannot silently regress (a
deleted "defensive-only" rule or a missing banned-markers list would otherwise
pass unnoticed).
"""

from __future__ import annotations

from pathlib import Path

PERSONA_PATH = Path(__file__).resolve().parents[2] / "src" / "friday" / "persona" / "friday.md"


def _persona_text() -> str:
    return PERSONA_PATH.read_text(encoding="utf-8")


def test_persona_file_exists() -> None:
    assert PERSONA_PATH.is_file(), f"persona spec missing at {PERSONA_PATH}"


def test_persona_is_non_trivial() -> None:
    # Guard against an empty/stub file passing the marker checks by accident.
    assert len(_persona_text().strip()) > 400


def test_addresses_owner_as_boss() -> None:
    text = _persona_text()
    assert "Boss" in text
    # The configurability of the address must be documented.
    assert "FRIDAY_OWNER_ADDRESS" in text


def test_tone_section_present() -> None:
    lower = _persona_text().lower()
    assert "tone" in lower
    # Core tonal qualities from the build-spec.
    for marker in ("confident", "dry", "irish"):
        assert marker in lower, f"tone marker {marker!r} missing"


def test_brevity_answer_first() -> None:
    lower = _persona_text().lower()
    assert "brevity" in lower
    assert "answer-first" in lower or "answer first" in lower


def test_honesty_rule_present() -> None:
    lower = _persona_text().lower()
    assert "honesty" in lower
    # The core honesty rule: never fabricate capability/data/confidence.
    assert "fabricate" in lower
    assert "uncertainty" in lower


def test_banned_tone_markers_section_present() -> None:
    text = _persona_text()
    lower = text.lower()
    # An explicit banned-markers section the orchestrator/tests can rely on.
    assert "banned" in lower
    for marker in ("sycophantic", "apolog", "enthusiasm", "padding"):
        assert marker in lower, f"banned-marker entry {marker!r} missing"


def test_safety_defensive_only_rule_present() -> None:
    lower = _persona_text().lower()
    assert "safety" in lower
    assert "defensive-only" in lower or "defensive only" in lower
    # Named out-of-scope refusals from the build-spec.
    assert "facial recognition" in lower
    assert "tracking" in lower
    assert "offensive cyber" in lower
