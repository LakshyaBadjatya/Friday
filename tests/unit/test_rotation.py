# © Lakshya Badjatya — Author
"""Unit tests for secret-rotation reminders."""

from __future__ import annotations

import pytest

from friday.security.rotation import RotationPolicy, SecretAge


def test_due_at_or_past_max_age() -> None:
    policy = RotationPolicy(max_age_seconds=100.0)
    fresh = SecretAge(name="api_key", last_rotated_ts=950.0)
    stale = SecretAge(name="old_token", last_rotated_ts=0.0)
    assert policy.status(fresh, now=1000.0).due is False  # age 50 < 100
    assert policy.status(stale, now=1000.0).due is True  # age 1000 >= 100


def test_boundary_age_is_due() -> None:
    policy = RotationPolicy(max_age_seconds=100.0)
    s = SecretAge(name="k", last_rotated_ts=0.0)
    assert policy.status(s, now=100.0).due is True  # age == max -> due


def test_future_timestamp_clamps_to_zero_age() -> None:
    policy = RotationPolicy(max_age_seconds=100.0)
    s = SecretAge(name="k", last_rotated_ts=2000.0)
    status = policy.status(s, now=1000.0)
    assert status.age_seconds == 0.0
    assert status.due is False


def test_due_filters_and_preserves_order() -> None:
    policy = RotationPolicy(max_age_seconds=10.0)
    secrets = [
        SecretAge(name="a", last_rotated_ts=0.0),   # due
        SecretAge(name="b", last_rotated_ts=95.0),  # fresh
        SecretAge(name="c", last_rotated_ts=80.0),  # due (age 20)
    ]
    assert policy.due(secrets, now=100.0) == ["a", "c"]


def test_nonpositive_max_age_rejected() -> None:
    with pytest.raises(ValueError):
        RotationPolicy(max_age_seconds=0.0)
