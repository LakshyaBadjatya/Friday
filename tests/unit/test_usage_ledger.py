# © Lakshya Badjatya — Author
"""Unit tests for the process usage/cost ledger (cost-dashboard data)."""

from __future__ import annotations

from friday.observability.usage import UsageLedger


def test_empty_snapshot_is_valid_and_zeroed() -> None:
    snap = UsageLedger().snapshot()
    assert snap == {
        "completions": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tokens": 0,
        "usd": 0.0,
        "by_model": {},
    }


def test_record_accumulates_totals_and_derives_tokens() -> None:
    ledger = UsageLedger()
    ledger.record("openrouter:gpt-oss-20b", prompt_tokens=10, completion_tokens=5)
    ledger.record("openrouter:gpt-oss-20b", prompt_tokens=3, completion_tokens=2)
    snap = ledger.snapshot()
    assert snap["completions"] == 2
    assert snap["prompt_tokens"] == 13
    assert snap["completion_tokens"] == 7
    assert snap["tokens"] == 20  # prompt + completion, surfaced for the dashboard
    assert snap["usd"] == 0.0


def test_per_model_breakdown_is_separate() -> None:
    ledger = UsageLedger()
    ledger.record("a:one", prompt_tokens=10, completion_tokens=0, usd=0.01)
    ledger.record("b:two", prompt_tokens=4, completion_tokens=4)
    by_model = ledger.snapshot()["by_model"]
    assert by_model["a:one"] == {
        "completions": 1,
        "prompt_tokens": 10,
        "completion_tokens": 0,
        "tokens": 10,
        "usd": 0.01,
    }
    assert by_model["b:two"]["tokens"] == 8
    assert by_model["b:two"]["usd"] == 0.0


def test_dollars_accumulate_and_round() -> None:
    ledger = UsageLedger()
    ledger.record("x:y", usd=0.1)
    ledger.record("x:y", usd=0.2)
    # 0.1 + 0.2 is 0.30000000000000004 in float; the snapshot rounds it off.
    assert ledger.snapshot()["usd"] == 0.3


def test_snapshot_is_a_copy_and_cannot_corrupt_live_tallies() -> None:
    ledger = UsageLedger()
    ledger.record("x:y", prompt_tokens=5)
    snap = ledger.snapshot()
    snap["completions"] = 999
    snap["by_model"]["x:y"]["tokens"] = 999
    snap["by_model"]["injected"] = {}
    fresh = ledger.snapshot()
    assert fresh["completions"] == 1
    assert fresh["by_model"]["x:y"]["tokens"] == 5
    assert "injected" not in fresh["by_model"]
