"""Unit tests for :mod:`friday.siri.arithmetic` — computed maths, not predicted.

Three things matter here and each has its own group below:

* **Correctness.** The whole point of the module is that the number is right, so
  every expected value is written out literally rather than recomputed in the
  test (a test asserting ``reply == f"That's {12 * 37}."`` would still pass if
  the evaluator and the test shared a bug).
* **Narrowness.** Firing on prose would replace a good LLM answer with "That's
  42." — so anything that is not unambiguously a sum must return ``None``.
* **Safety.** The input is spoken text arriving over the internet; the evaluator
  must reject names/calls and refuse to burn CPU on a huge exponent.
"""

from __future__ import annotations

import pytest

from friday.siri.arithmetic import arithmetic_reply


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # Plain symbolic arithmetic.
        ("what is 12 * 37", "That's 444."),
        ("23 * 47 + 19", "That's 1,100."),
        ("what is 15 minus 4", "That's 11."),
        ("(45 + 15) divided by 4", "That's 15."),
        # Operator precedence must survive the word -> symbol rewrite.
        ("5 plus 7 times 2", "That's 19."),
        # Spoken multiplication.
        ("2 x 3", "That's 6."),
        ("8 multiplied by 9", "That's 72."),
        # Percentages — the sum models most often get wrong.
        ("what is 20% of 80", "That's 16."),
        ("whats 17 percent of 2,480", "That's 421.6."),
        # Roots and powers.
        ("square root of 144", "That's 12."),
        ("12 squared", "That's 144."),
        ("5 cubed", "That's 125."),
        ("2 to the power of 10", "That's 1,024."),
        # Non-integer results are rounded to something speakable.
        ("what is 1 divided by 3", "That's 0.3333."),
        # Thousands separators are grouped so TTS reads them as words.
        ("1234 * 1000", "That's 1,234,000."),
    ],
)
def test_computes_pure_arithmetic(query: str, expected: str) -> None:
    assert arithmetic_reply(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        # Prose that merely mentions maths — belongs to the model.
        "how do you calculate compound interest",
        "explain the quadratic formula",
        "what is the formula for kinetic energy",
        "solve for x when 2x + 3 = 9",
        # A number with no operation is not a calculation.
        "what is 42",
        "set a reminder for 5 pm",
        # No digits at all.
        "what was the last topic",
        "tell me the weather in delhi",
        # Ambiguous spoken discount phrasing — deliberately not handled.
        "whats 20% off a 50 dollar shirt",
    ],
)
def test_falls_through_on_non_arithmetic(query: str) -> None:
    assert arithmetic_reply(query) is None


@pytest.mark.parametrize(
    "query",
    [
        "100 / 0",  # ZeroDivisionError must not surface as a 500
        "9**9**9",  # exponent bound — must not hang the worker
        "2 to the power of 100000",
        '__import__("os").system("ls")',  # names/calls are rejected outright
        "open('/etc/passwd').read()",
        "1 + [2]",
    ],
)
def test_unsafe_or_unevaluable_input_returns_none(query: str) -> None:
    assert arithmetic_reply(query) is None


def test_empty_and_blank_input_are_safe() -> None:
    assert arithmetic_reply("") is None
    assert arithmetic_reply("   ") is None


def test_conversational_lead_ins_are_stripped() -> None:
    """Siri sends whole spoken sentences, not bare expressions."""
    assert arithmetic_reply("hey friday what's 6 * 7?") == "That's 42."
    assert arithmetic_reply("calculate 100 - 1") == "That's 99."
