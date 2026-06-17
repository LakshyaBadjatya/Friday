# © Lakshya Badjatya — Author
"""Unit tests for the offline lexicon sentiment analyzer."""

from __future__ import annotations

from friday.nlp.sentiment import SentimentAnalyzer


def _a() -> SentimentAnalyzer:
    return SentimentAnalyzer()


def test_positive_text() -> None:
    r = _a().analyze("This is great, I love it — works perfectly. Thanks!")
    assert r.label == "positive"
    assert r.score > 0.1
    assert "love" in r.positive_hits


def test_negative_text() -> None:
    r = _a().analyze("Terrible. It crashed again and the error is frustrating.")
    assert r.label == "negative"
    assert r.score < -0.1
    assert "crashed" in r.negative_hits


def test_neutral_text_has_zero_score() -> None:
    r = _a().analyze("The meeting is scheduled for three o'clock on Tuesday.")
    assert r.label == "neutral"
    assert r.score == 0.0
    assert r.positive_hits == []
    assert r.negative_hits == []


def test_negation_flips_positive_to_negative() -> None:
    r = _a().analyze("This is not good.")
    # "good" negated -> counts as a negative hit.
    assert "good" in r.negative_hits
    assert r.label == "negative"


def test_negation_flips_negative_to_positive() -> None:
    r = _a().analyze("Honestly it's not bad at all.")
    assert "bad" in r.positive_hits
    assert r.label == "positive"


def test_contraction_negation_is_handled() -> None:
    r = _a().analyze("It isn't working.")
    # "working" is positive; "isn't" -> "is not" negates it.
    assert "working" in r.negative_hits


def test_score_is_normalized_between_minus_one_and_one() -> None:
    r = _a().analyze("good great love hate")  # 3 pos, 1 neg
    assert -1.0 <= r.score <= 1.0
    assert r.score == round((3 - 1) / 4, 4)
