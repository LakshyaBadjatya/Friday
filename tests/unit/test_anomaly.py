"""Unit tests for the rolling-z-score :class:`AnomalyDetector` (proactive slice).

Pure and deterministic: every case is a hand-built ``list[float]`` with a known
answer. No clock, no network, no LLM. The detector keeps a rolling
mean/standard-deviation over the preceding window and flags any point whose
absolute z-score meets the configured threshold.

Covered:
* An injected spike in an otherwise steady series is flagged with the right
  index/value and a z-score above threshold.
* A perfectly flat series (zero variance) yields no anomalies — a constant run
  must never be called anomalous.
* Determinism: the same series produces byte-identical results twice.
* ``z_threshold`` is honoured (a lower threshold flags a milder bump that a
  higher threshold ignores).
* Short series (fewer points than the warm-up window) yield nothing.
* A negative spike (downward outlier) is flagged just like an upward one.
"""

from __future__ import annotations

from friday.proactive import AnomalyDetector
from friday.proactive.anomaly import Anomaly


def test_spike_is_flagged() -> None:
    """A single large jump in a steady (noisy) series is detected at its index.

    The baseline has small natural variation so the rolling std is non-zero; the
    detector's zero-variance guard (a perfectly flat run is never anomalous) is
    exercised separately in :func:`test_flat_series_has_no_anomalies`.
    """
    series = [10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0, 50.0]
    detector = AnomalyDetector(z_threshold=3.0)

    anomalies = detector.detect(series)

    assert len(anomalies) == 1
    spike = anomalies[0]
    assert isinstance(spike, Anomaly)
    assert spike.index == 7
    assert spike.value == 50.0
    assert spike.zscore > 3.0


def test_flat_series_has_no_anomalies() -> None:
    """A constant (zero-variance) series is never anomalous."""
    detector = AnomalyDetector(z_threshold=3.0)

    assert detector.detect([5.0] * 12) == []


def test_empty_and_short_series_yield_nothing() -> None:
    """Series shorter than the warm-up window produce no anomalies."""
    detector = AnomalyDetector(z_threshold=3.0)

    assert detector.detect([]) == []
    assert detector.detect([1.0]) == []
    assert detector.detect([1.0, 2.0]) == []


def test_detection_is_deterministic() -> None:
    """Running the same input twice gives identical results."""
    series = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 20.0, 1.0, 2.0]
    detector = AnomalyDetector(z_threshold=3.0)

    first = detector.detect(series)
    second = detector.detect(series)

    assert [(a.index, a.value, a.zscore) for a in first] == [
        (a.index, a.value, a.zscore) for a in second
    ]


def test_threshold_is_honoured() -> None:
    """A milder bump is flagged at a low threshold but ignored at a high one."""
    series = [10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 15.0]

    lenient = AnomalyDetector(z_threshold=3.0).detect(series)
    strict = AnomalyDetector(z_threshold=8.0).detect(series)

    assert [a.index for a in lenient] == [6]
    assert strict == []


def test_negative_spike_is_flagged() -> None:
    """A downward outlier is flagged the same as an upward one."""
    series = [100.0, 101.0, 99.0, 100.0, 101.0, 99.0, 100.0, 10.0]
    detector = AnomalyDetector(z_threshold=3.0)

    anomalies = detector.detect(series)

    assert len(anomalies) == 1
    assert anomalies[0].index == 7
    assert anomalies[0].value == 10.0
    assert anomalies[0].zscore > 3.0
