"""Rolling z-score anomaly detection over a numeric series (proactive slice).

:class:`AnomalyDetector` flags points in a ``list[float]`` that deviate sharply
from their recent history. For each point it computes the mean and (population)
standard deviation of the *preceding* window and reports the point when the
absolute z-score ``|x - mean| / std`` meets the configured ``z_threshold``.

Design contract:

* **Pure & deterministic.** :meth:`AnomalyDetector.detect` has no clock, no I/O
  and no randomness; the same input always yields the same output. It does not
  mutate the input series.
* **Causal.** Each point is judged only against the points before it, so a spike
  never masks itself by inflating its own window's variance.
* **Zero-variance safe.** A flat (constant) window has ``std == 0``; such a
  window is never treated as anomalous — a steady signal is, by definition, not
  an outlier.

This module has **no** FRIDAY-config / app dependency and imports no LLM SDK; it
takes everything it needs as constructor or method arguments.
"""

from __future__ import annotations

from math import sqrt

from pydantic import BaseModel, Field

#: Default number of preceding points required before scoring begins. Below this
#: warm-up the rolling statistics are too thin to be meaningful.
_DEFAULT_MIN_WINDOW = 3


class Anomaly(BaseModel):
    """A single flagged outlier within a series.

    ``index`` is the position in the original series, ``value`` is the point's
    value and ``zscore`` is the (positive) absolute z-score against the rolling
    window that triggered the flag.
    """

    index: int
    value: float
    zscore: float = Field(ge=0.0)


class AnomalyDetector:
    """Detect outliers in a numeric series via a causal rolling z-score.

    Args:
        z_threshold: Minimum absolute z-score for a point to be flagged. Larger
            values are stricter (fewer flags). Must be positive.
        min_window: Minimum count of preceding points required before a point is
            scored at all (a warm-up). Defaults to :data:`_DEFAULT_MIN_WINDOW`.
        window: Optional fixed look-back size. When ``None`` (default) the whole
            preceding history is used (an expanding window); when set, only the
            most recent ``window`` points are considered (a sliding window).
    """

    def __init__(
        self,
        *,
        z_threshold: float = 3.0,
        min_window: int = _DEFAULT_MIN_WINDOW,
        window: int | None = None,
    ) -> None:
        if z_threshold <= 0.0:
            raise ValueError("z_threshold must be positive")
        if min_window < 2:
            raise ValueError("min_window must be at least 2")
        if window is not None and window < min_window:
            raise ValueError("window must be >= min_window when set")
        self.z_threshold = z_threshold
        self.min_window = min_window
        self.window = window

    def detect(self, series: list[float]) -> list[Anomaly]:
        """Return the anomalies in ``series`` in ascending index order.

        For each point past the warm-up, compute the mean/std of the preceding
        window and flag the point when ``|value - mean| / std >=`` the threshold.
        A constant (zero-variance) window never produces a flag. The input list
        is not modified.
        """
        anomalies: list[Anomaly] = []
        for index in range(self.min_window, len(series)):
            history = self._history(series, index)
            mean, std = _mean_std(history)
            if std == 0.0:
                continue
            zscore = abs(series[index] - mean) / std
            if zscore >= self.z_threshold:
                anomalies.append(
                    Anomaly(index=index, value=series[index], zscore=zscore)
                )
        return anomalies

    def _history(self, series: list[float], index: int) -> list[float]:
        """The preceding points used to score ``series[index]``."""
        start = 0 if self.window is None else max(0, index - self.window)
        return series[start:index]


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return the (mean, population standard deviation) of ``values``.

    ``values`` is assumed non-empty (callers only pass warm-up-sized windows).
    The population (not sample) standard deviation is used so that a single-value
    edge never divides by zero.
    """
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    return mean, sqrt(variance)
