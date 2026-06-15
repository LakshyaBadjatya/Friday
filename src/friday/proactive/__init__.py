"""Proactive intelligence (Tier 1): anomaly detection and rule-based foresight.

This package gives FRIDAY two small, pure, dependency-injected building blocks
for *proactive* behaviour — surfacing things the user has not asked about yet:

* :class:`~friday.proactive.anomaly.AnomalyDetector` — flags outliers in a
  numeric series via a causal rolling z-score (pure + deterministic).
* :class:`~friday.proactive.foresight.Foresight` — turns recent events into
  short, explainable suggestions via simple deterministic rules (rising metric,
  reminder due soon, recurring pattern), with an optional best-effort LLM
  phraser for nicer wording.

Both take every dependency as a parameter and import no FRIDAY config/app and no
LLM SDK, so they are trivially testable and safe to wire from the orchestrator.
"""

from __future__ import annotations

from friday.proactive.anomaly import Anomaly, AnomalyDetector
from friday.proactive.foresight import Foresight, Phraser, Suggestion

__all__ = [
    "Anomaly",
    "AnomalyDetector",
    "Foresight",
    "Phraser",
    "Suggestion",
]
