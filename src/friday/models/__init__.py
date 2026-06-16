# © Lakshya Badjatya — Author
"""Model catalog + multi-model gateway + per-turn budgeter for FRIDAY.

This package owns the listing of the verified free/free-tier LLM models
(:mod:`friday.models.catalog`), the :class:`~friday.models.gateway.ModelGateway`
that fronts the configured providers as a single drop-in
:class:`~friday.providers.llm.LLMProvider` — resolving a turn to an active model,
falling back on provider error, fanning out a side-by-side compare, and asking a
judge model to pick the best answer — and the pure per-turn cost/latency
:class:`~friday.models.budget.Budgeter`.

No module here imports an LLM SDK: the catalog is a pure data lookup, the budgeter
is pure arithmetic, and the gateway depends only on the
:class:`~friday.providers.llm.LLMProvider` contract, so the ``openai`` import
stays confined to :mod:`friday.providers.llm`.
"""

from __future__ import annotations

from friday.models.budget import Budgeter, TurnBudget
from friday.models.catalog import DEFAULT_CATALOG, ModelCatalog, ModelInfo
from friday.models.gateway import CompareResult, ModelGateway

__all__ = [
    "DEFAULT_CATALOG",
    "Budgeter",
    "CompareResult",
    "ModelCatalog",
    "ModelGateway",
    "ModelInfo",
    "TurnBudget",
]
