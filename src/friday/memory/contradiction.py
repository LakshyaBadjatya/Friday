# © Lakshya Badjatya — Author
"""Contradiction detection: flag a new fact that conflicts with stored memory.

Part of the honesty spine. Before FRIDAY commits a new fact, it can check whether
that fact contradicts something it already believes — so the owner is warned
about the conflict instead of memory silently holding two incompatible claims.

The check is one bounded LLM pass over the candidate fact and the existing facts;
the model returns a small JSON verdict that this module parses. It is **non-fatal
and conservative**: no stored facts, a provider error, or an unparseable verdict
all resolve to "no contradiction found" (with an honest explanation) rather than
raising or fabricating a conflict. Like the rest of the memory layer it imports no
LLM SDK — only the :class:`~friday.providers.llm.LLMProvider` contract — and reads
no configuration.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel

from friday.errors import ProviderError
from friday.memory.citations import Source
from friday.providers.llm import LLMProvider, Message

logger = logging.getLogger("friday.memory.contradiction")


class ContradictionResult(BaseModel):
    """The verdict on whether a new fact contradicts stored memory.

    Attributes:
        contradicts: Whether a conflict was found.
        conflicting_source_id: The id of the stored fact it conflicts with, if any.
        explanation: A short, human-readable rationale (always set).
    """

    contradicts: bool
    conflicting_source_id: str | None = None
    explanation: str = ""


class ContradictionDetector:
    """Checks a candidate fact against stored facts via one bounded LLM pass."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def check(
        self, new_fact: str, existing: list[Source]
    ) -> ContradictionResult:
        """Return whether ``new_fact`` contradicts any of ``existing`` (non-fatal)."""
        if not existing:
            return ContradictionResult(
                contradicts=False, explanation="no stored facts to compare against"
            )
        listing = "\n".join(f"[{s.source_id}] {s.text}" for s in existing)
        prompt = (
            "Decide whether the NEW fact contradicts any STORED fact. Reply with "
            'ONLY a JSON object: {"contradicts": bool, "source_id": string or '
            'null, "why": string}. source_id is the id of the conflicting stored '
            "fact when contradicts is true.\n\n"
            f"NEW: {new_fact}\n\nSTORED:\n{listing}"
        )
        try:
            response = await self._llm.complete(
                [Message(role="user", content=prompt)], None
            )
        except ProviderError as exc:
            logger.warning("contradiction check unavailable: %s", exc)
            return ContradictionResult(
                contradicts=False, explanation="contradiction check was unavailable"
            )
        return self._parse(response.text, existing)

    @staticmethod
    def _parse(
        text: str | None, existing: list[Source]
    ) -> ContradictionResult:
        """Parse the model's JSON verdict; conservative default on any malformed reply."""
        default = ContradictionResult(
            contradicts=False, explanation="no contradiction detected"
        )
        if not text:
            return default
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return default
        try:
            data = json.loads(text[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return default
        if not isinstance(data, dict) or not data.get("contradicts"):
            return default
        raw_id = data.get("source_id")
        known = {s.source_id for s in existing}
        source_id = raw_id if isinstance(raw_id, str) and raw_id in known else None
        why = data.get("why")
        return ContradictionResult(
            contradicts=True,
            conflicting_source_id=source_id,
            explanation=why if isinstance(why, str) and why else "a conflict was found",
        )
