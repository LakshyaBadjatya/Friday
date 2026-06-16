# © Lakshya Badjatya — Author
"""Context compaction: fold an over-long session into a compact summary + tail.

Short-term memory grows turn by turn; left unbounded it eventually crowds the
model's context window. *Compaction* keeps the window healthy by summarizing the
*older* part of a session into a single compact note (one bounded LLM pass) while
retaining the most recent turns verbatim. The caller then replaces the buffer
with ``[summary] + recent tail`` and may persist the summary to long-term memory
(through the existing write-consent policy), so nothing important is silently
dropped — it is condensed and remembered.

This module depends only on the :class:`~friday.providers.llm.LLMProvider`
contract (no LLM SDK) and reads no configuration — the thresholds are injected
and the enable flag is read by ``app.py``. It is **non-fatal**: too little
history, or a failed/empty summary pass, returns ``None`` so the caller simply
keeps the full buffer this turn rather than losing any of it.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from friday.errors import ProviderError
from friday.providers.llm import LLMProvider, Message

logger = logging.getLogger("friday.memory.compaction")


class CompactionResult(BaseModel):
    """The product of one compaction pass.

    Attributes:
        summary: A compact note condensing the older turns (facts, decisions,
            and still-open threads).
        kept: The most recent turns, retained verbatim.
        compacted_count: How many older messages the summary replaced.
    """

    summary: str
    kept: list[Message] = Field(default_factory=list)
    compacted_count: int = 0


class Compactor:
    """Summarizes the older part of a session once it grows past a threshold.

    Args:
        llm: Provider for the one summary pass (only ``complete`` is used).
        keep_recent: How many of the most recent messages to retain verbatim.
        trigger_at: Compaction is attempted only once the history exceeds this
            many messages, so short sessions are never touched.
    """

    def __init__(
        self, llm: LLMProvider, *, keep_recent: int = 6, trigger_at: int = 16
    ) -> None:
        if keep_recent < 0:
            raise ValueError("keep_recent must be non-negative")
        if trigger_at < keep_recent:
            raise ValueError("trigger_at must be >= keep_recent")
        self._llm = llm
        self._keep_recent = keep_recent
        self._trigger_at = trigger_at

    async def maybe_compact(
        self, history: list[Message]
    ) -> CompactionResult | None:
        """Compact the older turns when ``history`` is over threshold, else ``None``.

        Returns ``None`` (caller keeps the full buffer) when the history is at or
        below ``trigger_at``, when there is nothing older than the retained tail,
        or when the summary pass fails / comes back empty — so compaction can only
        ever condense, never drop, the conversation.
        """
        if len(history) <= self._trigger_at:
            return None
        older = history[: len(history) - self._keep_recent]
        if not older:
            return None
        summary = await self._summarize(older)
        if summary is None:
            return None
        kept = history[len(history) - self._keep_recent :] if self._keep_recent else []
        return CompactionResult(
            summary=summary, kept=kept, compacted_count=len(older)
        )

    async def _summarize(self, messages: list[Message]) -> str | None:
        """One bounded, non-fatal summary pass; ``None`` on failure/empty reply."""
        transcript = "\n".join(
            f"{m.role}: {m.content}" for m in messages if m.content
        )
        prompt = (
            "Summarize the following conversation into a compact note that "
            "preserves the concrete facts, decisions made, and any still-open "
            "threads. Be faithful — do not add anything not present.\n\n"
            f"{transcript}"
        )
        try:
            response = await self._llm.complete(
                [Message(role="user", content=prompt)], None
            )
        except ProviderError as exc:
            logger.warning("compaction summary failed; keeping full buffer: %s", exc)
            return None
        text = (response.text or "").strip()
        return text or None
