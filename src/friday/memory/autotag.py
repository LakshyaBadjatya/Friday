# © Lakshya Badjatya — Author
"""Auto-tagging: suggest tags for a note or document via one bounded LLM pass.

Feeds "smart collections" — journal entries and ingested documents get a small
set of normalized tags so they can be grouped and retrieved by theme. The model
proposes tags as a JSON array; this module normalizes them (lowercased, trimmed,
de-duplicated, order preserved) and, when an allow-list is supplied, keeps only
recognized tags so the vocabulary stays controlled.

Non-fatal and offline-shaped: a provider error or an unparseable reply yields
``[]`` (no tags) rather than raising. Depends only on the
:class:`~friday.providers.llm.LLMProvider` contract (no LLM SDK) and reads no
configuration — the allow-list and limit are injected.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable

from friday.errors import ProviderError
from friday.providers.llm import LLMProvider, Message

logger = logging.getLogger("friday.memory.autotag")


class AutoTagger:
    """Suggests normalized tags for text.

    Args:
        llm: Provider for the one tagging pass (only ``complete`` is used).
        allowed_tags: Optional controlled vocabulary; when given, suggested tags
            outside it are dropped. ``None`` permits any tag.
        max_tags: Upper bound on how many tags to return.
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        allowed_tags: Iterable[str] | None = None,
        max_tags: int = 5,
    ) -> None:
        if max_tags <= 0:
            raise ValueError("max_tags must be positive")
        self._llm = llm
        self._allowed = (
            {t.strip().lower() for t in allowed_tags if t.strip()}
            if allowed_tags is not None
            else None
        )
        self._max_tags = max_tags

    async def tag(self, text: str) -> list[str]:
        """Return up to ``max_tags`` normalized tags for ``text`` (``[]`` on failure)."""
        vocab = (
            f" Choose only from: {', '.join(sorted(self._allowed))}."
            if self._allowed
            else ""
        )
        prompt = (
            "Suggest a few short topic tags for the text below. Reply with ONLY a "
            f"JSON array of lowercase strings.{vocab}\n\n{text}"
        )
        try:
            response = await self._llm.complete(
                [Message(role="user", content=prompt)], None
            )
        except ProviderError as exc:
            logger.warning("auto-tagging unavailable: %s", exc)
            return []
        return self._normalize(response.text)

    def _normalize(self, text: str | None) -> list[str]:
        """Parse + normalize the model's JSON tag array (``[]`` on malformed input)."""
        if not text:
            return []
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end < start:
            return []
        try:
            raw = json.loads(text[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return []
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, str):
                continue
            tag = item.strip().lower()
            if not tag or tag in seen:
                continue
            if self._allowed is not None and tag not in self._allowed:
                continue
            seen.add(tag)
            out.append(tag)
            if len(out) >= self._max_tags:
                break
        return out
