"""Automatic filing: what kind of thing is this, and what is it about?

No folders, ever. A capture is classified into a kind (problem, document,
receipt, screenshot, photo, board), a subject and chapter when it is study
material, and a set of lowercase tags. That classification is the only
organisation the vault has, which is why it must be cheap and never blocking:
an unparseable reply — or a provider that fails outright — degrades to
"unknown" rather than failing the capture.

Depends only on the :class:`~friday.providers.llm.LLMProvider` contract.
"""

from __future__ import annotations

import json
import logging

from friday.providers.llm import LLMProvider, Message
from friday.vault.models import Classification

logger = logging.getLogger("friday.vault.organizer")

_PROMPT = """You are filing one capture into FRIDAY's vault.

Reply with ONLY a JSON object, no prose, no code fence:
{"kind": "...", "subject": "...", "chapter": "...", "tags": ["...", "..."]}

kind is one of: problem, document, receipt, screenshot, photo, board, unknown.
subject and chapter apply to study material only; use "" otherwise.
tags: at most five short lowercase keywords.

Extracted text:
---
%s
---
Caption: %s
"""


def _parse(reply: str) -> Classification:
    """Decode the model's reply, degrading to ``unknown`` rather than raising."""
    text = reply.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```")
        text = text.removesuffix("```").strip()

    # Tolerate prose wrapped around the JSON object by slicing to the
    # outermost braces before decoding.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return Classification()
    text = text[start : end + 1]

    try:
        raw = json.loads(text)
    except (ValueError, TypeError):
        return Classification()
    if not isinstance(raw, dict):
        return Classification()

    seen: list[str] = []
    tags = raw.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            if not isinstance(tag, str):
                continue
            lowered = tag.strip().lower()
            if lowered and lowered not in seen:
                seen.append(lowered)

    kind = raw.get("kind")
    subject = raw.get("subject")
    chapter = raw.get("chapter")
    return Classification(
        kind=kind if isinstance(kind, str) and kind else "unknown",
        subject=subject if isinstance(subject, str) else "",
        chapter=chapter if isinstance(chapter, str) else "",
        tags=seen[:5],
    )


class Organizer:
    """Classifies a capture from its extracted text and caption."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def classify(self, *, ocr_text: str, caption: str) -> Classification:
        """Classify one capture; never raises, never blocks the pipeline.

        A provider outage, timeout, or any other failure from ``complete``
        must not fail the capture it is describing — the classification is
        metadata, not a gate — so every exception degrades to ``unknown``
        just like an unparseable reply does.
        """
        if not (ocr_text.strip() or caption.strip()):
            return Classification()

        prompt = _PROMPT % (ocr_text[:4000], caption[:200])
        try:
            response = await self._llm.complete([Message(role="user", content=prompt)])
        except Exception:
            logger.warning("vault organizer: LLM call failed, filing as unknown", exc_info=True)
            return Classification()

        return _parse(response.text or "")
