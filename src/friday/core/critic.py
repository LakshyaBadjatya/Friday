"""Self-critique: a bounded, non-fatal review of a draft persona reply (Tier 2).

Before FRIDAY returns the final synthesized reply, the orchestrator may *review*
it once (gated behind ``FRIDAY_ENABLE_SELF_CRITIQUE``, off by default). The
review has two stages:

1. **Deterministic banned-tone scan (first).** The draft is scanned for the
   persona spec's "Banned Tone Markers" (sycophantic openers, over-apology, fake
   enthusiasm, padding). Any hit is recorded as an issue — this needs no model
   call and is always correct, so a banned opener is caught even if the LLM pass
   is skipped or errors.
2. **One LLM pass.** The model is asked whether the draft (a) answers the user's
   question, (b) avoids fabricated facts/figures, and (c) stays in the FRIDAY
   persona, returning a corrected ``revised`` draft when it does not. The verdict
   is a small JSON object the critic parses.

The review is **bounded** (exactly one LLM pass — the revision is never itself
re-critiqued) and **non-fatal**: ANY parse error, malformed verdict, or LLM/
provider failure yields ``Critique(ok=True, issues=[], revised=None)`` so the
original response is always kept rather than blocked. ``ok`` is true only when
there are no deterministic markers *and* the LLM says the draft is fine.

This module lives in ``core/`` and therefore — like the orchestrator — must NOT
import an LLM SDK (grep-enforced by ``tests/unit/test_architecture.py``). It
depends only on the injected :class:`~friday.providers.llm.LLMProvider`
abstraction; ``app.py`` wires the live provider in.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from friday.providers.llm import LLMProvider, Message

logger = logging.getLogger("friday.core.critic")


class Critique(BaseModel):
    """The outcome of one self-critique review of a draft reply.

    Attributes:
        ok: ``True`` when the draft passed — no deterministic banned-tone marker
            *and* the LLM judged it fine. ``False`` means at least one issue was
            found.
        issues: Short, human-readable descriptions of what was wrong (a banned
            marker that matched, and/or the LLM's stated reason). Empty when
            ``ok``.
        revised: A corrected draft to use *instead of* the original, set only
            when there is a concrete improvement to apply. ``None`` when the
            draft passed, or when a problem was noted but no rewrite was offered.
    """

    ok: bool = True
    issues: list[str] = Field(default_factory=list)
    revised: str | None = None


# The persona spec's "Banned Tone Markers" rendered as deterministic substring
# probes (matched case-insensitively). These mirror the forbidden patterns in
# ``persona/friday.md`` — sycophantic openers, over-apology, fake enthusiasm, and
# padding — so a draft that slipped one past the synthesis prompt is still caught
# without a model call. ``app.py`` passes this exact list in; it is also the
# module default so the critic is usable un-wired in narrow unit tests.
DEFAULT_PERSONA_MARKERS: tuple[str, ...] = (
    # Sycophantic openers / flattery.
    "great question",
    "what a great question",
    "what a fantastic",
    "excellent question",
    "i'd be happy to help",
    "i would be happy to help",
    "happy to help",
    "thanks for asking",
    "thank you for asking",
    # Over-apology / apology theatre.
    "i'm so sorry",
    "i am so sorry",
    "i deeply apologize",
    "i sincerely apologize",
    "my sincerest apologies",
    "i apologize for the confusion",
    # Fake enthusiasm / cheerleading.
    "i'm so excited",
    "i am so excited",
    "i'm thrilled",
    "i am thrilled",
    "absolutely fantastic",
    # Padding / empty preambles.
    "let me help you with that",
    "let me help you with this",
    "i'd be glad to assist",
    "i would be glad to assist",
    "as an ai language model",
    "as an ai assistant",
)


_REVIEW_INSTRUCTIONS = (
    "You are a strict editor reviewing a draft reply written by the FRIDAY "
    "assistant before it is sent to its owner. Judge the draft on three things "
    "ONLY:\n"
    "  (a) does it actually answer the user's message?\n"
    "  (b) is it free of fabricated facts, figures, citations, or tool results "
    "(it must not invent data it could not have)?\n"
    "  (c) does it stay in the FRIDAY persona — confident, direct, answer-first, "
    "honest, no sycophancy or padding?\n"
    "Respond with a SINGLE JSON object and nothing else, of the exact shape:\n"
    '  {"ok": <true|false>, "issues": [<short strings>], '
    '"revised": <a corrected draft string, or null>}\n'
    "Set ok=true and revised=null when the draft is fine. When it is not, set "
    "ok=false, list the concrete issues, and put a corrected reply in revised "
    "that fixes them while changing no real facts. Do not wrap the JSON in "
    "markdown fences or prose."
)


class SelfCritic:
    """A bounded, non-fatal reviewer for a draft persona reply.

    Args:
        llm: The provider used for the single LLM review pass. Only the abstract
            :class:`~friday.providers.llm.LLMProvider` is depended upon, so this
            module never imports an LLM SDK.
        persona_markers: The banned-tone markers (from the persona spec) the
            deterministic scan flags. Matched case-insensitively as substrings.
            Defaults to :data:`DEFAULT_PERSONA_MARKERS`.
    """

    def __init__(
        self,
        llm: LLMProvider,
        *,
        persona_markers: list[str] | tuple[str, ...] = DEFAULT_PERSONA_MARKERS,
    ) -> None:
        self._llm = llm
        # Pre-lower the markers once so the per-review scan is a plain membership
        # test; drop blanks so an empty entry never matches everything.
        self._markers: tuple[str, ...] = tuple(
            m.lower() for m in persona_markers if m and m.strip()
        )

    async def review(self, draft: str, *, user_text: str) -> Critique:
        """Review ``draft`` (the reply to ``user_text``); return a :class:`Critique`.

        Runs the deterministic banned-tone scan FIRST (always, no model call),
        then a single LLM pass. The result is:

        * ``ok`` iff there are no deterministic markers AND the LLM says the draft
          is fine;
        * ``issues`` the union of the matched markers and the LLM's stated issues;
        * ``revised`` the LLM's corrected draft, set only when it offered a
          concrete, non-empty rewrite that differs from the original.

        Non-fatal by construction: any parse/LLM/provider error leaves the LLM
        stage a no-op (it contributes no issue and no revision), so a failure can
        only ever *keep* the original response, never block it.
        """
        marker_issues = self._scan_markers(draft)
        llm_ok, llm_issues, llm_revised = await self._llm_review(draft, user_text)

        issues = [*marker_issues, *llm_issues]
        ok = not marker_issues and llm_ok
        return Critique(ok=ok, issues=issues, revised=llm_revised)

    # -- deterministic stage ---------------------------------------------- #
    def _scan_markers(self, draft: str) -> list[str]:
        """Return one issue per banned-tone marker found in ``draft``."""
        lowered = draft.lower()
        return [
            f"banned tone marker: {marker!r}"
            for marker in self._markers
            if marker in lowered
        ]

    # -- LLM stage (one pass, non-fatal) ----------------------------------- #
    async def _llm_review(
        self, draft: str, user_text: str
    ) -> tuple[bool, list[str], str | None]:
        """One LLM verdict pass. Returns ``(ok, issues, revised)``.

        Any failure (provider error, empty/garbled output, unparseable or
        malformed JSON) degrades to the non-fatal default ``(True, [], None)`` so
        the caller keeps the original response.
        """
        messages = [
            Message(role="system", content=_REVIEW_INSTRUCTIONS),
            Message(
                role="user",
                content=(
                    f"USER MESSAGE:\n{user_text}\n\n"
                    f"DRAFT REPLY:\n{draft}\n\n"
                    "Return the JSON verdict now."
                ),
            ),
        ]
        try:
            response = await self._llm.complete(messages, tools=None)
        except Exception as exc:  # non-fatal: any provider/LLM failure keeps original
            logger.warning("self-critique LLM pass failed; keeping draft: %s", exc)
            return True, [], None

        verdict = self._parse_verdict(response.text)
        if verdict is None:
            return True, [], None
        return verdict

    @staticmethod
    def _parse_verdict(text: str | None) -> tuple[bool, list[str], str | None] | None:
        """Parse the model's JSON verdict, or ``None`` on any malformed output.

        Tolerates an object embedded in surrounding prose / markdown fences by
        extracting the first ``{...}`` span. A non-object, missing ``ok``, or any
        type error yields ``None`` (the caller then treats the pass as a no-op).
        """
        if not text or not text.strip():
            return None
        raw = _extract_json_object(text)
        if raw is None:
            return None
        try:
            data: Any = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict) or "ok" not in data:
            return None

        ok = bool(data.get("ok"))

        issues_raw = data.get("issues", [])
        issues: list[str] = []
        if isinstance(issues_raw, list):
            issues = [str(item) for item in issues_raw if str(item).strip()]

        revised_raw = data.get("revised")
        revised: str | None = None
        if isinstance(revised_raw, str) and revised_raw.strip():
            revised = revised_raw.strip()

        return ok, issues, revised


# First balanced-ish ``{...}`` object span; good enough to peel a JSON verdict out
# of a model reply that wrapped it in prose or ```json fences.
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> str | None:
    """Return the first ``{...}`` JSON-object span in ``text``, or ``None``."""
    match = _JSON_OBJECT.search(text)
    return match.group(0) if match is not None else None
