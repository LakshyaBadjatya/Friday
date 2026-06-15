"""Unit tests for :class:`friday.core.critic.SelfCritic` (Tier 2 self-critique).

Offline, on :class:`~friday.providers.llm.FakeLLM` or a recording/raising stub —
zero network. The pinned behaviours mirror the plan contract:

* The deterministic banned-tone scan flags a draft with a banned opener even when
  the LLM pass returns "ok" (the scan runs FIRST and is authoritative).
* A scripted LLM verdict ``{ok:false, revised:"<fixed>"}`` surfaces the revision.
* A clean draft + an LLM "ok" passes unchanged (``ok=True``, no issues, no
  revision).
* Any LLM/parse error is NON-FATAL: the critique degrades to
  ``Critique(ok=True, issues=[], revised=None)``.
"""

from __future__ import annotations

import json

from friday.core.critic import DEFAULT_PERSONA_MARKERS, Critique, SelfCritic
from friday.providers.llm import (
    FakeLLM,
    LLMResponse,
    Message,
    ToolSpec,
    Usage,
)


def _verdict_resp(
    *, ok: bool, issues: list[str] | None = None, revised: str | None = None
) -> LLMResponse:
    """A scripted LLM response carrying a JSON self-critique verdict."""
    payload = {"ok": ok, "issues": issues or [], "revised": revised}
    return LLMResponse(text=json.dumps(payload), tool_calls=[], usage=Usage())


class _RaisingLLM:
    """An LLM stub whose ``complete`` always raises (to test non-fatal fallback)."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        raise RuntimeError("critic llm exploded")


class _RecordingLLM:
    """A spy LLM that records every ``complete`` call and never produces output.

    Used to assert the critic's LLM is (or is not) reached. ``calls`` counts
    invocations; if it is ever called it raises so a stray call also fails loudly.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        raise AssertionError("critic LLM must not be called in this test")


# --------------------------------------------------------------------------- #
# deterministic banned-tone scan (runs FIRST, no model call needed)
# --------------------------------------------------------------------------- #
async def test_deterministic_scan_flags_banned_opener() -> None:
    # The LLM says the draft is fine, but the deterministic scan must still flag
    # the sycophantic opener — the scan is authoritative and runs first.
    critic = SelfCritic(FakeLLM(responses=[_verdict_resp(ok=True)]))
    draft = "Great question! Two plus two is four, Boss."

    critique = await critic.review(draft, user_text="what's 2+2")

    assert critique.ok is False
    assert any("great question" in issue.lower() for issue in critique.issues)


async def test_scan_uses_injected_persona_markers() -> None:
    # A custom marker list is honored; a draft containing it is flagged.
    critic = SelfCritic(
        FakeLLM(responses=[_verdict_resp(ok=True)]),
        persona_markers=["super duper"],
    )
    critique = await critic.review(
        "Super duper, here's your answer.", user_text="hi"
    )
    assert critique.ok is False
    assert critique.issues


async def test_default_markers_cover_persona_banned_list() -> None:
    # Spot-check the module default carries the persona spec's example openers.
    lowered = {m.lower() for m in DEFAULT_PERSONA_MARKERS}
    assert "great question" in lowered
    assert "i'd be happy to help" in lowered


# --------------------------------------------------------------------------- #
# LLM verdict pass
# --------------------------------------------------------------------------- #
async def test_clean_draft_passes_unchanged() -> None:
    critic = SelfCritic(FakeLLM(responses=[_verdict_resp(ok=True)]))
    draft = "Four, Boss. Basic arithmetic holds."

    critique = await critic.review(draft, user_text="what's 2+2")

    assert critique == Critique(ok=True, issues=[], revised=None)


async def test_llm_flags_and_offers_revision() -> None:
    fixed = "Four, Boss."
    critic = SelfCritic(
        FakeLLM(
            responses=[
                _verdict_resp(
                    ok=False, issues=["did not answer"], revised=fixed
                )
            ]
        )
    )
    critique = await critic.review(
        "Let me think about that...", user_text="what's 2+2"
    )

    assert critique.ok is False
    assert critique.revised == fixed
    assert "did not answer" in critique.issues


async def test_llm_verdict_embedded_in_prose_is_parsed() -> None:
    # A model that wraps the JSON in fences/prose is still parsed (first {...}).
    payload = json.dumps({"ok": False, "issues": ["x"], "revised": "Fixed, Boss."})
    text = f"Here is my verdict:\n```json\n{payload}\n```\nThanks."
    critic = SelfCritic(FakeLLM(responses=[LLMResponse(text=text)]))

    critique = await critic.review("draft", user_text="q")

    assert critique.ok is False
    assert critique.revised == "Fixed, Boss."


# --------------------------------------------------------------------------- #
# NON-FATAL: any LLM/parse error keeps the original (passing critique)
# --------------------------------------------------------------------------- #
async def test_llm_error_is_non_fatal() -> None:
    critic = SelfCritic(_RaisingLLM())  # type: ignore[arg-type]

    critique = await critic.review(
        "Four, Boss.", user_text="what's 2+2"
    )  # must not raise

    assert critique == Critique(ok=True, issues=[], revised=None)


async def test_exhausted_fake_llm_is_non_fatal() -> None:
    # FakeLLM raises ProviderError when its script is empty; the critic swallows it.
    critic = SelfCritic(FakeLLM(responses=[]))

    critique = await critic.review("Four, Boss.", user_text="q")

    assert critique == Critique(ok=True, issues=[], revised=None)


async def test_unparseable_llm_output_is_non_fatal() -> None:
    critic = SelfCritic(FakeLLM(responses=[LLMResponse(text="not json at all")]))

    critique = await critic.review("Four, Boss.", user_text="q")

    assert critique == Critique(ok=True, issues=[], revised=None)


async def test_empty_llm_output_is_non_fatal() -> None:
    critic = SelfCritic(FakeLLM(responses=[LLMResponse(text=None)]))

    critique = await critic.review("Four, Boss.", user_text="q")

    assert critique == Critique(ok=True, issues=[], revised=None)


async def test_malformed_verdict_missing_ok_is_non_fatal() -> None:
    # Valid JSON but missing the required ``ok`` key -> treated as a no-op pass.
    critic = SelfCritic(
        FakeLLM(responses=[LLMResponse(text='{"issues": ["x"], "revised": "y"}')])
    )

    critique = await critic.review("Four, Boss.", user_text="q")

    assert critique == Critique(ok=True, issues=[], revised=None)


# --------------------------------------------------------------------------- #
# combined: deterministic + LLM
# --------------------------------------------------------------------------- #
async def test_revision_kept_even_when_only_llm_flags() -> None:
    # No banned marker present; the LLM alone flags + revises.
    critic = SelfCritic(
        FakeLLM(responses=[_verdict_resp(ok=False, revised="Better, Boss.")])
    )
    critique = await critic.review("A draft with no markers.", user_text="q")
    assert critique.ok is False
    assert critique.revised == "Better, Boss."


async def test_marker_and_llm_issues_are_unioned() -> None:
    critic = SelfCritic(
        FakeLLM(
            responses=[_verdict_resp(ok=False, issues=["fabricated a figure"])]
        )
    )
    critique = await critic.review(
        "Great question! The answer is 42%.", user_text="q"
    )
    assert critique.ok is False
    assert any("great question" in i.lower() for i in critique.issues)
    assert "fabricated a figure" in critique.issues
