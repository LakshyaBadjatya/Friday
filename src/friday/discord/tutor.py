"""Answering problems with a model that can actually do them.

Ordinary chat runs on a small free model, which is the right trade for banter:
fast, cheap, and nobody minds if a joke lands imperfectly. It is the wrong trade
entirely for a physics problem. Asked for the terminal voltage of a charging
cell it produced three separate errors in two attempts — an invented formula, a
current computed without subtracting the back-emf, and the discharging sign
convention used for charging — and delivered all of it with complete confidence.

The owner is sitting Class 12 PCM exams. A wrong answer in a confident tone is
worse than no answer, because it gets written down.

So problems are routed here instead, to a reasoning-grade model, with a prompt
that forces the working to be shown rather than skipped. Two things matter more
than fluency and are stated as such: the *form* of a law has to be checked
against the situation before it is used, and the result has to be substituted
back. Both are what the small model skipped.

Free, like everything else here — this is the same Gemini key already doing
transcription and vision.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any

import anyio

from friday.logging import get_logger

logger = get_logger("friday.discord.tutor")

#: A reasoning-grade model. The chat model is chosen for speed and personality;
#: this one is chosen for getting the number right.
_MODEL = "gemini-2.5-flash"

_PROMPT = (
    "You are solving a problem for a Class 12 PCM student preparing for exams. "
    "Accuracy is the only thing that matters here.\n\n"
    "Work it properly:\n"
    "1. List what is given, with units, and what is asked for.\n"
    "2. Name the principle you are using, and check its FORM against this "
    "situation before applying it. Sign conventions are where these go wrong: a "
    "cell being CHARGED has terminal voltage V = E + I·r, a cell DISCHARGING has "
    "V = E − I·r. When one source drives current against another emf, the net "
    "driving voltage is their difference, not the full source voltage.\n"
    "3. Show every step with its arithmetic. No skipped lines.\n"
    "4. Substitute the answer back into the original relation and confirm it "
    "holds. Say so.\n"
    "5. State the final answer on its own line with units.\n\n"
    "If a step is genuinely uncertain, say which one rather than presenting a "
    "guess as a result. If the student says their answer differs, find the "
    "specific step that is wrong rather than producing a second answer — two "
    "contradictory answers in a row are worse than one wrong one.\n\n"
    "Format for a chat message: plain text, no LaTeX, no markdown tables. Keep "
    "the working tight but complete.\n\n"
    "Start immediately with the first line of working. No preamble, no 'here is "
    "the solution', no 'following the steps you outlined', no mention of these "
    "instructions or of any format — the person reading this asked a physics "
    "question and wants the physics. If the message genuinely contains no "
    "problem to solve, say that in one short line and nothing else."
)


async def solve(settings: Any, question: str) -> str | None:
    """Work a problem carefully, or ``None`` when the model is unavailable.

    ``None`` means the caller falls back to the ordinary path — a slower answer
    from the usual model beats no answer, and this is an upgrade rather than a
    dependency.
    """
    secret = getattr(settings, "gemini_api_key", None)
    key = secret.get_secret_value() if secret is not None else ""
    if not key or not question.strip():
        return None
    return await anyio.to_thread.run_sync(_ask, key, question.strip())


#: ``{channel: (problem, answer)}``. Asked to "reanalyse your answer" she had no
#: idea which answer, and with nothing to work from the model filled the gap by
#: reciting the persona rules appended to the prompt — a short speech about who
#: is boss and who is queen, in place of the derivation that was asked for.
_LAST: dict[str, tuple[str, str]] = {}


def remember(channel: str, question: str, answer: str) -> None:
    """Hold the last problem and its answer so a follow-up has a referent."""
    if channel and question.strip() and answer.strip():
        _LAST[channel] = (question.strip(), answer.strip())


#: "reanalyse your answer", "are you sure", "explain that", "why", "check again".
#: What these have in common is that they carry no subject at all — the subject
#: is the previous message, which is exactly what she was missing.
_FOLLOW_UP = re.compile(
    r"\b(?:re-?(?:analyse|analyze|check|do|calculate|solve)"
    r"|are\s+you\s+sure|you\s+sure|check\s+(?:it\s+)?again|try\s+again"
    r"|explain\s+(?:it|that|this|your\s+answer|the\s+answer|again|properly)?"
    r"|continue|carry\s+on|go\s+on|finish\s+(?:it|that)"
    r"|elaborate|in\s+detail|step\s+by\s+step|show\s+(?:the\s+)?(?:working|steps)"
    r"|why\s+(?:is\s+)?(?:that|it|this)|how\s+did\s+you\s+get\s+that"
    r"|that(?:'s|\s+is)\s+(?:wrong|not\s+right|incorrect))\b",
    re.IGNORECASE,
)


def is_follow_up(channel: str, text: str) -> bool:
    """Whether this asks about the previous answer rather than a new problem."""
    return bool(_LAST.get(channel)) and bool(_FOLLOW_UP.search(text or ""))


async def revisit(settings: Any, channel: str, text: str) -> str | None:
    """Re-work the previous problem in light of what was just said.

    Deliberately re-derives from the original problem rather than editing the
    previous answer. Asked to check its work, a model handed its own answer will
    usually defend it; handed the problem again with the answer as a claim to
    test, it will actually redo the algebra.
    """
    remembered = _LAST.get(channel)
    secret = getattr(settings, "gemini_api_key", None)
    key = secret.get_secret_value() if secret is not None else ""
    if remembered is None or not key:
        return None
    problem, previous = remembered
    prompt = (
        f"Original problem:\n{problem}\n\n"
        f"The answer given previously was:\n{previous}\n\n"
        f'The student has now said: "{text.strip()}"\n\n'
        "Work the original problem again from scratch. Do not assume the "
        "previous answer is correct and do not defend it — derive it "
        "independently, then compare. If the previous answer was wrong, say "
        "plainly which step was wrong and what the right one is. If it was "
        "right, say so and show the working that proves it, addressing "
        "whatever the student actually asked about."
    )
    answer = await anyio.to_thread.run_sync(_ask, key, prompt)
    if answer:
        _LAST[channel] = (problem, answer)
    return answer


def _ask(key: str, question: str) -> str | None:
    """One reasoning call. Blocking; callers use a worker thread."""
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": f"{_PROMPT}\n\nProblem:\n{question}"}]}],
            "generationConfig": {
                # Deterministic. A physics answer should not vary between asks,
                # and sampling is what lets a plausible-looking wrong step
                # through.
                "temperature": 0.0,
                # This is a thinking model and the cap counts thought tokens.
                # At 1200 the model spent 1148 of them reasoning and had 48 left
                # to answer with, so every derivation stopped dead partway
                # through the list of givens — the reply looked like a bug in
                # the transport when it was the budget all along.
                "maxOutputTokens": 8192,
                # Enough thinking to get the sign conventions right, not so much
                # that it wanders. The reasoning is what makes this model worth
                # calling; it just cannot have the whole envelope.
                "thinkingConfig": {"thinkingBudget": 1024},
            },
        }
    ).encode()
    request = urllib.request.Request(  # noqa: S310
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{_MODEL}:generateContent?key={key}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:  # noqa: BLE001 - fall back to the ordinary path
        logger.warning("tutor: solve failed")
        return None
    answer = (text or "").strip()
    # No trimming to Discord's message limit here any more. That cap used to be
    # enforced by cutting the text, which reliably removed the last step — the
    # substitution check that makes the answer trustworthy — and left an ellipsis
    # where the confirmation should have been. The sender splits long replies
    # across messages now, so length is its problem and not this function's.
    # The ceiling that remains is a sanity bound, far above any real derivation.
    return answer[:6000] or None
