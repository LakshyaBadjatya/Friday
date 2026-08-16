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
    "the working tight but complete."
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


def _ask(key: str, question: str) -> str | None:
    """One reasoning call. Blocking; callers use a worker thread."""
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": f"{_PROMPT}\n\nProblem:\n{question}"}]}],
            # Deterministic. A physics answer should not vary between asks, and
            # sampling is what lets a plausible-looking wrong step through.
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1200},
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
    # Discord caps a message at 2000 characters and rejects the whole send if it
    # is longer, so a long derivation is trimmed at a line boundary rather than
    # mid-equation.
    if len(answer) > 1800:
        answer = answer[:1800].rsplit("\n", 1)[0] + "\n…"
    return answer or None
