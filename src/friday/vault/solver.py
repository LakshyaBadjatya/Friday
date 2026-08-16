"""The always-full ensemble: three drafts, one independent check, one judge.

Every captured problem runs the whole path — no fast path, no sampling. Three of
FRIDAY's roster operators draft a solution to the same problem, SymPy
independently re-derives the algebra without ever seeing what they said, and a
fourth model call (a judge) is spent only when the drafts disagree. The point of
the SymPy pass is that it is *not* a model: it is what would catch a confident
wrong answer that three models happen to agree on — see the module-level note
below on how far that check actually reaches.

Dissent is recorded on the :class:`~friday.vault.models.Solve` rather than
discarded — a minority answer that turns out right is exactly the thing worth
being able to look back at.

Depends only on the :class:`~friday.providers.llm.LLMProvider` contract (a list
of :class:`~friday.providers.llm.Message` in, an
:class:`~friday.providers.llm.LLMResponse` out) — it imports no LLM SDK.

A note on ``verify_with_sympy``, honestly: it only ever confirms something when
the text handed to it is *itself* already a bare, single-variable equation with
no surrounding words ("2x + 4 = 10", not "Solve for x: 2x + 4 = 10, in volts.").
Raw OCR of a photographed word problem almost never looks like that. Regexing
an equation out of the surrounding prose was tried and rejected: on
"2x + 4 = 10" it dropped the coefficient and silently changed the root from 3
to 6 — a verifier that corrupts what it checks is worse than one that declines.

So instead, each drafting operator is now *asked* to also emit the decisive
equation of its own solution in plain ASCII, and that — not the raw OCR
statement — is what gets verified. **This is weaker than a truly independent
derivation: the model supplying the equation is the same model being
checked.** A model that sets up the physics wrong and then solves its own
wrong equation consistently will still emit a self-consistent equation/answer
pair that SymPy happily confirms — this catches a *slip*, not a *misconception*.
What it does catch, honestly and reliably, is the single most common real
failure mode: correct setup, then an arithmetic error in the last step — the
model writes "2x + 4 = 10" and then claims x = 6. That case previously fell
through to "not verifiable" (raw OCR has no bare equation); now it is
correctly REFUTED. Multi-variable statements and genuinely unparseable
equations are still correctly declined rather than guessed at, and when the
chosen draft supplies no equation at all (prose, chemistry naming, and so on)
this falls back to attempting the raw OCR statement as before, so nothing that
used to verify stops verifying.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections import Counter
from datetime import UTC, datetime

import sympy  # type: ignore[import-untyped]
from sympy.parsing.sympy_parser import (  # type: ignore[import-untyped]
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from friday.logging import get_logger
from friday.providers.llm import LLMProvider, Message
from friday.vault.models import Consensus, Draft, Solve, Verification, VerificationStatus

logger = get_logger("friday.vault.solver")

_DRAFT_PROMPT = """You are %s, one of FRIDAY's operators, solving a problem she
has been shown. Work it fully, then reply with ONLY this JSON object:

{"subject": "...", "statement": "...", "latex": "...", "steps": ["...", "..."],
 "final_answer": "...", "equation": "...", "confidence": 0.0}

steps: each line one move, with the reasoning stated, not just the arithmetic.
final_answer: the answer alone, with its unit. confidence: your own 0-1 estimate.
equation: a single-variable equation in plain ASCII that SymPy can parse
(e.g. "2*x + 4 = 10") representing the decisive step of your solution — the
step that, if independently re-solved, would confirm your final_answer. Use
"" when the problem is not reducible to one equation (prose, chemistry
naming, and so on) — do not force one.

Extracted text of the problem:
---
%s
---
"""

_JUDGE_PROMPT = """FRIDAY's operators disagree about one problem. Decide which
answer is right, using the problem itself rather than a vote.

Problem:
---
%s
---
Drafts:
%s

Reply with ONLY: {"final_answer": "...", "steps": ["why the others went wrong"]}
"""

#: Trailing zeros in a decimal answer, so "9.0 V" and "9 V" compare equal —
#: but only a genuinely trailing run of zeros ("10.50" and "1.05" are left
#: alone; the lookahead refuses to fire mid-fraction).
_TRAILING_ZEROS = re.compile(r"(\d+)\.0+(?=\D|$)")

#: ``x^2`` (caret) and ``2x`` (implicit multiplication) both show up in
#: OCR'd/typed algebra even though bare SymPy parses neither — a vision model
#: has no reason to prefer ``**`` or ``2*x``. Broadening the grammar this way is
#: safe: it only accepts more strings, it never changes what an already-valid
#: expression means.
_SYMPY_TRANSFORMS = (
    *standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)


def normalise_answer(answer: str) -> str:
    """Reduce an answer to a comparable form for the agreement count."""
    text = answer.strip().strip("*` ").lower()
    text = _TRAILING_ZEROS.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def verify_with_sympy(statement: str, answer: str) -> Verification:
    """Re-derive the algebra independently of what any model claimed.

    Only single-variable equations are attempted, and only when ``statement``
    (after stripping whitespace) is *itself* one bare equation — no word-problem
    framing around it. Anything else reports ``NOT_VERIFIABLE``, which is
    honest — a false ``VERIFIED`` would be worse than no check at all.
    """
    text = statement.strip()
    if "=" not in text:
        return Verification(
            status=VerificationStatus.NOT_VERIFIABLE,
            detail="not verifiable: no equation found",
        )

    left, right = text.split("=", 1)
    try:
        left_expr = parse_expr(left, transformations=_SYMPY_TRANSFORMS)
        right_expr = parse_expr(right, transformations=_SYMPY_TRANSFORMS)
    except Exception:  # noqa: BLE001 - sympy's tokenizer raises whatever it
        # likes on genuinely malformed input (IndexError on stray parens, on
        # top of the usual Syntax/Type/Value/AttributeError) — an equation an
        # operator invented is untrusted input, so anything from the parser
        # must decline rather than crash the solve.
        return Verification(
            status=VerificationStatus.NOT_VERIFIABLE, detail="not verifiable: could not parse"
        )

    symbols = sorted(left_expr.free_symbols | right_expr.free_symbols, key=str)
    if len(symbols) != 1:
        return Verification(
            status=VerificationStatus.NOT_VERIFIABLE,
            detail="not verifiable: not single-variable",
        )

    try:
        solutions = sympy.solve(sympy.Eq(left_expr, right_expr), symbols[0])
    except (NotImplementedError, TypeError, ValueError) as exc:
        return Verification(
            status=VerificationStatus.NOT_VERIFIABLE,
            detail=f"not verifiable: could not solve ({exc})",
        )
    if not solutions:
        return Verification(
            status=VerificationStatus.NOT_VERIFIABLE, detail="not verifiable: no solution"
        )

    truth = {normalise_answer(str(s)) for s in solutions}
    claimed = normalise_answer(answer.split("=")[-1])
    if claimed in truth:
        return Verification(
            status=VerificationStatus.VERIFIED, detail=f"sympy agrees: {sorted(truth)}"
        )
    return Verification(
        status=VerificationStatus.REFUTED,
        detail=f"sympy gets {sorted(truth)}, draft said {claimed}",
    )


def _parse_draft(operator: str, reply: str) -> Draft | None:
    """Parse one operator's JSON reply into a :class:`Draft`, or ``None``.

    Never raises: a malformed reply (bad JSON, wrong shape, a non-numeric
    ``confidence``) drops that draft instead of sinking the whole panel — the
    same "one dead operator must not sink the solve" guarantee as a provider
    outright failing.
    """
    text = (
        reply.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        raw = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(raw, dict):
        return None

    try:
        confidence = float(raw.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    steps_raw = raw.get("steps")
    steps = [str(s) for s in steps_raw] if isinstance(steps_raw, list) else []

    return Draft(
        operator=operator,
        steps=steps,
        final_answer=str(raw.get("final_answer") or ""),
        equation=str(raw.get("equation") or ""),
        confidence=confidence,
    )


class Solver:
    """Runs the full panel over one problem.

    Args:
        llm: The provider used for every draft and the judge pass — the
            multi-model gateway in production, or any single provider /
            :class:`~friday.providers.llm.FakeLLM` in tests. Only the abstract
            ``complete`` contract is depended upon.
        operators: The roster code-names that draft a solution, in order.
    """

    def __init__(self, llm: LLMProvider, *, operators: list[str]) -> None:
        self._llm = llm
        self._operators = operators

    async def _draft(self, operator: str, ocr_text: str) -> Draft | None:
        """Produce one operator's draft, capturing (never raising) any failure."""
        messages = [Message(role="user", content=_DRAFT_PROMPT % (operator, ocr_text[:6000]))]
        try:
            response = await self._llm.complete(messages)
        except Exception as exc:  # noqa: BLE001 - one dead operator must not sink the panel
            logger.warning(
                "solver draft failed", extra={"operator": operator, "error": str(exc)}
            )
            return None
        return _parse_draft(operator, response.text or "")

    async def solve(self, *, item_ids: list[str], ocr_text: str) -> Solve:
        """Draft, verify, reconcile — and record what was contested."""
        results = await asyncio.gather(
            *(self._draft(op, ocr_text) for op in self._operators)
        )
        drafts = [d for d in results if d is not None]

        solve = Solve(
            id=uuid.uuid4().hex[:16],
            item_ids=list(item_ids),
            statement=ocr_text[:2000],
            drafts=drafts,
            created_at=datetime.now(UTC).isoformat(),
        )
        if not drafts:
            solve.consensus = Consensus(agreement="0/0")
            solve.verification = Verification(
                status=VerificationStatus.NOT_VERIFIABLE,
                detail="not verifiable: no usable drafts",
            )
            return solve

        counts = Counter(normalise_answer(d.final_answer) for d in drafts)
        winner, votes = counts.most_common(1)[0]
        chosen = next(d for d in drafts if normalise_answer(d.final_answer) == winner)
        solve.consensus = Consensus(
            final_answer=chosen.final_answer, agreement=f"{votes}/{len(drafts)}"
        )
        solve.dissent = [
            f"{d.operator}: {d.final_answer}"
            for d in drafts
            if normalise_answer(d.final_answer) != winner
        ]
        # Prefer the chosen draft's own equation over the raw OCR statement —
        # OCR of a word problem almost never is a bare equation itself, so
        # this is what actually gives the check something to work with. Only
        # fall back to the raw statement when the draft supplied nothing.
        solve.verification = (
            verify_with_sympy(chosen.equation, chosen.final_answer)
            if chosen.equation.strip()
            else verify_with_sympy(ocr_text, chosen.final_answer)
        )

        if votes < len(drafts):
            summary = "\n".join(f"- {d.operator}: {d.final_answer}" for d in drafts)
            messages = [
                Message(content=_JUDGE_PROMPT % (ocr_text[:4000], summary), role="user")
            ]
            try:
                response = await self._llm.complete(messages)
            except Exception as exc:  # noqa: BLE001 - fall back to the plurality
                logger.warning("solver judge failed", extra={"error": str(exc)})
                return solve
            judged = _parse_draft("JUDGE", response.text or "")
            if judged is not None and judged.final_answer:
                solve.consensus = Consensus(
                    final_answer=judged.final_answer,
                    agreement=f"{votes}/{len(drafts)}",
                    judged=True,
                )
        return solve
