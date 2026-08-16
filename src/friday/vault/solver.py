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
the text handed to it — statement, not answer — is *itself* already a bare,
single-variable equation with no surrounding words ("2x + 4 = 10", not "Solve
for x: 2x + 4 = 10, in volts."). OCR text of a photographed word problem almost
never looks like that, so on realistic captures this check declines ("not
verifiable") far more often than it confirms or denies. It still earns its seat:
when a problem *is* posed as a clean equation (a large share of pure-algebra
homework), it independently re-derives the root rather than trusting the panel,
and multi-variable statements are correctly refused rather than guessed at. But
it is not, and cannot honestly be sold as, a check that verifies every subject
in scope — physics and chemistry word problems mostly sail past it unchecked.
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
from friday.vault.models import Consensus, Draft, Solve, Verification

logger = get_logger("friday.vault.solver")

_DRAFT_PROMPT = """You are %s, one of FRIDAY's operators, solving a problem she
has been shown. Work it fully, then reply with ONLY this JSON object:

{"subject": "...", "statement": "...", "latex": "...", "steps": ["...", "..."],
 "final_answer": "...", "confidence": 0.0}

steps: each line one move, with the reasoning stated, not just the arithmetic.
final_answer: the answer alone, with its unit. confidence: your own 0-1 estimate.

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
    framing around it. Anything else reports ``not verifiable``, which is
    honest — a false ``ok`` would be worse than no check at all.
    """
    text = statement.strip()
    if "=" not in text:
        return Verification(ok=False, detail="not verifiable: no equation found")

    left, right = text.split("=", 1)
    try:
        left_expr = parse_expr(left, transformations=_SYMPY_TRANSFORMS)
        right_expr = parse_expr(right, transformations=_SYMPY_TRANSFORMS)
    except (SyntaxError, TypeError, ValueError, AttributeError):
        return Verification(ok=False, detail="not verifiable: could not parse")

    symbols = sorted(left_expr.free_symbols | right_expr.free_symbols, key=str)
    if len(symbols) != 1:
        return Verification(ok=False, detail="not verifiable: not single-variable")

    try:
        solutions = sympy.solve(sympy.Eq(left_expr, right_expr), symbols[0])
    except (NotImplementedError, TypeError, ValueError) as exc:
        return Verification(ok=False, detail=f"not verifiable: could not solve ({exc})")
    if not solutions:
        return Verification(ok=False, detail="not verifiable: no solution")

    truth = {normalise_answer(str(s)) for s in solutions}
    claimed = normalise_answer(answer.split("=")[-1])
    if claimed in truth:
        return Verification(ok=True, detail=f"sympy agrees: {sorted(truth)}")
    return Verification(
        ok=False, detail=f"sympy gets {sorted(truth)}, draft said {claimed}"
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
                ok=False, detail="not verifiable: no usable drafts"
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
        solve.verification = verify_with_sympy(ocr_text, chosen.final_answer)

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
