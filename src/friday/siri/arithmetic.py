"""Compute spoken arithmetic instead of predicting it.

A language model does not calculate — it predicts the tokens that usually follow a
sum, which is why "what's 17 percent of 2,480" can come back confidently wrong.
Any query that is *purely* a calculation is therefore evaluated here, in Python,
before a model is ever consulted. Anything that is not unambiguously arithmetic
returns ``None`` and falls through untouched, so word problems, algebra, and
"explain the quadratic formula" still reach the LLM.

Safety: the expression is parsed with :mod:`ast` and walked against an allowlist of
node types (numeric literals and the arithmetic operators). There is no ``eval`` of
arbitrary source — names, calls, attributes, and subscripts are rejected outright,
so a crafted query cannot reach the interpreter. Exponents are additionally bounded
so ``9**9**9`` cannot burn CPU.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any

#: AST node -> the Python operator it is allowed to perform. Anything absent is a
#: hard reject, which is what keeps this evaluator safe on untrusted spoken input.
_BINARY_OPS: dict[type[ast.AST], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS: dict[type[ast.AST], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

#: Largest exponent accepted — keeps ``2 ** 10000000`` from hanging the worker.
_MAX_EXPONENT = 1000
#: Magnitude past which a result is spoken in compact form, not full digits.
_MAX_SPEAKABLE = 1e15

#: Lead-ins a person says before a sum; stripped so the remainder is the expression.
_PREFIXES = (
    "what is", "what's", "whats", "what are",
    "how much is", "how much are",
    "calculate", "compute", "work out", "figure out", "solve",
    "tell me", "give me", "can you",
    "please", "hey friday", "friday",
)
#: Trailing politeness / punctuation to drop before parsing.
_SUFFIX = re.compile(r"[\s?.!,]+$")

#: Spoken operator words -> symbols. Order matters: multi-word phrases first, so
#: "divided by" is consumed before a bare word could be mistaken for anything.
_WORD_OPS: tuple[tuple[str, str], ...] = (
    ("multiplied by", "*"),
    ("divided by", "/"),
    ("raised to the power of", "**"),
    ("to the power of", "**"),
    ("plus", "+"),
    ("minus", "-"),
    ("times", "*"),
    ("modulo", "%"),
    ("mod", "%"),
)

#: ``17% of 240`` / ``17 percent of 240`` -> ``(17/100)*240``. Percent-of is the
#: single most common spoken sum and the one models most often fumble.
_PERCENT_OF = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)\s+of\s+([\d.,\s+\-*/()]+)",
    re.IGNORECASE,
)
#: Any surviving bare percentage becomes a plain fraction.
_PERCENT_BARE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)")

#: ``square root of 144`` -> ``144 ** 0.5``; cube root likewise.
_SQRT = re.compile(r"(?:square\s+root\s+of|sqrt\s+of|sqrt)\s*([\d.,]+)", re.IGNORECASE)
_CBRT = re.compile(r"cube\s+root\s+of\s*([\d.,]+)", re.IGNORECASE)
#: ``12 squared`` / ``5 cubed`` -> ``12 ** 2`` / ``5 ** 3``.
_SQUARED = re.compile(r"([\d.,]+)\s*squared", re.IGNORECASE)
_CUBED = re.compile(r"([\d.,]+)\s*cubed", re.IGNORECASE)

#: After normalisation the expression must contain only these characters, else it
#: is prose and belongs to the LLM. This is the gate that keeps the branch narrow.
_EXPRESSION_ONLY = re.compile(r"^[\d\s.+\-*/%()]+$")
#: At least one operator must survive, so a bare "what is 42" is not "computed".
_HAS_OPERATOR = re.compile(r"[+\-*/%]")
#: ``2 x 3`` — "x" as a spoken multiplication sign, only between numbers.
_X_TIMES = re.compile(r"(?<=\d)\s*[x×]\s*(?=\d)", re.IGNORECASE)


def arithmetic_reply(query: str) -> str | None:
    """Return a spoken answer for a pure calculation, else ``None`` to fall through.

    Handled: ``12 * 37``, ``what's 17 percent of 2480``, ``square root of 144``,
    ``(45 + 15) divided by 4``, ``2 to the power of 10``. Deliberately left to the
    model: ``how do you calculate compound interest``, ``solve for x``, and any
    phrasing that still contains words after normalisation.
    """
    expression = _to_expression(query)
    if expression is None:
        return None
    value = _safe_eval(expression)
    if value is None:
        return None
    return f"That's {_format_number(value)}."


def _to_expression(query: str) -> str | None:
    """Normalise a spoken query to a bare arithmetic expression, or ``None``."""
    text = (query or "").strip().lower()
    if not text or not any(ch.isdigit() for ch in text):
        return None

    text = _SUFFIX.sub("", text)
    # Strip conversational lead-ins repeatedly ("hey friday what's 2+2").
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            if text.startswith(prefix + " ") or text == prefix:
                text = text[len(prefix) :].strip()
                changed = True
    text = _SUFFIX.sub("", text)
    if not text:
        return None

    # Roots and powers before the generic word-operator pass, since "square root
    # of" contains no operator word and "squared" would survive as prose.
    text = _SQRT.sub(lambda m: f"({_num(m.group(1))} ** 0.5)", text)
    text = _CBRT.sub(lambda m: f"({_num(m.group(1))} ** (1/3))", text)
    text = _SQUARED.sub(lambda m: f"({_num(m.group(1))} ** 2)", text)
    text = _CUBED.sub(lambda m: f"({_num(m.group(1))} ** 3)", text)
    text = _PERCENT_OF.sub(lambda m: f"(({m.group(1)}/100) * ({m.group(2)}))", text)

    for word, symbol in _WORD_OPS:
        text = re.sub(rf"\b{re.escape(word)}\b", symbol, text)
    text = _X_TIMES.sub("*", text)
    text = _PERCENT_BARE.sub(lambda m: f"({m.group(1)}/100)", text)
    # Thousands separators: "2,480" -> "2480".
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)

    text = text.strip()
    if not _EXPRESSION_ONLY.match(text) or not _HAS_OPERATOR.search(text):
        return None
    return text


def _num(raw: str) -> str:
    """Strip thousands separators from a captured number."""
    return raw.replace(",", "")


def _safe_eval(expression: str) -> float | int | None:
    """Parse and evaluate ``expression`` under the node allowlist; ``None`` if unsafe."""
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    try:
        return _eval_node(tree.body)
    except (ArithmeticError, ValueError, TypeError, RecursionError):
        # Division by zero, overflow, a complex root — all mean "fall through to
        # the model" rather than speaking a wrong answer or raising.
        return None


def _eval_node(node: ast.AST) -> float | int:
    """Recursively evaluate one allowlisted AST node."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise TypeError("only numeric literals are allowed")
        return node.value
    if isinstance(node, ast.UnaryOp):
        unary = _UNARY_OPS.get(type(node.op))
        if unary is None:
            raise TypeError("unsupported unary operator")
        return _numeric(unary(_eval_node(node.operand)))
    if isinstance(node, ast.BinOp):
        binary = _BINARY_OPS.get(type(node.op))
        if binary is None:
            raise TypeError("unsupported binary operator")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("exponent too large to evaluate")
        return _numeric(binary(left, right))
    raise TypeError(f"unsupported expression node: {type(node).__name__}")


def _numeric(result: Any) -> float | int:
    """Assert an operator produced a finite real number, else reject the whole sum.

    ``**`` on a negative base with a fractional exponent yields a ``complex``, and
    overflow yields ``inf`` — neither is speakable, and both mean "fall through to
    the model" rather than reading nonsense aloud.
    """
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise TypeError("non-numeric result")
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("non-finite result")
    return result


def _format_number(value: float | int) -> str:
    """Render a result the way a person would say it.

    Integers keep thousands separators (TTS groups them correctly); floats are
    rounded to a sensible number of places and stripped of trailing zeros, so
    ``1/3`` speaks as "0.3333" rather than seventeen digits of noise.
    """
    if abs(value) >= _MAX_SPEAKABLE:
        return f"{value:.4g}"
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    rounded = round(float(value), 4)
    if rounded == 0:  # very small but non-zero
        return f"{value:.4g}"
    return f"{rounded:,.4f}".rstrip("0").rstrip(".")
