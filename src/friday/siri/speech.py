"""Turn an orchestrator reply into a short, plain string Siri can speak.

The core loop returns chat-style text — markdown, bullet lists, links, code
spans, and any length. Siri's "Speak Text" action wants a clean, bounded string.
:func:`for_speech` strips the markdown to its spoken words, collapses whitespace,
and truncates an over-long reply on a sentence boundary (falling back to a word
boundary) with a trailing ellipsis. It is pure and side-effect free so the route
stays thin and this shaping is unit-tested in isolation.
"""

from __future__ import annotations

import re

#: ``[label](url)`` -> ``label`` (keep what a listener would hear, drop the URL).
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
#: A leading ATX heading marker (``#`` .. ``######``) on a line.
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*")
#: A leading list marker: ``-``/``*``/``+`` bullets or ``1.``/``1)`` numbering.
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
#: Inline markdown punctuation that should not be spoken (emphasis, code, quote).
_MARKUP = re.compile(r"[*_~`>#]")
#: Any run of whitespace (incl. newlines/tabs), collapsed to a single space.
_WHITESPACE = re.compile(r"\s+")
#: Sentence-ending punctuation used to pick a clean truncation point.
_SENTENCE_END = re.compile(r"[.!?]")

# --- Spoken maths/science notation ------------------------------------------ #
# So a formula is READ, not mangled: "E = mc²" -> "E equals m c squared", not
# "E mc 2". Symbols map to their spoken names; powers become "squared"/"cubed"/
# "to the power of N" for every other exponent.
_SUPERSCRIPT = {
    "⁰": "0", "¹": "1", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8",
    "⁹": "9", "ⁿ": "n", "ⁱ": "i", "⁺": "plus", "⁻": "minus",
}
_SUBSCRIPT = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6",
    "₇": "7", "₈": "8", "₉": "9",
}
#: Single-character maths/science symbols -> spoken words (Greek + operators).
_SYMBOLS = {
    "√": " square root of ", "∛": " cube root of ", "±": " plus or minus ",
    "∓": " minus or plus ", "×": " times ", "⋅": " times ", "·": " times ",
    "÷": " divided by ", "≈": " approximately ", "≅": " approximately ",
    "≠": " not equal to ", "≤": " less than or equal to ",
    "≥": " greater than or equal to ", "≡": " is identical to ",
    "∝": " proportional to ", "→": " gives ", "⇒": " implies ", "∞": " infinity ",
    "∑": " the sum of ", "∏": " the product of ", "∫": " the integral of ",
    "∂": " partial ", "∇": " del ", "°": " degrees ", "′": " prime ",
    "α": " alpha ", "β": " beta ", "γ": " gamma ", "δ": " delta ", "Δ": " delta ",
    "ε": " epsilon ", "ζ": " zeta ", "η": " eta ", "θ": " theta ", "κ": " kappa ",
    "λ": " lambda ", "μ": " mu ", "ν": " nu ", "ξ": " xi ", "π": " pi ",
    "ρ": " rho ", "σ": " sigma ", "Σ": " sigma ", "τ": " tau ", "φ": " phi ",
    "ϕ": " phi ", "χ": " chi ", "ψ": " psi ", "ω": " omega ", "Ω": " omega ",
}
#: ``^2`` / ``^(n+1)`` / ``^9`` style powers.
_CARET = re.compile(r"\^\(?\s*([A-Za-z0-9+\-*/. ]+?)\s*\)?(?=[\s,.;)\]]|$)")
#: A run of unicode superscript characters (other than ² / ³, handled inline).
_SUP_RUN = re.compile(r"[⁰¹⁴⁵⁶⁷⁸⁹ⁿⁱ⁺⁻]+")


def _power_words(exponent: str) -> str:
    exponent = exponent.strip()
    if exponent == "2":
        return " squared"
    if exponent == "3":
        return " cubed"
    return f" to the power of {exponent}"


def _spoken_math(text: str) -> str:
    """Rewrite maths/science notation into words a TTS voice reads correctly."""
    text = _CARET.sub(lambda m: _power_words(m.group(1)), text)
    text = text.replace("²", " squared").replace("³", " cubed")
    text = _SUP_RUN.sub(
        lambda m: " to the power of "
        + "".join(_SUPERSCRIPT.get(c, c) for c in m.group()),
        text,
    )
    text = "".join(_SUBSCRIPT.get(ch, ch) for ch in text)
    for symbol, word in _SYMBOLS.items():
        text = text.replace(symbol, word)
    # "=" reads as "equals" (almost always a formula in this context).
    text = re.sub(r"\s*=\s*", " equals ", text)
    return text


def for_speech(text: str, max_chars: int = 600) -> str:
    """Return ``text`` reduced to a clean, bounded line suitable for TTS.

    Markdown emphasis/headings/bullets/links/code spans are removed, whitespace
    is collapsed, and the result is truncated to at most ``max_chars`` characters
    (plus a one-character ellipsis) at the last sentence boundary within the
    window, or the last word boundary if no sentence ends there. Empty or
    whitespace-only input yields ``""`` so callers can substitute a fallback.
    """
    if not text:
        return ""

    # Strip per-line markers first (headings, bullets) so list/heading text reads
    # as plain sentences once the newlines become spaces.
    stripped_lines = []
    for line in text.splitlines():
        line = _HEADING.sub("", line)
        line = _BULLET.sub("", line)
        stripped_lines.append(line)

    cleaned = " ".join(stripped_lines)
    cleaned = _LINK.sub(r"\1", cleaned)
    cleaned = _MARKUP.sub("", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    # Read formulas/symbols as words ("x²" -> "x squared", "√" -> "square root of").
    cleaned = _spoken_math(cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()

    if len(cleaned) <= max_chars:
        return cleaned

    window = cleaned[:max_chars]
    cut = None
    for match in _SENTENCE_END.finditer(window):
        cut = match.end()
    if cut is None:
        space = window.rfind(" ")
        cut = space if space > 0 else max_chars

    return cleaned[:cut].rstrip() + "…"
