"""Making synthesised speech sound like someone actually talking.

A text-to-speech voice reading a clean sentence is the giveaway: perfectly
metered, no hesitation, every clause landing on the beat. Real speech is full of
small failures — a breath before a hard word, "uhh" while the thought catches up,
a trailing "y'know" that means nothing. Those failures are the signal that
someone is thinking, and putting a few back is what stops her sounding like a
station announcement.

Three levers, in descending order of how much they matter:

* **Hesitation.** A filler at the start ("uhh", "hmm", "okay so") buys the
  listener the same beat a person takes, and is the single biggest change.
* **Punctuation.** TTS engines pause on commas and ellipses and nothing else. An
  ellipsis mid-sentence is a hesitation the voice will actually perform, so
  breaks are inserted as punctuation rather than requested as markup.
* **Prosody drift.** Identical rate and pitch on every utterance turns uncanny
  after about three sentences, so each reply is nudged a few percent either way.

Deliberately *not* here: a literal cough. Coughs, laughs and lip smacks are not
in any TTS voice's phoneme set — asking for one gets the word "cough" read aloud.
Doing them properly means mixing real audio samples into the PCM, which is a
different job and worth doing separately if the hesitations are not enough.

Everything is probabilistic and most of it does not fire. Disfluency on *every*
sentence is its own tell, and more annoying than the flat version.
"""

from __future__ import annotations

import re
import secrets

#: Openers that buy a beat. Weighted towards short ones — a long preamble on a
#: one-line answer sounds evasive rather than natural.
_OPENERS = (
    "uhh,", "uh,", "hmm,", "mm,", "mhm,", "okay so,", "ok so,", "right, so",
    "well,", "so,", "yeah so,", "honestly,", "I mean,",
)
#: How often a reply opens with one.
_OPENER_PERCENT = 45

#: Dropped between clauses, where a person would trail off briefly.
_MIDS = ("uhh", "like", "y'know", "I guess", "sort of")
#: How often a mid-sentence hesitation is inserted at all.
_MID_PERCENT = 22

#: Trailing sounds, for the end of a casual reply.
_TAILS = (" y'know?", " ...I think.", " or whatever.", " hmm.", " yeah.")
_TAIL_PERCENT = 15

#: Prosody jitter, in percent and hertz. Small on purpose: past roughly ±8% the
#: voice stops sounding like the same person between sentences.
_RATE_RANGE = 8
_PITCH_RANGE = 6

#: Clause boundaries — commas and the usual conjunctions — which is where a
#: person hesitates. Mid-word is not a hesitation, it is a glitch.
_CLAUSE = re.compile(r",\s+|\s+(?:but|and|so|because|though)\s+")
#: Below this, a reply is too short to hesitate inside without the hesitation
#: being longer than the sentence.
_MIN_WORDS_FOR_MID = 9


def humanize(text: str) -> str:
    """Add hesitations to a line about to be spoken aloud.

    Returns the text unchanged much of the time; the point is that it varies.
    """
    body = (text or "").strip()
    if not body:
        return body

    if secrets.randbelow(100) < _OPENER_PERCENT:
        opener = secrets.choice(_OPENERS)
        # Lower-case the original first letter, so "Uhh, Yes" does not happen.
        body = f"{opener} {body[0].lower()}{body[1:]}"

    if len(body.split()) >= _MIN_WORDS_FOR_MID and (
        secrets.randbelow(100) < _MID_PERCENT
    ):
        body = _hesitate_inside(body)

    if secrets.randbelow(100) < _TAIL_PERCENT and not body.endswith("?"):
        body = body.rstrip(".!") + secrets.choice(_TAILS)

    return body


def _hesitate_inside(body: str) -> str:
    """Drop a filler at one clause boundary, if there is one to use."""
    matches = list(_CLAUSE.finditer(body))
    if not matches:
        return body
    at = matches[secrets.randbelow(len(matches))]
    filler = secrets.choice(_MIDS)
    # Ellipses around the filler because TTS engines pause on punctuation and
    # ignore everything else — it is the only way to get a real hesitation out.
    return f"{body[: at.start()]}... {filler}... {body[at.end() :]}"


def jitter() -> tuple[str, str]:
    """A small random ``(rate, pitch)`` for one utterance.

    Returned in the ``+5%`` / ``-3Hz`` form edge-tts expects. Identical prosody
    on every line is uncanny within a few sentences; this is enough variation to
    read as the same person on a different breath.
    """
    rate = secrets.randbelow(_RATE_RANGE * 2 + 1) - _RATE_RANGE
    pitch = secrets.randbelow(_PITCH_RANGE * 2 + 1) - _PITCH_RANGE
    return f"{rate:+d}%", f"{pitch:+d}Hz"
