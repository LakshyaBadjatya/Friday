"""Speaking whatever language is being spoken to her.

Two separate problems that look like one:

**Being asked.** "Talk to me in Polish" is an instruction that should stick — for
the rest of the conversation, not just the next line — so the choice is held per
session and every later turn is told to answer in it.

**Being addressed.** In a voice call nobody announces a switch; they just start
speaking Polish. So the transcriber is asked to report which language it heard,
and the reply follows the speaker automatically. That is the difference between
a setting and a conversation.

The voice matters as much as the words. An English voice reading Polish text
produces something between comedy and gibberish — the phonemes are not there —
so each language maps to a native speaker from the Edge catalogue, and anything
unmapped falls back to English rather than being mangled.
"""

from __future__ import annotations

import re
from typing import Any

#: Language -> a native Edge voice. Chosen for the languages the owner is
#: plausibly going to use rather than padding the list: an entry here is a claim
#: that the voice was picked deliberately.
VOICES: dict[str, str] = {
    "en": "en-GB-SoniaNeural",
    "hi": "hi-IN-SwaraNeural",
    "pl": "pl-PL-ZofiaNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ar": "ar-EG-SalmaNeural",
    "tr": "tr-TR-EmelNeural",
    "nl": "nl-NL-ColetteNeural",
    "id": "id-ID-GadisNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "bn": "bn-IN-TanishaaNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "ur": "ur-PK-UzmaNeural",
}

#: Spoken names -> code. Both the English name and the endonym, because someone
#: switching to Polish may well ask for it in Polish.
_NAMES: dict[str, str] = {
    "english": "en", "angielski": "en",
    "hindi": "hi", "हिंदी": "hi",
    "polish": "pl", "polski": "pl", "polsku": "pl",
    "spanish": "es", "español": "es", "espanol": "es",
    "french": "fr", "français": "fr", "francais": "fr",
    "german": "de", "deutsch": "de",
    "italian": "it", "italiano": "it",
    "portuguese": "pt", "português": "pt",
    "russian": "ru", "русский": "ru",
    "japanese": "ja", "日本語": "ja",
    "korean": "ko", "한국어": "ko",
    "chinese": "zh", "mandarin": "zh", "中文": "zh",
    "arabic": "ar",
    "turkish": "tr", "türkçe": "tr",
    "dutch": "nl", "nederlands": "nl",
    "indonesian": "id", "bahasa": "id",
    "tamil": "ta", "telugu": "te", "bengali": "bn", "bangla": "bn",
    "marathi": "mr", "gujarati": "gu", "urdu": "ur",
}

#: "talk to me in polish", "reply in hindi", "switch to french". ``[^\W\d_]``
#: is the stdlib's way of writing "a letter in any script" without pulling in
#: the third-party regex module for ``\p{L}``.
_REQUEST = re.compile(
    # "in" is optional: people say "speak hindi" as readily as "speak in hindi",
    # and requiring the preposition silently ignored half the natural phrasings.
    r"\b(?:talk|speak|reply|answer|respond|write|say\s+it)\s+(?:to\s+me\s+)?"
    r"(?:back\s+)?(?:in\s+)?(?P<name>[^\W\d_]+)\b"
    r"|\bswitch\s+to\s+(?P<name2>[^\W\d_]+)\b"
    r"|\bin\s+(?P<name3>[^\W\d_]+)\s+please\b",
    re.IGNORECASE | re.UNICODE,
)
#: "back to english", "english again" — returning to the default.
_RESET = re.compile(
    r"\b(?:back\s+to|return\s+to)\s+english\b|\benglish\s+again\b", re.IGNORECASE
)


def requested(text: str) -> str | None:
    """The language code just asked for, or ``None``.

    Matched on the *name*, so an unknown word — "talk to me in riddles" — is not
    mistaken for a language and cannot silently change anything.
    """
    body = (text or "").strip()
    if not body:
        return None
    if _RESET.search(body):
        return "en"
    match = _REQUEST.search(body)
    if match is None:
        return None
    for group in ("name", "name2", "name3"):
        found = match.group(group)
        if found and found.lower() in _NAMES:
            return _NAMES[found.lower()]
    return None


def voice_for(code: str | None) -> str:
    """The Edge voice for a language code, falling back to English.

    A missing mapping falls back deliberately rather than guessing at a voice
    name: an invalid voice makes edge-tts fail outright and she loses her speech
    entirely, whereas English reading a foreign phrase is merely bad.
    """
    return VOICES.get((code or "en").lower()[:2], VOICES["en"])


def name_of(code: str | None) -> str:
    """The English name of a code, for putting in a prompt."""
    for name, mapped in _NAMES.items():
        if mapped == code and name.isascii():
            return name
    return "english"


def instruction(code: str | None) -> str:
    """A prompt fragment telling the model which language to answer in."""
    if not code or code == "en":
        return ""
    language = name_of(code)
    return (
        f"\n\nReply ONLY in {language}. The user is speaking {language} and "
        f"expects the same back — do not translate, do not add an English "
        f"version, and do not remark on having switched."
    )


# --- per-session preference ------------------------------------------------- #
def get(state: Any, session: str) -> str | None:
    """The language pinned to this conversation, if any."""
    return _store(state).get(session)


def set_for(state: Any, session: str, code: str | None) -> None:
    """Pin (or clear) the language for a conversation."""
    store = _store(state)
    if code and code != "en":
        store[session] = code
    else:
        store.pop(session, None)


def _store(state: Any) -> dict[str, str]:
    existing = getattr(state, "_languages", None)
    if not isinstance(existing, dict):
        existing = {}
        state._languages = existing  # noqa: SLF001 - app state is a namespace
    return existing
