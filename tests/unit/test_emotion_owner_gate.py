"""Phase-3B: the emotion analyzer gates the signal to the owner's voice."""

from __future__ import annotations

import asyncio

from friday.providers.emotion import Emotion, FakeEmotion
from friday.voice.emotion_stream import EmotionStreamAnalyzer
from friday.voice.voiceprint import FakeVoiceprint, OwnerIdentity

OWNER = b"\x10\x11" * 8000  # the enrolled owner window (16000 bytes)
OTHER = b"\x20\x21" * 8000  # someone else


def _owner_identity() -> OwnerIdentity:
    vp = FakeVoiceprint()
    profile = vp.enroll([OWNER])
    return OwnerIdentity(verifier=vp, profile=profile)


def _analyzer(seen: list[Emotion]) -> EmotionStreamAnalyzer:
    an = EmotionStreamAnalyzer(
        FakeEmotion(valence=0.2, arousal=0.3, dominance=0.4),
        window_s=0.5, hop_s=0.25, owner=_owner_identity(), owner_only=True,
    )
    an.on_emotion(seen.append)
    return an


def test_owner_only_suppresses_non_owner() -> None:
    seen: list[Emotion] = []
    an = _analyzer(seen)
    asyncio.run(an.push(OTHER))
    assert seen == [] and an.last() is None


def test_owner_only_emits_for_owner() -> None:
    seen: list[Emotion] = []
    an = _analyzer(seen)
    asyncio.run(an.push(OWNER))
    assert len(seen) >= 1 and an.last() is not None
