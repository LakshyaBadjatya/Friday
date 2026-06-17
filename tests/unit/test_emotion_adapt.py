"""Phase-2 emotion adaptation mappers: tone hint + TTS speed nudge (model-free)."""

from __future__ import annotations

from friday.providers.emotion import Emotion, emotion_hint


def _emo(label="sad", v=0.2, a=0.3, d=0.4, intensity=0.7, conf=0.6) -> Emotion:
    return Emotion(valence=v, arousal=a, dominance=d, label=label,
                   intensity=intensity, confidence=conf, ts=0.0)


def test_emotion_hint_mentions_label_and_guardrails() -> None:
    h = emotion_hint(_emo("sad"))
    assert "sad" in h.lower()
    # Honesty guardrail: never assert the feeling as fact / no clinical claims.
    assert "not" in h.lower() and "fact" in h.lower()


def test_emotion_hint_includes_intensity_and_confidence() -> None:
    h = emotion_hint(_emo(intensity=0.7, conf=0.6))
    assert "70%" in h and "60%" in h


def test_voice_for_emotion_slows_for_low_arousal() -> None:
    from friday.providers.tts import VoiceConfig, voice_for_emotion

    v = voice_for_emotion(VoiceConfig(), _emo(a=0.1))
    assert v.speed < 1.0


def test_voice_for_emotion_neutral_arousal_unchanged() -> None:
    from friday.providers.tts import VoiceConfig, voice_for_emotion

    v = voice_for_emotion(VoiceConfig(speed=1.0), _emo(a=0.5))
    assert abs(v.speed - 1.0) < 1e-9


def test_voice_for_emotion_preserves_voice_id_and_clamps() -> None:
    from friday.providers.tts import VoiceConfig, voice_for_emotion

    v = voice_for_emotion(VoiceConfig(voice_id="amy", speed=1.0), _emo(a=1.0))
    assert v.voice_id == "amy"
    assert 0.7 <= v.speed <= 1.3
