"""GraphState carries the sensed emotion for a turn (Phase 1)."""

from __future__ import annotations

from friday.core.state import GraphState
from friday.providers.emotion import Emotion


def test_graphstate_emotion_defaults_none_and_roundtrips() -> None:
    s = GraphState(session_id="s", user_input="hi")
    assert s.emotion is None
    s.emotion = Emotion(
        valence=0.2, arousal=0.3, dominance=0.4, label="sad",
        intensity=0.5, confidence=0.7, ts=1.0,
    )
    assert GraphState.model_validate_json(s.model_dump_json()).emotion == s.emotion
