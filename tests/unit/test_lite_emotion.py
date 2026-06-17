"""Phase-3C: the lazy LiteEmotion provider (custom Kaggle-trained head)."""

from __future__ import annotations

import asyncio
import os

import pytest

from friday.errors import ProviderError
from friday.providers.emotion import LiteEmotion


def test_liteemotion_missing_model_raises_providererror() -> None:
    with pytest.raises(ProviderError):
        LiteEmotion(model_path="/nonexistent/emotion_head.onnx")


@pytest.mark.skipif(
    not os.environ.get("FRIDAY_EMOTION_LITE_MODEL"),
    reason="no lite emotion ONNX head provided",
)
def test_liteemotion_smoke() -> None:
    import numpy as np

    prov = LiteEmotion(os.environ["FRIDAY_EMOTION_LITE_MODEL"])
    pcm = np.zeros(16000, dtype=np.int16).tobytes()
    e = asyncio.run(prov.analyze(pcm))
    assert 0.0 <= e.valence <= 1.0 and 0.0 <= e.arousal <= 1.0
