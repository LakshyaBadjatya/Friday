"""Tests for the studio generator: procedural LLM-codegen + hi-fi fallback.

All offline against :class:`~friday.providers.llm.FakeLLM` (scripted) and the
:class:`~friday.studio.generator.FakeText3D` stub — no network, no key.

Covered:
* ``ProceduralGenerator`` with a FakeLLM scripted to a valid Scene JSON returns a
  validated :class:`Scene`.
* A scripted bad-then-good pair drives exactly one bounded repair re-prompt and
  yields the repaired Scene.
* A second failure raises a typed :class:`~friday.errors.ProviderError` (honest:
  no fabricated model).
* ``StudioService`` fast path returns a procedural Scene; the hi-fi path falls
  back to procedural when no hi-fi provider is wired, when the provider is
  keyless, and when it raises (never paywalls the user).
"""

from __future__ import annotations

import json

import pytest

from friday.errors import ProviderError
from friday.providers.llm import FakeLLM, LLMResponse, Usage
from friday.studio.generator import (
    FakeText3D,
    ProceduralGenerator,
    StudioService,
    Text3DProvider,
)
from friday.studio.scene import Scene

_VALID_SCENE_JSON = json.dumps(
    {
        "name": "red-cube",
        "background": "#101014",
        "nodes": [
            {
                "id": "cube",
                "type": "box",
                "params": {"w": 1.0, "h": 1.0, "d": 1.0},
                "color": "#ff0000",
            }
        ],
    }
)


def _llm(*texts: str) -> FakeLLM:
    """A :class:`FakeLLM` scripted to return each text in order."""
    return FakeLLM(
        responses=[LLMResponse(text=t, tool_calls=[], usage=Usage()) for t in texts]
    )


async def test_procedural_generator_returns_scene_from_valid_json() -> None:
    """A FakeLLM scripted to valid Scene JSON yields a validated :class:`Scene`."""
    gen = ProceduralGenerator(_llm(_VALID_SCENE_JSON))

    scene = await gen.generate("a red cube")

    assert isinstance(scene, Scene)
    assert scene.name == "red-cube"
    assert scene.nodes[0].type == "box"
    assert scene.nodes[0].color == "#ff0000"


async def test_procedural_generator_strips_markdown_fences() -> None:
    """JSON wrapped in a ```json fenced block is still parsed (LLMs love fences)."""
    fenced = f"```json\n{_VALID_SCENE_JSON}\n```"
    gen = ProceduralGenerator(_llm(fenced))

    scene = await gen.generate("a red cube")

    assert scene.name == "red-cube"


async def test_procedural_generator_repairs_bad_then_good() -> None:
    """Invalid JSON triggers exactly one repair re-prompt; the good retry wins."""
    gen = ProceduralGenerator(_llm("not json at all {", _VALID_SCENE_JSON))

    scene = await gen.generate("a red cube")

    assert isinstance(scene, Scene)
    assert scene.name == "red-cube"


async def test_procedural_generator_repairs_invalid_scene_shape() -> None:
    """Well-formed JSON that violates the Scene contract also triggers a repair."""
    bad_scene = json.dumps({"name": "x", "nodes": [{"id": "n", "type": "teapot"}]})
    gen = ProceduralGenerator(_llm(bad_scene, _VALID_SCENE_JSON))

    scene = await gen.generate("a red cube")

    assert scene.name == "red-cube"


async def test_procedural_generator_raises_after_second_failure() -> None:
    """Two bad outputs (original + repair) raise a typed :class:`ProviderError`."""
    gen = ProceduralGenerator(_llm("garbage {", "still garbage }"))

    with pytest.raises(ProviderError):
        await gen.generate("a red cube")


async def test_studio_service_fast_returns_scene() -> None:
    """The fast (default) quality returns a procedural scene envelope."""
    service = StudioService(ProceduralGenerator(_llm(_VALID_SCENE_JSON)))

    result = await service.generate("a red cube", quality="fast")

    assert result["kind"] == "scene"
    assert isinstance(result["scene"], dict)
    assert result["scene"]["name"] == "red-cube"


async def test_studio_service_hifi_none_falls_back_to_procedural() -> None:
    """With no hi-fi provider wired, ``quality='hifi'`` falls back to procedural."""
    service = StudioService(ProceduralGenerator(_llm(_VALID_SCENE_JSON)), hifi=None)

    result = await service.generate("a red cube", quality="hifi")

    assert result["kind"] == "scene"
    assert result["scene"]["name"] == "red-cube"


async def test_studio_service_hifi_keyless_falls_back_to_procedural() -> None:
    """A keyless hi-fi provider (``available() is False``) falls back, never errors."""
    service = StudioService(
        ProceduralGenerator(_llm(_VALID_SCENE_JSON)),
        hifi=FakeText3D(available=False),
    )

    result = await service.generate("a red cube", quality="hifi")

    assert result["kind"] == "scene"
    assert result["scene"]["name"] == "red-cube"


async def test_studio_service_hifi_raises_falls_back_to_procedural() -> None:
    """A hi-fi provider that raises mid-call still falls back to procedural."""

    class _BoomText3D:
        def available(self) -> bool:
            return True

        async def generate_mesh(self, description: str) -> dict[str, object]:
            raise ProviderError("quota exceeded")

    provider: Text3DProvider = _BoomText3D()
    service = StudioService(
        ProceduralGenerator(_llm(_VALID_SCENE_JSON)), hifi=provider
    )

    result = await service.generate("a red cube", quality="hifi")

    assert result["kind"] == "scene"
    assert result["scene"]["name"] == "red-cube"


async def test_studio_service_hifi_available_returns_mesh() -> None:
    """When a hi-fi provider is available it returns a mesh envelope (no fallback)."""
    service = StudioService(
        ProceduralGenerator(_llm(_VALID_SCENE_JSON)),
        hifi=FakeText3D(available=True),
    )

    result = await service.generate("a red cube", quality="hifi")

    assert result["kind"] == "mesh"
    assert result["format"] == "glb"
    assert "url" in result
