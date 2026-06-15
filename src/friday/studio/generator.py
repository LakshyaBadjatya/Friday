"""Studio generation: free procedural LLM-codegen + optional hi-fi mesh (fallback).

Two generation strategies behind one service:

* :class:`ProceduralGenerator` — the **free, default** path. It prompts the live
  :class:`~friday.providers.llm.LLMProvider` to emit a JSON
  :class:`~friday.studio.scene.Scene` (schema + worked examples baked into the
  system prompt), parses + validates it, and on the first invalid output does
  exactly **one** bounded repair re-prompt. A second failure raises a typed
  :class:`~friday.errors.ProviderError` — honest, never a fabricated scene.

* :class:`Text3DProvider` — a Protocol for an external text-to-3D mesh service.
  :class:`MeshyText3D` is a lazy, flagged, keyless-safe adapter over Meshy's API;
  :class:`FakeText3D` is the offline stub used in tests. Neither is required.

:class:`StudioService` ties them together. ``quality="fast"`` always uses the
procedural path; ``quality="hifi"`` tries the external mesh provider but **falls
back to procedural** whenever the provider is absent, keyless, or raises (e.g.
over quota). The user is never paywalled into an error.

The LLM emits validated JSON only — there is no execution of model output here.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import ValidationError

from friday.errors import ProviderError
from friday.logging import get_logger
from friday.providers.llm import LLMProvider, Message
from friday.studio.scene import Scene, validate_scene

logger = get_logger("friday.studio.generator")

Quality = Literal["fast", "hifi"]

# A compact statement of the Scene contract + worked examples, baked into the
# system prompt so the model returns ONLY JSON matching the schema. This is
# instruction text, not executable code.
_SYSTEM_PROMPT = """\
You are a 3D scene generator. Given a description, output ONLY a single JSON
object (no prose, no markdown fences) describing a scene as a graph of geometry
primitives. The browser builds a Three.js scene from this JSON; you never write
JavaScript.

JSON schema:
{
  "name": string,
  "background": string (hex color, default "#101014"),
  "nodes": [ SceneNode, ... ]
}
SceneNode = {
  "id": string (unique),
  "type": one of "box" | "sphere" | "cylinder" | "cone" | "torus" | "plane" | "group",
  "params": object of numbers, per type:
      box: {"w","h","d"}, sphere: {"r"}, cylinder: {"r","h"}, cone: {"r","h"},
      torus: {"r","tube"}, plane: {"w","h"}, group: {},
  "position": [x,y,z]   (default [0,0,0]),
  "rotation": [x,y,z]   (Euler radians, default [0,0,0]),
  "scale":    [x,y,z]   (default [1,1,1]),
  "color":    string hex (default "#cccccc"),
  "metalness": number 0..1 (default 0.0),
  "roughness": number 0..1 (default 0.8),
  "children": [ SceneNode, ... ]   (nested under this node's transform)
}
Use a "group" to compose sub-parts; place children with positions relative to
the parent. Keep it to a few dozen nodes at most. Output JSON only.

Example 1 — "a red cube":
{"name":"red-cube","nodes":[{"id":"c","type":"box","params":{"w":1,"h":1,"d":1},"color":"#ff0000"}]}

Example 2 — "a simple snowman":
{"name":"snowman","nodes":[{"id":"snowman","type":"group","params":{},"children":[
{"id":"base","type":"sphere","params":{"r":1.0},"position":[0,0,0],"color":"#ffffff"},
{"id":"torso","type":"sphere","params":{"r":0.7},"position":[0,1.4,0],"color":"#ffffff"},
{"id":"head","type":"sphere","params":{"r":0.45},"position":[0,2.4,0],"color":"#ffffff"},
{"id":"nose","type":"cone","params":{"r":0.08,"h":0.4},"position":[0,2.4,0.45],"rotation":[1.57,0,0],"color":"#ff8c00"}]}]}
"""


def _extract_json(text: str) -> str:
    """Best-effort strip of markdown fences / surrounding prose around JSON.

    LLMs often wrap JSON in ```json ... ``` fences or add a sentence around it.
    We strip a leading/trailing fence and, failing that, slice from the first
    ``{`` to the last ``}``. This is pure string munging — the result is still
    validated by :func:`~friday.studio.scene.validate_scene` before use.
    """
    s = text.strip()
    if s.startswith("```"):
        # Drop the opening fence line (``` or ```json) and any trailing fence.
        s = s.split("\n", 1)[-1] if "\n" in s else s
        if s.endswith("```"):
            s = s[: -len("```")]
        s = s.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


class ProceduralGenerator:
    """Generate a validated :class:`Scene` from a description via the LLM (free path).

    Prompts ``llm`` for JSON-only output, parses + validates it, and performs at
    most one bounded repair re-prompt on invalid output before raising a typed
    :class:`ProviderError`.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def generate(self, description: str) -> Scene:
        """Return a validated :class:`Scene` for ``description``.

        One initial attempt plus one repair attempt (feeding back the parse/
        validation error). If both fail, raise :class:`ProviderError` rather than
        return a fabricated scene.
        """
        messages = [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=description),
        ]
        text = await self._complete(messages)
        scene, error = self._try_parse(text)
        if scene is not None:
            return scene

        logger.info(
            "studio: procedural output invalid, attempting one bounded repair",
            extra={"error": error},
        )
        repair_messages = [
            *messages,
            Message(role="assistant", content=text),
            Message(
                role="user",
                content=(
                    "That was not valid per the schema "
                    f"({error}). Reply with ONLY the corrected JSON Scene object, "
                    "no prose, no markdown fences."
                ),
            ),
        ]
        repaired_text = await self._complete(repair_messages)
        scene, error = self._try_parse(repaired_text)
        if scene is not None:
            return scene

        raise ProviderError(
            f"procedural generation failed to produce a valid Scene after one "
            f"repair attempt: {error}"
        )

    async def _complete(self, messages: list[Message]) -> str:
        """Call the LLM and return its text, or raise :class:`ProviderError`."""
        response = await self._llm.complete(messages)
        if response.text is None:
            raise ProviderError("LLM returned no text for studio generation")
        return response.text

    @staticmethod
    def _try_parse(text: str) -> tuple[Scene | None, str]:
        """Parse ``text`` into a :class:`Scene`; return ``(scene|None, error)``."""
        raw = _extract_json(text)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc}"
        if not isinstance(data, dict):
            return None, "top-level JSON value is not an object"
        try:
            return validate_scene(data), ""
        except ValidationError as exc:
            return None, f"schema validation failed: {exc.error_count()} error(s)"


@runtime_checkable
class Text3DProvider(Protocol):
    """A text-to-3D mesh provider (optional, high-fidelity, external).

    ``available`` reports whether the provider is usable *right now* (flagged on
    and keyed); the service only attempts ``generate_mesh`` when it returns True,
    and falls back to procedural otherwise. ``generate_mesh`` returns a mesh
    envelope (at minimum ``{"format": ..., "url" | "bytes": ...}``).
    """

    def available(self) -> bool:
        """Whether this provider is configured and usable (flag + key present)."""
        ...

    async def generate_mesh(self, description: str) -> dict[str, Any]:
        """Generate a mesh for ``description`` and return its envelope."""
        ...


class FakeText3D:
    """An offline stub :class:`Text3DProvider` for tests (no network, no key).

    ``available`` is whatever was passed at construction so a test can exercise
    both the "use the mesh" and the "fall back to procedural" branches.
    """

    def __init__(self, available: bool = True) -> None:
        self._available = available

    def available(self) -> bool:
        return self._available

    async def generate_mesh(self, description: str) -> dict[str, Any]:
        return {"format": "glb", "url": "https://example.invalid/fake-mesh.glb"}


class MeshyText3D:
    """Lazy, flagged, keyless-safe :class:`Text3DProvider` over Meshy's API.

    Constructed from config; ``available`` is True only when a Meshy API key is
    present. The ``httpx`` client and the network call are both lazy (inside
    ``generate_mesh``), so wiring this provider performs no I/O and importing it
    needs no extra dependency at module load. Any transport/quota failure is
    wrapped in :class:`ProviderError` so :class:`StudioService` cleanly falls back
    to the free procedural path.
    """

    _BASE_URL = "https://api.meshy.ai/v2/text-to-3d"

    def __init__(
        self, api_key: str | None, model: str = "", timeout: float = 60.0
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def available(self) -> bool:
        return bool(self._api_key)

    async def generate_mesh(self, description: str) -> dict[str, Any]:
        if not self._api_key:  # pragma: no cover - guarded by available()
            raise ProviderError("Meshy: no API key configured")
        import httpx  # lazy: keeps import-time + offline tests dependency-free

        payload: dict[str, Any] = {"mode": "preview", "prompt": description}
        if self._model:
            payload["model"] = self._model
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    self._BASE_URL, json=payload, headers=headers
                )
                resp.raise_for_status()
                body: dict[str, Any] = resp.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Meshy request failed: {exc}") from exc
        url = body.get("model_url") or body.get("result")
        return {"format": "glb", "url": url, "raw": body}


class StudioService:
    """Front door for studio generation: procedural by default, hi-fi with fallback.

    ``quality="fast"`` (the default) always returns a procedural
    ``{"kind": "scene", "scene": {...}}`` envelope. ``quality="hifi"`` tries the
    external :class:`Text3DProvider` and returns ``{"kind": "mesh", ...}`` only
    when it is available and succeeds; otherwise it logs the reason and falls back
    to the free procedural path. The user is never paywalled into an error.
    """

    def __init__(
        self, procedural: ProceduralGenerator, hifi: Text3DProvider | None = None
    ) -> None:
        self._procedural = procedural
        self._hifi = hifi

    async def generate(
        self, description: str, quality: Quality = "fast"
    ) -> dict[str, Any]:
        """Generate for ``description`` at ``quality`` and return a JSON envelope."""
        if quality == "hifi":
            mesh = await self._try_hifi(description)
            if mesh is not None:
                return mesh
        scene = await self._procedural.generate(description)
        return {"kind": "scene", "scene": scene.model_dump(mode="json")}

    async def _try_hifi(self, description: str) -> dict[str, Any] | None:
        """Attempt the external mesh path; return ``None`` to signal fallback."""
        if self._hifi is None:
            logger.info("studio hi-fi requested but no provider wired; using procedural")
            return None
        if not self._hifi.available():
            logger.info(
                "studio hi-fi provider unavailable (no key/flag); using procedural"
            )
            return None
        try:
            mesh = await self._hifi.generate_mesh(description)
        except Exception as exc:  # noqa: BLE001 - falling back to procedural is the contract
            logger.warning(
                "studio hi-fi provider failed; falling back to procedural",
                extra={"error": str(exc)},
            )
            return None
        return {"kind": "mesh", **mesh}
