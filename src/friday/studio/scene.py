"""The Scene / SceneNode schema — the JSON contract shared with the frontend.

This is the single source of truth for the generative 3D scene-graph. The LLM is
prompted to emit JSON matching this schema, which is then ``pydantic``-validated
*before* it ever reaches the browser; the no-build Three.js frontend builds meshes
from exactly this shape. Because the model output is validated structured data
(never JavaScript), there is no code execution of anything the LLM produced.

``SceneNode`` is recursive (``children``), and the node ``type`` is a closed
``Literal`` set, so an unknown/unmappable geometry is rejected by validation
rather than failing silently in the renderer.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: A 3D vector (x, y, z) used for position, rotation (Euler radians), and scale.
Vec3 = tuple[float, float, float]

#: The closed set of geometry primitives the frontend knows how to build. Each
#: maps to a Three.js geometry; ``group`` is a transform-only container.
NodeType = Literal["box", "sphere", "cylinder", "cone", "torus", "plane", "group"]

#: Case-insensitive map from a real-model param *synonym* to the canonical
#: abbreviated key the contract + Three.js factory expect. Real LLMs emit full
#: Three.js names (``width``/``height``/``radius``/``tubeRadius``...) while the
#: contract uses ``w``/``h``/``d``/``r``/``tube``; this canonicalizes them so the
#: canvas renders the geometry it was asked for. Keys are pre-lowercased.
_PARAM_SYNONYMS: dict[str, str] = {
    "width": "w",
    "height": "h",
    "depth": "d",
    "radius": "r",
    "radiustop": "r",  # cylinder/cone: use whichever radius the model provided
    "radiusbottom": "r",
    "length": "h",  # cylinder/cone height synonym
    "size": "w",  # some models emit {size} for a box edge
    "tube": "tube",
    "tuberadius": "tube",
    "tube_radius": "tube",
    # Already-canonical keys map to themselves so a mixed payload stays stable.
    "w": "w",
    "h": "h",
    "d": "d",
    "r": "r",
}

#: Segment-count / tessellation keys the renderer does not take from the LLM;
#: dropped during normalization so they never pollute ``params``. Pre-lowercased.
_DROPPED_PARAM_KEYS: frozenset[str] = frozenset(
    {
        "widthsegments",
        "heightsegments",
        "depthsegments",
        "radialsegments",
        "tubularsegments",
        "thetasegments",
        "phisegments",
        "segments",
    }
)


def _normalize_params(params: dict[str, Any]) -> dict[str, float]:
    """Canonicalize a raw ``params`` mapping to the abbreviated contract keys.

    Maps known synonyms (case-insensitive) onto ``w``/``h``/``d``/``r``/``tube``,
    drops segment-count keys, and keeps unknown *numeric* keys as-is. When several
    synonyms map to the same canonical key (e.g. ``radiusTop``/``radiusBottom`` ->
    ``r``), the first one encountered wins so an already-set canonical value is
    never clobbered. Non-numeric values are dropped (the contract is numbers).
    """
    out: dict[str, float] = {}
    for raw_key, value in params.items():
        key = raw_key.lower()
        if key in _DROPPED_PARAM_KEYS:
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        canonical = _PARAM_SYNONYMS.get(key, raw_key)
        # First writer of a canonical key wins (don't clobber an existing value).
        out.setdefault(canonical, num)
    return out


class SceneNode(BaseModel):
    """A single node in the scene-graph: a primitive (or group) with a transform.

    ``params`` carries the per-type geometry numbers (e.g. ``box -> {w, h, d}``,
    ``sphere -> {r}``, ``cylinder``/``cone -> {r, h}``, ``torus -> {r, tube}``,
    ``plane -> {w, h}``, ``group -> {}``). ``children`` nests recursively under the
    node's transform. Transform/material fields default to the contract values so a
    terse LLM payload still produces a sensible mesh.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    type: NodeType
    params: dict[str, float] = Field(default_factory=dict)
    position: Vec3 = (0.0, 0.0, 0.0)
    rotation: Vec3 = (0.0, 0.0, 0.0)
    scale: Vec3 = (1.0, 1.0, 1.0)
    color: str = "#cccccc"
    metalness: float = 0.0
    roughness: float = 0.8
    children: list[SceneNode] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _canonicalize_params(cls, data: Any) -> Any:
        """Canonicalize ``params`` to abbreviated keys *before* field validation.

        Real models emit full Three.js param names; this rewrites them to the
        contract's abbreviated keys (``w``/``h``/``d``/``r``/``tube``) so both the
        API path and :func:`validate_scene` produce canonical params, and the
        no-build frontend renders the requested geometry. Recursion into
        ``children`` is handled by pydantic re-running this validator per child.
        """
        if isinstance(data, dict):
            params = data.get("params")
            if isinstance(params, dict):
                data = {**data, "params": _normalize_params(params)}
        return data


class Scene(BaseModel):
    """A named scene: a dark background plus a list of root :class:`SceneNode`."""

    model_config = ConfigDict(extra="forbid")

    name: str
    background: str = "#101014"
    nodes: list[SceneNode] = Field(default_factory=list)


def validate_scene(data: dict[str, object]) -> Scene:
    """Validate a raw mapping into a :class:`Scene` (the API↔frontend contract).

    Recurses through ``children`` and rejects any node ``type`` outside the closed
    :data:`NodeType` set. Raises :class:`pydantic.ValidationError` on any contract
    violation so callers (the generator's repair loop, the route) can react to bad
    LLM output without trusting it.
    """
    return Scene.model_validate(data)
