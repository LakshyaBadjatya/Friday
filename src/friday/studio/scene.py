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

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: A 3D vector (x, y, z) used for position, rotation (Euler radians), and scale.
Vec3 = tuple[float, float, float]

#: The closed set of geometry primitives the frontend knows how to build. Each
#: maps to a Three.js geometry; ``group`` is a transform-only container.
NodeType = Literal["box", "sphere", "cylinder", "cone", "torus", "plane", "group"]


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
