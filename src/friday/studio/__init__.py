"""The 3D Studio feature (Phase 7) — off by default behind ``FRIDAY_ENABLE_STUDIO``.

Describe a 3D model by text or voice and get back a **validated JSON scene-graph**
(:class:`~friday.studio.scene.Scene`) that the no-build Three.js frontend renders.

Safety: the LLM emits this validated JSON, never JavaScript — there is no eval of
model output anywhere. Generation is free by default (the existing LLM drives the
procedural path); an optional high-fidelity external adapter is lazy, flagged, and
keyless-safe — it always falls back to procedural rather than paywalling the user.
"""

from __future__ import annotations

from friday.studio.scene import Scene, SceneNode, Vec3, validate_scene

__all__ = ["Scene", "SceneNode", "Vec3", "validate_scene"]
