"""Tests for the Scene/SceneNode pydantic schema (Phase 7, Stage 1).

The Scene schema is the JSON contract shared by the backend generator and the
no-build Three.js frontend. These tests pin the contract: a well-formed scene
validates (including recursive children + defaults), and an unknown node type is
rejected so the LLM can never smuggle an unmappable geometry past validation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from friday.studio.scene import Scene, SceneNode, validate_scene


def _good_scene_data() -> dict[str, object]:
    """A minimal-but-complete valid scene with a nested child group."""
    return {
        "name": "snowman",
        "background": "#101014",
        "nodes": [
            {
                "id": "body",
                "type": "sphere",
                "params": {"r": 1.0},
                "position": [0, 0, 0],
                "color": "#ffffff",
                "children": [
                    {
                        "id": "head",
                        "type": "sphere",
                        "params": {"r": 0.6},
                        "position": [0, 1.4, 0],
                    }
                ],
            }
        ],
    }


def test_validate_scene_accepts_a_good_scene() -> None:
    """A well-formed scene validates and preserves its (recursive) structure."""
    scene = validate_scene(_good_scene_data())

    assert isinstance(scene, Scene)
    assert scene.name == "snowman"
    assert scene.background == "#101014"
    assert len(scene.nodes) == 1

    body = scene.nodes[0]
    assert body.id == "body"
    assert body.type == "sphere"
    assert body.params == {"r": 1.0}
    # Recursive children parse into SceneNode instances.
    assert len(body.children) == 1
    assert isinstance(body.children[0], SceneNode)
    assert body.children[0].id == "head"
    assert body.children[0].type == "sphere"


def test_scenenode_applies_schema_defaults() -> None:
    """Omitted transform/material fields fall back to the contract defaults."""
    node = SceneNode(id="b", type="box", params={"w": 1.0, "h": 1.0, "d": 1.0})

    assert node.position == (0.0, 0.0, 0.0)
    assert node.rotation == (0.0, 0.0, 0.0)
    assert node.scale == (1.0, 1.0, 1.0)
    assert node.color == "#cccccc"
    assert node.metalness == 0.0
    assert node.roughness == 0.8
    assert node.children == []


def test_scene_applies_background_default() -> None:
    """An omitted ``background`` defaults to the dark studio backdrop."""
    scene = Scene(name="empty", nodes=[])
    assert scene.background == "#101014"


def test_validate_scene_rejects_unknown_node_type() -> None:
    """A node ``type`` outside the allowed literal set is rejected."""
    data = _good_scene_data()
    data["nodes"] = [{"id": "x", "type": "teapot", "params": {}}]

    with pytest.raises(ValidationError):
        validate_scene(data)


def test_validate_scene_rejects_unknown_nested_node_type() -> None:
    """Validation recurses: an unknown type on a *child* node is rejected too."""
    data = _good_scene_data()
    nodes = data["nodes"]
    assert isinstance(nodes, list)
    nodes[0]["children"] = [{"id": "bad", "type": "dragon", "params": {}}]

    with pytest.raises(ValidationError):
        validate_scene(data)


def test_validate_scene_rejects_non_dict() -> None:
    """A non-mapping payload (e.g. a bare list) is a validation error, not a crash."""
    with pytest.raises(ValidationError):
        validate_scene([])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "node_type", ["box", "sphere", "cylinder", "cone", "torus", "plane", "group"]
)
def test_every_contract_node_type_validates(node_type: str) -> None:
    """Each of the seven contract geometry types is accepted."""
    scene = validate_scene(
        {"name": "t", "nodes": [{"id": "n", "type": node_type, "params": {}}]}
    )
    assert scene.nodes[0].type == node_type
