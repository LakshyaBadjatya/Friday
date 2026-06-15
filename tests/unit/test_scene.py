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


# --------------------------------------------------------------------------- #
# Param normalization: real models emit full param names (width/height/radius);
# the contract + Three.js factory expect abbreviated keys (w/h/d/r/tube). The
# schema canonicalizes them BEFORE validation so the canvas renders correctly.
# --------------------------------------------------------------------------- #
def test_box_full_param_names_normalize_to_abbreviated() -> None:
    """``box`` with ``{width,height,depth}`` canonicalizes to ``{w,h,d}``."""
    node = SceneNode(
        id="b",
        type="box",
        params={"width": 2, "height": 4, "depth": 1},  # type: ignore[dict-item]
    )
    assert node.params == {"w": 2.0, "h": 4.0, "d": 1.0}


def test_sphere_radius_normalizes_to_r() -> None:
    """``sphere`` with ``{radius}`` canonicalizes to ``{r}``."""
    node = SceneNode(id="s", type="sphere", params={"radius": 3})  # type: ignore[dict-item]
    assert node.params == {"r": 3.0}


def test_cylinder_radius_height_normalize_to_r_h() -> None:
    """``cylinder`` with ``{radius,height}`` canonicalizes to ``{r,h}``."""
    node = SceneNode(
        id="c",
        type="cylinder",
        params={"radius": 0.5, "height": 2},  # type: ignore[dict-item]
    )
    assert node.params == {"r": 0.5, "h": 2.0}


def test_torus_radius_tuberadius_normalize_to_r_tube() -> None:
    """``torus`` with ``{radius,tubeRadius}`` canonicalizes to ``{r,tube}``."""
    node = SceneNode(
        id="t",
        type="torus",
        params={"radius": 1, "tubeRadius": 0.2},  # type: ignore[dict-item]
    )
    assert node.params == {"r": 1.0, "tube": 0.2}


def test_normalization_is_case_insensitive() -> None:
    """Synonym matching ignores case (``Width``/``RADIUS`` etc.)."""
    node = SceneNode(
        id="b",
        type="box",
        params={"Width": 2, "HEIGHT": 4, "Depth": 1},  # type: ignore[dict-item]
    )
    assert node.params == {"w": 2.0, "h": 4.0, "d": 1.0}


def test_normalization_drops_segment_counts() -> None:
    """Segment-count keys (widthSegments/radialSegments/...) are dropped."""
    node = SceneNode(
        id="s",
        type="sphere",
        params={  # type: ignore[arg-type]
            "radius": 1,
            "widthSegments": 32,
            "heightSegments": 16,
        },
    )
    assert node.params == {"r": 1.0}


def test_normalization_handles_length_and_radiustop() -> None:
    """``length`` is an ``h`` synonym; ``radiusTop``/``radiusBottom`` -> ``r``."""
    cyl = SceneNode(
        id="c",
        type="cylinder",
        params={"radiusTop": 0.5, "radiusBottom": 0.5, "length": 3},  # type: ignore[dict-item]
    )
    assert cyl.params == {"r": 0.5, "h": 3.0}


def test_normalization_handles_size_synonym() -> None:
    """``size`` is a ``w`` synonym (some models emit ``{size}`` for a box)."""
    node = SceneNode(id="b", type="box", params={"size": 2})  # type: ignore[dict-item]
    assert node.params == {"w": 2.0}


def test_normalization_handles_tube_underscore_synonyms() -> None:
    """``tube_radius`` (snake) canonicalizes to ``tube`` like ``tubeRadius``."""
    node = SceneNode(
        id="t",
        type="torus",
        params={"radius": 1, "tube_radius": 0.3},  # type: ignore[dict-item]
    )
    assert node.params == {"r": 1.0, "tube": 0.3}


def test_already_abbreviated_params_unchanged() -> None:
    """Params already in canonical abbreviated form pass through untouched."""
    node = SceneNode(
        id="b", type="box", params={"w": 1.0, "h": 2.0, "d": 3.0}
    )
    assert node.params == {"w": 1.0, "h": 2.0, "d": 3.0}


def test_normalization_keeps_unknown_numeric_keys() -> None:
    """An unrecognized numeric key is preserved (not every model param is mapped)."""
    node = SceneNode(
        id="b",
        type="box",
        params={"width": 1, "twist": 0.5},  # type: ignore[dict-item]
    )
    assert node.params == {"w": 1.0, "twist": 0.5}


def test_normalization_recurses_into_children() -> None:
    """Nested children are normalized too (recursive, like the rest of the schema)."""
    scene = validate_scene(
        {
            "name": "nested",
            "nodes": [
                {
                    "id": "parent",
                    "type": "group",
                    "params": {},
                    "children": [
                        {
                            "id": "child",
                            "type": "box",
                            "params": {"width": 2, "height": 4, "depth": 1},
                        }
                    ],
                }
            ],
        }
    )
    child = scene.nodes[0].children[0]
    assert child.params == {"w": 2.0, "h": 4.0, "d": 1.0}


def test_validate_scene_normalizes_full_param_names() -> None:
    """The API path (``validate_scene``) also canonicalizes full param names."""
    scene = validate_scene(
        {
            "name": "box",
            "nodes": [
                {
                    "id": "b",
                    "type": "box",
                    "params": {"width": 2, "height": 4, "depth": 1},
                }
            ],
        }
    )
    assert scene.nodes[0].params == {"w": 2.0, "h": 4.0, "d": 1.0}
