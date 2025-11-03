"""Parameterised builder for the three-cubes MuJoCo scene.

This module uses :mod:`tools.mjcf_generator` to construct the XML that
was previously embedded as a large literal in ``testing/3-cubes.py``.
The intent is to make experimentation easier: tweak body placements,
colours, spring parameters, or adjacency matrices in Python and emit a
fresh XML without manual editing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from tools.mjcf_generator import BodyTemplate, SceneBuilder


Vec3 = Sequence[float]
Color = Sequence[float]


@dataclass
class SceneParameters:
    """High-level knobs describing the scene."""

    # Geometry defaults
    default_geom_friction: Vec3 = (1.0, 0.05, 0.01)
    default_joint_damping: float = 2.0

    # Floor appearance
    floor_pos: Vec3 = (0.0, 0.0, -0.3)
    floor_size: Vec3 = (2.0, 2.0, 0.1)
    floor_friction: Vec3 = (1.2, 0.05, 0.02)

    # Crane parameters
    crane_pos: Vec3 = (0.0, 0.0, 0.8)
    crane_joint_name: str = "up_down"
    crane_joint_axis: Vec3 = (0.0, 0.0, 1.0)
    crane_joint_range: Vec3 = (-0.5, 1.0)
    crane_joint_damping: float = 3.0
    crane_capsule_fromto: Sequence[float] = (-0.5, 0.0, 0.0, 0.5, 0.0, 0.0)
    crane_capsule_radius: float = 0.05
    crane_site_pos: Vec3 = (0.0, 0.0, -0.05)

    # Cubes
    box_size: Vec3 = (0.2, 0.2, 0.2)  # MuJoCo box size = half-extent
    box_anchor_offset: float = 0.5
    box_site_radius: float = 0.02
    box_friction: Vec3 = (0.9, 0.05, 0.01)

    red_pos: Vec3 = (-0.5, 0.1, 0.3)
    green_pos: Vec3 = (0.5, 0.0, 0.3)
    blue_pos: Vec3 = (0.0, 0.0, 0.3)

    red_rgba: Color = (1.0, 0.0, 0.0, 1.0)
    green_rgba: Color = (0.0, 1.0, 0.0, 1.0)
    blue_rgba: Color = (0.0, 0.0, 1.0, 1.0)

    # Spheres
    blue_radius: float = 0.2
    dummy_radius: float = 0.2
    dummy_pos: Vec3 = (0.0, 0.0, 0.0)
    dummy_rgba: Color = (1.0, 0.0, 1.0, 0.3)
    dummy_lateral_offset: float = 0.2
    dummy_collision: Dict[str, str] = field(default_factory=lambda: {"contype": "0", "conaffinity": "0"})

    # Spring network
    spring_rgba: str = "0 0 1 1"
    spring1_stiffness: float = 5000.0
    spring1_length: float = 0.1
    spring2_stiffness: float = 5000.0
    spring2_damping: float = 10.0
    spring2_length: float = 0.3
    spring4_stiffness: float = 1000.0
    spring4_damping: float = 100.0
    spring4_length: float = -0.5

    # Chain tendons to crane
    chain_range: Vec3 = (0.0, 0.5)
    chain_red_rgba: str = "1 0 0 1"
    chain_green_rgba: str = "0 1 0 1"

    # Actuator
    actuator_name: str = "lift"
    actuator_kp: float = 7000.0
    actuator_kv: float = 2000.0
    actuator_ctrlrange: Vec3 = (-0.5, 2.0)

    # Horizontal springs adjacency (red_far <-> green_far, red_near <-> green_near)
    horizontal_nodes: Mapping[str, str] = field(default_factory=lambda: {
        "red_far": "red_far",
        "green_far": "green_far",
        "red_near": "red_near",
        "green_near": "green_near",
    })
    horizontal_adjacency: Sequence[Sequence[int]] = field(default_factory=lambda: (
        (0, 1, 0, 0),
        (1, 0, 0, 0),
        (0, 0, 0, 1),
        (0, 0, 1, 0),
    ))


# ---------------------------------------------------------------------------
# Templates


BOX_TEMPLATE = BodyTemplate(
    freejoint={},
    geoms=[
        {
            "name": "{name}",
            "type": "box",
            "size": "{box_size}",
            "rgba": "{rgba}",
            "friction": "{friction}",
        }
    ],
    sites=[
        {"name": "{short}_bottom", "pos": "{bottom_pos}", "size": "{site_radius}"},
        {"name": "{short}_top", "pos": "{top_pos}", "size": "{site_radius}"},
        {"name": "{short}_far", "pos": "{far_pos}", "size": "{site_radius}"},
        {"name": "{short}_near", "pos": "{near_pos}", "size": "{site_radius}"},
        {"name": "{short}_mid", "pos": "{mid_pos}", "size": "{site_radius}"},
    ],
)


SPHERE_TEMPLATE = BodyTemplate(
    freejoint={"name": "{joint_name}"},
    geoms=[
        {
            "name": "{name}",
            "type": "sphere",
            "size": "{radius}",
            "rgba": "{rgba}",
            "friction": "{friction}",
        }
    ],
)


DUMMY_TEMPLATE = BodyTemplate(
    freejoint={"name": "{joint_name}"},
    geoms=[
        {
            "name": "dummy",
            "type": "sphere",
            "size": "{radius}",
            "rgba": "{rgba}",
            **{"contype": "{contype}", "conaffinity": "{conaffinity}"},
        }
    ],
    sites=[
        {"name": "dummy_bottom", "pos": "{bottom_pos}", "size": "{site_radius}"},
        {"name": "dummy_far", "pos": "{far_pos}", "size": "{site_radius}"},
        {"name": "dummy_near", "pos": "{near_pos}", "size": "{site_radius}"},
        {"name": "dummy_green", "pos": "{green_anchor}", "size": "{site_radius}"},
        {"name": "dummy_red", "pos": "{red_anchor}", "size": "{site_radius}"},
    ],
)


CRANE_TEMPLATE = BodyTemplate(
    body_attrs={"pos": "{pos}"},
    joints=[
        {
            "name": "{joint_name}",
            "type": "slide",
            "axis": "{axis}",
            "range": "{joint_range}",
            "damping": "{joint_damping}",
        }
    ],
    geoms=[
        {
            "name": "crane",
            "type": "capsule",
            "fromto": "{fromto}",
            "size": "{radius}",
            "rgba": "1 1 0 1",
        }
    ],
    sites=[{"name": "crane_site", "pos": "{site_pos}", "size": "0.02"}],
)


# ---------------------------------------------------------------------------
# Builder entry point


def _box_mapping(short: str, color: Color, params: SceneParameters) -> Dict[str, str]:
    sx, _, sz = params.box_size
    anchor = params.box_anchor_offset
    return {
        "box_size": params.box_size,
        "rgba": color,
        "friction": params.box_friction,
        "site_radius": params.box_site_radius,
        "bottom_pos": (sx, 0.0, -sz),
        "top_pos": (sx, 0.0, sz),
        "far_pos": (0.0, anchor, 0.0),
        "near_pos": (0.0, -anchor, 0.0),
        "mid_pos": (sx, 0.0, 0.0),
        "short": short,
        "name": f"{short}_box",
    }


def build_scene(params: Optional[SceneParameters] = None) -> SceneBuilder:
    params = params or SceneParameters()
    builder = SceneBuilder()

    # Defaults and assets
    builder.add_default("geom", friction=params.default_geom_friction)
    builder.add_default("joint", damping=params.default_joint_damping)

    builder.add_texture(
        name="grid",
        type="2d",
        builtin="checker",
        rgb1=".1 .2 .3",
        rgb2=".2 .3 .4",
        width="300",
        height="300",
        mark="none",
    )
    builder.add_material(
        name="grid",
        texture="grid",
        texrepeat="1 1",
        texuniform="true",
        reflectance=".2",
    )

    # World elements
    builder.add_light(pos="0 0 3")
    builder.add_geom(
        attrs={
            "name": "floor",
            "type": "plane",
            "pos": params.floor_pos,
            "size": params.floor_size,
            "material": "grid",
            "friction": params.floor_friction,
        }
    )

    # Crane assembly
    CRANE_TEMPLATE.instantiate(
        builder,
        name="crane_base",
        mapping={
            "pos": params.crane_pos,
            "joint_name": params.crane_joint_name,
            "axis": params.crane_joint_axis,
            "joint_range": params.crane_joint_range,
            "joint_damping": params.crane_joint_damping,
            "fromto": params.crane_capsule_fromto,
            "radius": params.crane_capsule_radius,
            "site_pos": params.crane_site_pos,
        },
    )

    # Boxes
    BOX_TEMPLATE.instantiate(
        builder,
        name="red_box",
        pos=params.red_pos,
        mapping=_box_mapping("red", params.red_rgba, params),
    )
    BOX_TEMPLATE.instantiate(
        builder,
        name="green_box",
        pos=params.green_pos,
        mapping=_box_mapping("green", params.green_rgba, params),
    )

    # Blue sphere (target)
    SPHERE_TEMPLATE.instantiate(
        builder,
        name="blue_box",
        pos=params.blue_pos,
        mapping={
            "joint_name": "blue_joint",
            "radius": params.blue_radius,
            "rgba": params.blue_rgba,
            "friction": params.box_friction,
        },
    )

    # Dummy helper body
    DUMMY_TEMPLATE.instantiate(
        builder,
        name="dummy",
        pos=params.dummy_pos,
        mapping={
            "joint_name": "dummy_joint",
            "radius": params.dummy_radius,
            "rgba": params.dummy_rgba,
            "bottom_pos": (0.0, 0.0, -params.dummy_radius),
            "far_pos": (0.0, params.dummy_lateral_offset, 0.0),
            "near_pos": (0.0, -params.dummy_lateral_offset, 0.0),
            "green_anchor": (params.dummy_radius, 0.0, 0.0),
            "red_anchor": (-params.dummy_radius, 0.0, 0.0),
            **params.dummy_collision,
            "site_radius": params.box_site_radius,
        },
    )

    # Tendons
    builder.add_spatial_tendon(
        name="spring1",
        attrs={
            "rgba": params.spring_rgba,
            "stiffness": params.spring1_stiffness,
            "springlength": params.spring1_length,
        },
        path=[
            {"site": "red_bottom"},
            {"geom": "dummy"},
            {"site": "dummy_bottom"},
            {"geom": "dummy"},
            {"site": "green_bottom"},
        ],
    )

    builder.add_spatial_from_adjacency(
        name_prefix="spring",
        start_index=2,
        nodes=params.horizontal_nodes,
        adjacency=params.horizontal_adjacency,
        base_attrs={
            "rgba": params.spring_rgba,
            "stiffness": params.spring2_stiffness,
            "damping": params.spring2_damping,
            "springlength": params.spring2_length,
        },
    )

    builder.add_spatial_tendon(
        name="spring4",
        attrs={
            "rgba": params.spring_rgba,
            "stiffness": params.spring4_stiffness,
            "damping": params.spring4_damping,
            "springlength": params.spring4_length,
        },
        path=[
            {"site": "red_mid"},
            {"geom": "dummy"},
            {"site": "dummy_red"},
        ],
    )

    builder.add_spatial_tendon(
        name="spring5",
        attrs={
            "rgba": params.spring_rgba,
            "stiffness": params.spring4_stiffness,
            "damping": params.spring4_damping,
            "springlength": params.spring4_length,
        },
        path=[
            {"site": "green_mid"},
            {"geom": "dummy"},
            {"site": "dummy_green"},
        ],
    )

    builder.add_spatial_tendon(
        name="chain_red",
        attrs={"limited": "true", "range": params.chain_range, "rgba": params.chain_red_rgba},
        path=[{"site": "red_top"}, {"site": "crane_site"}],
    )

    builder.add_spatial_tendon(
        name="chain_green",
        attrs={"limited": "true", "range": params.chain_range, "rgba": params.chain_green_rgba},
        path=[{"site": "green_top"}, {"site": "crane_site"}],
    )

    # Actuation
    builder.add_position_actuator(
        name=params.actuator_name,
        joint=params.crane_joint_name,
        kp=params.actuator_kp,
        kv=params.actuator_kv,
        ctrlrange=params.actuator_ctrlrange,
    )

    return builder


def build_xml(params: Optional[SceneParameters] = None) -> str:
    """Convenience to obtain the XML string directly."""

    return build_scene(params).to_string()
