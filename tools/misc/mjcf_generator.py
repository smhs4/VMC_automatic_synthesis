"""Utility helpers for building MuJoCo XML programmatically.

The goal is to avoid large hand-authored XML strings by providing a
lightweight Python builder that understands common MuJoCo constructs
(defaults, assets, worldbody, tendons, actuators) and supports modular
body templates that can be instantiated with simple transforms.

The builder focuses on:
* Formatting numeric attributes consistently ("0.1 0.2 0.3" styles).
* Allowing templates to express attributes via ``str.format``
  placeholders (e.g. ``"{name}_bottom"``).
* Basic transform support: translation, axis-angle rotation, and
  coordinate-axis reflections for positioning multiple instances of a
  template (e.g. mirroring a body about the YZ plane).
* Generating spatial tendons from adjacency matrices so spring networks
  can be described compactly.

Example
-------

>>> from tools.mjcf_generator import SceneBuilder, BodyTemplate
>>> box = BodyTemplate(
...     geoms=[{"name": "{name}_geom", "type": "box", "size": [0.2, 0.2, 0.2],
...              "rgba": "{rgba}", "friction": "{friction}"}],
...     sites=[{"name": "{name}_top", "pos": [0, 0, 0.2], "size": 0.02}])
>>> builder = SceneBuilder()
>>> box.instantiate(builder, name="red", pos=[-0.3, 0, 0.2],
...                 mapping={"rgba": [1, 0, 0, 1], "friction": [0.9, 0.05, 0.01]})
>>> xml = builder.to_string()

The module also exposes a simple CLI to emit XML from a JSON config.
This keeps dependency footprint minimal while providing a reusable
foundation for bespoke generators.
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import xml.etree.ElementTree as ET

Number = Union[int, float]
Vec3 = Sequence[Number]


# ---------------------------------------------------------------------------
# Formatting helpers


def _float_to_str(value: Number) -> str:
    """Format floats compactly while keeping ints intact."""

    if isinstance(value, int):
        return str(value)
    return f"{float(value):.12g}"  # up to ~12 significant figures, no trailing zeros


def _stringify(value: Any) -> str:
    """Recursively convert supported attribute values to MuJoCo strings."""

    if value is None:
        raise ValueError("Cannot stringify None for MJCF attribute")
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return _float_to_str(value)
    if isinstance(value, (list, tuple)):
        return " ".join(_stringify(v) for v in value)
    raise TypeError(f"Unsupported attribute type: {type(value)!r}")


def _prepare_mapping(mapping: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    if not mapping:
        return {}
    return {key: _stringify(value) for key, value in mapping.items()}


def _format_value(value: Any, mapping: Optional[Mapping[str, str]] = None) -> str:
    """Format an attribute value possibly containing placeholders."""

    if isinstance(value, str):
        fmt_map = mapping or {}
        return value.format(**fmt_map)
    if isinstance(value, (list, tuple)):
        return " ".join(_format_value(v, mapping) for v in value)
    if value is None:
        raise ValueError("Attempted to format None attribute")
    return _float_to_str(value)

def _prettify_xml(xml_string):
    """Add proper indentation to XML string."""
    import xml.dom.minidom
    dom = xml.dom.minidom.parseString(xml_string)
    return dom.toprettyxml(indent="  ")

def _serialize_attrs(attrs: Mapping[str, Any], mapping: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    fmt_map = mapping or {}
    result: Dict[str, str] = {}
    for key, value in attrs.items():
        if value is None:
            continue
        result[key] = _format_value(value, fmt_map)
    return result


# ---------------------------------------------------------------------------
# Quaternion utilities


def _normalize(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0:
        raise ValueError("Zero-length vector cannot be normalised")
    return [v / norm for v in vec]


def _axis_angle_to_quat(axis: Sequence[float], angle_deg: float) -> List[float]:
    ax = _normalize(axis)
    angle_rad = math.radians(angle_deg)
    half = angle_rad / 2.0
    s = math.sin(half)
    w = math.cos(half)
    x, y, z = (component * s for component in ax)
    return [w, x, y, z]


def _quat_multiply(a: Sequence[float], b: Sequence[float]) -> List[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _is_identity_quat(q: Sequence[float], tol: float = 1e-9) -> bool:
    return abs(q[0] - 1.0) < tol and all(abs(comp) < tol for comp in q[1:])


def _ensure_quat(value: Optional[Sequence[float]]) -> Optional[List[float]]:
    if value is None:
        return None
    if len(value) != 4:
        raise ValueError(f"Quaternion must have length 4, got {len(value)}")
    return [float(v) for v in value]


def _vector(value: Optional[Sequence[float]]) -> Optional[List[float]]:
    if value is None:
        return None
    if len(value) != 3:
        raise ValueError(f"3-vector expected, got length {len(value)}")
    return [float(v) for v in value]


def _combine_positions(parts: Iterable[Optional[Sequence[float]]], reflect_axes: Optional[Iterable[str]] = None) -> Optional[List[float]]:
    total = [0.0, 0.0, 0.0]
    contributing = False
    for part in parts:
        if part is None:
            continue
        contributing = True
        vec = _vector(part)
        total = [a + b for a, b in zip(total, vec)]
    if not contributing:
        return None
    if reflect_axes:
        for axis in reflect_axes:
            idx = "xyz".index(axis.lower())
            total[idx] = -total[idx]
    return total


def _combine_quaternions(quats: Iterable[Optional[Sequence[float]]]) -> Optional[List[float]]:
    result = [1.0, 0.0, 0.0, 0.0]
    contributing = False
    for quat in quats:
        if quat is None:
            continue
        contributing = True
        result = _quat_multiply(result, _ensure_quat(quat))
    if not contributing or _is_identity_quat(result):
        return None
    return result


# ---------------------------------------------------------------------------
# Scene builder primitives


class SceneBuilder:
    """Incrementally build up an MJCF document."""

    def __init__(self) -> None:
        self.root = ET.Element("mujoco")
        self._default: Optional[ET.Element] = None
        self._asset: Optional[ET.Element] = None
        self.worldbody = ET.SubElement(self.root, "worldbody")
        self._tendon: Optional[ET.Element] = None
        self._actuator: Optional[ET.Element] = None

    # -- Defaults ----------------------------------------------------------------

    def _ensure_default(self) -> ET.Element:
        if self._default is None:
            self._default = ET.SubElement(self.root, "default")
        return self._default

    def add_default(self, element: str, **attrs: Any) -> None:
        serialized = _serialize_attrs(attrs)
        if serialized:
            ET.SubElement(self._ensure_default(), element, serialized)

    # -- Assets ------------------------------------------------------------------

    def _ensure_asset(self) -> ET.Element:
        if self._asset is None:
            self._asset = ET.SubElement(self.root, "asset")
        return self._asset

    def add_texture(self, **attrs: Any) -> None:
        ET.SubElement(self._ensure_asset(), "texture", _serialize_attrs(attrs))

    def add_material(self, **attrs: Any) -> None:
        ET.SubElement(self._ensure_asset(), "material", _serialize_attrs(attrs))

    # -- Worldbody ---------------------------------------------------------------

    def add_option(self, **attrs: Any) -> None:
        ET.SubElement(self.root, "option", _serialize_attrs(attrs))

    def add_light(self, **attrs: Any) -> None:
        ET.SubElement(self.worldbody, "light", _serialize_attrs(attrs))

    def add_geom(self, parent: Optional[ET.Element] = None, *, attrs: Mapping[str, Any], mapping: Optional[Mapping[str, str]] = None) -> ET.Element:
        parent_elem = parent or self.worldbody
        return ET.SubElement(parent_elem, "geom", _serialize_attrs(attrs, mapping))

    def add_site(self, parent: ET.Element, *, attrs: Mapping[str, Any], mapping: Optional[Mapping[str, str]] = None) -> ET.Element:
        return ET.SubElement(parent, "site", _serialize_attrs(attrs, mapping))

    def add_joint(self, parent: ET.Element, *, attrs: Mapping[str, Any], mapping: Optional[Mapping[str, str]] = None) -> ET.Element:
        return ET.SubElement(parent, "joint", _serialize_attrs(attrs, mapping))

    def add_freejoint(self, parent: ET.Element, attrs: Optional[Mapping[str, Any]] = None, mapping: Optional[Mapping[str, str]] = None) -> ET.Element:
        parameters = _serialize_attrs(attrs or {}, mapping)
        return ET.SubElement(parent, "freejoint", parameters)

    def add_body(self, *, name: str, pos: Optional[Sequence[float]] = None, quat: Optional[Sequence[float]] = None,
                 attrs: Optional[Mapping[str, Any]] = None, parent: Optional[ET.Element] = None,
                 mapping: Optional[Mapping[str, str]] = None) -> ET.Element:
        attributes: Dict[str, str] = {"name": name}
        if pos is not None:
            attributes["pos"] = _stringify(pos)
        if quat is not None:
            attributes["quat"] = _stringify(quat)
        if attrs:
            attributes.update(_serialize_attrs(attrs, mapping))
        return ET.SubElement(parent or self.worldbody, "body", attributes)

    # -- Tendons -----------------------------------------------------------------

    def _ensure_tendon(self) -> ET.Element:
        if self._tendon is None:
            self._tendon = ET.SubElement(self.root, "tendon")
        return self._tendon

    def add_spatial_tendon(self, *, name: str, attrs: Optional[Mapping[str, Any]] = None,
                           path: Sequence[Mapping[str, Any]], mapping: Optional[Mapping[str, str]] = None) -> ET.Element:
        tendon = self._ensure_tendon()
        tendon_attrs = {"name": name}
        if attrs:
            tendon_attrs.update(_serialize_attrs(attrs, mapping))
        spatial = ET.SubElement(tendon, "spatial", tendon_attrs)
        for segment in path:
            if len(segment) != 1:
                raise ValueError("Each tendon path segment must specify exactly one target (site/geom/body)")
            (tag, value), = segment.items()
            ET.SubElement(spatial, tag, _serialize_attrs({tag: value}, mapping))
        return spatial

    def add_spatial_from_adjacency(self, *, name_prefix: str, nodes: Mapping[str, str], adjacency: Sequence[Sequence[Any]],
                                   base_attrs: Optional[Mapping[str, Any]] = None,
                                   path_template: Optional[Sequence[Mapping[str, Any]]] = None,
                                   start_index: int = 1,
                                   mapping: Optional[Mapping[str, str]] = None) -> List[str]:
        """Create spatial tendons for every non-zero entry in an adjacency matrix.

        ``nodes`` maps logical node names to site identifiers. ``adjacency`` is a
        square matrix (list of lists) aligned with ``nodes`` ordering. Non-zero
        entries produce a tendon. Entries may be dictionaries containing
        per-edge attribute overrides and an optional ``"path"`` list that
        overrides ``path_template``.
        """

        node_items = list(nodes.items())
        if any(len(row) != len(node_items) for row in adjacency):
            raise ValueError("Adjacency matrix must be square with size equal to node count")

        created: List[str] = []
        counter = start_index
        fmt_base = mapping or {}

        for i, (name_i, site_i) in enumerate(node_items):
            for j in range(i + 1, len(node_items)):
                edge_data = adjacency[i][j]
                if not edge_data:
                    continue

                name_j, site_j = node_items[j]
                attributes = dict(base_attrs or {})
                edge_path_template = path_template

                if isinstance(edge_data, Mapping):
                    overrides = dict(edge_data)  # shallow copy
                    edge_path_template = overrides.pop("path", edge_path_template)
                    attributes.update(overrides.pop("attrs", overrides))
                elif isinstance(edge_data, (int, float)):
                    # treat numeric zero as skip, non-zero as present without changes
                    pass
                else:
                    raise TypeError("Adjacency entries must be 0/False, numeric, or dict")

                tendon_name = f"{name_prefix}{counter}"
                counter += 1

                edge_mapping = {
                    "from": name_i,
                    "to": name_j,
                    "from_site": site_i,
                    "to_site": site_j,
                    "index": str(counter - 1),
                }
                edge_mapping.update(fmt_base)
                fmt_map = _prepare_mapping(edge_mapping)

                default_path = path_template or (
                    {"site": "{from_site}"}, {"site": "{to_site}"}
                )
                steps = edge_path_template or default_path
                formatted_steps = []
                for step in steps:
                    formatted_steps.append({tag: _format_value(value, fmt_map) for tag, value in step.items()})

                self.add_spatial_tendon(name=tendon_name, attrs=attributes, path=formatted_steps, mapping=fmt_map)
                created.append(tendon_name)

        return created

    # -- Actuators ---------------------------------------------------------------

    def _ensure_actuator(self) -> ET.Element:
        if self._actuator is None:
            self._actuator = ET.SubElement(self.root, "actuator")
        return self._actuator

    def add_position_actuator(self, **attrs: Any) -> ET.Element:
        return ET.SubElement(self._ensure_actuator(), "position", _serialize_attrs(attrs))

    # -- Output ------------------------------------------------------------------

    def to_string(self) -> str:
        xml_string = ET.tostring(self.root, encoding="unicode")
        return _prettify_xml(xml_string)


# ---------------------------------------------------------------------------
# Body templates


@dataclass
class BodyTemplate:
    """Reusable description of a body, parameterised via format placeholders."""

    body_attrs: Mapping[str, Any] = field(default_factory=dict)
    joints: List[Mapping[str, Any]] = field(default_factory=list)
    freejoint: Optional[Mapping[str, Any]] = None
    geoms: List[Mapping[str, Any]] = field(default_factory=list)
    sites: List[Mapping[str, Any]] = field(default_factory=list)
    children: List["BodyTemplate"] = field(default_factory=list)

    def instantiate(
        self,
        builder: SceneBuilder,
        *,
        name: str,
        parent: Optional[ET.Element] = None,
        pos: Optional[Sequence[float]] = None,
        quat: Optional[Sequence[float]] = None,
        transform: Optional[Mapping[str, Any]] = None,
        mapping: Optional[Mapping[str, Any]] = None,
        body_overrides: Optional[Mapping[str, Any]] = None,
    ) -> ET.Element:
        """Create an instance of this body template.

        Parameters
        ----------
        builder: SceneBuilder
            Target builder accumulating the MJCF document.
        name: str
            Name for the instantiated body.
        parent: xml.etree.ElementTree.Element, optional
            Optional parent body (defaults to ``worldbody``).
        pos / quat: sequence of 3 / 4 numbers, optional
            Additional translation/rotation applied on top of template
            defaults and transforms.
        transform: mapping, optional
            Special keys ``translate`` (vec3), ``rotate`` (dict with
            ``axis`` and ``angle``), and ``reflect`` (iterable of axes)
            provide reusable placement transforms.
        mapping: mapping, optional
            Extra placeholder values available when formatting attribute
            strings inside the template.
        body_overrides: mapping, optional
            Additional body attributes applied after template defaults
            (e.g. change ``gravcomp`` or add ``mass``).
        """

        template_mapping = {"name": name, "NAME": name.upper()}
        if mapping:
            template_mapping.update(mapping)
        fmt_mapping = _prepare_mapping(template_mapping)

        base_attrs = deepcopy(dict(self.body_attrs))
        if body_overrides:
            base_attrs.update(body_overrides)

        base_pos_raw = base_attrs.pop("pos", None)
        base_quat_raw = base_attrs.pop("quat", None)

        def _coerce_vector(value: Any, expected_len: int) -> Optional[List[float]]:
            if value is None:
                return None
            formatted = _format_value(value, fmt_mapping)
            components = [float(part) for part in formatted.split()]
            if expected_len > 0 and len(components) != expected_len:
                raise ValueError(
                    f"Expected {expected_len} components, got {len(components)} for value '{formatted}'"
                )
            return components

        base_pos = _coerce_vector(base_pos_raw, 3)
        base_quat = _coerce_vector(base_quat_raw, 4)

        transform = transform or {}
        reflect_axes = transform.get("reflect")
        transform_translate = transform.get("translate")
        transform_rotate = None
        rotate_spec = transform.get("rotate")
        if rotate_spec:
            axis = rotate_spec.get("axis", [0, 0, 1])
            angle = rotate_spec.get("angle", 0.0)
            transform_rotate = _axis_angle_to_quat(axis, angle)

        final_pos = _combine_positions([base_pos, transform_translate, pos], reflect_axes)
        final_quat = _combine_quaternions([base_quat, transform_rotate, quat])

        body_elem = builder.add_body(
            name=name,
            pos=final_pos,
            quat=final_quat,
            attrs=base_attrs,
            parent=parent,
            mapping=fmt_mapping,
        )

        if self.freejoint is not None:
            builder.add_freejoint(body_elem, self.freejoint, fmt_mapping)

        for joint in self.joints:
            builder.add_joint(body_elem, attrs=joint, mapping=fmt_mapping)

        for geom in self.geoms:
            builder.add_geom(body_elem, attrs=geom, mapping=fmt_mapping)

        for site in self.sites:
            builder.add_site(body_elem, attrs=site, mapping=fmt_mapping)

        for child in self.children:
            child.instantiate(builder, name=f"{name}_{child.body_attrs.get('name', 'child')}", parent=body_elem)

        return body_elem


# ---------------------------------------------------------------------------
# CLI for JSON-driven generation (optional convenience)


def _load_json_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _builder_from_json(config: Mapping[str, Any]) -> SceneBuilder:
    """Naive JSON-to-MJCF converter using the same primitives.

    The JSON grammar is intentionally minimal: it mirrors the public
    methods on :class:`SceneBuilder` and :class:`BodyTemplate`. This is
    primarily a convenience for quick experimentation and is not aimed
    at covering full MJCF surface area.
    """

    builder = SceneBuilder()

    for geom_default in config.get("defaults", {}).get("geom", []):
        builder.add_default("geom", **geom_default)
    for joint_default in config.get("defaults", {}).get("joint", []):
        builder.add_default("joint", **joint_default)

    for texture in config.get("textures", []):
        builder.add_texture(**texture)
    for material in config.get("materials", []):
        builder.add_material(**material)

    for option in config.get("options", []):
        builder.add_option(**option)
    for light in config.get("lights", []):
        builder.add_light(**light)
    for geom in config.get("geoms", []):
        builder.add_geom(attrs=geom)

    body_templates: Dict[str, BodyTemplate] = {}
    for template_name, template_data in config.get("templates", {}).items():
        body_templates[template_name] = BodyTemplate(
            body_attrs=template_data.get("body", {}),
            joints=template_data.get("joints", []),
            freejoint=template_data.get("freejoint"),
            geoms=template_data.get("geoms", []),
            sites=template_data.get("sites", []),
        )

    for body in config.get("bodies", []):
        template_ref = body.get("template")
        if not template_ref:
            raise ValueError("JSON config bodies must reference a template")
        template = body_templates[template_ref]
        template.instantiate(
            builder,
            name=body["name"],
            pos=body.get("pos"),
            quat=body.get("quat"),
            transform=body.get("transform"),
            mapping=body.get("mapping"),
            body_overrides=body.get("body_overrides"),
        )

    if "tendons" in config:
        for tendon in config["tendons"].get("spatial", []):
            builder.add_spatial_tendon(
                name=tendon["name"],
                attrs=tendon.get("attrs"),
                path=tendon["path"],
            )

    for actuator in config.get("actuators", {}).get("position", []):
        builder.add_position_actuator(**actuator)

    return builder


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MuJoCo XML via a JSON description")
    parser.add_argument("config", help="Path to a JSON configuration file")
    parser.add_argument("--out", "-o", help="Destination file (defaults to stdout)")
    args = parser.parse_args()

    builder = _builder_from_json(_load_json_config(args.config))
    xml_string = builder.to_string()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(xml_string)
    else:
        print(xml_string)


if __name__ == "__main__":
    main()
