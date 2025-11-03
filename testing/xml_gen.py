import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

#!/usr/bin/env python3
# xml_gen.py
# Build MuJoCo XML from a structured spec (JSON/YAML), with templates, instancing, and safe referencing.
# Child names are auto-prefixed with their body name for uniqueness (bodyName__childName).
# References use "body/child" (e.g., site: "red_box/red_top") and are resolved to final unique names.

import xml.etree.ElementTree as ET

try:
    import yaml  # optional for YAML input
except ImportError:
    yaml = None


# ----------------------------- Utilities -----------------------------

def to_str_vec(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v)
    return str(v)

def norm_attrs(attrs: Dict[str, Any]) -> Dict[str, str]:
    return {k: to_str_vec(v) for k, v in attrs.items() if v is not None}

def indent(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            indent(e, level + 1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

def deep_format(obj: Any, vars: Dict[str, Any]) -> Any:
    # Recursively format strings with {var} placeholders using vars.
    if isinstance(obj, str):
        try:
            return obj.format_map(DefaultDict(vars))
        except Exception:
            return obj
    if isinstance(obj, list):
        return [deep_format(x, vars) for x in obj]
    if isinstance(obj, dict):
        return {k: deep_format(v, vars) for k, v in obj.items()}
    return obj

class DefaultDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


# ----------------------------- Registry -----------------------------

class NameRegistry:
    def __init__(self, auto_prefix_children: bool = True):
        self.auto_prefix_children = auto_prefix_children
        # Maps for global lookup
        self.bodies: Dict[str, ET.Element] = {}
        self.sites: Dict[str, str] = {}   # alias "body/local" -> final name
        self.geoms: Dict[str, str] = {}
        self.joints: Dict[str, str] = {}
        self.actuators: Dict[str, str] = {}
        # Reverse: final name -> type
        self.all_used: Dict[str, str] = {}  # name -> kind

    def make_child_name(self, body_name: str, child_name: str, abs_name: Optional[bool] = None) -> str:
        # If child_name starts with "::", treat as absolute and strip marker.
        if abs_name is True or (isinstance(child_name, str) and child_name.startswith("::")):
            return child_name.lstrip(":")
        if not self.auto_prefix_children:
            return child_name
        return f"{body_name}__{child_name}"

    def reserve(self, kind: str, final_name: str) -> str:
        if final_name in self.all_used and self.all_used[final_name] != kind:
            # If collision with different kind, add numeric suffix.
            i = 2
            base = final_name
            while final_name in self.all_used:
                final_name = f"{base}_{i}"
                i += 1
        self.all_used[final_name] = kind
        return final_name

    def add_body(self, name: str, elem: ET.Element):
        final = self.reserve("body", name)
        self.bodies[final] = elem
        return final

    def add_child_alias(self, kind: str, alias_key: str, final_name: str):
        final = self.reserve(kind, final_name)
        if kind == "site":
            self.sites[alias_key] = final
        elif kind == "geom":
            self.geoms[alias_key] = final
        elif kind == "joint":
            self.joints[alias_key] = final
        elif kind == "actuator":
            self.actuators[alias_key] = final
        return final

    def resolve(self, kind: str, ref: str) -> str:
        # Pass-through if already a final name and exists
        if ref in self.all_used and (self.all_used[ref] == kind or kind == "any"):
            return ref
        # Allow "body/local"
        if "/" in ref:
            body, local = ref.split("/", 1)
            key = f"{body}/{local}"
            if kind == "site":
                name = self.sites.get(key)
            elif kind == "geom":
                name = self.geoms.get(key)
            elif kind == "joint":
                name = self.joints.get(key)
            elif kind == "actuator":
                name = self.actuators.get(key)
            else:
                name = None
            if not name:
                raise KeyError(f"Unknown {kind} reference '{ref}'")
            return name
        # Fallback: if unique match by final name for the kind
        mapping = {"site": self.sites, "geom": self.geoms, "joint": self.joints, "actuator": self.actuators}
        if kind in mapping:
            # ref might already be the final name but not registered via alias; check global
            if ref in self.all_used and (self.all_used[ref] == kind or kind == "any"):
                return ref
        raise KeyError(f"Unresolvable {kind} reference '{ref}'. Use 'body/local' form.")


# ----------------------------- Builder -----------------------------

class MjModelBuilder:
    def __init__(self, spec: Dict[str, Any], auto_prefix_children: bool = True):
        self.spec = spec
        self.registry = NameRegistry(auto_prefix_children=auto_prefix_children)

    def build(self) -> ET.ElementTree:
        mujoco_attrs = norm_attrs(self.spec.get("mujoco", {}))
        root = ET.Element("mujoco", mujoco_attrs)

        self._add_optional(root, "compiler", self.spec.get("compiler"))
        self._add_optional(root, "option", self.spec.get("option"))

        self._add_defaults(root, self.spec.get("defaults"))
        self._add_asset(root, self.spec.get("asset"))

        worldbody = ET.SubElement(root, "worldbody")
        self._add_world_children(worldbody, self.spec.get("worldbody", {}))

        self._add_tendon(root, self.spec.get("tendon"))
        self._add_actuators(root, self.spec.get("actuator"))

        indent(root)
        return ET.ElementTree(root)

    # ---------- Sections ----------

    def _add_optional(self, root: ET.Element, tag: str, attrs: Optional[Dict[str, Any]]):
        if not attrs:
            return
        ET.SubElement(root, tag, norm_attrs(attrs))

    def _add_defaults(self, root: ET.Element, defaults: Optional[Dict[str, Any]]):
        if not defaults:
            return
        default = ET.SubElement(root, "default")
        for tag, attrs in defaults.items():
            if isinstance(attrs, list):
                for a in attrs:
                    ET.SubElement(default, tag, norm_attrs(a))
            elif isinstance(attrs, dict):
                ET.SubElement(default, tag, norm_attrs(attrs))

    def _add_asset(self, root: ET.Element, asset: Optional[Dict[str, Any]]):
        if not asset:
            return
        asset_el = ET.SubElement(root, "asset")
        for tag, items in asset.items():
            if isinstance(items, dict):
                ET.SubElement(asset_el, tag, norm_attrs(items))
            elif isinstance(items, list):
                for it in items:
                    ET.SubElement(asset_el, tag, norm_attrs(it))

    def _add_world_children(self, world: ET.Element, wb: Dict[str, Any]):
        # Lights
        for light in wb.get("lights", []):
            ET.SubElement(world, "light", norm_attrs(light))

        # Top-level geoms (e.g., floor)
        for geom in wb.get("geoms", []):
            name = geom.get("name")
            if name:
                self.registry.reserve("geom", name)
            ET.SubElement(world, "geom", norm_attrs(geom))

        # Templates
        templates = self.spec.get("templates", {})

        # Bodies (instances or raw)
        for body in wb.get("bodies", []):
            if "use" in body:
                tmpl_name = body["use"]
                if tmpl_name not in templates:
                    raise KeyError(f"Unknown template '{tmpl_name}'")
                vars_map = body.get("vars", {})
                tmpl_body = deep_format(copy.deepcopy(templates[tmpl_name]), vars_map)
                # Allow overrides at body level
                base = {"name": body.get("name"), **{k: v for k, v in body.items() if k not in ("use", "vars", "overrides")}}
                merged = merge_body_specs(base, tmpl_body)
                elem = self._build_body(world, merged)
                # Overrides for nested elements (sites/geoms/joints) by local name
                overrides = body.get("overrides", {})
                if overrides:
                    self._apply_body_overrides(elem, overrides)
            else:
                self._build_body(world, body)

    def _apply_body_overrides(self, body_el: ET.Element, overrides: Dict[str, Any]):
        # overrides = {"sites": [{"name":"local_name","set":{...}}, ...], "geoms":[...], "joints":[...]}
        by_tag = {"sites": "site", "geoms": "geom", "joints": "joint", "bodies": "body"}
        for key, tag in by_tag.items():
            for rule in overrides.get(key, []):
                local = rule.get("name")
                sets = rule.get("set", {})
                if not local:
                    continue
                # Names are prefixed in XML as bodyName__local
                pref = find_child_by_prefixed_local_name(body_el, tag, local)
                if pref is not None:
                    for k, v in sets.items():
                        pref.set(k, to_str_vec(v))

    def _build_body(self, parent: ET.Element, body: Dict[str, Any]) -> ET.Element:
        name = body.get("name")
        if not name:
            raise ValueError("Every body must have a 'name'")
        attrs = {k: v for k, v in body.items() if k not in ("geoms", "sites", "joints", "bodies", "freejoint")}
        attrs = norm_attrs(attrs)
        b_el = ET.SubElement(parent, "body", attrs)
        self.registry.add_body(name, b_el)

        # Freejoint
        fj = body.get("freejoint")
        if fj:
            if isinstance(fj, dict):
                jname = fj.get("name")
                if jname:
                    final = self.registry.make_child_name(name, jname, abs_name=fj.get("abs"))
                    self.registry.add_child_alias("joint", f"{name}/{jname}", final)
                    fj = {"name": final, **{k: v for k, v in fj.items() if k not in ("name", "abs")}}
                ET.SubElement(b_el, "freejoint", norm_attrs(fj))
            else:
                ET.SubElement(b_el, "freejoint")

        # Joints
        for j in body.get("joints", []):
            j = copy.deepcopy(j)
            if "name" in j:
                final = self.registry.make_child_name(name, j["name"], abs_name=j.get("abs"))
                self.registry.add_child_alias("joint", f"{name}/{j['name']}", final)
                j["name"] = final
                j.pop("abs", None)
            ET.SubElement(b_el, "joint", norm_attrs(j))

        # Geoms
        for g in body.get("geoms", []):
            g = copy.deepcopy(g)
            if "name" in g:
                final = self.registry.make_child_name(name, g["name"], abs_name=g.get("abs"))
                self.registry.add_child_alias("geom", f"{name}/{g['name']}", final)
                g["name"] = final
                g.pop("abs", None)
            ET.SubElement(b_el, "geom", norm_attrs(g))

        # Sites
        for s in body.get("sites", []):
            s = copy.deepcopy(s)
            if "name" in s:
                final = self.registry.make_child_name(name, s["name"], abs_name=s.get("abs"))
                self.registry.add_child_alias("site", f"{name}/{s['name']}", final)
                s["name"] = final
                s.pop("abs", None)
            ET.SubElement(b_el, "site", norm_attrs(s))

        # Sub-bodies
        for sb in body.get("bodies", []):
            self._build_body(b_el, sb)

        return b_el

    def _add_tendon(self, root: ET.Element, tendon: Optional[Dict[str, Any]]):
        if not tendon:
            return
        tend_el = ET.SubElement(root, "tendon")
        # Spatial tendons
        for sp in tendon.get("spatial", []) + tendon.get("spatials", []):
            attrs = {k: v for k, v in sp.items() if k not in ("segments",)}
            sp_el = ET.SubElement(tend_el, "spatial", norm_attrs(attrs))
            for seg in sp.get("segments", []):
                if "site" in seg:
                    ref = self.registry.resolve("site", seg["site"])
                    ET.SubElement(sp_el, "site", {"site": ref})
                elif "geom" in seg:
                    ref = self.registry.resolve("geom", seg["geom"])
                    ET.SubElement(sp_el, "geom", {"geom": ref})
                elif "body" in seg:
                    # Rarely used in tendons, but allow passthrough
                    ET.SubElement(sp_el, "body", {"body": to_str_vec(seg["body"])})
                else:
                    # Allow raw segment tag + attrs: e.g., {"site": ..., "side":"left"}
                    # Already handled above; ignore else.
                    pass

    def _add_actuators(self, root: ET.Element, act: Optional[Dict[str, Any]]):
        if not act:
            return
        act_el = ET.SubElement(root, "actuator")
        # Common actuator types: position, velocity, motor, general
        for tag, items in act.items():
            for a in items:
                a = copy.deepcopy(a)
                # Resolve joint references that use body/local form
                if "joint" in a and "/" in str(a["joint"]):
                    a["joint"] = self.registry.resolve("joint", a["joint"])
                if "name" in a:
                    self.registry.add_child_alias("actuator", a["name"], a["name"])
                ET.SubElement(act_el, tag, norm_attrs(a))


# ----------------------------- Helpers -----------------------------

def merge_body_specs(base: Dict[str, Any], tmpl: Dict[str, Any]) -> Dict[str, Any]:
    # Merge template body spec into base instance; base has priority for attrs and lists extend
    out = copy.deepcopy(tmpl)
    out.update({k: v for k, v in base.items() if k not in ("geoms", "sites", "joints", "bodies")})
    for key in ("geoms", "sites", "joints", "bodies"):
        base_list = base.get(key, [])
        tmpl_list = out.get(key, [])
        if base_list and tmpl_list:
            out[key] = tmpl_list + base_list
        elif base_list:
            out[key] = base_list
        else:
            out[key] = tmpl_list
    return out

def find_child_by_prefixed_local_name(body_el: ET.Element, tag: str, local: str) -> Optional[ET.Element]:
    # Find first child where name endswith "__{local}" or equals local
    for ch in body_el.findall(tag):
        nm = ch.get("name", "")
        if nm.endswith(f"__{local}") or nm == local:
            return ch
    return None


# ----------------------------- IO -----------------------------

def load_spec(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        txt = f.read()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("pyyaml not installed. Install with: pip install pyyaml")
        return yaml.safe_load(txt)
    return json.loads(txt)

def save_xml(tree: ET.ElementTree, path: str) -> None:
    tree.write(path, encoding="utf-8", xml_declaration=True)


# ----------------------------- Demo Spec -----------------------------

def demo_spec() -> Dict[str, Any]:
    # Recreates the provided example with safe naming and references.
    return {
        "mujoco": {},
        "defaults": {
            "geom": {"friction": "1.0 0.05 0.01"},
            "joint": {"damping": "2"},
        },
        "asset": {
            "texture": [{
                "name": "grid", "type": "2d", "builtin": "checker",
                "rgb1": ".1 .2 .3", "rgb2": ".2 .3 .4",
                "width": "300", "height": "300", "mark": "none"
            }],
            "material": [{
                "name": "grid", "texture": "grid", "texrepeat": "1 1",
                "texuniform": "true", "reflectance": ".2"
            }]
        },
        "templates": {
            "color_box": {
                "name": "{name}",
                "pos": "{pos}",
                "freejoint": True,
                "geoms": [
                    {"name": "box", "type": "box", "size": ".2 .2 .2", "rgba": "{rgba}", "friction": "0.9 0.05 0.01"}
                ],
                "sites": [
                    {"name": "bottom", "pos": "{s_bottom}", "size": ".02"},
                    {"name": "top",    "pos": "{s_top}",    "size": ".02"},
                    {"name": "far",    "pos": "{s_far}",    "size": ".02"},
                    {"name": "near",   "pos": "{s_near}",   "size": ".02"},
                    {"name": "mid",    "pos": "{s_mid}",    "size": ".02"},
                    {"name": "top_middle",    "pos": "{s_top_middle}",    "size": ".02"},
                    {"name": "bottom_middle", "pos": "{s_bottom_middle}", "size": ".02"}
                ]
            }
        },
        "worldbody": {
            "lights": [{"pos": "0 0 3"}],
            "geoms": [{
                "name": "floor", "type": "plane", "pos": "0 0 -.3", "size": "2 2 .1",
                "material": "grid", "friction": "1.2 0.05 0.02"
            }],
            "bodies": [
                {
                    "name": "crane_base", "pos": "0 0 0.8",
                    "joints": [
                        {"name": "up_down", "type": "slide", "axis": "0 0 1", "range": "-0.5 1", "damping": "3"}
                    ],
                    "geoms": [
                        {"name": "crane", "type": "capsule", "fromto": "-0.5 0 0 0.5 0 0", "size": ".05", "rgba": "1 1 0 1"}
                    ],
                    "sites": [{"name": "crane_site", "pos": "0 0 -0.05", "size": ".02"}]
                },
                {
                    "use": "color_box",
                    "name": "red_box",
                    "vars": {
                        "name": "red_box", "pos": "-0.5 0 0.3", "rgba": "1 0 0 1",
                        "s_bottom": "0.2 0 -0.2", "s_top": "0.2 0 0.2",
                        "s_far": "0 0.5 0", "s_near": "0 -0.5 0", "s_mid": "0.2 0 0",
                        "s_top_middle": "0 0 0.5", "s_bottom_middle": "0 0 -0.5"
                    }
                },
                {
                    "name": "blue_box", "pos": "0.05 0 0.3",
                    "freejoint": {"name": "blue_joint"},
                    "geoms": [{"name": "blue_box", "type": "sphere", "size": ".2", "rgba": "0 0 1 1", "friction": "0.9 0.05 0.01"}]
                },
                {
                    "name": "dummy", "pos": "0 0 0",
                    "freejoint": {"name": "dummy_joint"},
                    "geoms": [{"name": "dummy", "type": "sphere", "size": ".2", "rgba": "1 0 1 0.3", "contype": "0", "conaffinity": "0"}],
                    "sites": [
                        {"name": "bottom", "pos": "0 0 -0.2", "size": ".02"},
                        {"name": "far",    "pos": "0 0.2 0",  "size": ".02"},
                        {"name": "near",   "pos": "0 -0.2 0", "size": ".02"},
                        {"name": "green",  "pos": "0.2 0 0",  "size": ".02"},
                        {"name": "red",    "pos": "-0.2 0 0", "size": ".02"}
                    ]
                },
                {
                    "use": "color_box",
                    "name": "green_box",
                    "vars": {
                        "name": "green_box", "pos": "0.5 0.1 0.3", "rgba": "0 1 0 1",
                        "s_bottom": "-0.2 0 -0.2", "s_top": "-0.2 0 0.2",
                        "s_far": "0 0.5 0", "s_near": "0 -0.5 0", "s_mid": "-0.2 0 0",
                        "s_top_middle": "0 0 0.5", "s_bottom_middle": "0 0 -0.5"
                    }
                }
            ]
        },
        "tendon": {
            "spatial": [
                {
                    "name": "spring1", "rgba": "0 0 1 1", "stiffness": "100", "damping": "100", "springlength": "0.7",
                    "segments": [
                        {"site": "red_box/bottom_middle"},
                        {"site": "green_box/bottom_middle"}
                    ]
                },
                {
                    "name": "spring2", "rgba": "0 0 1 1", "stiffness": "100", "damping": "100", "springlength": "0.72",
                    "segments": [
                        {"site": "red_box/far"},
                        {"site": "green_box/far"}
                    ]
                },
                {
                    "name": "spring3", "rgba": "0 0 1 1", "stiffness": "100", "damping": "100", "springlength": "0.72",
                    "segments": [
                        {"site": "red_box/near"},
                        {"site": "green_box/near"}
                    ]
                },
                {
                    "name": "spring6", "rgba": "0 0 1 1", "stiffness": "100", "damping": "100", "springlength": "0.72",
                    "segments": [
                        {"site": "red_box/top_middle"},
                        {"site": "green_box/top_middle"}
                    ]
                },
                {
                    "name": "spring4", "rgba": "0 0 1 1", "stiffness": "0", "damping": "0", "springlength": "0.00",
                    "segments": [
                        {"site": "red_box/mid"},
                        {"geom": "dummy/dummy"},
                        {"site": "dummy/red"}
                    ]
                },
                {
                    "name": "spring5", "rgba": "0 0 1 1", "stiffness": "0", "damping": "0", "springlength": "0.00",
                    "segments": [
                        {"site": "green_box/mid"},
                        {"geom": "dummy/dummy"},
                        {"site": "dummy/green"}
                    ]
                },
                {
                    "name": "chain_red", "limited": "true", "range": "0 0.5", "rgba": "1 0 0 1",
                    "segments": [
                        {"site": "red_box/top"}, {"site": "crane_base/crane_site"}
                    ]
                },
                {
                    "name": "chain_green", "limited": "true", "range": "0 0.5", "rgba": "0 1 0 1",
                    "segments": [
                        {"site": "green_box/top"}, {"site": "crane_base/crane_site"}
                    ]
                }
            ]
        },
        "actuator": {
            "position": [
                {"name": "lift", "joint": "crane_base/up_down", "kp": "7000", "kv": "2000", "ctrlrange": "-0.5 2.0"}
            ]
        }
    }


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate MuJoCo XML from a structured spec with templates and safe references.")
    ap.add_argument("--spec", type=str, help="Path to spec file (.json or .yaml/.yml). If omitted, uses a built-in demo spec.")
    ap.add_argument("--out", type=str, default="testing/model.xml", help="Output XML path.")
    ap.add_argument("--no-prefix", action="store_true", help="Disable auto-prefixing child names with body name.")
    args = ap.parse_args()

    if args.spec:
        spec = load_spec(args.spec)
    else:
        spec = demo_spec()

    builder = MjModelBuilder(spec, auto_prefix_children=not args.no_prefix)
    tree = builder.build()
    save_xml(tree, args.out)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()