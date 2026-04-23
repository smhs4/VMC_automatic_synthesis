#!/usr/bin/env python3
"""
Generate a MuJoCo XML with a virtual spring mechanism that pulls a robot into
an initial pose.

The mechanism is created by:
1. Loading a base XML and applying a desired joint-position array.
2. Computing world-space anchor points for each selected joint.
3. Adding multiple robot sites near each joint and matching fixed world sites.
4. Connecting each site pair with a stiff/damped tendon.

Multiple sites per joint provide both translation and rotation constraints.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional
import xml.etree.ElementTree as ET
import time

import mujoco
from mujoco import viewer
import networkx as nx
import numpy as np

try:
    from tools.mujoco_sim_template import ModelBuilder
except ImportError:
    from mujoco_sim_template import ModelBuilder


DEFAULT_JOINT_NAMES = [
    "waist_yaw_joint", "neck_yaw_joint", "neck_pitch_joint",
    "r_arm_joint1", "r_arm_joint2", "r_arm_joint3", "r_arm_joint4",
    "r_arm_joint5", "r_arm_joint6", "r_arm_joint7", "r_hand_mimic_joint",
    "l_arm_joint1", "l_arm_joint2", "l_arm_joint3", "l_arm_joint4",
    "l_arm_joint5", "l_arm_joint6", "l_arm_joint7", "l_hand_mimic_joint",
]


def parse_float_list(text: str) -> List[float]:
    values = np.fromstring(text.replace(",", " "), sep=" ")
    if values.size == 0:
        raise ValueError("No numeric values parsed from input.")
    return values.astype(float).tolist()


def load_qpos(args: argparse.Namespace) -> List[float]:
    if args.qpos is not None:
        return parse_float_list(args.qpos)

    if args.qpos_file is None:
        raise ValueError("Provide either --qpos or --qpos-file.")

    qpos_path = Path(args.qpos_file)
    raw = qpos_path.read_text().strip()
    if qpos_path.suffix.lower() == ".json":
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("JSON qpos file must contain a list of floats.")
        return [float(x) for x in data]

    return parse_float_list(raw)


def apply_joint_targets_to_data(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_values: List[float],
    joint_names: List[str],
) -> None:
    """Write scalar joint targets into data.qpos, clipped to joint limits."""
    if len(qpos_values) != len(joint_names):
        raise ValueError(f"qpos length ({len(qpos_values)}) must match number of joints ({len(joint_names)}).")

    for jname, qval in zip(joint_names, qpos_values):
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if j_id == -1:
            raise ValueError(f"Joint '{jname}' not found in model.")
        qadr = int(model.jnt_qposadr[j_id])
        jtype = int(model.jnt_type[j_id])
        nq = 7 if jtype == int(mujoco.mjtJoint.mjJNT_FREE) else 1
        if nq != 1:
            raise ValueError(f"Joint '{jname}' has nq={nq}; this script expects scalar joints.")
        qval_use = float(qval)
        if int(model.jnt_limited[j_id]) == 1:
            lo = float(model.jnt_range[j_id, 0])
            hi = float(model.jnt_range[j_id, 1])
            qval_clip = float(np.clip(qval_use, lo, hi))
            if abs(qval_clip - qval_use) > 1e-12:
                print(f"Warning: joint '{jname}' value {qval_use:.6f} clipped to [{lo:.6f}, {hi:.6f}] -> {qval_clip:.6f}")
            qval_use = qval_clip
        data.qpos[qadr] = qval_use


def fix_meshdir_in_xml(base_xml_path: Path, output_path: Path) -> None:
    """Rewrite compiler meshdir in output XML to an absolute path based on base XML."""
    base_tree = ET.parse(str(base_xml_path))
    base_compiler = base_tree.getroot().find("compiler")
    base_meshdir = "meshes"
    if base_compiler is not None and base_compiler.get("meshdir"):
        base_meshdir = base_compiler.get("meshdir")
    abs_meshdir = (base_xml_path.parent / base_meshdir).resolve()

    out_tree = ET.parse(str(output_path))
    out_root = out_tree.getroot()
    out_compiler = out_root.find("compiler")
    if out_compiler is None:
        out_compiler = ET.SubElement(out_root, "compiler")
    out_compiler.set("meshdir", str(abs_meshdir))
    out_tree.write(str(output_path), encoding="utf-8", xml_declaration=False)


def local_offsets(num_sites: int, radius: float) -> List[np.ndarray]:
    base = [
        np.array([0.0, 0.0, 0.0], dtype=float),
        np.array([radius, 0.0, 0.0], dtype=float),
        np.array([0.0, radius, 0.0], dtype=float),
        np.array([0.0, 0.0, radius], dtype=float),
        np.array([-radius, 0.0, 0.0], dtype=float),
        np.array([0.0, -radius, 0.0], dtype=float),
        np.array([0.0, 0.0, -radius], dtype=float),
    ]
    if num_sites < 1:
        raise ValueError("--sites-per-joint must be >= 1")
    if num_sites > len(base):
        raise ValueError(f"--sites-per-joint must be <= {len(base)}")
    return base[:num_sites]


def generate_virtual_pose_xml(
    base_xml: str,
    output_xml: str,
    qpos_values: List[float],
    joint_names: List[str] = None,
    output_json: Optional[str] = None,
    stiffness: float = 1500.0,
    damping: float = 80.0,
    sites_per_joint: int = 3,
    site_offset_radius: float = 0.01,
    site_size: float = 0.003,
    anchor_geom_size: float = 0.002,
    pre_sim_steps: int = 0,
    preview_passive: bool = False,
    preview_duration: float = 5.0,
    preview_realtime_speed: float = 1.0,
) -> int:
    """
    Generate an XML with virtual tendons that pull the robot to a target pose.

    Returns:
        Number of generated virtual tendons.
    """
    base_xml_path = Path(base_xml).resolve()
    output_path = Path(output_xml)
    descriptor_path = Path(output_json) if output_json is not None else output_path.with_name(f"{output_path.stem}_tendons.json")
    joint_names = DEFAULT_JOINT_NAMES if joint_names is None else joint_names

    model = mujoco.MjModel.from_xml_path(str(base_xml_path))
    data = mujoco.MjData(model)
    apply_joint_targets_to_data(model, data, qpos_values, joint_names)

    mujoco.mj_forward(model, data)

    builder = ModelBuilder(nx.Graph())
    builder.from_xml_path(str(base_xml_path))
    offsets = local_offsets(sites_per_joint, site_offset_radius)

    tendon_count = 0
    descriptor_entries = []
    for jname in joint_names:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        body_id = int(model.jnt_bodyid[j_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name is None:
            continue

        joint_local = np.array(model.jnt_pos[j_id], dtype=float)
        body_pos = np.array(data.xpos[body_id], dtype=float)
        body_rot = np.array(data.xmat[body_id], dtype=float).reshape(3, 3)

        for k, off in enumerate(offsets):
            robot_local = joint_local + off
            anchor_world = body_pos + body_rot.dot(robot_local)

            robot_site = f"vm_{jname}_robot_site_{k}"
            anchor_body = f"vm_{jname}_anchor_body_{k}"
            anchor_site = f"vm_{jname}_anchor_site_{k}"
            tendon_name = f"vm_{jname}_tendon_{k}"

            builder.add_site(
                name=robot_site,
                body_name=body_name,
                pos=robot_local.tolist(),
                size=site_size,
                rgba=[1.0, 0.2, 0.2, 0.9],
            )
            builder.add_body(
                name=anchor_body,
                pos=anchor_world.tolist(),
                geom_type="sphere",
                geom_size=[anchor_geom_size],
                free_joint=False,
                geom_rgba=[0.2, 1.0, 0.2, 0.7],
                mass=1e-6,
                intersection=False,
            )
            builder.add_site(
                name=anchor_site,
                body_name=anchor_body,
                pos=[0.0, 0.0, 0.0],
                size=site_size,
                rgba=[0.2, 1.0, 0.2, 0.9],
            )
            builder.add_tendon(
                name=tendon_name,
                sites=[robot_site, anchor_site],
                stiffness=stiffness,
                damping=damping,
                springlength=[0.0, 0.0],
                rgba=[1.0, 1.0, 0.0, 0.9],
                width=0.002,
            )
            tendon_count += 1
            descriptor_entries.append({
                "name": tendon_name,
                "constant": float(stiffness),
                "cosntant": float(stiffness),  # compatibility with existing typo in some pipelines
                "damping": float(damping),
                "rest_length": 0.0,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_out = builder.compile(keep_spec=True)

    # Verify virtual entities exist in compiled model before any export.
    model_vm_site_count = 0
    for s_id in range(model_out.nsite):
        s_name = mujoco.mj_id2name(model_out, mujoco.mjtObj.mjOBJ_SITE, s_id)
        if s_name is not None and s_name.startswith("vm_"):
            model_vm_site_count += 1
    model_vm_tendon_count = 0
    for t_id in range(model_out.ntendon):
        t_name = mujoco.mj_id2name(model_out, mujoco.mjtObj.mjOBJ_TENDON, t_id)
        if t_name is not None and t_name.startswith("vm_"):
            model_vm_tendon_count += 1
    if tendon_count > 0 and (model_vm_site_count == 0 or model_vm_tendon_count == 0):
        raise RuntimeError(
            "Compiled model is missing vm_* entities before export. "
            "Virtual mechanism was not created correctly."
        )

    # Optional simulation sanity-check before writing XML.
    vm_tendon_ids = []
    for t_id in range(model_out.ntendon):
        t_name = mujoco.mj_id2name(model_out, mujoco.mjtObj.mjOBJ_TENDON, t_id)
        if t_name is not None and t_name.startswith("vm_"):
            vm_tendon_ids.append(t_id)

    pre_sim_mean_len_before = None
    pre_sim_mean_len_after = None
    if pre_sim_steps > 0:
        data_out = mujoco.MjData(model_out)
        if vm_tendon_ids:
            pre_sim_mean_len_before = float(np.mean(data_out.ten_length[vm_tendon_ids]))
        for _ in range(pre_sim_steps):
            mujoco.mj_step(model_out, data_out)
        if vm_tendon_ids:
            pre_sim_mean_len_after = float(np.mean(data_out.ten_length[vm_tendon_ids]))

    # Optional passive preview of the generated model before writing XML.
    if preview_passive:
        data_preview = mujoco.MjData(model_out)
        with viewer.launch_passive(model_out, data_preview) as v:
            t0 = data_preview.time
            step_sleep = model_out.opt.timestep / max(preview_realtime_speed, 1e-6)
            while v.is_running() and (data_preview.time - t0) < preview_duration:
                mujoco.mj_step(model_out, data_preview)
                v.sync()
                time.sleep(step_sleep)

    # Export from updated schema first; fallback to model-based save if needed.
    export_errors = []
    exported = False
    if getattr(builder, "spec", None) is not None:
        try:
            builder.spec.to_file(str(output_path))
            exported = True
        except Exception as e:
            export_errors.append(f"spec.to_file failed: {e}")
    if not exported:
        try:
            mujoco.mj_saveLastXML(str(output_path), model_out)
            exported = True
        except Exception as e:
            export_errors.append(f"mj_saveLastXML failed: {e}")
    if not exported:
        raise RuntimeError("Failed to export XML. " + " | ".join(export_errors))

    # Make mesh paths robust regardless of where output XML is saved.
    fix_meshdir_in_xml(base_xml_path, output_path)

    # Sanity-check that generated virtual mechanism was written.
    verify_root = ET.parse(str(output_path)).getroot()
    vm_site_count = sum(
        1 for elem in verify_root.iter("site")
        if elem.get("name", "").startswith("vm_")
    )
    vm_tendon_count = sum(
        1 for elem in verify_root.iter("spatial")
        if elem.get("name", "").startswith("vm_")
    )
    if tendon_count > 0 and (vm_site_count == 0 or vm_tendon_count == 0):
        raise RuntimeError(
            "Virtual mechanism missing from output XML (vm_* sites/tendons not found). "
            f"Compiled model had vm_sites={model_vm_site_count}, vm_tendons={model_vm_tendon_count}; "
            f"XML has vm_sites={vm_site_count}, vm_tendons={vm_tendon_count}."
        )

    if pre_sim_mean_len_before is not None and pre_sim_mean_len_after is not None:
        print(
            "Pre-sim vm tendon mean length: "
            f"{pre_sim_mean_len_before:.6f} -> {pre_sim_mean_len_after:.6f}"
        )

    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    with open(descriptor_path, "w") as f:
        json.dump(descriptor_entries, f, indent=2)
    print(f"Saved VMC descriptor JSON: {descriptor_path}")

    return tendon_count


def generate_static_pose_xml(
    base_xml: str,
    output_xml: str,
    qpos_values: List[float],
    joint_names: Optional[List[str]] = None,
) -> None:
    """
    Generate a static pose XML by baking target joint values into model qpos0.

    This is useful for visual comparison against the virtual spring mechanism.
    """
    base_xml_path = Path(base_xml).resolve()
    output_path = Path(output_xml)
    joint_names = DEFAULT_JOINT_NAMES if joint_names is None else joint_names

    model = mujoco.MjModel.from_xml_path(str(base_xml_path))
    data = mujoco.MjData(model)
    apply_joint_targets_to_data(model, data, qpos_values, joint_names)
    mujoco.mj_forward(model, data)

    # Bake desired pose as model reset state.
    model.qpos0[:] = data.qpos[:]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(output_path), model)
    fix_meshdir_in_xml(base_xml_path, output_path)

    # Quick consistency check against reloaded model.
    reloaded = mujoco.MjModel.from_xml_path(str(output_path))
    qerr = float(np.max(np.abs(reloaded.qpos0 - model.qpos0)))
    print(f"Static pose qpos0 max error after reload: {qerr:.6e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create virtual pose mechanism XML.")
    parser.add_argument("--mode", choices=["virtual", "static"], default="virtual",
                        help="Generation mode: virtual (tendons+anchors) or static (baked qpos0).")
    parser.add_argument("--base-xml", required=True, help="Path to base MuJoCo XML.")
    parser.add_argument("--output-xml", required=True, help="Output XML path.")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Output JSON path for generated VMC descriptor (virtual mode).")
    parser.add_argument("--qpos", type=str, default=None, help="Joint array as comma/space-separated values.")
    parser.add_argument("--qpos-file", type=str, default=None, help="Text/JSON file with joint array.")
    parser.add_argument("--joint-names", type=str, default=None,
                        help="Comma-separated joint names matching qpos order. Default uses Sciurus list.")
    parser.add_argument("--stiffness", type=float, default=1500.0, help="Virtual tendon stiffness.")
    parser.add_argument("--damping", type=float, default=80.0, help="Virtual tendon damping.")
    parser.add_argument("--sites-per-joint", type=int, default=3, help="Number of site pairs per joint.")
    parser.add_argument("--site-offset-radius", type=float, default=0.01,
                        help="Offset radius around each joint for rotational locking.")
    parser.add_argument("--site-size", type=float, default=0.003, help="Visualization size for added sites.")
    parser.add_argument("--anchor-geom-size", type=float, default=0.002, help="Visualization size for anchor geoms.")
    parser.add_argument("--pre-sim-steps", type=int, default=0,
                        help="Run N simulation steps on generated model before saving XML.")
    parser.add_argument("--preview-passive", action="store_true",
                        help="Preview generated model in passive viewer before saving XML.")
    parser.add_argument("--preview-duration", type=float, default=5.0,
                        help="Passive preview duration in seconds.")
    parser.add_argument("--preview-speed", type=float, default=1.0,
                        help="Passive preview realtime speed multiplier.")
    args = parser.parse_args()

    qpos_values = load_qpos(args)
    joint_names = DEFAULT_JOINT_NAMES if args.joint_names is None else [x.strip() for x in args.joint_names.split(",")]
    if args.mode == "virtual":
        tendon_count = generate_virtual_pose_xml(
            base_xml=args.base_xml,
            output_xml=args.output_xml,
            qpos_values=qpos_values,
            joint_names=joint_names,
            output_json=args.output_json,
            stiffness=args.stiffness,
            damping=args.damping,
            sites_per_joint=args.sites_per_joint,
            site_offset_radius=args.site_offset_radius,
            site_size=args.site_size,
            anchor_geom_size=args.anchor_geom_size,
            pre_sim_steps=args.pre_sim_steps,
            preview_passive=args.preview_passive,
            preview_duration=args.preview_duration,
            preview_realtime_speed=args.preview_speed,
        )
        print(f"Saved XML: {args.output_xml}")
        print(f"Added virtual tendons: {tendon_count}")
        print(f"Sites per joint: {args.sites_per_joint}")
        print("Mechanism ready: stiff springs should pull joints toward the desired pose.")
    else:
        generate_static_pose_xml(
            base_xml=args.base_xml,
            output_xml=args.output_xml,
            qpos_values=qpos_values,
            joint_names=joint_names,
        )
        print(f"Saved static pose XML: {args.output_xml}")


if __name__ == "__main__":
    main()
