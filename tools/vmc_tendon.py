#!/usr/bin/env python3
"""
Torque-based VMC controller integrated with the tools simulation environment.

Key behavior:
- Discovers tendons from the MuJoCo model.
- Extracts each tendon's physical parameters (stiffness, damping, rest length).
- Sets physical tendon stiffness/damping to zero so tendons are visual-only.
- Computes virtual tendon forces each control step.
- Maps resulting generalized torque to actuator controls.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import mujoco
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running this file directly: `python tools/vmc_tendon.py ...`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mujoco_sim_template import MuJoCoSimulation, ModelBuilder, quat_from_euler


@dataclass
class VirtualTendon:
    tendon_id: int
    name: str
    stiffness: float
    damping: float
    rest_length: float


class TendonTorqueVMC:
    """
    Virtual tendon controller that applies actuator torques from tendon Jacobians.
    """

    def __init__(
        self,
        sim: MuJoCoSimulation,
        tendon_names: Optional[List[str]] = None,
        zero_physical_tendons: bool = True,
        zero_damping_too: bool = True,
        use_current_length_as_rest: bool = False,
        torque_clip: Optional[float] = 3.0,
    ):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.torque_clip = torque_clip
        self.virtual_tendons: List[VirtualTendon] = self._discover_virtual_tendons(
            tendon_names=tendon_names,
            use_current_length_as_rest=use_current_length_as_rest,
        )

        if zero_physical_tendons:
            self._disable_physical_tendon_forces(zero_damping_too=zero_damping_too)

        self._actuator_to_dof = self._build_actuator_to_dof_map()
        self._history_t: List[float] = []
        self._history_tau: List[np.ndarray] = []
        self._history_ctrl: List[np.ndarray] = []
        self._history_tendon_force: List[np.ndarray] = []

    def _discover_virtual_tendons(
        self,
        tendon_names: Optional[List[str]],
        use_current_length_as_rest: bool,
    ) -> List[VirtualTendon]:
        mujoco.mj_forward(self.model, self.data)

        if tendon_names is None:
            tendon_ids = list(range(self.model.ntendon))
        else:
            tendon_ids = []
            for name in tendon_names:
                t_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TENDON, name)
                if t_id == -1:
                    raise ValueError(f"Tendon '{name}' not found in model.")
                tendon_ids.append(t_id)

        vtendons: List[VirtualTendon] = []
        for t_id in tendon_ids:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_TENDON, t_id) or f"tendon_{t_id}"
            k = float(np.atleast_1d(self.model.tendon_stiffness[t_id])[0])
            b = float(np.atleast_1d(self.model.tendon_damping[t_id])[0])
            l0_model = float(np.atleast_1d(self.model.tendon_lengthspring[t_id])[-1])
            l0 = float(self.data.ten_length[t_id]) if use_current_length_as_rest else l0_model
            vtendons.append(
                VirtualTendon(
                    tendon_id=t_id,
                    name=name,
                    stiffness=k,
                    damping=b,
                    rest_length=l0,
                )
            )

        return vtendons

    def _disable_physical_tendon_forces(self, zero_damping_too: bool) -> None:
        for vt in self.virtual_tendons:
            self.model.tendon_stiffness[vt.tendon_id] = 0.0
            if zero_damping_too:
                self.model.tendon_damping[vt.tendon_id] = 0.0

    def _build_actuator_to_dof_map(self) -> List[Dict[str, float]]:
        mapping = []
        for i in range(self.model.nu):
            jnt_id = int(self.model.actuator_trnid[i, 0])
            if jnt_id < 0 or jnt_id >= self.model.njnt:
                mapping.append({"dof_adr": -1, "gear": 1.0})
                continue
            dof_adr = int(self.model.jnt_dofadr[jnt_id])
            gear = float(self.model.actuator_gear[i, 0])
            if abs(gear) < 1e-9:
                gear = 1.0
            mapping.append({"dof_adr": dof_adr, "gear": gear})
        return mapping

    @staticmethod
    def _virtual_tendon_force(length: float, velocity: float, stiffness: float, damping: float, rest_length: float) -> float:
        return -stiffness * (length - rest_length) - damping * velocity

    def _current_physical_tendon_force(self, tendon_id: int) -> float:
        k = float(np.atleast_1d(self.model.tendon_stiffness[tendon_id])[0])
        b = float(np.atleast_1d(self.model.tendon_damping[tendon_id])[0])
        l0 = float(np.atleast_1d(self.model.tendon_lengthspring[tendon_id])[-1])
        l = float(self.data.ten_length[tendon_id])
        v = float(self.data.ten_velocity[tendon_id])
        return -k * (l - l0) - b * v

    def compute_generalized_torque(self) -> np.ndarray:
        # ten_J shape is [ntendon, nv] for tendon Jacobian wrt generalized velocities.
        ten_j = self.data.ten_J
        tau = np.zeros(self.model.nv, dtype=np.float64)

        for vt in self.virtual_tendons:
            t_id = vt.tendon_id
            l = float(self.data.ten_length[t_id])
            v = float(self.data.ten_velocity[t_id])
            f = self._virtual_tendon_force(
                length=l,
                velocity=v,
                stiffness=vt.stiffness,
                damping=vt.damping,
                rest_length=vt.rest_length,
            )
            tau += ten_j[t_id] * f

        if self.torque_clip is not None:
            tau = np.clip(tau, -self.torque_clip, self.torque_clip)
        return tau

    def apply_tau_to_actuators(self, tau: np.ndarray) -> np.ndarray:
        # Start from zero every step so this controller fully specifies actuator commands.
        self.data.ctrl[:] = 0.0

        for i in range(self.model.nu):
            dof_adr = self._actuator_to_dof[i]["dof_adr"]
            if dof_adr < 0 or dof_adr >= self.model.nv:
                continue
            gear = self._actuator_to_dof[i]["gear"]
            ctrl = float(tau[dof_adr] / gear)

            if int(self.model.actuator_ctrllimited[i]) == 1:
                ctrl_min = float(self.model.actuator_ctrlrange[i, 0])
                ctrl_max = float(self.model.actuator_ctrlrange[i, 1])
                ctrl = float(np.clip(ctrl, ctrl_min, ctrl_max))

            self.data.ctrl[i] = ctrl

        return self.data.ctrl.copy()

    def step_control(self, _sim: MuJoCoSimulation) -> None:
        tau = self.compute_generalized_torque()
        ctrl = self.apply_tau_to_actuators(tau)
        self._history_t.append(float(self.data.time))
        self._history_tau.append(tau.copy())
        self._history_ctrl.append(ctrl.copy())
        phys_forces = np.array(
            [self._current_physical_tendon_force(vt.tendon_id) for vt in self.virtual_tendons],
            dtype=np.float64,
        )
        self._history_tendon_force.append(phys_forces)

    def as_callback(self):
        def _callback(sim: MuJoCoSimulation):
            self.step_control(sim)
        return _callback

    def summary(self) -> str:
        lines = [
            f"Virtual tendons: {len(self.virtual_tendons)}",
            f"Actuators: {self.model.nu}",
        ]
        for vt in self.virtual_tendons[:10]:
            lines.append(
                f"  {vt.name}: k={vt.stiffness:.3f}, b={vt.damping:.3f}, l0={vt.rest_length:.5f}"
            )
        if len(self.virtual_tendons) > 10:
            lines.append("  ...")
        return "\n".join(lines)

    def save_torque_plot(self, save_path: str = "vmc_torque_plot.png", max_lines: int = 8) -> None:
        if not self._history_t:
            print("No control history collected yet; skipping torque plot.")
            return

        t = np.asarray(self._history_t)
        tau_hist = np.asarray(self._history_tau)  # [T, nv]
        ctrl_hist = np.asarray(self._history_ctrl)  # [T, nu]
        ten_force_hist = np.asarray(self._history_tendon_force)  # [T, n_virtual_tendons]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # 1) Generalized torque produced by the VMC.
        axes[0].plot(t, np.linalg.norm(tau_hist, axis=1), color="tab:blue", linewidth=2)
        axes[0].set_ylabel("||tau||")
        axes[0].set_title("VMC Generalized Torque Norm")
        axes[0].grid(True, alpha=0.3)

        # 2) Actuator controls (actual commanded torques/efforts).
        n_lines = min(ctrl_hist.shape[1], max_lines)
        for i in range(n_lines):
            a_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}"
            axes[1].plot(t, ctrl_hist[:, i], label=a_name, linewidth=1.2)
        axes[1].set_ylabel("actuator ctrl")
        axes[1].set_title("Controller Output (Actuator Commands)")
        axes[1].grid(True, alpha=0.3)
        if n_lines > 0:
            axes[1].legend(loc="upper right", fontsize=8)

        # 3) Physical tendon force sensor values (should be near-zero after disabling).
        if ten_force_hist.size > 0:
            for idx, vt in enumerate(self.virtual_tendons[:max_lines]):
                axes[2].plot(t, ten_force_hist[:, idx], label=vt.name, linewidth=1.2)
        axes[2].set_xlabel("time (s)")
        axes[2].set_ylabel("ten_force")
        axes[2].set_title("Physical Tendon Forces (Validation: should be ~0)")
        axes[2].grid(True, alpha=0.3)
        if self.virtual_tendons:
            axes[2].legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved torque plot to: {save_path}")


def set_default_joint_pose(sim: MuJoCoSimulation) -> None:
    initial_joint_positions = np.array([
        0.05062136600022616, -0.3436116964863836, -1.3959225169759335,
        0.10584467436410924, -1.5539225381281545, 0.09817477042468103, 2.741223667951641,
        0.02761165418194154, -1.7916895602504288, -1.6122138080678088, 0.2, 0.0,
        -0.0798, 1.53, 0.145, -2.26,
        0.0372, 1.07, -2.0, -0.2, 0.5
    ], dtype=np.float32)
    joint_names = [
        "waist_yaw_joint", "neck_yaw_joint", "neck_pitch_joint",
        "r_arm_joint1", "r_arm_joint2", "r_arm_joint3", "r_arm_joint4",
        "r_arm_joint5", "r_arm_joint6", "r_arm_joint7", "r_hand_joint", "r_hand_mimic_joint",
        "l_arm_joint1", "l_arm_joint2", "l_arm_joint3", "l_arm_joint4",
        "l_arm_joint5", "l_arm_joint6", "l_arm_joint7", "l_hand_joint", "l_hand_mimic_joint"
    ]
    for name, pos in zip(joint_names, initial_joint_positions):
        try:
            sim.reg_joint(name, name).qpos = pos
        except Exception:
            pass
    sim.forward()


def build_demo_sim_with_builder(xml_path: str, remove_end_links: bool = False) -> MuJoCoSimulation:
    """
    Build a test model and add tendons using the same ModelBuilder workflow as GA scripts.
    """
    graph = nx.Graph()
    builder = ModelBuilder(graph)
    builder.from_xml_path(xml_path)

    radius = 0.01
    half_length = 0.09

    if remove_end_links:
        builder.remove_body("r_link7")
        builder.remove_body("l_link7")

    builder.add_body(
        "left_chopstick",
        pos=[0, half_length, 0],
        quat=quat_from_euler(np.pi / 2, 0, 0),
        geom_type="box",
        geom_size=[radius, radius, half_length],
        parent="l_link6",
        free_joint=False,
        geom_rgba=[1, 0, 0, 1],
        mass=0.1,
        friction=[2.0, 0.01, 0.001],
    )
    builder.add_body(
        "right_chopstick",
        pos=[0, -half_length, 0],
        quat=quat_from_euler(np.pi / 2, 0, 0),
        geom_type="box",
        geom_size=[radius, radius, half_length],
        parent="r_link6",
        free_joint=False,
        geom_rgba=[0, 0, 1, 1],
        mass=0.1,
        friction=[2.0, 0.01, 0.001],
    )

    # Sites and tendons in the same style as training/ga_tendon_optimizer_v2.py
    builder.add_site("left_hand_site", body_name="l_link6", pos=[0, 0, 0], size=0.01)
    builder.add_site("left_shoulder_site", body_name="body_link", pos=[0.2, 0.05, 0.4], size=0.01)
    builder.add_site("right_hand_site", body_name="r_link6", pos=[0, 0, 0], size=0.01)
    builder.add_site("right_shoulder_site", body_name="body_link", pos=[0.2, -0.05, 0.4], size=0.01)
    builder.add_site("left_chopstick_mid", body_name="left_chopstick", pos=[0, 0, 0], size=0.004)
    builder.add_site("right_chopstick_mid", body_name="right_chopstick", pos=[0, 0, 0], size=0.004)

    builder.add_tendon(
        "left_lifter",
        tendon_type="spatial",
        sites=["left_hand_site", "left_shoulder_site"],
        stiffness=18.0,
        damping=2.0,
        springlength=[0.0, 0.12],
        rgba=[0, 1, 1, 0.8],
    )
    builder.add_tendon(
        "right_lifter",
        tendon_type="spatial",
        sites=["right_hand_site", "right_shoulder_site"],
        stiffness=18.0,
        damping=2.0,
        springlength=[0.0, 0.12],
        rgba=[0, 1, 1, 0.8],
    )
    builder.add_tendon(
        "chopstick_coupler",
        tendon_type="spatial",
        sites=["left_chopstick_mid", "right_chopstick_mid"],
        stiffness=10.0,
        damping=1.0,
        springlength=[0.0, 0.01],
        rgba=[1, 1, 0, 0.8],
    )

    sim = MuJoCoSimulation.from_builder(builder)
    set_default_joint_pose(sim)
    return sim


def parse_args():
    parser = argparse.ArgumentParser(description="Run torque-based tendon VMC in MuJoCoSimulation")
    parser.add_argument("--xml", required=True, help="Path to MuJoCo XML")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration (s)")
    parser.add_argument(
        "--tendons",
        nargs="*",
        default=None,
        help="Optional explicit tendon names to control (default: all tendons)",
    )
    parser.add_argument(
        "--keep-physical-damping",
        action="store_true",
        help="Do not zero physical tendon damping (stiffness is still zeroed).",
    )
    parser.add_argument(
        "--use-current-rest",
        action="store_true",
        help="Use current tendon length as virtual rest length.",
    )
    parser.add_argument(
        "--torque-clip",
        type=float,
        default=3.0,
        help="Absolute clip for generalized torque before actuator mapping.",
    )
    parser.add_argument(
        "--headless-steps",
        type=int,
        default=0,
        help="Run fixed number of steps without viewer (for quick testing).",
    )
    parser.add_argument(
        "--remove-end-links",
        action="store_true",
        help="Remove r_link7 and l_link7 before adding chopsticks.",
    )
    parser.add_argument(
        "--raw-xml",
        action="store_true",
        help="Skip builder tendon setup and load xml directly.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="vmc_torque_plot.png",
        help="Path to save controller torque plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.raw_xml:
        sim = MuJoCoSimulation(xml_path=args.xml)
        set_default_joint_pose(sim)
    else:
        sim = build_demo_sim_with_builder(
            xml_path=args.xml,
            remove_end_links=args.remove_end_links,
        )

    selected_tendons = args.tendons
    if selected_tendons is None and not args.raw_xml:
        selected_tendons = ["left_lifter", "right_lifter", "chopstick_coupler"]

    vmc = TendonTorqueVMC(
        sim=sim,
        tendon_names=selected_tendons,
        zero_physical_tendons=True,
        zero_damping_too=not args.keep_physical_damping,
        use_current_length_as_rest=args.use_current_rest,
        torque_clip=args.torque_clip,
    )

    print(vmc.summary())
    sim.set_control_callback(vmc.as_callback(), gravity_comp=False)
    if args.headless_steps > 0:
        for _ in range(args.headless_steps):
            sim.step()
        tau = vmc.compute_generalized_torque()
        print(f"Headless test complete. Tau norm={np.linalg.norm(tau):.4f}")
    else:
        sim.run(passive=True, duration=args.duration, control_preset=True)
    vmc.save_torque_plot(args.plot_path)
    sim.close()


if __name__ == "__main__":
    main()
