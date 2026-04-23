import argparse
import os
import sys

import networkx as nx
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.mujoco_sim_template import MuJoCoSimulation, ModelBuilder


XML_PATH = "Sciurus17_mujoco_sim_example/URDFs/sciurus17_description/urdf/sciurus17.xml"


INITIAL_JOINT_POSITIONS = np.array([
    0.05062136600022616, -0.3436116964863836, -1.3959225169759335,
    0.10584467436410924, -1.5539225381281545, 0.09817477042468103, 2.741223667951641,
    0.02761165418194154, -1.7916895602504288, -1.6122138080678088, 0.3,
    -0.0798, 1.53, 0.145, -2.26,
    0.0372, 1.07, -2, -0.3
], dtype=np.float32)


JOINT_NAMES = [
    "waist_yaw_joint", "neck_yaw_joint", "neck_pitch_joint",
    "r_arm_joint1", "r_arm_joint2", "r_arm_joint3", "r_arm_joint4",
    "r_arm_joint5", "r_arm_joint6", "r_arm_joint7", "r_hand_joint", "r_hand_mimic_joint",
    "l_arm_joint1", "l_arm_joint2", "l_arm_joint3", "l_arm_joint4",
    "l_arm_joint5", "l_arm_joint6", "l_arm_joint7", "l_hand_joint", "l_hand_mimic_joint",
]


def set_initial_pose(sim: MuJoCoSimulation):
    # Small mirrored opening angle for the gripper fingers.
    hand_open = 0.2

    base_positions = list(INITIAL_JOINT_POSITIONS)
    # Right hand ranges are positive, left hand ranges are negative in this model.
    base_positions.insert(10, hand_open)    # r_hand_joint
    base_positions.insert(19, -hand_open)   # l_hand_joint

    for joint_name, joint_pos in zip(JOINT_NAMES, base_positions):
        try:
            sim.reg_joint(joint_name, joint_name).qpos = joint_pos
        except Exception as exc:
            print(f"[WARN] Could not set joint '{joint_name}': {exc}")
    sim.forward()


def print_body_positions(sim: MuJoCoSimulation):
    body_names = [
        "r_link6", "l_link6",
        "r_link7", "l_link7",
        "r_handA_link", "r_handB_link",
        "l_handA_link", "l_handB_link",
    ]
    print("\nBody positions (world frame):")
    for name in body_names:
        body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id == -1:
            print(f"  {name:12s} : MISSING")
        else:
            print(f"  {name:12s} : {sim.data.xpos[body_id]}")


def build_sim(remove_end_links: bool) -> MuJoCoSimulation:
    graph = nx.Graph()
    builder = ModelBuilder(graph)
    builder.from_xml_path(XML_PATH)

    if remove_end_links:
        builder.remove_body("r_link7")
        builder.remove_body("l_link7")

    sim = MuJoCoSimulation.from_builder(builder)
    set_initial_pose(sim)
    print_body_positions(sim)
    return sim


def main():
    parser = argparse.ArgumentParser(description="Debug extra hand placement in Sciurus17 model.")
    parser.add_argument(
        "--remove-end-links",
        action="store_true",
        help="Remove r_link7 and l_link7 before compile (same as optimizer path).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Viewer run duration in seconds.",
    )
    args = parser.parse_args()

    print(f"Loading model from: {XML_PATH}")
    print(f"remove_end_links = {args.remove_end_links}")
    sim = build_sim(remove_end_links=args.remove_end_links)
    sim.run(passive=True, duration=args.duration)
    sim.close()


if __name__ == "__main__":
    main()
