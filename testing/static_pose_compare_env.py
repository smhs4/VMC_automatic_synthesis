#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.generate_virtual_pose_mechanism import generate_static_pose_xml, DEFAULT_JOINT_NAMES
from tools.mujoco_sim_template import MuJoCoSimulation


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_xml = os.path.join(
        root,
        "Sciurus17_mujoco_sim_example/URDFs/sciurus17_description/urdf/sciurus17.xml",
    )
    output_xml = os.path.join(root, "testing/sciurus17_static_target_pose.xml")

    qpos = [
        0.050621366, -0.343611696, -1.395922517,
        0.105844674, -1.553922538, 0.098174770, 2.741223668,
        0.027611654, -1.791689560, -1.612213808, 0.3,
        -0.0798, 1.53, 0.145, -2.26,
        0.0372, 1.07, -2.0, -0.3,
    ]

    generate_static_pose_xml(
        base_xml=base_xml,
        output_xml=output_xml,
        qpos_values=qpos,
    )
    print(f"Generated static pose XML: {output_xml}")

    sim = MuJoCoSimulation(xml_path=output_xml)

    # Verify loaded initial pose against requested target; force-set if mismatch.
    max_err = 0.0
    for name, target in zip(DEFAULT_JOINT_NAMES, qpos):
        try:
            joint = sim.reg_joint(name, name)
            err = abs(float(joint.qpos[0]) - float(target))
            max_err = max(max_err, err)
        except Exception:
            pass
    print(f"Loaded pose max abs joint error: {max_err:.6e}")

    if max_err > 1e-4:
        for name, target in zip(DEFAULT_JOINT_NAMES, qpos):
            try:
                joint = sim.reg_joint(name, name)
                joint.qpos = float(target)
            except Exception:
                pass
        sim.forward()
        print("Applied target qpos directly in sim for visual comparison.")

    sim.run(
        passive=True,
        duration=10.0,
        realtime_speed=1.0,
    )


if __name__ == "__main__":
    main()
