#!/usr/bin/env python3
import os
import sys
import xml.etree.ElementTree as ET

# Add project root to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.generate_virtual_pose_mechanism import generate_virtual_pose_xml
from tools.mujoco_sim_template import MuJoCoSimulation


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_xml = os.path.join(
        root,
        "Sciurus17_mujoco_sim_example/URDFs/sciurus17_description/urdf/sciurus17.xml",
    )
    output_xml = os.path.join(root, "testing/sciurus17_with_virtual_pose.xml")
    output_json = os.path.join(root, "testing/sciurus17_with_virtual_pose_tendons.json")

    qpos = [
        0.050621366, -0.343611696, -1.395922517,
        0.105844674, -1.553922538, 0.098174770, 2.741223668,
        0.027611654, -1.791689560, -1.612213808, 0.3,
        -0.0798, 1.53, 0.145, -2.26,
        0.0372, 1.07, -2.0, -0.3,
    ]

    tendon_count = generate_virtual_pose_xml(
        base_xml=base_xml,
        output_xml=output_xml,
        qpos_values=qpos,
        output_json=output_json,
        stiffness=1500.0,
        damping=80.0,
        sites_per_joint=3,
        site_offset_radius=0.01,
        site_size=0.008,
        anchor_geom_size=0.01,
        pre_sim_steps=500,
    )
    print(f"Generated {output_xml} with {tendon_count} virtual tendons")
    print(f"Generated descriptor JSON: {output_json}")

    root_xml = ET.parse(output_xml).getroot()
    vm_sites = sum(
        1 for elem in root_xml.iter("site")
        if elem.get("name", "").startswith("vm_")
    )
    vm_tendons = sum(
        1 for elem in root_xml.iter("spatial")
        if elem.get("name", "").startswith("vm_")
    )
    print(f"VM check: sites={vm_sites}, tendons={vm_tendons}")

    sim = MuJoCoSimulation(xml_path=output_xml)
    sim.run(
        passive=True,
        duration=10.0,
        realtime_speed=1.0,
    )


if __name__ == "__main__":
    main()
