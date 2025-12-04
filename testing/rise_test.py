import sys
import os
# Add parent directory to path to import from tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mujoco_sim_template import (
    MuJoCoSimulation, ModelBuilder,
    quat_mul, quat_conj, quat_to_axis_angle
)
import numpy as np
import argparse
import networkx as nx

model = """
<mujoco model="rise_test">
<worldbody>
    <light cutoff="100" diffuse=".8 .8 .8" dir="-0.5 0 -1" pos="0 0 3"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba=".9 .9 .9 1"/>
    <body name="plate" pos="0 0 0.1">
    <joint name="plate_joint" axis="0 0 1" type="slide"/>
    <site name="top_plate" pos="0 0 0.02"/>
    <geom name="circle" type="cylinder" size="0.5 0.02" rgba="0 1 0 1"/>
    </body>
    <body name="box" pos="0 0 0.2">
        <freejoint name="box_joint"/>
        <geom name="box_geom" type="box" size="0.1 0.1 0.1" rgba="0 0 1 1"/>
    </body>
    <body name="shadow" pos="0 0 0.2">
        <freejoint name="shadow_joint"/>
        <site name="shadow_site" pos="0 0 0.5"/>
        <geom name="shadow_geom" type="box" size="0.1 0.1 0.1" rgba="0 0 0 0.3" contype="0" conaffinity="0"/>
    </body>
    
</worldbody>
<tendon>
    <spatial name="spring4" rgba="0 0 1 1" stiffness="1000" damping="0" springlength="0.00">
        <site site="top_plate"/>
        <site site="shadow_site"/>
    </spatial>
    </tendon>
</mujoco>
"""

sim = MuJoCoSimulation(xml_string=model)


box_joint = sim.reg_joint('box_joint', 'box_joint')
shadow_joint = sim.reg_joint('shadow_joint', 'shadow_joint')

def control_shadow(box_joint, shadow_joint):
    def callback(self: MuJoCoSimulation):
        shadow_joint.qpos = box_joint.qpos
        shadow_joint.qvel = box_joint.qvel
    return callback

sim.run(
    passive=True,
    realtime_speed=0.2,
    control_callback=control_shadow(box_joint, shadow_joint),
    record_video=True
)
    