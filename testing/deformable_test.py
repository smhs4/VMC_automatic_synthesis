import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mujoco_sim_template import MuJoCoSimulation, ModelBuilder
import numpy as np
import networkx as nx

model = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <body name="FL_1">
      <inertial pos="0 0 0" mass="1" diaginertia="1.66667e-05 1.66667e-05 1.66667e-05"/>
      <joint pos="0 0 0" axis="1 0 0" type="slide"/>
      <joint pos="0 0 0" axis="0 1 0" type="slide"/>
      <joint pos="0 0 0" axis="0 0 1" type="slide"/>
    </body>
    <body name="FL_2" pos="0.2 0 0">
      <inertial pos="0 0 0" mass="1" diaginertia="1.66667e-05 1.66667e-05 1.66667e-05"/>
      <joint pos="0 0 0" axis="1 0 0" type="slide"/>
      <joint pos="0 0 0" axis="0 1 0" type="slide"/>
      <joint pos="0 0 0" axis="0 0 1" type="slide"/>
    </body>
  </worldbody>
  <deformable>
    <flex name="FL" dim="1" body="world FL_1 FL_2" vertex="-0.2 0 0 0 0 0 0 0 0" element="0 1 1 2"/>
  </deformable>
  <equality>
    <flex flex="FL"/>
  </equality>
</mujoco>
"""

sim = MuJoCoSimulation(xml_string=model)
sim.run(
    passive=True,
    viewer_distance=2,
    viewer_lookat=[0,0,0.3],
    realtime_speed=1,
    control_preset=False,
)

