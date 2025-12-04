import sys
import os

import mujoco
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mujoco_sim_template import MuJoCoSimulation, ModelBuilder
import numpy as np
import networkx as nx

model1 = """
<mujoco model="flex_sphere_slope_stable">
  <!-- More conservative integrator + tighter solver -->

  <default>
    <!-- Softer contact, moderate friction, 3D contact space -->
    <geom friction="0.8 0.02 0.001"
          condim="3"
          solref="0.04 1"
          solimp="0.95 0.95 0.002"
          margin="0.003"
          gap="0.001"/>
  </default>

  <worldbody>
    <light pos="0 0 3"/>

    <!-- Slope: 20° about Y (downslope toward +X) -->
    <geom name="slope" type="plane" size="5 5 0.1"
          quat="0.985 0 0.174 0"
          rgba="0.85 0.9 0.95 1"/>

    <!-- Stopper WALL as a vertical plane with normal -X.
         Plane normal = local +Z; rotate -90° about Y to point -X:
         quat = [cos(45°), 0, -sin(45°), 0] ≈ [0.7071, 0, -0.7071, 0] -->
    <geom name="stopper_wall" type="plane" size="5 5 0.1"
          pos="0.8 0 0.3"
          quat="0.7071068 0 -0.7071068 0"
          rgba="0.6 0.6 0.8 1"
          solref="0.05 1" solimp="0.9 0.9 0.005"/>

    <!-- Deformable sphere carried by a free body -->
    <body name="soft_body" pos="0 0 0.6">
      <freejoint/>
      <!-- Minimal rigid inertia so free body is valid -->
      <inertial mass="1" diaginertia="2 2 2" pos="0 0 0"/>

      <flexcomp
          name="softball"
          type="ellipsoid"
        
          >

        <!-- No self-collision for trilinear, and no internal self contacts -->
        <contact selfcollide="none" />

        <!-- More damping to avoid chatter -->
        <edge damping="3.5"/>
        <elasticity young="2e4" poisson="0.3" damping="0.08"/>
      </flexcomp>
    </body>
  </worldbody>
</mujoco>
"""

model2 = """
<mujoco model="flex_sphere_slope_stable">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <worldbody>
    <light pos="0 0 3"/>
    <geom name="slope" type="plane" size="5 5 0.1"
          
          rgba="0.85 0.9 0.95 1"/>
    <body name="crane_base" pos="0 0 0.5">

    <freejoint/>
      <geom name="crane_base_geom" type="box" size="0.3 0.3 0.1" rgba="0.7 0.7 0.7 1" euler="0 90 0"/>

      <flexcomp
          pos="0.2 0 0"
          
          name="crane_arm"
          type="box"
          dim="3"
          dof="trilinear"
          count="10 10 10"
          spacing="0.02 0.02 0.02"
          mass="1.0">
        <contact selfcollide="none"/>
        <edge damping="5.0"/>
        <elasticity young="9.2e2" poisson="0.4" damping="0.8"/>
        <pin id="0 1 2 3"/>
      </flexcomp>
    </body>
  </worldbody>

  
</mujoco>

"""

model3 = """
<mujoco model="soft_gripper_attached">
  <!-- Stable integrator & smaller dt for deformables -->
  <option gravity="0 0 -9.81" timestep="0.001" integrator="implicitfast" solver="Newton" iterations="120"/>

  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.90 0.95 1 1"/>

    <!-- Make the base fixed (no freejoint) to avoid the whole assembly drifting -->
    <body name="crane_base" pos="0 0 0.5">
        <freejoint/>
      <geom name="crane_base_geom" type="box" size="0.3 0.3 0.1" rgba="0.7 0.7 0.7 1"/>

      <!-- Soft gripper pad: a volumetric grid, pinned along the face touching the block -->
      <flexcomp
          name="gripper"
          type="grid"           
          dim="3"
          dof="trilinear"      
          pos="0.31 0 0"       
          count="8 6 4"        
          spacing="0.015 0.015 0.015"
          radius="0.004"       
          mass="0.25">

        <!-- Robustness: prevent element inversion / self-intersection inside the pad -->
        <contact selfcollide="none"/>

        <!-- Material model (SVK): soft but not floppy; moderate Rayleigh damping -->
        <elasticity young="2.0e5" poisson="0.35" damping="0.03"/>

        <!-- Shape-preserving edge-length constraints (no need for huge Young's) -->
        <edge equality="true" solref="0.02 1"/>

        <!-- Pin the entire X=0 face of the grid to the base, using grid indices -->
        <!-- gridrange: [xmin ymin zmin  xmax ymax zmax] -->
        <pin gridrange="0 0 0  0 5 3"/>
      </flexcomp>
    </body>
  </worldbody>
</mujoco>
"""

sim = MuJoCoSimulation(xml_string=model3)
print(mujoco.mjMINVAL)
sim.run(
    passive=True,
    viewer_distance=2,
    viewer_lookat=[0,0,0.3],
    realtime_speed=1,
    control_preset=False,
)

