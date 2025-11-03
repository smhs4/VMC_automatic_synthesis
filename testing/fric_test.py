import argparse
import mujoco
from mujoco import viewer
import numpy as np
import time

xml = """
<mujoco>
  <!-- Make integration/contact settings explicit -->
  

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300" mark="none"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3"/>
    <!-- High-friction floor -->
    <geom name="floor" type="plane" pos="0 0 0" size="2 2 .1" material="grid" />

    <!-- RED box: named freejoint so we can set initial velocities -->
    <body name="red_box" pos="-0.3 0 0.6">
      <freejoint name="red_free"/>
      <geom type="box" size=".2 .2 .2" rgba="1 0 0 1" />
    </body>

    
  </worldbody>

  
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data  = mujoco.MjData(model)


# For named freejoints, find DOF addresses (each freejoint has 6 dofs: 3 ang + 3 lin)
jid_red  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_free")
dof_red  = model.jnt_dofadr[jid_red]  # index into qvel for the first of 6 vel components

"""
Controls
- Set `init_speed_xy` to control initial horizontal velocity of the red box.
- Toggle `controller_enabled` to engage/disengage the crane position controller.
  When disabled, the crane will coast and the boxes will settle under friction.
"""

# Initial velocity for the red box in the horizontal plane (x, y)
init_speed_xy = np.array([0, 0, 0])  # m/s (x, y, z)


# Give the RED box an initial horizontal velocity so friction is obvious:
# For a free joint: qvel[dof:dof+3] = angular vel, qvel[dof+3:dof+6] = linear vel (x,y,z) in world frame
data.qvel[dof_red + 0] = float(init_speed_xy[0])   # +x m/s
data.qvel[dof_red + 1] = float(init_speed_xy[1])   # +y m/s
data.qvel[dof_red + 2] = float(init_speed_xy[2])   # +z m/s


parser = argparse.ArgumentParser(description="Friction test with passive/non-passive viewer")
parser.add_argument("--passive", action="store_true", help="Run with passive viewer (no built-in sim loop)")
args = parser.parse_args()

if args.passive:
    with viewer.launch_passive(model, data) as v:
        # Wider, higher vantage to see more of the scene
        v.cam.distance = 4.5       # zoom out
        v.cam.elevation = -15.0    # slight downward tilt
        v.cam.azimuth = 160.0      # rotated view for depth
        v.cam.lookat[:] = [0.0, 0.0, 0.3]


        while v.is_running():

            # Slow down the simulation by adding a time delay
            time.sleep(0.01)  # 10 ms delay to slow down the simulation

            data.qpos[dof_red] = np.random.uniform(-np.pi, np.pi)
            # Step physics: this computes collisions, contacts and friction just like non-passive
            mujoco.mj_step(model, data)

            
            # Sync drawing
            v.sync()
else:

    viewer.launch(model, data)
