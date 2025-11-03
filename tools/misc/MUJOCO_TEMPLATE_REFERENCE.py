"""
MUJOCO SIMULATION TEMPLATE - QUICK REFERENCE
=============================================

SETUP
-----
from mujoco_sim_template import MuJoCoSimulation

sim = MuJoCoSimulation(xml_path="model.xml")
# or
sim = MuJoCoSimulation(xml_string=xml_content)


REGISTER ENTITIES
-----------------
joint = sim.add_joint("alias", "joint_name")
body = sim.add_body("alias", "body_name")
actuator = sim.add_actuator("alias", "actuator_name")
site = sim.add_site("alias", "site_name")


JOINT PROPERTIES
----------------
joint.qpos          # Get/set position
joint.qvel          # Get/set velocity
joint.qfrc_applied  # Get/set applied forces
joint.add_qfrc(f)   # Add force

joint.id            # MuJoCo ID
joint.qpos_adr      # Position address
joint.qvel_adr      # Velocity address


BODY PROPERTIES
---------------
body.pos            # World position [x, y, z]
body.quat           # Orientation quaternion [w, x, y, z]
body.xmat           # 3x3 rotation matrix

body.cvel           # COM velocity [wx, wy, wz, vx, vy, vz]
body.angular_vel    # Angular velocity [wx, wy, wz]
body.linear_vel     # Linear velocity [vx, vy, vz]

body.add_force(f)   # Add force to linear DOFs
body.add_torque(t)  # Add torque to angular DOFs

body.id             # MuJoCo ID
body.dof_adr        # DOF address
body.dof_num        # Number of DOFs


ACTUATOR PROPERTIES
-------------------
actuator.ctrl       # Get/set control signal
actuator.force      # Get actuator force
actuator.id         # MuJoCo ID


SITE PROPERTIES
---------------
site.pos            # World position [x, y, z]
site.mat            # 3x3 orientation matrix
site.id             # MuJoCo ID


SIMULATION CONTROL
------------------
sim.time            # Current simulation time
sim.reset()         # Reset to initial state
sim.step()          # Advance one timestep
sim.forward()       # Forward kinematics only


CONTROL CALLBACK
----------------
def control(sim: MuJoCoSimulation):
    # Access entities
    joint = sim.joints["alias"]
    body = sim.bodies["alias"]
    
    # Compute control
    target = compute(body.pos, sim.time)
    joint.qpos = target

sim.set_control_callback(control)


RUN SIMULATION
--------------
sim.run(
    control_callback=my_control,  # Optional
    passive=True,                 # True=passive, False=active viewer
    duration=10.0,                # Seconds (None=infinite)
    realtime_speed=1.0,           # Speed multiplier
    viewer_distance=2.5,          # Camera distance
    viewer_lookat=[0, 0, 0.5]     # Camera target
)


UTILITY FUNCTIONS
-----------------
from mujoco_sim_template import (
    quat_mul,              # Quaternion multiplication
    quat_conj,             # Quaternion conjugate
    quat_to_axis_angle,    # Quat to axis-angle
    quat_error             # Orientation error
)


COMPLETE EXAMPLE
----------------
from mujoco_sim_template import MuJoCoSimulation
import numpy as np

# Setup
sim = MuJoCoSimulation(xml_path="model.xml")
joint = sim.add_joint("j", "my_joint")
actuator = sim.add_actuator("a", "my_actuator")

# Control
def pd_control(sim):
    Kp, Kd = 100, 10
    target = 1.0
    
    error = target - joint.qpos[0]
    derror = 0 - joint.qvel[0]
    
    actuator.ctrl = Kp*error + Kd*derror

# Run
sim.run(control_callback=pd_control, passive=True)


COMPARISON
----------
OLD WAY:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "crane")
    qpos_adr = model.jnt_qposadr[joint_id]
    pos = data.qpos[qpos_adr]
    data.qpos[qpos_adr] = new_pos

NEW WAY:
    crane = sim.add_joint("crane", "crane")
    pos = crane.qpos
    crane.qpos = new_pos


TIPS
----
• Register all entities at the start
• Use descriptive aliases for entities
• Access entities via properties (not arrays)
• Use control callbacks for clean code
• Free joints: 7 qpos (x,y,z,qw,qx,qy,qz), 6 qvel (vx,vy,vz,wx,wy,wz)
• Body forces/torques work for free joints (6+ DOFs)
"""

print(__doc__)
