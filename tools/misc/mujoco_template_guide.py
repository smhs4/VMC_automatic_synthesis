#!/usr/bin/env python3
"""
GUIDE: Using the MuJoCo Simulation Template

This guide shows various ways to use the object-oriented MuJoCo template.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Note: Import examples are shown below, but not executed here
# to avoid dependency issues in the guide script

print("="*70)
print("MuJoCo Simulation Template - Usage Guide")
print("="*70)

print("""
The template provides an object-oriented wrapper around MuJoCo that makes
your simulation code much cleaner and more maintainable.

KEY CONCEPTS:
1. MuJoCoSimulation - Main simulation object
2. Joint, Body, Actuator, Site - Entity wrappers with property access
3. Control callbacks - Clean function-based control definition

""")

print("="*70)
print("EXAMPLE 1: Basic Setup")
print("="*70)

print("""
# Create simulation from XML file
sim = MuJoCoSimulation(xml_path="path/to/model.xml")

# Or from XML string
xml_string = '''<mujoco>...</mujoco>'''
sim = MuJoCoSimulation(xml_string=xml_string)

# Register entities you want to work with
crane = sim.add_joint("crane", "crane_base__up_down")
box = sim.add_body("box", "red_box")
motor = sim.add_actuator("motor", "lift")
grasp_site = sim.add_site("grasp", "grasp_point")
""")

print("="*70)
print("EXAMPLE 2: Accessing Properties")
print("="*70)

print("""
# JOINTS - Access position, velocity, forces
crane = sim.add_joint("crane", "crane_joint")

# Get/set position
current_pos = crane.qpos      # Returns numpy array
crane.qpos = np.array([0.5])  # Set position

# Get/set velocity
current_vel = crane.qvel
crane.qvel = np.array([0.1])

# Apply forces
crane.add_qfrc(np.array([10.0]))  # Add 10N force

# For free joints (7 DOF position, 6 DOF velocity)
free_joint = sim.add_joint("obj", "object_joint")
pos_quat = free_joint.qpos  # [x, y, z, qw, qx, qy, qz]
lin_ang_vel = free_joint.qvel  # [vx, vy, vz, wx, wy, wz]


# BODIES - Access world-frame properties
box = sim.add_body("box", "red_box")

# Position and orientation
position = box.pos           # World position [x, y, z]
orientation = box.quat       # Quaternion [w, x, y, z]
rot_matrix = box.xmat        # 3x3 rotation matrix

# Velocities
angular_vel = box.angular_vel    # [wx, wy, wz]
linear_vel = box.linear_vel      # [vx, vy, vz]
full_cvel = box.cvel             # [wx, wy, wz, vx, vy, vz]

# Apply forces/torques
box.add_force(np.array([10, 0, 0]))     # Force in world frame
box.add_torque(np.array([0, 0, 1]))    # Torque in world frame


# ACTUATORS - Control signals
motor = sim.add_actuator("motor", "lift")

# Get/set control
motor.ctrl = 0.5         # Set control signal
current = motor.ctrl     # Get current control
force = motor.force      # Get actuator force


# SITES - Attachment points
site = sim.add_site("grasp", "grasp_point")

# Get site position and orientation
site_pos = site.pos      # World position
site_mat = site.mat      # 3x3 orientation matrix
""")

print("="*70)
print("EXAMPLE 3: Control Callbacks")
print("="*70)

print("""
# Define a control function that takes the simulation object
def my_control(sim: MuJoCoSimulation):
    # Access registered entities through sim
    crane = sim.joints["crane"]
    box = sim.bodies["box"]
    motor = sim.actuators["motor"]
    
    # Compute control
    target = compute_target(box.pos, sim.time)
    motor.ctrl = target
    
    # Apply forces
    if box.pos[2] < 0.5:
        box.add_force(np.array([0, 0, 10]))

# Run with control callback
sim.run(control_callback=my_control, passive=True)
""")

print("="*70)
print("EXAMPLE 4: Complete Simulation Example")
print("="*70)

print("""
def main():
    # 1. Create simulation
    sim = MuJoCoSimulation(xml_path="model.xml")
    
    # 2. Register entities
    slider = sim.add_joint("slider", "slider_joint")
    lift = sim.add_actuator("lift", "lift_actuator")
    payload = sim.add_body("payload", "payload_body")
    
    # 3. Define parameters
    target_height = 1.0
    Kp = 100.0
    Kd = 10.0
    
    # 4. Define control
    def pd_control(sim):
        # PD controller for slider
        pos_error = target_height - slider.qpos[0]
        vel_error = 0 - slider.qvel[0]
        
        control_signal = Kp * pos_error + Kd * vel_error
        lift.ctrl = control_signal
        
        # Print diagnostics
        if int(sim.time * 100) % 100 == 0:  # Every second
            print(f"t={sim.time:.2f}, pos={slider.qpos[0]:.3f}, "
                  f"payload_height={payload.pos[2]:.3f}")
    
    # 5. Run
    sim.run(
        control_callback=pd_control,
        passive=True,
        duration=10.0,  # Run for 10 seconds
        realtime_speed=1.0
    )

if __name__ == "__main__":
    main()
""")

print("="*70)
print("EXAMPLE 5: Orientation Control")
print("="*70)

print("""
from mujoco_sim_template import quat_mul, quat_conj, quat_error

def orientation_control(sim):
    box_a = sim.bodies["box_a"]
    box_b = sim.bodies["box_b"]
    
    # Compute orientation error (B relative to A)
    ang_error = quat_error(box_a.quat, box_b.quat)
    
    # Angular velocity error
    w_error = box_b.angular_vel - box_a.angular_vel
    
    # PD control torque
    Kp, Kd = 20.0, 5.0
    torque = -Kp * ang_error - Kd * w_error
    
    # Apply torques
    box_a.add_torque(-torque)
    box_b.add_torque(torque)

sim.run(control_callback=orientation_control, passive=True)
""")

print("="*70)
print("EXAMPLE 6: Copying State Between Objects")
print("="*70)

print("""
# Useful for dummy objects that follow targets
def make_dummy_follow(sim):
    target = sim.joints["target"]
    dummy = sim.joints["dummy"]
    
    # Copy full state (position and orientation for free joint)
    dummy.qpos = target.qpos
    
    # Or copy just position
    dummy.qpos[:3] = target.qpos[:3]
    
    # Keep identity orientation
    dummy.qpos[3:7] = [1, 0, 0, 0]  # w, x, y, z
""")

print("="*70)
print("COMPARISON: Before and After")
print("="*70)

print("""
# BEFORE (manual index management):
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "crane")
qpos_adr = model.jnt_qposadr[joint_id]
current_pos = data.qpos[qpos_adr]
data.qpos[qpos_adr] = new_pos

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
box_pos = data.xpos[body_id].copy()
dof_adr = model.body_dofadr[body_id]
data.qfrc_applied[dof_adr+3:dof_adr+6] += torque

# AFTER (clean property access):
crane = sim.add_joint("crane", "crane")
current_pos = crane.qpos
crane.qpos = new_pos

box = sim.add_body("box", "box")
box_pos = box.pos
box.add_torque(torque)
""")

print("="*70)
print("KEY BENEFITS")
print("="*70)

print("""
✓ No manual index management
✓ Clear, readable property names
✓ Type safety and IDE autocompletion
✓ Less error-prone (no off-by-one errors)
✓ Easier to refactor and maintain
✓ Self-documenting code
✓ Reusable across projects
""")

print("\n" + "="*70)
print("Ready to use! Check out:")
print("  - mujoco_sim_template.py - Main template")
print("  - 3-cubes-oop.py - Example recreation of 3-cubes")
print("="*70)
