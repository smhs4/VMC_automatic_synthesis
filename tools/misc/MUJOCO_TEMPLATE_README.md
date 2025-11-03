# MuJoCo Simulation Template

An object-oriented wrapper for MuJoCo that makes simulation code cleaner, more maintainable, and easier to write.

## Why Use This Template?

**Before (Manual Index Management):**
```python
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "crane")
qpos_adr = model.jnt_qposadr[joint_id]
current_pos = data.qpos[qpos_adr]
data.qpos[qpos_adr] = new_pos

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
box_pos = data.xpos[body_id].copy()
dof_adr = model.body_dofadr[body_id]
data.qfrc_applied[dof_adr+3:dof_adr+6] += torque
```

**After (Clean OOP Interface):**
```python
crane = sim.add_joint("crane", "crane")
current_pos = crane.qpos
crane.qpos = new_pos

box = sim.add_body("box", "box")
box_pos = box.pos
box.add_torque(torque)
```

## Features

### Core OOP Interface
- ✅ **No manual index management** - No more tracking addresses and offsets
- ✅ **Property-based access** - Use `joint.qpos` instead of `data.qpos[addr]`
- ✅ **Type-safe** - Get IDE autocompletion and type hints
- ✅ **Self-documenting** - Clear, readable code
- ✅ **Reusable** - Works across different MuJoCo models
- ✅ **Error-resistant** - No off-by-one indexing errors

### Model Building (NEW!)
- ✅ **Dynamic site creation** - Add attachment points to existing bodies
- ✅ **Programmatic tendons** - Create spring/cable connections between sites
- ✅ **Runtime object addition** - Add new bodies and geoms on the fly
- ✅ **Model specification** - Build complete scenes programmatically

## Quick Start

### 1. Import and Create Simulation

```python
from mujoco_sim_template import MuJoCoSimulation

# From XML file
sim = MuJoCoSimulation(xml_path="model.xml")

# Or from XML string
sim = MuJoCoSimulation(xml_string=xml_content)
```

### 2. Register Entities

```python
# Register the entities you want to work with
crane = sim.add_joint("crane", "crane_base__up_down")
box = sim.add_body("box", "red_box")
motor = sim.add_actuator("motor", "lift")
grasp_site = sim.add_site("grasp", "grasp_point")
```

### 3. Define Control

```python
def control(sim):
    # Access properties easily
    if box.pos[2] < 0.5:  # If box is too low
        motor.ctrl = 1.0  # Lift up
    else:
        motor.ctrl = crane.qpos[0]  # Hold position
```

### 4. Run Simulation

```python
sim.run(
    control_callback=control,
    passive=True,
    duration=10.0
)
```

## Complete Example

```python
from mujoco_sim_template import MuJoCoSimulation
import numpy as np

# Create simulation
sim = MuJoCoSimulation(xml_path="model.xml")

# Register entities
slider = sim.add_joint("slider", "slider_joint")
lift = sim.add_actuator("lift", "lift_actuator")
payload = sim.add_body("payload", "payload_body")

# Control parameters
target_height = 1.0
Kp, Kd = 100.0, 10.0

# Define PD controller
def pd_control(sim):
    pos_error = target_height - slider.qpos[0]
    vel_error = 0 - slider.qvel[0]
    lift.ctrl = Kp * pos_error + Kd * vel_error

# Run
sim.run(control_callback=pd_control, passive=True)
```

## Entity Types

### Joints
Access joint positions, velocities, and forces:

```python
joint = sim.add_joint("alias", "joint_name")

# Position (1 DOF for hinge/slide, 7 for free joint)
joint.qpos           # Get position
joint.qpos = value   # Set position

# Velocity (1 DOF for hinge/slide, 6 for free joint)
joint.qvel           # Get velocity
joint.qvel = value   # Set velocity

# Forces
joint.qfrc_applied   # Get/set applied forces
joint.add_qfrc(f)    # Add force
```

### Bodies
Access body positions, orientations, and velocities in world frame:

```python
body = sim.add_body("alias", "body_name")

# Position & Orientation
body.pos             # World position [x, y, z]
body.quat            # Quaternion [w, x, y, z]
body.xmat            # 3x3 rotation matrix

# Velocities
body.angular_vel     # Angular velocity [wx, wy, wz]
body.linear_vel      # Linear velocity [vx, vy, vz]
body.cvel            # Full [wx, wy, wz, vx, vy, vz]

# Apply forces (for free joints)
body.add_force(f)    # Linear force
body.add_torque(t)   # Torque
```

### Actuators
Control actuators:

```python
actuator = sim.add_actuator("alias", "actuator_name")

actuator.ctrl        # Get/set control signal
actuator.force       # Get actuator force
```

### Sites
Access site positions:

```python
site = sim.reg_site("alias", "site_name")

site.pos             # World position [x, y, z]
site.mat             # 3x3 orientation matrix
```

### Tendons
Access tendon properties:

```python
tendon = sim.reg_tendon("alias", "tendon_name")

tendon.length        # Current length
tendon.velocity      # Rate of length change
tendon.force         # Force in tendon
tendon.stiffness     # Spring stiffness
tendon.damping       # Damping coefficient
tendon.springlength  # Rest length
```

## Model Building (Dynamic Manipulation)

Add sites, tendons, and objects to existing models:

```python
from tools.mujoco_sim_template import ModelBuilder, MuJoCoSimulation

# Load base model
builder = ModelBuilder()
builder.from_xml_path("robot.xml")

# Add sites to existing bodies
builder.add_site("gripper_tip", "gripper_link", pos=[0, 0, 0.05])
builder.add_site("target_point", "target_object", pos=[0.1, 0, 0])

# Create tendon between sites
builder.add_tendon("gripper_spring",
                  sites=["gripper_tip", "target_point"],
                  stiffness=100.0,
                  damping=5.0)

# Add new objects
builder.add_body("new_box",
                pos=[0.5, 0, 0.3],
                geom_type="box",
                geom_size=[0.1, 0.1, 0.1])

# Compile and simulate
sim = MuJoCoSimulation.from_builder(builder)
```

See [MODEL_BUILDER_REFERENCE.md](MODEL_BUILDER_REFERENCE.md) for complete API documentation.

## Utility Functions

```python
from mujoco_sim_template import (
    quat_mul,           # Multiply quaternions
    quat_conj,          # Quaternion conjugate
    quat_to_axis_angle, # Convert to axis-angle
    quat_error          # Orientation error
)

# Orientation control example
qA = body_a.quat
qB = body_b.quat
ang_error = quat_error(qA, qB)
```

## Advanced Usage

### Orientation Synchronization

```python
def sync_orientation(sim):
    box_a = sim.bodies["box_a"]
    box_b = sim.bodies["box_b"]
    
    # Compute orientation error
    ang_error = quat_error(box_a.quat, box_b.quat)
    w_error = box_b.angular_vel - box_a.angular_vel
    
    # PD control
    Kp, Kd = 20.0, 5.0
    torque = -Kp * ang_error - Kd * w_error
    
    box_a.add_torque(-torque)
    box_b.add_torque(torque)
```

### Copying State (Dummy Objects)

```python
def make_dummy_follow(sim):
    target = sim.joints["target"]
    dummy = sim.joints["dummy"]
    
    # Copy full state
    dummy.qpos = target.qpos
    
    # Or copy just position, keep identity rotation
    dummy.qpos[:3] = target.qpos[:3]
    dummy.qpos[3:7] = [1, 0, 0, 0]
```

### Trajectory Generation

```python
def trajectory_control(sim):
    crane = sim.joints["crane"]
    lift = sim.actuators["lift"]
    
    t = sim.time
    target_height = 1.5
    t_rise = 3.0
    
    if 0.5 < t < 0.5 + t_rise:
        alpha = 0.5 - 0.5*np.cos(np.pi * (t - 0.5) / t_rise)
        target = crane.qpos[0]*(1-alpha) + target_height*alpha
    elif t <= 0.5:
        target = 0.0
    else:
        target = target_height
    
    lift.ctrl = target
```

## Files

### Core Template
- **`mujoco_sim_template.py`** - Main template with OOP wrappers and ModelBuilder
- **`3-cubes-oop.py`** - Example: Basic simulation with OOP template

### Documentation
- **`MUJOCO_TEMPLATE_README.md`** - This file (overview and quickstart)
- **`MUJOCO_TEMPLATE_REFERENCE.py`** - Quick reference card for OOP interface
- **`MODEL_BUILDER_REFERENCE.md`** - Complete guide to model building/manipulation
- **`mujoco_template_guide.py`** - Detailed usage examples for OOP interface
- **`INTEGRATION_GUIDE.md`** - How to integrate with existing code

### Examples
- **`model_builder_examples.py`** - Examples of dynamic model building
  - Adding tendons to robots
  - Creating new objects
  - Building scenes programmatically

## Documentation

For more examples and detailed explanations:

```bash
# View the usage guide
python3 tools/mujoco_template_guide.py

# View quick reference
python3 tools/MUJOCO_TEMPLATE_REFERENCE.py

# Try model building examples
python3 tools/model_builder_examples.py 4  # Quick demo
python3 tools/model_builder_examples.py 2  # Objects with tendons
```

## Benefits

1. **Cleaner Code**: No manual index management
2. **Fewer Bugs**: No off-by-one errors or wrong indices
3. **Better Readability**: `box.pos` vs `data.xpos[body_id]`
4. **IDE Support**: Autocomplete and type hints
5. **Maintainability**: Easy to refactor and understand
6. **Reusability**: Works with any MuJoCo model

## Migration Guide

To convert existing code:

1. Create simulation: `sim = MuJoCoSimulation(xml_path="...")`
2. Replace ID lookups with entity registration
3. Replace array access with properties
4. Move control logic into callback function
5. Run with `sim.run()`

See `3-cubes-oop.py` for a complete migration example.
