# Integration Guide: OOP Template with Existing MuJoCo Code

This guide shows how to integrate the OOP simulation template with existing MuJoCo patterns like gravity compensation, custom controllers, and complex simulations.

## Pattern 1: Pure OOP (Simplest)

For new simulations where you only need the clean OOP interface:

```python
from tools.mujoco_sim_template import MuJoCoSimulation

sim = MuJoCoSimulation(xml_path="model.xml")

# Register entities
joint = sim.add_joint("my_joint", "joint_name")
body = sim.add_body("my_body", "body_name")

def control(sim):
    # Pure OOP control
    if body.pos[2] < 0.5:
        joint.add_qfrc([10.0])

sim.run(control_callback=control, passive=True)
```

## Pattern 2: Hybrid (OOP + Raw MuJoCo)

When you need to access raw MuJoCo features alongside OOP convenience:

```python
from tools.mujoco_sim_template import MuJoCoSimulation
import mujoco

sim = MuJoCoSimulation(xml_path="model.xml")

# Register entities you want OOP access to
body = sim.add_body("my_body", "body_name")

def control(sim):
    # Access raw model and data when needed
    model = sim.model
    data = sim.data
    
    # Mix raw MuJoCo operations
    mujoco.mj_forward(model, data)
    
    # With OOP convenience
    pos = body.pos
    body.add_force([0, 0, 10.0])

sim.run(control_callback=control, passive=True)
```

## Pattern 3: Gravity Compensation Integration

Combining gravity compensation with task control:

```python
from tools.mujoco_sim_template import MuJoCoSimulation
import mujoco
import numpy as np

sim = MuJoCoSimulation(xml_path="robot.xml")

# Register task-relevant entities
gripper = sim.add_body("gripper", "gripper_link")
target = sim.add_body("target", "target_object")

def control(sim):
    model = sim.model
    data = sim.data
    
    # === STEP 1: Gravity Compensation ===
    # Compute bias forces (gravity + Coriolis)
    acc_zero = np.zeros(model.nv)
    qacc_old = data.qacc.copy()
    data.qacc[:] = acc_zero
    mujoco.mj_rne(model, data, 0, data.qfrc_bias)
    data.qacc[:] = qacc_old
    
    # Map to actuator space
    data.ctrl[:] = 0.0
    for i in range(model.nu):
        jnt_id = model.actuator_trnid[i, 0]
        if jnt_id >= 0 and jnt_id < model.njnt:
            dof_adr = model.jnt_dofadr[jnt_id]
            gear = model.actuator_gear[i, 0]
            data.ctrl[i] = (data.qfrc_bias[dof_adr] / gear) if gear != 0 else data.qfrc_bias[dof_adr]
    
    # === STEP 2: Task Control ===
    # Add task-specific forces using OOP interface
    if target.pos[2] < 0.5:
        gripper.add_force([0, 0, 10.0])

sim.run(control_callback=control, passive=True)
```

## Pattern 4: Reusable Gravity Compensation

Create a reusable gravity compensation function:

```python
from tools.mujoco_sim_template import MuJoCoSimulation
import mujoco
import numpy as np

def apply_gravity_compensation(model, data):
    """Reusable gravity compensation - call this from any control callback."""
    acc_zero = np.zeros(model.nv)
    qacc_old = data.qacc.copy()
    data.qacc[:] = acc_zero
    mujoco.mj_rne(model, data, 0, data.qfrc_bias)
    data.qacc[:] = qacc_old
    
    data.ctrl[:] = 0.0
    for i in range(model.nu):
        jnt_id = model.actuator_trnid[i, 0]
        if jnt_id >= 0 and jnt_id < model.njnt:
            dof_adr = model.jnt_dofadr[jnt_id]
            gear = model.actuator_gear[i, 0]
            data.ctrl[i] = (data.qfrc_bias[dof_adr] / gear) if gear != 0 else data.qfrc_bias[dof_adr]

# === Usage ===
sim = MuJoCoSimulation(xml_path="robot.xml")
gripper = sim.add_body("gripper", "gripper_link")

def control(sim):
    # Apply gravity compensation
    apply_gravity_compensation(sim.model, sim.data)
    
    # Then add task control
    gripper.add_force([0, 0, 10.0])

sim.run(control_callback=control, passive=True)
```

## Pattern 5: Multi-Phase Control

Different control modes at different times:

```python
from tools.mujoco_sim_template import MuJoCoSimulation
import numpy as np

sim = MuJoCoSimulation(xml_path="robot.xml")

# Register entities
joint = sim.add_joint("arm", "shoulder_joint")
actuator = sim.add_actuator("motor", "shoulder_motor")
end_effector = sim.add_body("ee", "end_effector")

# Control parameters
target_positions = [0.0, 1.0, 0.5]
phase_duration = 3.0

def control(sim):
    t = sim.time
    
    # Determine current phase
    phase = int(t / phase_duration) % len(target_positions)
    target = target_positions[phase]
    
    # PD control
    Kp, Kd = 100.0, 10.0
    error = target - joint.qpos[0]
    derror = 0.0 - joint.qvel[0]
    
    actuator.ctrl = Kp * error + Kd * derror
    
    # Phase-specific additional control
    if phase == 1:  # During second phase, apply extra force
        end_effector.add_force([0, 0, 5.0])

sim.run(control_callback=control, passive=True, duration=15.0)
```

## Pattern 6: Extending the Template

Add your own wrapper classes for custom functionality:

```python
from tools.mujoco_sim_template import MuJoCoSimulation, Body
import numpy as np

class RobotArm:
    """High-level wrapper for a robot arm."""
    
    def __init__(self, sim: MuJoCoSimulation, joints: list, end_effector: str):
        self.sim = sim
        self.joints = [sim.add_joint(f"j{i}", name) for i, name in enumerate(joints)]
        self.ee = sim.add_body("ee", end_effector)
    
    @property
    def q(self):
        """Get all joint positions."""
        return np.array([j.qpos[0] for j in self.joints])
    
    @q.setter
    def q(self, values):
        """Set all joint positions."""
        for joint, val in zip(self.joints, values):
            joint.qpos = val
    
    @property
    def ee_pos(self):
        """Get end effector position."""
        return self.ee.pos

# Usage
sim = MuJoCoSimulation(xml_path="robot.xml")
arm = RobotArm(sim, ["j1", "j2", "j3", "j4"], "end_effector_link")

def control(sim):
    # Use high-level interface
    current_pos = arm.q
    ee_position = arm.ee_pos
    
    # Inverse kinematics / trajectory planning here
    target_q = compute_ik(ee_position)
    arm.q = target_q

sim.run(control_callback=control, passive=True)
```

## Key Integration Points

### Accessing Raw MuJoCo from OOP Template

The simulation object always provides access to underlying MuJoCo:

```python
def control(sim):
    model = sim.model  # mujoco.MjModel
    data = sim.data    # mujoco.MjData
    
    # Now use any MuJoCo function
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    mujoco.mj_inverse(model, data)
```

### Entity Properties Map to MuJoCo Arrays

Understanding the mapping helps debugging:

```python
# These are equivalent:
joint.qpos              <==>  data.qpos[model.jnt_qposadr[joint.id]:...]
body.pos                <==>  data.xpos[body.id]
actuator.ctrl           <==>  data.ctrl[actuator.id]
site.pos                <==>  data.site_xpos[site.id]
```

### When to Use What

**Use OOP interface when:**
- You want clean, readable code
- Working with specific named entities
- Building high-level controllers
- Prototyping quickly

**Use raw MuJoCo when:**
- Implementing low-level algorithms (gravity comp, dynamics)
- Need maximum performance
- Using advanced MuJoCo features not wrapped
- Working with many entities at once (batch operations)

**Use both when:**
- Integrating existing code with new features
- Need both convenience and control
- Building complex systems

## Migration Strategy

To migrate existing code:

1. **Wrap in template**: Create `MuJoCoSimulation` instance
2. **Register key entities**: Add the bodies/joints you interact with frequently
3. **Move control to callback**: Put control logic in callback function
4. **Gradually refactor**: Replace index-based access with OOP properties
5. **Keep raw access**: Use `sim.model` and `sim.data` for existing low-level code

You don't need to convert everything at once - the patterns can coexist!
