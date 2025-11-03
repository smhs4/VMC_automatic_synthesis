# Contact Sensor Feature

## Overview

The `ContactSensor` class provides an easy way to monitor contacts between a body and other specified bodies in MuJoCo simulations. It automatically detects contacts and computes contact forces.

## Features

- **Multi-body monitoring**: Track contacts with multiple bodies simultaneously
- **Force measurement**: Get normal force, tangent force, and total force magnitude
- **Contact details**: Access contact position and normal direction
- **Convenience methods**: Check if specific bodies are in contact, get total forces, etc.

## Quick Start

```python
from tools.mujoco_sim_template import MuJoCoSimulation

# Create simulation
sim = MuJoCoSimulation("model.xml")

# Register a contact sensor
sensor = sim.reg_contact_sensor(
    "target_contacts",          # Alias for the sensor
    "target_block",             # Body to monitor
    ["left_gripper", "right_gripper"]  # Bodies to detect contacts with
)

# In your control loop or after simulation step:
contacts = sensor.get_contacts()
for contact in contacts:
    print(f"Contact with {contact.body_name}")
    print(f"  Normal force: {contact.normal_force:.2f} N")
    print(f"  Total force:  {contact.total_force:.2f} N")
    print(f"  Position:     {contact.contact_pos}")
```

## API Reference

### ContactSensor Class

#### Constructor
```python
ContactSensor(body_name: str, monitored_bodies: List[str], 
              model: MjModel, data: MjData)
```

**Parameters:**
- `body_name`: Name of the body to monitor contacts for
- `monitored_bodies`: List of body names to detect contacts with
- `model`: MuJoCo model
- `data`: MuJoCo data

**Note:** Typically created via `sim.reg_contact_sensor()` rather than directly.

#### Methods

##### `get_contacts() -> List[ContactInfo]`
Get list of all current contacts with monitored bodies.

**Returns:**
- List of `ContactInfo` objects, one for each monitored body currently in contact

**Example:**
```python
contacts = sensor.get_contacts()
for contact in contacts:
    print(f"{contact.body_name}: {contact.total_force:.2f} N")
```

##### `is_in_contact(body_name: str) -> bool`
Check if a specific monitored body is currently in contact.

**Parameters:**
- `body_name`: Name of the monitored body to check

**Returns:**
- `True` if in contact, `False` otherwise

**Example:**
```python
if sensor.is_in_contact("left_gripper"):
    print("Left gripper is touching the target!")
```

##### `get_total_contact_force() -> float`
Get sum of all contact forces with monitored bodies.

**Returns:**
- Total contact force magnitude (N)

**Example:**
```python
total_force = sensor.get_total_contact_force()
print(f"Total contact force: {total_force:.2f} N")
```

##### `get_contact_force_by_body(body_name: str) -> float`
Get contact force with a specific monitored body.

**Parameters:**
- `body_name`: Name of the monitored body

**Returns:**
- Contact force magnitude (N), or 0.0 if not in contact

**Example:**
```python
left_force = sensor.get_contact_force_by_body("left_gripper")
right_force = sensor.get_contact_force_by_body("right_gripper")
print(f"Left: {left_force:.2f} N, Right: {right_force:.2f} N")
```

### ContactInfo Dataclass

Information about a single contact.

**Attributes:**
- `body_name` (str): Name of the contacting body
- `body_id` (int): MuJoCo body ID
- `geom_id` (int): MuJoCo geometry ID
- `normal_force` (float): Normal force magnitude (N)
- `tangent_force` (np.ndarray): Tangent forces [2] (N)
- `total_force` (float): Total force magnitude (N)
- `contact_pos` (np.ndarray): Contact position in world frame [3]
- `contact_normal` (np.ndarray): Contact normal direction [3]

## Usage Examples

### Example 1: Basic Contact Detection

```python
# Setup
sim = MuJoCoSimulation("model.xml")
target = sim.reg_body("target", "target_block")
sensor = sim.reg_contact_sensor("contacts", "target_block", 
                                ["gripper_left", "gripper_right"])

# In control callback
def control(sim):
    contacts = sensor.get_contacts()
    
    if contacts:
        print(f"Detected {len(contacts)} contact(s)")
        for c in contacts:
            print(f"  {c.body_name}: {c.total_force:.2f} N")
    else:
        print("No contacts")
```

### Example 2: Grasp Force Monitoring

```python
# Monitor grasp quality by checking bilateral contact
sensor = sim.reg_contact_sensor("grasp", "object", 
                                ["finger_left", "finger_right"])

def control(sim):
    left_contact = sensor.is_in_contact("finger_left")
    right_contact = sensor.is_in_contact("finger_right")
    
    if left_contact and right_contact:
        # Good grasp - both fingers in contact
        left_force = sensor.get_contact_force_by_body("finger_left")
        right_force = sensor.get_contact_force_by_body("finger_right")
        
        # Check force balance
        force_diff = abs(left_force - right_force)
        if force_diff < 5.0:  # N
            print("Balanced grasp!")
        else:
            print(f"Unbalanced: {force_diff:.2f} N difference")
    else:
        print("Incomplete grasp")
```

### Example 3: Collision Avoidance

```python
# Monitor collisions with obstacles
sensor = sim.reg_contact_sensor("collision", "robot_arm", 
                                ["obstacle_1", "obstacle_2", "wall"])

def control(sim):
    contacts = sensor.get_contacts()
    
    if contacts:
        # Collision detected - emergency stop
        for actuator in sim.actuators.values():
            actuator.ctrl = 0.0
        
        print("COLLISION DETECTED!")
        for c in contacts:
            print(f"  Hit {c.body_name} at {c.contact_pos}")
```

### Example 4: Fitness Function for GA

```python
# Use in genetic algorithm to optimize grasping
def evaluate_fitness(individual):
    # Build and run simulation
    sim = build_model(individual)
    sensor = sim.reg_contact_sensor("target", "target_block",
                                   ["left_chopstick", "right_chopstick"])
    
    total_contact = 0.0
    for _ in range(num_steps):
        sim.step()
        total_contact += sensor.get_total_contact_force()
    
    # Higher contact force = better grasp
    fitness = total_contact
    return fitness
```

## Implementation Details

### Contact Detection

The sensor works by:
1. Getting all geom IDs associated with the monitored body
2. Getting all geom IDs for each monitored body
3. Iterating through MuJoCo's contact array (`data.contact`)
4. Checking if each contact involves our body and a monitored body
5. Extracting force information from the contact frame

### Force Calculation

MuJoCo stores contact forces in `contact.frame`:
- `frame[0]`: Normal force (perpendicular to surface)
- `frame[1]`: Tangent force 1 (friction)
- `frame[2]`: Tangent force 2 (friction)

The total force is computed as:
```python
total_force = sqrt(normal_force^2 + tangent_force[0]^2 + tangent_force[1]^2)
```

### Performance

- **Efficient**: Only iterates through active contacts in `data.ncon`
- **Scales well**: Performance independent of total number of bodies
- **Real-time**: Suitable for control loops at typical simulation rates

## Limitations

- Only detects contacts between registered bodies (won't detect contacts with unmonitored bodies)
- Requires bodies to have geometry (geoms) for collision detection
- Contact forces are instantaneous (not integrated over time)

## Demo

Run the contact sensor demo:

```bash
# Interactive viewer
mjpython testing/contact_sensor_demo.py

# Passive viewer (non-interactive)
mjpython testing/contact_sensor_demo.py --passive

# Custom duration
mjpython testing/contact_sensor_demo.py --duration 10.0
```

The demo shows:
- A target block (green)
- Three manipulators (red, blue, yellow)
- Real-time contact force monitoring
- Force visualization in terminal

## Integration with Existing Code

The contact sensor integrates seamlessly with existing simulation code:

```python
# Before - manual contact detection
for i in range(data.ncon):
    contact = data.contact[i]
    # Complex geom ID checking...
    # Manual force calculation...

# After - with ContactSensor
sensor = sim.reg_contact_sensor("my_sensor", "body", ["other_body"])
contacts = sensor.get_contacts()
# Clean, simple API!
```

## See Also

- `tools/mujoco_sim_template.py` - Main implementation
- `testing/contact_sensor_demo.py` - Demo script
- `training/ga_tendon_optimizer_deap.py` - Example usage in GA fitness function
