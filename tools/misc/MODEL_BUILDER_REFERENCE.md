# Model Builder Reference - Dynamic Model Manipulation

## Overview

The `ModelBuilder` class allows you to programmatically modify MuJoCo models by:
- Adding sites to existing bodies
- Creating spatial and fixed tendons between sites/joints
- Adding new bodies/objects
- Building complete scenes from scratch
- **NEW**: Automatic tendon network generation from NetworkX graphs

This uses MuJoCo's `MjSpec` API for model specification.

## Quick Start

```python
from tools.mujoco_sim_template import MuJoCoSimulation, ModelBuilder

# 1. Create builder and load base model
builder = ModelBuilder()
builder.from_xml_path("my_robot.xml")

# 2. Add sites to existing bodies
builder.add_site("hand_site", "gripper_link", pos=[0, 0, 0.05])
builder.add_site("target_site", "target_object", pos=[0.1, 0, 0])

# 3. Create tendon between sites
builder.add_tendon("gripper_tendon",
                  sites=["hand_site", "target_site"],
                  stiffness=100.0,
                  damping=5.0)

# 4. Compile and create simulation
sim = MuJoCoSimulation.from_builder(builder)

# 5. Use as normal
tendon = sim.reg_tendon("my_tendon", "gripper_tendon")
print(f"Tendon length: {tendon.length}")
```

## API Reference

### ModelBuilder Class

#### Loading Base Model

```python
builder = ModelBuilder()

# From file
builder.from_xml_path("path/to/model.xml")

# From string
builder.from_xml_string(xml_content)
```

#### Adding Sites

```python
builder.add_site(
    name="site_name",           # Unique site name
    body_name="parent_body",    # Existing body to attach to
    pos=[0, 0, 0],              # Position relative to body
    quat=None,                  # Orientation [w,x,y,z] (optional)
    size=0.01,                  # Visual size
    rgba=[1, 0, 0, 1]           # Color [r,g,b,a]
)
```

**Example:**
```python
# Add site 5cm above gripper
builder.add_site("grip_tip", "gripper_link", pos=[0, 0, 0.05])

# Add site with orientation
builder.add_site("sensor_pos", "arm_link",
                pos=[0.1, 0, 0],
                quat=[1, 0, 0, 0],  # Identity rotation
                rgba=[0, 1, 0, 1])  # Green
```

#### Creating Tendons

```python
builder.add_tendon(
    name="tendon_name",         # Unique tendon name
    sites=["site1", "site2"],   # List of site names (2+)
    stiffness=0.0,              # Spring stiffness (0 = no spring)
    damping=0.0,                # Damping coefficient
    springlength=None,          # Rest length (None = auto)
    rgba=[0.9, 0.7, 0.3, 1]     # Color [r,g,b,a]
)
```

**Examples:**

```python
# Simple spring between two points
builder.add_tendon("spring",
                  sites=["point_a", "point_b"],
                  stiffness=100.0,
                  damping=5.0)

# Multi-segment tendon (cable routing)
builder.add_tendon("cable",
                  sites=["start", "waypoint1", "waypoint2", "end"],
                  stiffness=50.0)

# Damper without spring
builder.add_tendon("damper",
                  sites=["body_a", "body_b"],
                  stiffness=0.0,
                  damping=10.0)
```

#### Adding Bodies/Objects

```python
builder.add_body(
    name="body_name",           # Unique body name
    pos=[0, 0, 0],              # Initial position
    quat=None,                  # Initial orientation [w,x,y,z]
    mass=1.0,                   # Body mass
    geom_type="box",            # Geometry: "box", "sphere", "cylinder", "capsule"
    geom_size=[0.05, 0.05, 0.05],  # Size (interpretation depends on type)
    geom_rgba=[0.5, 0.5, 0.8, 1],   # Color [r,g,b,a]
    free_joint=True,            # Add free joint (6 DOF)
    parent=None                 # Parent body (None = world)
)
```

**Geom Size Interpretation:**
- **box**: `[x_half, y_half, z_half]` - Half-widths in each dimension
- **sphere**: `[radius]`
- **cylinder**: `[radius, half_length]`
- **capsule**: `[radius, half_length]`

**Examples:**

```python
# Add a box
builder.add_body("my_box",
                pos=[0, 0, 1.0],
                mass=2.0,
                geom_type="box",
                geom_size=[0.1, 0.1, 0.1],  # 20cm cube
                geom_rgba=[1, 0, 0, 1])     # Red

# Add a sphere
builder.add_body("ball",
                pos=[0.5, 0, 0.5],
                mass=0.5,
                geom_type="sphere",
                geom_size=[0.08],           # 8cm radius
                geom_rgba=[0, 1, 0, 1])     # Green

# Add a fixed cylinder (no free joint)
builder.add_body("pillar",
                pos=[1, 0, 0.5],
                geom_type="cylinder",
                geom_size=[0.05, 0.5],      # 5cm radius, 1m tall
                free_joint=False,
                parent="floor_body")
```

#### Compiling Model

```python
# Compile to MuJoCo model
model = builder.compile()

# Or create simulation directly
sim = MuJoCoSimulation.from_builder(builder)
```

### Tendon Class (Wrapper)

Once you have a tendon in your simulation:

```python
tendon = sim.reg_tendon("my_tendon", "tendon_name_in_xml")

# Read properties
length = tendon.length          # Current length
velocity = tendon.velocity      # Rate of length change
force = tendon.force            # Force in tendon
stiffness = tendon.stiffness    # Spring stiffness
damping = tendon.damping        # Damping coefficient
rest_length = tendon.springlength  # Rest length
```

## Common Patterns

### Pattern 1: Add Tendon to Existing Robot

```python
# Load robot
builder = ModelBuilder()
builder.from_xml_path("robot.xml")

# Add attachment points
builder.add_site("r_hand_attach", "right_hand", pos=[0, 0, 0.05])
builder.add_site("l_hand_attach", "left_hand", pos=[0, 0, 0.05])

# Connect with spring
builder.add_tendon("hand_spring",
                  sites=["r_hand_attach", "l_hand_attach"],
                  stiffness=50.0,
                  damping=2.0)

# Simulate
sim = MuJoCoSimulation.from_builder(builder)
```

### Pattern 2: Build Scene from Scratch

```python
# Minimal base
minimal = """
<mujoco>
    <worldbody>
        <light pos="0 0 3"/>
        <geom type="plane" size="5 5 0.1"/>
    </worldbody>
</mujoco>
"""

builder = ModelBuilder().from_xml_string(minimal)

# Add objects
builder.add_body("box1", pos=[0, 0, 0.5], geom_size=[0.1, 0.1, 0.1])
builder.add_body("box2", pos=[0.3, 0, 0.5], geom_size=[0.1, 0.1, 0.1])

# Add sites
builder.add_site("s1", "box1", pos=[0.1, 0, 0])
builder.add_site("s2", "box2", pos=[-0.1, 0, 0])

# Connect
builder.add_tendon("connection", sites=["s1", "s2"], stiffness=100.0)

sim = MuJoCoSimulation.from_builder(builder)
```

### Pattern 3: Cable Routing

```python
# Create a cable that routes through multiple points
builder.add_site("start", "actuator_body", pos=[0, 0, 0])
builder.add_site("pulley1", "support1", pos=[0, 0, 0])
builder.add_site("pulley2", "support2", pos=[0, 0, 0])
builder.add_site("end", "load_body", pos=[0, 0, 0])

builder.add_tendon("cable_route",
                  sites=["start", "pulley1", "pulley2", "end"],
                  stiffness=500.0,
                  damping=10.0)
```

### Pattern 4: Monitoring Tendon Forces

```python
sim = MuJoCoSimulation.from_builder(builder)
tendon = sim.reg_tendon("monitor", "my_tendon")

def control(sim):
    # Check if tendon is under tension
    if tendon.force > 10.0:
        print(f"High tension detected: {tendon.force:.2f}N")
    
    # Monitor stretch
    stretch = tendon.length - tendon.springlength
    print(f"Tendon stretched by {stretch:.3f}m")

sim.run(control_callback=control, passive=True)
```

### Pattern 5: Dynamic Stiffness (via direct modification)

```python
# Note: Stiffness is set at compile time, but you can access model directly
sim = MuJoCoSimulation.from_builder(builder)
tendon = sim.reg_tendon("spring", "my_spring")

def control(sim):
    # Directly modify model (careful - this persists!)
    if sim.time > 5.0:
        sim.model.tendon_stiffness[tendon.id] = 200.0  # Increase stiffness
    
    # Force will respond to new stiffness
    print(f"Force: {tendon.force:.2f}N at stiffness {tendon.stiffness:.1f}")

sim.run(control_callback=control, passive=True)
```

### Pattern 6: Network-Based Tendon Generation (NEW)

```python
import networkx as nx

# Load model and add sites
builder = ModelBuilder.from_xml_path("robot.xml")
for i, body in enumerate(['box1', 'box2', 'box3', 'box4']):
    builder.add_site(f"site{i}", body, pos=[0, 0, 0])

# Create tendon network graph
G = nx.Graph()
G.add_edge('site0', 'site1', stiffness=200, springlength=0.3)
G.add_edge('site1', 'site2', stiffness=150, springlength=0.4)
G.add_edge('site2', 'site3', stiffness=180, springlength=0.35)
G.add_edge('site3', 'site0', stiffness=200, springlength=0.3)  # Close the loop

# Automatically generate all tendons from graph
builder.add_tendons_from_graph(
    graph=G,
    default_damping=2.0,
    tendon_name_format="net_{i}_{j}"
)

sim = MuJoCoSimulation.from_builder(builder)
```

**See [`NETWORKX_TENDONS_GUIDE.md`](./NETWORKX_TENDONS_GUIDE.md) for comprehensive NetworkX usage.**

## Tips and Best Practices

### Site Placement

- Place sites where you want the tendon to attach
- Sites inherit parent body motion
- Use visualization (rgba parameter) to verify placement

### Tendon Parameters

- **Stiffness = 0**: Pure constraint (fixed length, no spring)
- **Stiffness > 0**: Elastic spring behavior
- **Damping**: Dissipates energy, stabilizes oscillations
- **Spring length**: Set to `None` for auto (uses initial configuration)

### Performance

- Each tendon adds computational cost
- Multiple sites in a tendon (cable routing) is efficient
- Many separate tendons are more expensive

### Debugging

```python
# Check if sites exist
try:
    builder.add_tendon("test", sites=["site1", "site2"])
except ValueError as e:
    print(f"Error: {e}")  # Will tell you which site is missing

# Visualize sites
builder.add_site("debug", "body", size=0.05, rgba=[1, 0, 0, 1])  # Large red site
```

## Examples

See `model_builder_examples.py` for complete working examples:

```bash
# Run quick demo
python tools/model_builder_examples.py 4

# Add tendons to robot
python tools/model_builder_examples.py 1

# Create objects with tendons
python tools/model_builder_examples.py 2

# Complex manipulation scene
python tools/model_builder_examples.py 3
```

## Limitations

- Sites and tendons added dynamically cannot be removed (MjSpec limitation)
- Must recompile to make changes (creates new model)
- Tendon properties (stiffness, damping) set at compile time
- Cannot modify site positions after compilation (they move with parent body)

## Advanced: Direct MjSpec Access

For features not wrapped by ModelBuilder:

```python
builder = ModelBuilder()
builder.from_xml_path("model.xml")

# Access spec directly
spec = builder.spec

# Use any MjSpec API
custom_body = spec.worldbody.add_body()
custom_body.name = "custom"
# ... full MjSpec API available

# Compile as normal
sim = MuJoCoSimulation.from_builder(builder)
```

See [MuJoCo documentation](https://mujoco.readthedocs.io/) for full MjSpec API.
