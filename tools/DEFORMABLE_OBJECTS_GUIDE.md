# Deformable Objects in MuJoCo

## Overview

The `add_deformable_object()` method creates soft, squishy objects that can be grasped, squeezed, and deformed using MuJoCo's composite/flexcomp feature. These objects are made of particles connected by springs, allowing realistic deformation.

## Basic Usage

```python
from tools.mujoco_sim_template import ModelBuilder
import networkx as nx

G = nx.Graph()
builder = ModelBuilder(G)
builder.from_xml_path("base_model.xml")

# Add a soft sponge
builder.add_deformable_object(
    name="sponge",
    pos=[0.5, 0, 0.2],
    shape="box",
    size=[0.05, 0.05, 0.05],
    spacing=0.015,
    stiffness=500.0,
    damping=5.0,
    rgba=[1, 1, 0, 0.6]
)

sim = MuJoCoSimulation.from_builder(builder)
sim.run(passive=True)
```

## Parameters

### Shape and Size

**`shape`**: `"box"`, `"sphere"`, `"ellipsoid"`, or `"cylinder"`

**`size`**: Dimensions (interpretation depends on shape)
- **box**: `[half_width, half_depth, half_height]`
  - Example: `[0.05, 0.05, 0.05]` = 10cm cube
- **sphere**: `[radius, 0, 0]`
  - Example: `[0.04, 0, 0]` = 4cm radius sphere
- **ellipsoid**: `[radius_x, radius_y, radius_z]`
  - Example: `[0.05, 0.04, 0.03]` = stretched ellipsoid
- **cylinder**: `[radius, half_height, 0]`
  - Example: `[0.03, 0.08, 0]` = 3cm radius, 16cm tall

### Mechanical Properties

**`spacing`**: Distance between particles (meters)
- Smaller = more particles = smoother deformation = slower simulation
- Typical: 0.01 - 0.02 meters
- Rule of thumb: spacing ≈ size / 5

**`stiffness`**: Spring stiffness (N/m)
- How much the object resists deformation
- Soft gel: 100-500
- Sponge: 500-1500
- Firm rubber: 1500-5000
- Stiff foam: 5000+

**`damping`**: Spring damping (N·s/m)
- Energy dissipation during deformation
- Low (5-15): bouncy, elastic
- Medium (15-30): realistic
- High (30-50): gel-like, sluggish

**`mass`**: Total mass (kg)
- Distributed evenly across all particles
- Each particle gets: mass / num_particles

### Visual Properties

**`rgba`**: Color `[red, green, blue, alpha]`
- RGB values: 0.0 - 1.0
- Alpha: transparency (0 = invisible, 1 = opaque)
- Recommended: 0.6-0.8 for soft objects (shows internal structure)

**`particle_size`**: Visual radius of each particle sphere
- Typical: 0.005 - 0.01 meters
- Should be ≤ spacing for best appearance

### Contact Properties (Advanced)

**`solref`**: `[timeconst, dampratio]`
- Controls contact dynamics
- Default: `[0.01, 1.0]` (soft, compliant)
- Smaller timeconst = softer contacts

**`solimp`**: `[dmin, dmax, width, mid, power]`
- Contact impedance parameters
- Default: `[0.9, 0.95, 0.001, 0.5, 2]`
- Allows penetration for soft bodies

## Examples

### 1. Soft Sponge (High Deformability)

```python
builder.add_deformable_object(
    name="sponge",
    pos=[0.0, 0.0, 0.15],
    shape="box",
    size=[0.06, 0.06, 0.06],
    spacing=0.015,
    stiffness=500.0,    # Soft
    damping=20.0,
    mass=0.5,
    rgba=[1.0, 0.8, 0.2, 0.7],  # Yellow/orange
    particle_size=0.008
)
```

### 2. Rubber Ball (Medium Stiffness)

```python
builder.add_deformable_object(
    name="ball",
    pos=[0.3, 0.0, 0.1],
    shape="sphere",
    size=[0.05, 0, 0],  # 5cm radius
    spacing=0.012,
    stiffness=1500.0,   # Medium-stiff
    damping=15.0,
    mass=0.3,
    rgba=[0.2, 0.6, 1.0, 0.7],  # Blue
    particle_size=0.006
)
```

### 3. Gel/Jelly (Very Soft, High Damping)

```python
builder.add_deformable_object(
    name="gel",
    pos=[-0.3, 0.0, 0.08],
    shape="ellipsoid",
    size=[0.05, 0.04, 0.03],
    spacing=0.01,
    stiffness=300.0,    # Very soft
    damping=30.0,       # High damping
    mass=0.4,
    rgba=[0.2, 1.0, 0.4, 0.6],  # Green, transparent
    particle_size=0.005
)
```

### 4. Firm Foam Cylinder

```python
builder.add_deformable_object(
    name="foam_roll",
    pos=[0.0, 0.0, 0.1],
    shape="cylinder",
    size=[0.04, 0.1, 0],  # 4cm radius, 20cm tall
    spacing=0.015,
    stiffness=2000.0,   # Stiff
    damping=10.0,
    mass=0.6,
    rgba=[0.9, 0.5, 0.3, 0.8],  # Orange
    particle_size=0.008
)
```

## Performance Considerations

### Particle Count

Number of particles ≈ (size / spacing)³ for box

Examples:
- **10cm box, 2cm spacing**: ~125 particles (fast)
- **10cm box, 1cm spacing**: ~1000 particles (medium)
- **10cm box, 0.5cm spacing**: ~8000 particles (slow)

### Optimization Tips

1. **Use larger spacing** for background/non-critical objects
2. **Reduce particle_size** if visuals don't matter
3. **Increase timestep** (but may reduce stability)
4. **Limit number** of deformable objects in scene
5. **Use simpler shapes** (sphere/cylinder faster than box)

## Common Use Cases

### Grasping Simulation

```python
# Soft object to grasp
builder.add_deformable_object(
    name="target",
    pos=[0.5, 0, 0.2],
    shape="sphere",
    size=[0.04, 0, 0],
    spacing=0.012,
    stiffness=1000.0,
    damping=20.0,
    mass=0.3,
    rgba=[1, 0, 0, 0.7]
)

# Add gripper...
# Object will deform when grasped
```

### Soft Robot Components

```python
# Soft actuator/muscle
builder.add_deformable_object(
    name="actuator",
    pos=[0.0, 0.0, 0.1],
    shape="cylinder",
    size=[0.02, 0.08, 0],
    spacing=0.01,
    stiffness=800.0,
    damping=15.0,
    mass=0.2,
    rgba=[0.8, 0.2, 0.2, 0.7]
)
```

### Cushion/Padding

```python
# Protective padding
builder.add_deformable_object(
    name="cushion",
    pos=[0.0, 0.0, 0.05],
    shape="box",
    size=[0.15, 0.15, 0.03],  # Flat cushion
    spacing=0.015,
    stiffness=400.0,
    damping=25.0,
    mass=0.5,
    rgba=[0.5, 0.5, 1.0, 0.6]
)
```

## Troubleshooting

### Object falls through ground
- Increase `stiffness`
- Reduce `spacing` (more particles)
- Check `solref`/`solimp` parameters

### Object too bouncy
- Increase `damping`
- Adjust `solref` dampratio

### Simulation unstable/exploding
- Reduce timestep in model options
- Increase `damping`
- Increase `spacing` (fewer particles)
- Check mass isn't too large

### Object doesn't deform enough
- Decrease `stiffness`
- Reduce `spacing` (more particles)
- Check gripper force is sufficient

### Poor visual appearance
- Reduce `particle_size`
- Decrease `spacing` for smoother surface
- Adjust `rgba` alpha for better transparency

## Demo

Run the demo to see deformable objects in action:

```bash
mjpython testing/deformable_object_demo.py
```

This shows:
- Yellow sponge (moderately deformable)
- Blue ball (stiffer)
- Green gel (very soft)
- Gripper jaws that squeeze the objects

## Technical Details

### Under the Hood

Deformable objects use MuJoCo's **composite** feature with:
- **Particles**: Individual spheres with mass and collision
- **Springs/Tendons**: Connect nearby particles
- **Skin**: Visual mesh for smooth appearance
- **Constraints**: Internal structure for stability

### Limitations

- Not true FEM (Finite Element Method) - uses particle-spring model
- Best for moderately soft objects (not liquids or very stiff solids)
- Performance scales with particle count (O(n²) for constraints)
- May require tuning for specific materials

### Alternatives

For very specific needs:
- **Rigid objects**: Use standard `add_body()`
- **Cables/ropes**: Use spatial tendons
- **Cloth**: Use composite with cloth type
- **Liquids**: Not supported (use particle fluids externally)
