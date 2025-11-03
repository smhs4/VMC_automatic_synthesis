# Testing Directory

Examples and tests for the MuJoCo simulation template.

## Files

### Examples

**`3-cubes-oop.py`** - Three-cubes manipulation with OOP template
- Demonstrates clean OOP interface
- Shows ModelBuilder capabilities
- Two modes: standard and enhanced

**`graph_tendons_example.py`** (NEW) - NetworkX-based tendon network generation
- Example 1: Triangle network connecting three boxes
- Example 2: 2x2 grid with cross-bracing diagonals
- Example 3: Custom properties and named tendons

**`testing_env.py`** - Sciurus17 robot test environment
- Gravity compensation integration
- OOP interface for robot manipulation
- Example of hybrid control (raw MuJoCo + OOP)

### Configuration

**`model.xml`** - Three-cubes scene definition
- Crane with lifting actuator
- Red and green boxes connected by springs
- Blue target box
- Dummy object for constraint implementation

**`three_cubes_config.py`** - Configuration for three-cubes scene

**`__init__.py`** - Package initialization

## Running Examples

### 3-Cubes Simulation

**Standard mode** (use existing model as-is):
```bash
python testing/3-cubes-oop.py --passive
```

**Enhanced mode** (add features with ModelBuilder):
```bash
python testing/3-cubes-oop.py --passive --add-features
```

The enhanced mode adds:
- Center monitoring sites on all boxes
- Cross-stabilization tendon between red and green boxes
- Extra telemetry output

### NetworkX Tendon Networks (NEW)

**Prerequisites:**
```bash
pip install networkx
```

**Run examples:**
```bash
# Triangle network - 3 boxes connected in triangle
python testing/graph_tendons_example.py --example 1

# Grid network - 2x2 grid with cross-bracing
python testing/graph_tendons_example.py --example 2

# Custom properties - varied tendon parameters
python testing/graph_tendons_example.py --example 3
```

Features demonstrated:
- Automatic tendon generation from NetworkX graphs
- Edge attributes → tendon properties
- Custom stiffness, damping, springlength per connection
- Color-coded tendons by function

### Sciurus17 Robot Test

```bash
python testing/testing_env.py
```

Features:
- Gravity compensation for 21-DOF robot
- Body-level force application
- Integration of low-level and high-level control

## What the Examples Demonstrate

### 3-Cubes Example Shows:

1. **OOP Entity Registration**
   ```python
   crane = sim.reg_joint("crane", "crane_base__up_down")
   red_box = sim.reg_body("red_box", "red_box")
   spring = sim.reg_tendon("spring1", "spring1")
   ```

2. **Property-Based Access**
   ```python
   crane.qpos = target_height
   position = red_box.pos
   force = spring.force
   ```

3. **ModelBuilder Enhancement** (with `--add-features`)
   ```python
   builder = ModelBuilder()
   builder.from_xml_path("model.xml")
   builder.add_site("red_center", "red_box", pos=[0, 0, 0])
   builder.add_tendon("cross", sites=["red_center", "green_center"],
                     stiffness=50.0, damping=20.0)
   sim = MuJoCoSimulation.from_builder(builder)
   ```

4. **Clean Control Callbacks**
   ```python
   def control(sim):
       crane.qpos = compute_target()
       if red_box.pos[2] < 0.5:
           red_box.add_force([0, 0, 10.0])
   ```

### Testing Environment Shows:

1. **Gravity Compensation Integration**
   - Computing bias forces with `mj_rne`
   - Mapping to actuator space
   - Combined with task control

2. **Hybrid Control Pattern**
   - Low-level: Direct `model` and `data` access
   - High-level: OOP property access
   - Both in same callback

3. **Body Force Application**
   ```python
   r_hand.add_force([0, 0, -100.0])  # Apply downward force
   ```

## Model Structure (3-Cubes)

```
crane_base (slide joint, position actuator)
  └─ crane beam with hanging sites

red_box (free joint)
  ├─ Spring connections to green_box
  └─ Chain connection to crane

green_box (free joint)
  ├─ Spring connections to red_box
  └─ Chain connection to crane

blue_box (free joint, target)
  └─ Dummy constraint via tendons

dummy (free joint, non-colliding)
  └─ Follows blue_box exactly
```

## Comparison: Before vs After

### Before (Manual Index Management)

```python
# Find IDs
crane_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "crane")
qpos_adr = model.jnt_qposadr[crane_id]

# Access state
crane_pos = data.qpos[qpos_adr]

# Set position
data.qpos[qpos_adr] = target

# Get body position
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_box")
pos = data.xpos[body_id].copy()
```

### After (OOP Template)

```python
# Register once
crane = sim.reg_joint("crane", "crane_base__up_down")
red_box = sim.reg_body("red_box", "red_box")

# Use throughout
crane_pos = crane.qpos
crane.qpos = target
pos = red_box.pos
```

### With ModelBuilder

```python
# Enhance model programmatically
builder = ModelBuilder()
builder.from_xml_path("model.xml")

# Add features
builder.add_site("new_attach", "red_box", pos=[0.1, 0, 0])
builder.add_tendon("new_spring", 
                  sites=["new_attach", "green_center"],
                  stiffness=100.0)

# Use enhanced model
sim = MuJoCoSimulation.from_builder(builder)
new_spring = sim.reg_tendon("spring", "new_spring")
print(f"New spring force: {new_spring.force}N")
```

## Tips

1. **Start simple**: Use standard mode first, add enhancements later
2. **Register only what you need**: Don't register every entity, just the ones you'll use
3. **Use ModelBuilder for**: Experiments with different tendon configurations
4. **Monitor tendons**: Tendons provide force, length, velocity information
5. **Combine patterns**: Low-level control (mj_rne) + high-level access (properties)

## Related Documentation

- `../tools/MUJOCO_TEMPLATE_README.md` - Complete OOP template guide
- `../tools/MODEL_BUILDER_REFERENCE.md` - ModelBuilder API reference
- `../tools/NETWORKX_TENDONS_GUIDE.md` - NetworkX tendon generation guide (NEW)
- `../tools/INTEGRATION_GUIDE.md` - Integration patterns
- `../tools/model_builder_examples.py` - More ModelBuilder examples
- `../tools/test_networkx_tendons.py` - NetworkX unit tests (NEW)
