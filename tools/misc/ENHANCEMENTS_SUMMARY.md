# Template Enhancements Summary

## What's New

The MuJoCo simulation template has been enhanced with **dynamic model manipulation** capabilities using MuJoCo's `MjSpec` API, including:
- ✅ Tendon system with force/length monitoring
- ✅ Dynamic site, tendon, and body creation
- ✅ **NEW**: Automatic tendon network generation from NetworkX graphs
- ✅ Support for both spatial and fixed tendon types

## New Features

### 1. Tendon Class
Access tendon properties in your simulations:

```python
tendon = sim.reg_tendon("my_tendon", "tendon_name")
print(f"Length: {tendon.length}m")
print(f"Force: {tendon.force}N")
print(f"Velocity: {tendon.velocity}m/s")
```

### 2. ModelBuilder Class
Programmatically modify MuJoCo models:

#### Add Sites to Existing Bodies
```python
builder.add_site("gripper_tip", "gripper_link", pos=[0, 0, 0.05])
```

#### Create Tendons Between Sites
```python
builder.add_tendon("spring",
                  sites=["site1", "site2"],
                  stiffness=100.0,
                  damping=5.0)
```

#### Add New Bodies/Objects
```python
builder.add_body("new_box",
                pos=[0.5, 0, 0.3],
                geom_type="box",
                geom_size=[0.1, 0.1, 0.1])
```

### 3. from_builder Constructor
Create simulations from modified models:

```python
builder = ModelBuilder()
builder.from_xml_path("base_model.xml")
# ... add sites, tendons, bodies ...
sim = MuJoCoSimulation.from_builder(builder)
```

### 4. NetworkX-Based Tendon Generation (NEW!)
Automatically create complex tendon networks from graphs:

```python
import networkx as nx

# Create graph representing tendon network
G = nx.Graph()
G.add_edge('site1', 'site2', stiffness=200, springlength=0.3)
G.add_edge('site2', 'site3', stiffness=150, springlength=0.4)
G.add_edge('site3', 'site1', stiffness=180, springlength=0.35)

# Automatically generate all tendons
builder.add_tendons_from_graph(
    graph=G,
    default_damping=2.0,
    tendon_name_format="net_{i}_{j}"
)
```

**Perfect for:**
- Complex cable networks
- Systematic tendon routing
- Programmatic constraint systems
- Research reproducibility

See **[`NETWORKX_TENDONS_GUIDE.md`](./NETWORKX_TENDONS_GUIDE.md)** for comprehensive examples!

## Use Cases

### Robotics Research
- Add tendon actuators to existing robot models
- Create variable stiffness connections
- Build soft robotics simulations
- Model cable-driven systems

### Manipulation Tasks
- Add attachment points to grippers
- Create elastic connections to objects
- Build deformable object interactions
- Model spring-loaded mechanisms

### Scene Building
- Programmatically generate test scenarios
- Add obstacles and targets dynamically
- Build parametric environments
- Create randomized scenes

## Example Workflow

```python
# 1. Load your robot
builder = ModelBuilder()
builder.from_xml_path("my_robot.xml")

# 2. Add sites where you want tendons
builder.add_site("r_hand_attach", "right_hand", pos=[0, 0, 0.05])
builder.add_site("target_attach", "target_object", pos=[0.1, 0, 0])

# 3. Create tendon
builder.add_tendon("gripper_tendon",
                  sites=["r_hand_attach", "target_attach"],
                  stiffness=50.0,
                  damping=2.0)

# 4. Simulate with gravity compensation
sim = MuJoCoSimulation.from_builder(builder)
tendon = sim.reg_tendon("my_tendon", "gripper_tendon")

def control(sim):
    # Gravity comp + monitor tendon
    apply_gravity_compensation(sim.model, sim.data)
    print(f"Tendon force: {tendon.force:.2f}N")

sim.run(control_callback=control, passive=True)
```

## Files Added/Updated

1. **Enhanced `mujoco_sim_template.py`**:
   - `Tendon` class with properties
   - `ModelBuilder` class with full API (spatial & fixed tendons)
   - `add_tendons_from_graph()` for NetworkX integration
   - `from_builder()` constructor

2. **`MODEL_BUILDER_REFERENCE.md`**:
   - Complete API documentation
   - Common patterns including NetworkX
   - Tips and best practices

3. **`NETWORKX_TENDONS_GUIDE.md`** (NEW):
   - Comprehensive NetworkX graph-based tendon generation guide
   - Multiple examples (triangle, grid, custom properties)
   - Graph algorithm examples

4. **`model_builder_examples.py`**:
   - 4 complete examples
   - Robot tendons, object creation, manipulation scenes

5. **`test_model_builder.py`**:
   - Verification tests for ModelBuilder API
   
6. **`test_networkx_tendons.py`** (NEW):
   - Tests for NetworkX-based generation
   - Edge attribute propagation tests
   - Graph type compatibility tests

7. **`graph_tendons_example.py`** (NEW):
   - 3 runnable NetworkX examples
   - Triangle network, grid with cross-bracing, custom properties

8. **Updated documentation**:
   - `MUJOCO_TEMPLATE_README.md` - Overview with new features
   - `INTEGRATION_GUIDE.md` - How to combine with existing code
   - `ENHANCEMENTS_SUMMARY.md` - This file!

## Quick Test

```bash
# Run verification tests
python tools/test_model_builder.py
python tools/test_networkx_tendons.py  # NEW: NetworkX tests

# Try examples  
python tools/model_builder_examples.py 4         # Quick demo
python tools/model_builder_examples.py 2         # Objects with tendons
python testing/graph_tendons_example.py --example 1  # Triangle network
python testing/graph_tendons_example.py --example 2  # Grid with cross-bracing
python testing/graph_tendons_example.py --example 3  # Custom properties
```

**Note:** For NetworkX examples, install with: `pip install networkx`

**Note:** All MjSpec API issues have been fixed. See `MJSPEC_API_FIXES.md` for details on the corrections made.

## API Summary

### ModelBuilder Methods
```python
builder = ModelBuilder()
builder.from_xml_path(path)                      # Load base model
builder.from_xml_string(xml)                     # Load from string
builder.add_site(name, body, pos, ...)           # Add attachment point
builder.add_tendon(name, sites=..., joints=..., tendon_type='spatial', ...)  # Create tendon
builder.add_tendons_from_graph(graph, ...)       # Generate from NetworkX graph (NEW)
builder.add_body(name, pos, ...)                 # Add new object
model = builder.compile()                        # Get MuJoCo model
```

### Tendon Properties
```python
tendon.length       # Current length [m]
tendon.velocity     # Length rate [m/s]
tendon.force        # Tension [N]
tendon.stiffness    # Spring constant [N/m]
tendon.damping      # Damping [N·s/m]
tendon.springlength # Rest length [m]
```

## Benefits

1. **No XML Editing**: Add features programmatically
2. **Reusable**: Same code works across models
3. **Dynamic**: Build models on-the-fly
4. **Type-Safe**: Python API with IDE support
5. **Maintainable**: Clear, documented code

## Compatibility

- ✅ Works with existing OOP template features
- ✅ Compatible with gravity compensation
- ✅ Integrates with control callbacks
- ✅ Supports all entity types (joints, bodies, actuators, sites, tendons)
- ✅ Maintains backward compatibility

## Next Steps

1. Try the examples: `mjpython tools/model_builder_examples.py`
2. Read the reference: `MODEL_BUILDER_REFERENCE.md`
3. Build your first dynamic model
4. Apply to your research problem

## Questions?

- See `MODEL_BUILDER_REFERENCE.md` for complete API
- See `model_builder_examples.py` for working examples
- See `INTEGRATION_GUIDE.md` for combining with existing code
