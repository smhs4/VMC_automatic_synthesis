# Quick Reference: Tendon System with NetworkX

## Installation

```bash
pip install networkx  # Optional, for graph-based generation
```

## Tendon Types

| Type | Connects | Use Case | MjSpec Method |
|------|----------|----------|---------------|
| **Spatial** | Sites | Cables, springs wrapping around geometry | `tendon.wrap_site()` |
| **Fixed** | Joints | Direct joint coupling | `tendon.wrap_joint()` |

## Basic Tendon Creation

### Spatial Tendon
```python
builder.add_tendon(
    name="cable",
    sites=["site1", "site2", "site3"],
    tendon_type='spatial',
    stiffness=200.0,
    damping=2.0,
    springlength=0.5
)
```

### Fixed Tendon
```python
builder.add_tendon(
    name="coupling",
    joints=["joint1", "joint2"],
    tendon_type='fixed',
    stiffness=100.0
)
```

## NetworkX Graph Generation

### Step 1: Create Graph
```python
import networkx as nx

G = nx.Graph()
G.add_edge('site1', 'site2', stiffness=200, springlength=0.3)
G.add_edge('site2', 'site3', stiffness=150, springlength=0.4)
```

### Step 2: Generate Tendons
```python
builder.add_tendons_from_graph(
    graph=G,
    default_damping=2.0,
    tendon_name_format="net_{i}_{j}"
)
```

### Complete Example
```python
from mujoco_sim_template import ModelBuilder
import networkx as nx

# Load model
builder = ModelBuilder.from_xml_path("robot.xml")

# Add sites
for name, body in [('s1', 'box1'), ('s2', 'box2'), ('s3', 'box3')]:
    builder.add_site(name=name, body_name=body, pos=[0, 0, 0])

# Create triangle network
G = nx.Graph()
G.add_edge('s1', 's2', stiffness=200, springlength=0.3)
G.add_edge('s2', 's3', stiffness=150, springlength=0.4)
G.add_edge('s3', 's1', stiffness=180, springlength=0.35)

# Generate tendons
builder.add_tendons_from_graph(G, default_damping=2.0)

# Compile and run
model = builder.compile()
```

## Edge Attributes (NetworkX)

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `stiffness` | float | Spring constant (N/m) | `200.0` |
| `damping` | float | Damping (N·s/m) | `2.0` |
| `springlength` | float or list | Rest length | `0.5` or `[0.5, 0.6]` |
| `rgba` | list[4] | Color [R,G,B,A] | `[1, 0, 0, 1]` |
| `type` | str | `'spatial'` or `'fixed'` | `'spatial'` |
| `name` | str | Custom name | `'stabilizer'` |

## Common Graph Patterns

### Triangle
```python
G = nx.Graph()
G.add_edge('s1', 's2')
G.add_edge('s2', 's3')
G.add_edge('s3', 's1')
nx.set_edge_attributes(G, 100, 'stiffness')
```

### Complete Network
```python
sites = ['center', 'north', 'south', 'east', 'west']
G = nx.complete_graph(sites)
nx.set_edge_attributes(G, 100, 'stiffness')
```

### Grid
```python
G = nx.grid_2d_graph(3, 3)
G = nx.relabel_nodes(G, {(i,j): f's{i}{j}' for i,j in G.nodes()})
```

### Cycle/Ring
```python
G = nx.cycle_graph(['s1', 's2', 's3', 's4'])
```

## Tendon Properties (Runtime)

```python
tendon = sim.reg_tendon("my_tendon", "tendon_name")

# Read properties
length = tendon.length          # Current length (m)
velocity = tendon.velocity      # Rate of change (m/s)
force = tendon.force            # Tension (N)
stiffness = tendon.stiffness    # Spring constant (N/m)
damping = tendon.damping        # Damping (N·s/m)
rest_length = tendon.springlength  # Rest length (m)
```

## Method Signatures

### add_tendon()
```python
def add_tendon(
    self,
    name: str,
    sites: Optional[List[str]] = None,
    joints: Optional[List[str]] = None,
    tendon_type: str = 'spatial',  # 'spatial' or 'fixed'
    stiffness: float = 0.0,
    damping: float = 0.0,
    springlength: Optional[Union[float, List[float]]] = None,
    rgba: List[float] = [0.9, 0.7, 0.3, 1]
) -> ModelBuilder
```

### add_tendons_from_graph()
```python
def add_tendons_from_graph(
    self,
    graph,  # NetworkX Graph
    site_prefix: str = "",
    default_stiffness: float = 100.0,
    default_damping: float = 1.0,
    default_springlength: Optional[Union[float, List[float]]] = None,
    default_rgba: List[float] = [0.9, 0.7, 0.3, 1],
    tendon_name_format: str = "tendon_{i}_{j}"
) -> ModelBuilder
```

## Examples to Run

```bash
# Unit tests
python tools/test_model_builder.py
python tools/test_networkx_tendons.py

# Examples
python tools/model_builder_examples.py 4
python testing/graph_tendons_example.py --example 1
python testing/graph_tendons_example.py --example 2
python testing/graph_tendons_example.py --example 3
```

## Documentation Files

| File | Description |
|------|-------------|
| `MODEL_BUILDER_REFERENCE.md` | Complete ModelBuilder API |
| `NETWORKX_TENDONS_GUIDE.md` | Comprehensive NetworkX guide |
| `ENHANCEMENTS_SUMMARY.md` | Overview of all features |
| `FIXED_SPATIAL_NETWORKX_IMPLEMENTATION.md` | Implementation details |

## Workflow

1. **Load model**: `builder = ModelBuilder.from_xml_path("model.xml")`
2. **Add sites**: `builder.add_site(name, body, pos)`
3. **Create graph**: `G = nx.Graph()` + `G.add_edge(...)`
4. **Generate tendons**: `builder.add_tendons_from_graph(G)`
5. **Compile**: `model = builder.compile()`
6. **Simulate**: `sim = MuJoCoSimulation.from_builder(builder)`

## Tips

- **Springlength = None**: Auto-compute from initial configuration
- **Stiffness = 0**: Pure constraint (no spring)
- **2-element springlength**: Different springs at each end `[L1, L2]`
- **Custom names**: Use edge attribute `name='custom'`
- **Color coding**: Use `rgba` to distinguish tendon types
- **Debugging**: Make sites large and colored: `size=0.05, rgba=[1,0,0,1]`

## Common Use Cases

| Use Case | Graph Type | Notes |
|----------|------------|-------|
| Stabilization | Complete graph | All-to-all connections |
| Cable routing | Path graph | Linear sequence |
| Cross-bracing | Grid + diagonals | Structural support |
| Radial support | Star graph | Central hub |
| Circular constraint | Cycle graph | Closed loop |

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| "NetworkX not found" | `pip install networkx` |
| "Site not found" | Ensure sites created before graph generation |
| Tendons too stiff | Reduce stiffness or increase damping |
| Tendons unstable | Increase damping |
| Wrong rest length | Set `springlength` explicitly |

---

**See full documentation in:** `tools/NETWORKX_TENDONS_GUIDE.md`
