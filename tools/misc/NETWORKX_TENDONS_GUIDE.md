# NetworkX-Based Tendon Network Generation

## Overview

The `ModelBuilder.add_tendons_from_graph()` method enables automatic generation of tendon networks from NetworkX graph structures. This is particularly useful for:

- Complex tendon networks with many connections
- Programmatically defined constraint systems
- Research simulations requiring reproducible tendon configurations
- Cable-driven robot models with systematic routing

## Installation

First, install NetworkX:

```bash
pip install networkx
```

## Basic Concept

**Nodes** represent sites (for spatial tendons) or joints (for fixed tendons).  
**Edges** represent tendons connecting them.  
**Edge attributes** define tendon properties (stiffness, damping, springlength, rgba, etc.).

## Quick Start

```python
import networkx as nx
from mujoco_sim_template import ModelBuilder

# Load base model
builder = ModelBuilder.from_xml_path("model.xml")

# Add sites (if not already in model)
builder.add_site(name="site1", body_name="box1", pos=[0, 0, 0])
builder.add_site(name="site2", body_name="box2", pos=[0, 0, 0])
builder.add_site(name="site3", body_name="box3", pos=[0, 0, 0])

# Create graph
G = nx.Graph()
G.add_edge('site1', 'site2', stiffness=200, springlength=0.5)
G.add_edge('site2', 'site3', stiffness=150, springlength=0.3)

# Generate tendons from graph
builder.add_tendons_from_graph(G, default_damping=2.0)

# Compile
model = builder.compile()
```

## Method Signature

```python
def add_tendons_from_graph(
    graph,                    # NetworkX graph
    site_prefix: str = "",    # Prefix for site names
    default_stiffness: float = 100.0,
    default_damping: float = 1.0,
    default_springlength: Optional[Union[float, List[float]]] = None,
    default_rgba: List[float] = [0.9, 0.7, 0.3, 1],
    tendon_name_format: str = "tendon_{i}_{j}"
) -> ModelBuilder:
```

## Parameters

### Graph Structure
- **graph**: NetworkX Graph or DiGraph
  - Nodes: Site names (or joint names for fixed tendons)
  - Edges: Tendon connections

### Site/Joint Naming
- **site_prefix**: String to prepend to node names
  - Example: `site_prefix="box_"` converts node `"center"` to site `"box_center"`

### Default Tendon Properties
Used when edge doesn't specify a property:

- **default_stiffness**: Spring stiffness (N/m)
- **default_damping**: Damping coefficient (N·s/m)
- **default_springlength**: Rest length (m), can be scalar or `[L1, L2]` for spatial
- **default_rgba**: Color `[R, G, B, A]`

### Naming
- **tendon_name_format**: Format string for auto-generated names
  - Available: `{i}`, `{j}` (node names)
  - Example: `"cable_{i}_to_{j}"`

## Edge Attributes

Edge attributes override defaults:

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `stiffness` | float | Spring constant (N/m) | `200.0` |
| `damping` | float | Damping coefficient | `1.5` |
| `springlength` | float or list | Rest length(s) | `0.5` or `[0.5, 0.6]` |
| `rgba` | list[4] | Color | `[1, 0, 0, 1]` |
| `type` | str | `'spatial'` or `'fixed'` | `'spatial'` |
| `name` | str | Custom tendon name | `'stabilizer_1'` |

### Spring Length Details

**Spatial tendons** support 2-element springlength:
- `springlength=0.5` → Equal spring at both ends
- `springlength=[0.5, 0.6]` → Different rest lengths at each end

**Fixed tendons** use scalar springlength.

## Examples

### Example 1: Triangle Network

```python
import networkx as nx

builder = ModelBuilder.from_xml_path("model.xml")

# Add sites
for name, body in [('s1', 'box1'), ('s2', 'box2'), ('s3', 'box3')]:
    builder.add_site(name=name, body_name=body, pos=[0, 0, 0], size=0.02)

# Create triangle
G = nx.Graph()
G.add_edge('s1', 's2', stiffness=200, springlength=0.3)
G.add_edge('s2', 's3', stiffness=150, springlength=0.4)
G.add_edge('s3', 's1', stiffness=180, springlength=0.35)

builder.add_tendons_from_graph(G, default_damping=2.0)
```

### Example 2: Grid with Cross-Bracing

```python
# 2x2 grid with diagonals
G = nx.Graph()

# Grid edges (stiff)
G.add_edge('s00', 's01', stiffness=300, springlength=0.4)
G.add_edge('s00', 's10', stiffness=300, springlength=0.4)
G.add_edge('s01', 's11', stiffness=300, springlength=0.4)
G.add_edge('s10', 's11', stiffness=300, springlength=0.4)

# Diagonals (soft, colored)
G.add_edge('s00', 's11', stiffness=150, 
           springlength=[0.5, 0.6], rgba=[1, 0.5, 0, 1])
G.add_edge('s01', 's10', stiffness=150, 
           springlength=[0.5, 0.6], rgba=[1, 0.5, 0, 1])

builder.add_tendons_from_graph(G, tendon_name_format="grid_{i}_{j}")
```

### Example 3: Complete Graph

```python
# Fully connected network
sites = ['center', 'north', 'south', 'east', 'west']
G = nx.complete_graph(sites)

# Set properties for all edges
for i, j in G.edges():
    G[i][j]['stiffness'] = 100
    G[i][j]['damping'] = 1.0
    G[i][j]['springlength'] = 0.5

builder.add_tendons_from_graph(G, site_prefix="arm_")
```

### Example 4: Custom Names and Properties

```python
G = nx.Graph()

# Each edge with unique properties and name
G.add_edge('top', 'bottom',
           name='vertical_stabilizer',
           stiffness=500,
           damping=5.0,
           springlength=0.2,
           rgba=[1, 0, 0, 1])

G.add_edge('left', 'right',
           name='horizontal_spring',
           stiffness=200,
           damping=2.0,
           springlength=0.3,
           rgba=[0, 1, 0, 1])

builder.add_tendons_from_graph(G)
# Will use custom names instead of tendon_name_format
```

### Example 5: Directed Graph (Order Matters)

```python
# DiGraph for specific tendon routing order
G = nx.DiGraph()
G.add_edge('start', 'waypoint1', stiffness=150)
G.add_edge('waypoint1', 'waypoint2', stiffness=150)
G.add_edge('waypoint2', 'end', stiffness=150)

builder.add_tendons_from_graph(G, default_springlength=0.4)
# Creates tendons in specified order
```

### Example 6: Algorithmic Graph Generation

```python
import networkx as nx

# Use NetworkX algorithms to generate structure
G = nx.random_geometric_graph(10, 0.3)  # Random spatial network
# Or: G = nx.ladder_graph(5)  # Ladder structure
# Or: G = nx.circular_ladder_graph(6)  # Circular ladder

# Rename nodes to match site names
mapping = {i: f"site_{i}" for i in G.nodes()}
G = nx.relabel_nodes(G, mapping)

# Set uniform properties
nx.set_edge_attributes(G, 200, 'stiffness')
nx.set_edge_attributes(G, 0.5, 'springlength')

builder.add_tendons_from_graph(G, default_damping=2.0)
```

## Advanced Usage

### Combining with Manual Tendon Creation

```python
# Automatic network
G = nx.Graph()
G.add_edge('s1', 's2', stiffness=100)
G.add_edge('s2', 's3', stiffness=100)
builder.add_tendons_from_graph(G)

# Plus manual tendons
builder.add_tendon(
    name="special_cable",
    sites=['s1', 's3'],
    stiffness=300,
    springlength=0.8,
    rgba=[1, 0, 0, 1]
)
```

### Multiple Networks

```python
# Structural network
G_struct = nx.Graph()
G_struct.add_edge('s1', 's2', stiffness=500)
builder.add_tendons_from_graph(G_struct, tendon_name_format="struct_{i}_{j}")

# Control network
G_control = nx.Graph()
G_control.add_edge('s3', 's4', stiffness=100)
builder.add_tendons_from_graph(G_control, tendon_name_format="ctrl_{i}_{j}")
```

### Dynamic Graph Modification

```python
# Build graph programmatically
G = nx.Graph()
for i in range(5):
    for j in range(i+1, 5):
        distance = compute_distance(i, j)
        if distance < threshold:
            G.add_edge(f'site{i}', f'site{j}',
                      stiffness=compute_stiffness(distance),
                      springlength=distance)

builder.add_tendons_from_graph(G)
```

## Common Patterns

### Star Topology
```python
G = nx.star_graph(['center', 'p1', 'p2', 'p3', 'p4'])
```

### Ring Topology
```python
G = nx.cycle_graph(['s1', 's2', 's3', 's4', 's5'])
```

### Grid Topology
```python
G = nx.grid_2d_graph(3, 3)  # 3x3 grid
G = nx.relabel_nodes(G, {(i,j): f's{i}{j}' for i,j in G.nodes()})
```

### Tree Structure
```python
G = nx.balanced_tree(r=2, h=3)  # Binary tree, height 3
```

## Workflow

### Typical Usage Flow

1. **Load base model**
   ```python
   builder = ModelBuilder.from_xml_path("robot.xml")
   ```

2. **Add sites** (if needed)
   ```python
   for name, body in site_locations:
       builder.add_site(name=name, body_name=body, pos=[0,0,0])
   ```

3. **Create graph**
   ```python
   G = nx.Graph()
   # Add edges with properties
   ```

4. **Generate tendons**
   ```python
   builder.add_tendons_from_graph(G, default_damping=2.0)
   ```

5. **Compile and simulate**
   ```python
   model = builder.compile()
   sim = MuJoCoSimulation.from_builder(builder)
   sim.run(passive=True)
   ```

## Tips & Best Practices

### Performance
- For large graphs (>100 edges), consider batching site creation first
- Use `default_*` parameters to avoid repetitive edge attributes

### Debugging
- Print graph info: `print(G.number_of_nodes(), G.number_of_edges())`
- Visualize: `nx.draw(G, with_labels=True)`
- Check edge data: `print(G.get_edge_data('s1', 's2'))`

### Reproducibility
- Use `nx.to_dict_of_dicts(G)` to save graph structure
- Load with `nx.from_dict_of_dicts(data)`
- Or use `nx.write_gml(G, "network.gml")` for persistence

### Error Handling
- Ensure all nodes correspond to existing sites/joints
- Check that site_prefix matches your naming convention
- Verify edge attributes have correct types (float for stiffness, etc.)

## See Also

- `ModelBuilder` class documentation
- `add_tendon()` method reference
- NetworkX documentation: https://networkx.org/
- Examples: `testing/graph_tendons_example.py`

## Testing

Run the examples:

```bash
python testing/graph_tendons_example.py --example 1  # Triangle
python testing/graph_tendons_example.py --example 2  # Grid
python testing/graph_tendons_example.py --example 3  # Custom properties
```
