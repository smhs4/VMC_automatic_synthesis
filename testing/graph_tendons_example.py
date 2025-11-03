#!/usr/bin/env python3
"""
Example demonstrating automatic tendon network generation from NetworkX graphs.

This example shows how to:
1. Create a tendon network using NetworkX
2. Specify tendon properties via edge attributes
3. Generate spatial and fixed tendons automatically
"""

import sys
sys.path.append('/Users/hassanshahristani/Documents/IIB/4th_year_project/tools')

import networkx as nx
import numpy as np
from mujoco_sim_template import MuJoCoSimulation, ModelBuilder

# ============================================================================
# Example 1: Simple Triangle Network
# ============================================================================

def example_triangle_network():
    """Create a triangle tendon network connecting three boxes."""
    
    # Start with the base 3-cubes model
    xml_path = "/Users/hassanshahristani/Documents/IIB/4th_year_project/testing/model.xml"
    builder = ModelBuilder.from_xml_path(xml_path)
    
    # Add sites to the three boxes (one per box)
    builder.add_site(name="red_center", body_name="red_box", pos=[0, 0, 0], 
                     size=0.02, rgba=[1, 0, 0, 1])
    builder.add_site(name="green_center", body_name="green_box", pos=[0, 0, 0], 
                     size=0.02, rgba=[0, 1, 0, 1])
    builder.add_site(name="blue_center", body_name="blue_box", pos=[0, 0, 0], 
                     size=0.02, rgba=[0, 0, 1, 1])
    
    # Create a NetworkX graph representing the tendon network
    G = nx.Graph()
    
    # Add edges with custom properties
    # Each edge will become a tendon
    G.add_edge('red_center', 'green_center', 
               stiffness=200, damping=2.0, springlength=0.3)
    G.add_edge('green_center', 'blue_center', 
               stiffness=150, damping=1.5, springlength=0.4)
    G.add_edge('blue_center', 'red_center', 
               stiffness=180, damping=1.8, springlength=0.35)
    
    # Automatically generate tendons from the graph
    builder.add_tendons_from_graph(
        graph=G,
        site_prefix="",  # No prefix needed since sites already have full names
        tendon_name_format="tri_{i}_{j}"
    )
    
    # Compile and create simulation
    model = builder.compile()
    sim = MuJoCoSimulation.from_builder(builder)
    
    # Register tendons to inspect them
    t1 = sim.reg_tendon("tri_red_green", "tri_red_center_green_center")
    t2 = sim.reg_tendon("tri_green_blue", "tri_green_center_blue_center")
    t3 = sim.reg_tendon("tri_blue_red", "tri_blue_center_red_center")
    
    def control(sim):
        """Print tendon information."""
        print(f"\nTendon Lengths: {t1.length:.3f}, {t2.length:.3f}, {t3.length:.3f}")
        print(f"Tendon Forces: {t1.force:.3f}, {t2.force:.3f}, {t3.force:.3f}")
    
    # Run simulation
    print("=== Triangle Network Example ===")
    print("Three boxes connected in a triangle with tendons")
    sim.run(control_callback=control, passive=True, duration=5.0)


# ============================================================================
# Example 2: Grid Network with Multiple Spring Lengths
# ============================================================================

def example_grid_network():
    """Create a 2x2 grid of boxes with cross-bracing tendons."""
    
    xml_path = "/Users/hassanshahristani/Documents/IIB/4th_year_project/testing/model.xml"
    builder = ModelBuilder.from_xml_path(xml_path)
    
    # Add 4 boxes in a grid pattern
    positions = {
        'box_00': [-0.2, -0.2, 0.5],
        'box_01': [-0.2, 0.2, 0.5],
        'box_10': [0.2, -0.2, 0.5],
        'box_11': [0.2, 0.2, 0.5]
    }
    
    for name, pos in positions.items():
        builder.add_body(
            name=name,
            pos=pos,
            mass=0.1,
            geom_type='box',
            geom_size=[0.05, 0.05, 0.05],
            geom_rgba=[0.5, 0.5, 0.8, 1],
            free_joint=True
        )
        # Add site at center of each box
        builder.add_site(name=f"{name}_site", body_name=name, 
                        pos=[0, 0, 0], size=0.01)
    
    # Create graph with grid edges + diagonals
    G = nx.Graph()
    
    # Horizontal and vertical edges (stiffer)
    G.add_edge('box_00_site', 'box_01_site', stiffness=300, springlength=0.4)
    G.add_edge('box_00_site', 'box_10_site', stiffness=300, springlength=0.4)
    G.add_edge('box_01_site', 'box_11_site', stiffness=300, springlength=0.4)
    G.add_edge('box_10_site', 'box_11_site', stiffness=300, springlength=0.4)
    
    # Diagonal edges (softer, with 2-element springlength for spatial tendons)
    G.add_edge('box_00_site', 'box_11_site', 
               stiffness=150, springlength=[0.5, 0.6], rgba=[1, 0.5, 0, 1])
    G.add_edge('box_01_site', 'box_10_site', 
               stiffness=150, springlength=[0.5, 0.6], rgba=[1, 0.5, 0, 1])
    
    # Generate tendons from graph
    builder.add_tendons_from_graph(
        graph=G,
        default_damping=2.0,
        tendon_name_format="grid_{i}_{j}"
    )
    
    # Compile and run
    model = builder.compile()
    sim = MuJoCoSimulation.from_builder(builder)
    
    print("\n=== Grid Network Example ===")
    print("2x2 grid of boxes with cross-bracing tendons")
    print("Orange diagonals have different spring properties")
    sim.run(passive=True, duration=5.0)


# ============================================================================
# Example 3: Custom Graph Properties
# ============================================================================

def example_custom_properties():
    """Demonstrate using NetworkX node/edge attributes for tendon properties."""
    
    xml_path = "/Users/hassanshahristani/Documents/IIB/4th_year_project/testing/model.xml"
    builder = ModelBuilder.from_xml_path(xml_path)
    
    # Create sites on the existing boxes
    sites = {
        'r_top': ('red_box', [0, 0, 0.1]),
        'r_bottom': ('red_box', [0, 0, -0.1]),
        'g_top': ('green_box', [0, 0, 0.1]),
        'g_bottom': ('green_box', [0, 0, -0.1]),
        'b_top': ('blue_box', [0, 0, 0.1]),
        'b_bottom': ('blue_box', [0, 0, -0.1])
    }
    
    for site_name, (body, pos) in sites.items():
        builder.add_site(name=site_name, body_name=body, 
                        pos=pos, size=0.015, rgba=[1, 1, 0, 1])
    
    # Create directed graph with varying tendon properties
    G = nx.Graph()
    
    # Strong vertical stabilizers
    G.add_edge('r_top', 'r_bottom', 
               name='red_vertical',
               stiffness=500, damping=5.0, springlength=0.2,
               rgba=[1, 0, 0, 1])
    G.add_edge('g_top', 'g_bottom',
               name='green_vertical', 
               stiffness=500, damping=5.0, springlength=0.2,
               rgba=[0, 1, 0, 1])
    G.add_edge('b_top', 'b_bottom',
               name='blue_vertical',
               stiffness=500, damping=5.0, springlength=0.2,
               rgba=[0, 0, 1, 1])
    
    # Weak cross-connections
    G.add_edge('r_top', 'g_top',
               stiffness=100, damping=1.0, springlength=0.5,
               rgba=[0.5, 0.5, 0, 1])
    G.add_edge('g_top', 'b_top',
               stiffness=100, damping=1.0, springlength=0.5,
               rgba=[0, 0.5, 0.5, 1])
    
    # Generate all tendons from graph
    builder.add_tendons_from_graph(G)
    
    model = builder.compile()
    sim = MuJoCoSimulation.from_builder(builder)
    
    # Register all tendons
    red_v = sim.reg_tendon("red_v", "red_vertical")
    green_v = sim.reg_tendon("green_v", "green_vertical")
    blue_v = sim.reg_tendon("blue_v", "blue_vertical")
    
    def control(sim):
        """Monitor tendon states."""
        print(f"\nVertical Tendons - Lengths: R={red_v.length:.3f}, "
              f"G={green_v.length:.3f}, B={blue_v.length:.3f}")
        print(f"Vertical Tendons - Forces: R={red_v.force:.2f}, "
              f"G={green_v.force:.2f}, B={blue_v.force:.2f}")
    
    print("\n=== Custom Properties Example ===")
    print("Boxes with vertical stabilizers (strong) and horizontal links (weak)")
    sim.run(control_callback=control, passive=True, duration=5.0)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NetworkX Tendon Graph Examples")
    parser.add_argument('--example', type=int, choices=[1, 2, 3], default=1,
                       help='Which example to run (1=triangle, 2=grid, 3=custom)')
    args = parser.parse_args()
    
    if args.example == 1:
        example_triangle_network()
    elif args.example == 2:
        example_grid_network()
    elif args.example == 3:
        example_custom_properties()
