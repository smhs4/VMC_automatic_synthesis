#!/usr/bin/env python3
"""
Practical example: Using MJCF Generator to create a scene for Sciurus17 robot testing.
This demonstrates how to use the generator for robotics applications.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from mjcf_generator import SceneBuilder, BodyTemplate


def create_manipulation_scene():
    """Create a scene with objects for robot manipulation testing."""
    
    print("Creating manipulation scene for robot testing...")
    
    builder = SceneBuilder()
    
    # Basic scene setup
    builder.add_option(gravity=[0, 0, -9.81], timestep=0.002)
    
    # Add defaults for realistic physics
    builder.add_default("geom", friction=[0.9, 0.05, 0.01], condim=4)
    builder.add_default("joint", damping=0.1, armature=0.01)
    
    # Create visual assets
    builder.add_texture(
        name="checker", 
        type="2d", 
        builtin="checker", 
        rgb1=[0.2, 0.3, 0.4], 
        rgb2=[0.3, 0.4, 0.5], 
        width=512, 
        height=512
    )
    builder.add_material(name="floor_mat", texture="checker", texrepeat=[5, 5], reflectance=0.1)
    
    # Lighting
    builder.add_light(pos=[2, 2, 4], dir=[-1, -1, -2])
    builder.add_light(pos=[-2, 1, 3], dir=[1, -0.5, -1])
    
    # Floor
    builder.add_geom(attrs={
        "name": "floor",
        "type": "plane",
        "pos": [0, 0, 0],
        "size": [3, 3, 0.1],
        "material": "floor_mat"
    })
    
    # Create a manipulable object template
    manipulable_object = BodyTemplate(
        freejoint={},
        geoms=[{
            "name": "{name}_geom",
            "type": "{shape}",
            "size": "{size}",
            "rgba": "{color}",
            "friction": [0.8, 0.05, 0.01]
        }],
        sites=[
            {"name": "{name}_grasp_top", "pos": [0, 0, "{grasp_height}"], "size": 0.01, "rgba": [1, 1, 0, 1]},
            {"name": "{name}_grasp_side", "pos": ["{grasp_offset}", 0, 0], "size": 0.01, "rgba": [0, 1, 1, 1]},
            {"name": "{name}_center", "pos": [0, 0, 0], "size": 0.005, "rgba": [1, 0, 1, 0.5]}
        ]
    )
    
    # Create various objects for manipulation
    objects = [
        {
            "name": "red_box",
            "pos": [0.4, -0.3, 0.15],
            "mapping": {
                "shape": "box",
                "size": [0.05, 0.05, 0.05],
                "color": [1, 0, 0, 1],
                "grasp_height": 0.06,
                "grasp_offset": 0.06
            }
        },
        {
            "name": "blue_cylinder",
            "pos": [0.5, 0, 0.1],
            "mapping": {
                "shape": "cylinder",
                "size": [0.03, 0.08],
                "color": [0, 0, 1, 1],
                "grasp_height": 0.09,
                "grasp_offset": 0.04
            }
        },
        {
            "name": "green_sphere",
            "pos": [0.3, 0.3, 0.125],
            "mapping": {
                "shape": "sphere",
                "size": 0.04,
                "color": [0, 1, 0, 1],
                "grasp_height": 0.05,
                "grasp_offset": 0.05
            }
        }
    ]
    
    # Instantiate objects
    for obj in objects:
        manipulable_object.instantiate(
            builder,
            name=obj["name"],
            pos=obj["pos"],
            mapping=obj["mapping"]
        )
    
    # Create a goal area
    goal_area = builder.add_body(name="goal_area", pos=[0.8, 0, 0.01])
    builder.add_geom(goal_area, attrs={
        "name": "goal_platform",
        "type": "box",
        "size": [0.15, 0.15, 0.01],
        "rgba": [1, 1, 0, 0.5],
        "contype": 0,  # No collision
        "conaffinity": 0
    })
    
    # Add some connecting tendons for interesting dynamics
    # Get all object center sites for tendon network
    object_sites = {obj["name"]: f"{obj['name']}_center" for obj in objects}
    
    # Create a simple spring network (triangle)
    adjacency = [
        [0, 1, 1],  # red_box connects to blue_cylinder and green_sphere
        [0, 0, 1],  # blue_cylinder connects to green_sphere
        [0, 0, 0]   # green_sphere (upper triangle matrix)
    ]
    
    created_tendons = builder.add_spatial_from_adjacency(
        name_prefix="connection_",
        nodes=object_sites,
        adjacency=adjacency,
        base_attrs={
            "rgba": [0.5, 0.5, 1, 0.7],
            "stiffness": 20,
            "damping": 2,
            "springlength": 0.2
        }
    )
    
    print(f"Created {len(created_tendons)} connecting tendons: {created_tendons}")
    
    return builder.to_string()


def create_tendon_network_scene():
    """Create a scene with a complex tendon network."""
    
    print("Creating complex tendon network scene...")
    
    builder = SceneBuilder()
    
    # Basic setup
    builder.add_option(gravity=[0, 0, -9.81], timestep=0.001)
    builder.add_light(pos=[0, 0, 4])
    builder.add_geom(attrs={"name": "floor", "type": "plane", "size": [2, 2, 0.1]})
    
    # Create a network of connected masses
    network_node = BodyTemplate(
        freejoint={},
        geoms=[{
            "name": "{name}_mass",
            "type": "sphere",
            "size": 0.02,
            "rgba": "{color}",
            "mass": 0.1
        }],
        sites=[{
            "name": "{name}_site",
            "pos": [0, 0, 0],
            "size": 0.005
        }]
    )
    
    # Create a 3x3 grid of masses
    grid_positions = []
    grid_names = []
    grid_sites = {}
    
    for i in range(3):
        for j in range(3):
            name = f"mass_{i}_{j}"
            pos = [-0.2 + i*0.2, -0.2 + j*0.2, 0.5]
            color = [i/2.0, j/2.0, 1.0, 1.0]
            
            network_node.instantiate(
                builder,
                name=name,
                pos=pos,
                mapping={"color": color}
            )
            
            grid_names.append(name)
            grid_sites[name] = f"{name}_site"
            grid_positions.append(pos)
    
    # Create adjacency matrix for nearest-neighbor connections
    adjacency = [[0 for _ in range(9)] for _ in range(9)]
    
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            # Connect to right neighbor
            if j < 2:
                right_idx = i * 3 + (j + 1)
                adjacency[idx][right_idx] = 1
            # Connect to bottom neighbor
            if i < 2:
                bottom_idx = (i + 1) * 3 + j
                adjacency[idx][bottom_idx] = 1
    
    # Generate spring network
    created_springs = builder.add_spatial_from_adjacency(
        name_prefix="spring_",
        nodes=grid_sites,
        adjacency=adjacency,
        base_attrs={
            "rgba": [0, 1, 0, 1],
            "stiffness": 100,
            "damping": 10,
            "springlength": 0.2
        }
    )
    
    print(f"Created {len(created_springs)} springs in the network")
    
    return builder.to_string()


def save_practical_examples():
    """Save practical examples to files."""
    
    examples = [
        ("manipulation_scene", create_manipulation_scene),
        ("tendon_network", create_tendon_network_scene)
    ]
    
    output_dir = "/Users/hassanshahristani/Documents/IIB/4th_year_project/tools/practical_examples"
    os.makedirs(output_dir, exist_ok=True)
    
    for name, func in examples:
        print(f"\nGenerating {name}...")
        xml_content = func()
        
        output_file = os.path.join(output_dir, f"{name}.xml")
        with open(output_file, 'w') as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write(xml_content)
        
        print(f"Saved to {output_file}")
        
        # Also show a snippet of the content
        print("\nGenerated XML snippet:")
        print("="*40)
        print(xml_content[:500] + "..." if len(xml_content) > 500 else xml_content)
        print("="*40)


if __name__ == "__main__":
    print("MJCF Generator - Practical Examples for Robotics")
    print("="*60)
    
    save_practical_examples()
    
    print("\nPractical examples demonstrate:")
    print("• Object creation for manipulation tasks")
    print("• Tendon networks for soft-body simulation") 
    print("• Template-based scene generation")
    print("• Integration with robot testing workflows")