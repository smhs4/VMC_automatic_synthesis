#!/usr/bin/env python3
"""
Test script for the MJCF generator to understand its capabilities and XML output format.
This demonstrates various features of the mjcf_generator module.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from mjcf_generator import SceneBuilder, BodyTemplate


def test_basic_scene():
    """Test basic scene generation with lights, geoms, and bodies."""
    print("=== Test 1: Basic Scene ===")
    
    builder = SceneBuilder()
    
    # Add basic options
    builder.add_option(gravity=[0, 0, -9.81], timestep=0.001)
    
    # Add defaults
    builder.add_default("geom", friction=[1.0, 0.05, 0.01])
    builder.add_default("joint", damping=2)
    
    # Add assets
    builder.add_texture(
        name="grid", 
        type="2d", 
        builtin="checker", 
        rgb1=[0.1, 0.2, 0.3], 
        rgb2=[0.2, 0.3, 0.4], 
        width=300, 
        height=300
    )
    builder.add_material(name="grid_mat", texture="grid", texrepeat=[1, 1])
    
    # Add lights
    builder.add_light(pos=[0, 0, 3])
    
    # Add floor
    builder.add_geom(attrs={
        "name": "floor",
        "type": "plane", 
        "pos": [0, 0, -0.3],
        "size": [2, 2, 0.1],
        "material": "grid_mat"
    })
    
    # Add a simple box
    box_body = builder.add_body(
        name="test_box",
        pos=[0, 0, 0.3]
    )
    builder.add_freejoint(box_body)
    builder.add_geom(box_body, attrs={
        "name": "box_geom",
        "type": "box",
        "size": [0.2, 0.2, 0.2],
        "rgba": [1, 0, 0, 1]
    })
    
    xml_output = builder.to_string()
    print(xml_output)
    print("\n" + "="*50 + "\n")
    return xml_output


def test_body_template():
    """Test body template functionality."""
    print("=== Test 2: Body Template ===")
    
    # Create a reusable box template
    box_template = BodyTemplate(
        freejoint={},  # Empty dict means default freejoint
        geoms=[{
            "name": "{name}_geom",
            "type": "box",
            "size": [0.2, 0.2, 0.2],
            "rgba": "{rgba}",
            "friction": "{friction}"
        }],
        sites=[{
            "name": "{name}_top",
            "pos": [0, 0, 0.2],
            "size": 0.02,
            "rgba": [0, 1, 0, 1]
        }, {
            "name": "{name}_bottom",
            "pos": [0, 0, -0.2],
            "size": 0.02,
            "rgba": [1, 1, 0, 1]
        }]
    )
    
    builder = SceneBuilder()
    
    # Add basic setup
    builder.add_option(gravity=[0, 0, -9.81])
    builder.add_light(pos=[0, 0, 3])
    builder.add_geom(attrs={"name": "floor", "type": "plane", "size": [5, 5, 0.1]})
    
    # Instantiate multiple boxes using the template
    box_template.instantiate(
        builder,
        name="red_box",
        pos=[-0.5, 0, 0.3],
        mapping={
            "rgba": [1, 0, 0, 1],
            "friction": [0.9, 0.05, 0.01]
        }
    )
    
    box_template.instantiate(
        builder,
        name="blue_box",
        pos=[0.5, 0, 0.3],
        mapping={
            "rgba": [0, 0, 1, 1],
            "friction": [0.8, 0.04, 0.02]
        }
    )
    
    # Create a sphere template with different structure
    sphere_template = BodyTemplate(
        freejoint={},
        geoms=[{
            "name": "{name}_sphere",
            "type": "sphere",
            "size": 0.15,
            "rgba": "{color}"
        }],
        sites=[{
            "name": "{name}_center",
            "pos": [0, 0, 0],
            "size": 0.01
        }]
    )
    
    sphere_template.instantiate(
        builder,
        name="green_sphere",
        pos=[0, 0.7, 0.3],
        mapping={"color": [0, 1, 0, 1]}
    )
    
    xml_output = builder.to_string()
    print(xml_output)
    print("\n" + "="*50 + "\n")
    return xml_output


def test_tendons():
    """Test tendon generation capabilities."""
    print("=== Test 3: Tendons ===")
    
    builder = SceneBuilder()
    
    # Basic setup
    builder.add_option(gravity=[0, 0, -9.81])
    builder.add_light(pos=[0, 0, 3])
    builder.add_geom(attrs={"name": "floor", "type": "plane", "size": [3, 3, 0.1]})
    
    # Create two boxes with sites
    box1 = builder.add_body(name="box1", pos=[-0.5, 0, 0.3])
    builder.add_freejoint(box1)
    builder.add_geom(box1, attrs={
        "name": "box1_geom", "type": "box", "size": [0.1, 0.1, 0.1], "rgba": [1, 0, 0, 1]
    })
    builder.add_site(box1, attrs={"name": "box1_right", "pos": [0.15, 0, 0], "size": 0.02})
    
    box2 = builder.add_body(name="box2", pos=[0.5, 0, 0.3])
    builder.add_freejoint(box2)
    builder.add_geom(box2, attrs={
        "name": "box2_geom", "type": "box", "size": [0.1, 0.1, 0.1], "rgba": [0, 0, 1, 1]
    })
    builder.add_site(box2, attrs={"name": "box2_left", "pos": [-0.15, 0, 0], "size": 0.02})
    
    # Add a direct spatial tendon
    builder.add_spatial_tendon(
        name="connection",
        attrs={"rgba": [0, 1, 0, 1], "stiffness": 100, "damping": 10},
        path=[
            {"site": "box1_right"},
            {"site": "box2_left"}
        ]
    )
    
    xml_output = builder.to_string()
    print(xml_output)
    print("\n" + "="*50 + "\n")
    return xml_output


def test_adjacency_tendons():
    """Test adjacency matrix tendon generation."""
    print("=== Test 4: Adjacency Matrix Tendons ===")
    
    builder = SceneBuilder()
    
    # Basic setup
    builder.add_option(gravity=[0, 0, -9.81])
    builder.add_light(pos=[0, 0, 3])
    builder.add_geom(attrs={"name": "floor", "type": "plane", "size": [3, 3, 0.1]})
    
    # Create three bodies in a triangle formation
    positions = [(-0.5, -0.3, 0.3), (0.5, -0.3, 0.3), (0, 0.5, 0.3)]
    node_names = ["node_a", "node_b", "node_c"]
    site_mapping = {}
    
    for i, (name, pos) in enumerate(zip(node_names, positions)):
        body = builder.add_body(name=name, pos=pos)
        builder.add_freejoint(body)
        builder.add_geom(body, attrs={
            "name": f"{name}_geom",
            "type": "sphere",
            "size": 0.08,
            "rgba": [1, i*0.5, 0, 1]
        })
        site_name = f"{name}_site"
        builder.add_site(body, attrs={
            "name": site_name,
            "pos": [0, 0, 0],
            "size": 0.02
        })
        site_mapping[name] = site_name
    
    # Create adjacency matrix (triangle: all nodes connected)
    adjacency = [
        [0, 1, 1],  # node_a connects to node_b and node_c
        [0, 0, 1],  # node_b connects to node_c (symmetry handled automatically)
        [0, 0, 0]   # node_c (upper triangle only)
    ]
    
    # Generate tendons from adjacency matrix
    created_tendons = builder.add_spatial_from_adjacency(
        name_prefix="spring_",
        nodes=site_mapping,
        adjacency=adjacency,
        base_attrs={
            "rgba": [0, 1, 1, 1],
            "stiffness": 50,
            "damping": 5
        }
    )
    
    print(f"Created tendons: {created_tendons}")
    
    xml_output = builder.to_string()
    print(xml_output)
    print("\n" + "="*50 + "\n")
    return xml_output


def test_transforms():
    """Test transformation features."""
    print("=== Test 5: Transforms ===")
    
    # Create a template with an offset
    offset_template = BodyTemplate(
        body_attrs={"pos": [0.2, 0, 0]},  # Template has built-in offset
        freejoint={},
        geoms=[{
            "name": "{name}_geom",
            "type": "capsule",
            "size": [0.05, 0.3],
            "rgba": "{color}"
        }]
    )
    
    builder = SceneBuilder()
    builder.add_option(gravity=[0, 0, -9.81])
    builder.add_light(pos=[0, 0, 3])
    builder.add_geom(attrs={"name": "floor", "type": "plane", "size": [3, 3, 0.1]})
    
    # Instantiate with various transforms
    offset_template.instantiate(
        builder,
        name="original",
        pos=[0, 0, 0.5],
        mapping={"color": [1, 0, 0, 1]}
    )
    
    offset_template.instantiate(
        builder,
        name="translated",
        pos=[0, 0, 0.5],
        transform={"translate": [0, 1, 0]},
        mapping={"color": [0, 1, 0, 1]}
    )
    
    offset_template.instantiate(
        builder,
        name="rotated",
        pos=[0, 0, 0.5],
        transform={"rotate": {"axis": [0, 0, 1], "angle": 45}},
        mapping={"color": [0, 0, 1, 1]}
    )
    
    offset_template.instantiate(
        builder,
        name="reflected",
        pos=[0, 0, 0.5],
        transform={"reflect": ["y"]},
        mapping={"color": [1, 1, 0, 1]}
    )
    
    xml_output = builder.to_string()
    print(xml_output)
    print("\n" + "="*50 + "\n")
    return xml_output


def test_actuators():
    """Test actuator generation."""
    print("=== Test 6: Actuators ===")
    
    builder = SceneBuilder()
    builder.add_option(gravity=[0, 0, -9.81])
    builder.add_light(pos=[0, 0, 3])
    builder.add_geom(attrs={"name": "floor", "type": "plane", "size": [3, 3, 0.1]})
    
    # Create a body with a sliding joint
    slider_body = builder.add_body(name="slider", pos=[0, 0, 0.5])
    builder.add_joint(slider_body, attrs={
        "name": "slide_joint",
        "type": "slide",
        "axis": [0, 0, 1],
        "range": [-0.5, 1.0]
    })
    builder.add_geom(slider_body, attrs={
        "name": "slider_geom",
        "type": "box",
        "size": [0.1, 0.1, 0.1],
        "rgba": [0.5, 0.5, 1, 1]
    })
    
    # Add position actuator
    builder.add_position_actuator(
        name="lift",
        joint="slide_joint",
        kp=1000,
        kv=100,
        ctrlrange=[-0.5, 1.0]
    )
    
    xml_output = builder.to_string()
    print(xml_output)
    print("\n" + "="*50 + "\n")
    return xml_output


def save_test_outputs():
    """Save all test outputs to files for inspection."""
    print("=== Saving Test Outputs ===")
    
    tests = [
        ("basic_scene", test_basic_scene),
        ("body_template", test_body_template),
        ("tendons", test_tendons),
        ("adjacency_tendons", test_adjacency_tendons),
        ("transforms", test_transforms),
        ("actuators", test_actuators)
    ]
    
    output_dir = "/Users/hassanshahristani/Documents/IIB/4th_year_project/tools/test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        xml_content = test_func()
        
        output_file = os.path.join(output_dir, f"{test_name}.xml")
        with open(output_file, 'w') as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write(xml_content)
        
        print(f"Saved to {output_file}")
    
    print(f"\nAll test outputs saved to: {output_dir}")


if __name__ == "__main__":
    print("Testing MJCF Generator")
    print("="*50)
    
    # Run individual tests
    test_basic_scene()
    test_body_template()
    test_tendons()
    test_adjacency_tendons()
    test_transforms()
    test_actuators()
    
    # Save outputs for inspection
    save_test_outputs()
    
    print("All tests completed!")