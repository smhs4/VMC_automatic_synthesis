#!/usr/bin/env python3
"""
Test script for exploring mesh loading and ghost/shadow body setup.

This script tests:
1. Loading mesh files (STL) using ModelBuilder.add_mesh()
2. Creating a target body with mesh geometry
3. Creating a ghost body that follows the target (no collision)
4. Attaching tendons between ghost body and other objects

Run with: mjpython testing/mesh_ghost_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mujoco_sim_template import (
    MuJoCoSimulation, ModelBuilder, quat_from_euler
)
import numpy as np
import networkx as nx
import math

# ============================================================================
# Test 1: Basic mesh loading with inline XML
# ============================================================================
def test_basic_mesh():
    """Test basic mesh loading using ModelBuilder."""
    print("\n=== Test 1: Basic Mesh Loading ===")
    
    # Base XML with minimal setup
    base_xml = """
    <mujoco model="mesh_test">
        <option gravity="0 0 -9.81"/>
        <worldbody>
            <light cutoff="100" diffuse=".8 .8 .8" dir="-0.5 0 -1" pos="0 0 3"/>
            <geom name="floor" type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
        </worldbody>
    </mujoco>
    """
    
    G = nx.DiGraph()
    builder = ModelBuilder(G)
    builder.from_xml_string(base_xml)
    
    # Add mesh from file
    # STL is in millimeters (~70mm), scale to meters: 0.001 = 7cm, 0.002 = 14cm
    mesh_path = "/Users/hassanshahristani/Documents/IIB/4th_year_project/meshes/tetrahedron.stl"
    mesh_scale = 0.001  # Convert mm to m (results in ~7cm object)
    
    try:
        builder.add_mesh("tetra_mesh", file=mesh_path, scale=[mesh_scale, mesh_scale, mesh_scale])
        print(f"  ✓ Mesh loaded from: {mesh_path}")
        print(f"  ✓ Scale: {mesh_scale} (mesh is ~{70*mesh_scale*100:.1f}cm)")
    except Exception as e:
        print(f"  ✗ Failed to load mesh: {e}")
        return None
    
    # Add body using the mesh
    try:
        builder.add_body("tetra_body", 
                        pos=[0, 0, 0.5],
                        geom_type="mesh",
                        geom_mesh="tetra_mesh",
                        geom_rgba=[0, 1, 0, 1],
                        free_joint=True)  # 500g - reasonable for a small object
        print("  ✓ Body with mesh geometry created (mass=0.5kg)")
    except Exception as e:
        print(f"  ✗ Failed to create body with mesh: {e}")
        return None
    
    # Build simulation
    try:
        sim = MuJoCoSimulation.from_builder(builder)
        print("  ✓ Simulation compiled successfully")
        return sim
    except Exception as e:
        print(f"  ✗ Failed to compile simulation: {e}")
        return None


# ============================================================================
# Test 2: Ghost body setup (target + ghost that follows)
# ============================================================================
def test_ghost_setup():
    """Test ghost body setup without mesh (using boxes)."""
    print("\n=== Test 2: Ghost Body Setup (Boxes) ===")
    
    base_xml = """
    <mujoco model="ghost_test">
        <option gravity="0 0 -9.81"/>
        <worldbody>
            <light cutoff="100" diffuse=".8 .8 .8" dir="-0.5 0 -1" pos="0 0 3"/>
            <geom name="floor" type="plane" size="5 5 0.1" rgba=".9 .9 .9 1"/>
        </worldbody>
    </mujoco>
    """
    
    G = nx.DiGraph()
    builder = ModelBuilder(G)
    builder.from_xml_string(base_xml)
    
    # Add platform
    builder.add_body("platform", pos=[0, 0, 0.05],
                    geom_type="box", geom_size=[0.3, 0.3, 0.05],
                    free_joint=False, geom_rgba=[0.5, 0.5, 0.5, 1], mass=10)
    
    # Add a lifting post with a site
    builder.add_body("post", pos=[0.5, 0, 0.3],
                    geom_type="cylinder", geom_size=[0.02, 0.3],
                    free_joint=False, geom_rgba=[0.3, 0.3, 0.3, 1], mass=1)
    builder.add_site("post_site", body_name="post", pos=[0, 0, 0.3], size=0.02)
    
    # Add target body (the real physical object)
    builder.add_body("target", pos=[0, 0, 0.2],
                    geom_type="box", geom_size=[0.05, 0.05, 0.05],
                    free_joint=True, geom_rgba=[0, 1, 0, 1], mass=0.3)
    
    # Add ghost body (follows target, no collision, for tendon attachment)
    builder.add_body("ghost", pos=[0, 0, 0.2],
                    geom_type="box", geom_size=[0.05, 0.05, 0.05],
                    free_joint=True, geom_rgba=[1, 1, 1, 0.3], mass=0.1,
                    intersection=False)  # No collision!
    
    # Add site on ghost for tendon attachment
    builder.add_site("ghost_site", body_name="ghost", pos=[0, 0, 0.05], size=0.01)
    
    # Add tendon connecting ghost to post (this pulls the ghost up)
    builder.add_tendon("lift_tendon",
                      sites=["ghost_site", "post_site"],
                      stiffness=100.0,
                      damping=5.0,
                      springlength=0.1,
                      rgba=[1, 0, 0, 1])
    
    try:
        sim = MuJoCoSimulation.from_builder(builder)
        print("  ✓ Ghost setup compiled successfully")
        
        # Register joints for control
        target_joint = sim.reg_joint('target_joint', 'target_freejoint')
        ghost_joint = sim.reg_joint('ghost_joint', 'ghost_freejoint')
        
        # Control callback: ghost follows target
        def ghost_follow(sim):
            ghost_joint.qpos = target_joint.qpos
            ghost_joint.qvel = target_joint.qvel
        
        sim.set_control_callback(ghost_follow)
        print("  ✓ Ghost follow controller set up")
        
        return sim
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Test 3: Full mesh + ghost setup
# ============================================================================
def test_mesh_with_ghost():
    """Test mesh loading combined with ghost body setup."""
    print("\n=== Test 3: Mesh + Ghost Setup ===")
    
    base_xml = """
    <mujoco model="mesh_ghost_test">
        <option gravity="0 0 -9.81"/>
        <worldbody>
            <light cutoff="100" diffuse=".8 .8 .8" dir="-0.5 0 -1" pos="0 0 3"/>
            <geom name="floor" type="plane" size="5 5 0.1" rgba=".9 .9 .9 1"/>
        </worldbody>
    </mujoco>
    """
    
    G = nx.DiGraph()
    builder = ModelBuilder(G)
    builder.from_xml_string(base_xml)
    
    mesh_path = "/Users/hassanshahristani/Documents/IIB/4th_year_project/meshes/tetrahedron.stl"
    
    # Add platform
    builder.add_body("platform", pos=[0, 0, 0.05],
                    geom_type="box", geom_size=[0.3, 0.3, 0.05],
                    free_joint=False, geom_rgba=[0.5, 0.5, 0.5, 1], mass=10)
    
    # Add lifting post
    builder.add_body("post", pos=[0.5, 0, 0.5],
                    geom_type="cylinder", geom_size=[0.02, 0.5],
                    free_joint=False, geom_rgba=[0.3, 0.3, 0.3, 1], mass=1)
    builder.add_site("post_site", body_name="post", pos=[0, 0, 0.5], size=0.02)
    
    # Try to add mesh
    try:
        builder.add_mesh("tetra_mesh", file=mesh_path, scale=[0.002, 0.002, 0.002])
        print(f"  ✓ Mesh loaded")
        
        # Add target with mesh
        builder.add_body("target", pos=[0, 0, 0.3],
                        geom_type="mesh",
                        geom_mesh="tetra_mesh",
                        free_joint=True, geom_rgba=[0, 1, 0, 1])
        print("  ✓ Target body with mesh created")
        
        # Add ghost with mesh (no collision)
        builder.add_body("ghost", pos=[0, 0, 0.3],
                        geom_type="mesh",
                        geom_mesh="tetra_mesh",
                        free_joint=True, geom_rgba=[1, 1, 1, 0.3],
                        intersection=False)
        print("  ✓ Ghost body with mesh created")
        
    except Exception as e:
        print(f"  ! Mesh failed, falling back to boxes: {e}")
        # Fallback to boxes
        builder.add_body("target", pos=[0, 0, 0.3],
                        geom_type="box", geom_size=[0.05, 0.05, 0.05],
                        free_joint=True, geom_rgba=[0, 1, 0, 1], mass=0.3)
        builder.add_body("ghost", pos=[0, 0, 0.3],
                        geom_type="box", geom_size=[0.05, 0.05, 0.05],
                        free_joint=True, geom_rgba=[1, 1, 1, 0.3], mass=0.1,
                        intersection=False)
    
    # Add site on ghost for tendon
    builder.add_site("ghost_site", body_name="ghost", pos=[0, 0, 0], size=0.01)
    builder.add_site("target_site", body_name="target", pos=[0, 0, 0], size=0.01)
    
    # Add lifting tendon
    builder.add_tendon("lift_tendon",
                      sites=["ghost_site", "post_site"],
                      stiffness=50.0,
                      damping=5.0,
                      springlength=[0.2,0.2],
                      rgba=[1, 0, 0, 1])
    
    try:
        sim = MuJoCoSimulation.from_builder(builder)
        print("  ✓ Simulation compiled")
        
        target_joint = sim.reg_joint('target_joint', 'target_freejoint')
        ghost_joint = sim.reg_joint('ghost_joint', 'ghost_freejoint')
        
        def ghost_follow(sim):
            ghost_joint.qpos = target_joint.qpos
            ghost_joint.qvel = target_joint.qvel
        
        sim.set_control_callback(ghost_follow)
        return sim
    except Exception as e:
        print(f"  ✗ Failed to compile: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Test 4: Different mesh scales
# ============================================================================
def test_mesh_scales():
    """Test different mesh scale factors to find the right size."""
    print("\n=== Test 4: Mesh Scale Exploration ===")
    
    base_xml = """
    <mujoco model="scale_test">
        <option gravity="0 0 -9.81"/>
        <worldbody>
            <light cutoff="100" diffuse=".8 .8 .8" dir="-0.5 0 -1" pos="0 0 3"/>
            <geom name="floor" type="plane" size="5 5 0.1" rgba=".9 .9 .9 1"/>
        </worldbody>
    </mujoco>
    """
    
    mesh_path = "/Users/hassanshahristani/Documents/IIB/4th_year_project/meshes/tetrahedron.stl"
    
    # Test different scales
    scales_to_test = [
        [0.001, 0.001, 0.001],  # mm to m
        [0.01, 0.01, 0.01],    # cm to m
        [0.1, 0.1, 0.1],
        [1.0, 1.0, 1.0],
    ]
    
    for i, scale in enumerate(scales_to_test):
        G = nx.DiGraph()
        builder = ModelBuilder(G)
        builder.from_xml_string(base_xml)
        
        try:
            builder.add_mesh(f"tetra_{i}", file=mesh_path, scale=scale)
            builder.add_body(f"tetra_body_{i}", 
                           pos=[i * 0.3, 0, 0.5],
                           geom_type="mesh",
                           geom_mesh=f"tetra_{i}",
                           geom_rgba=[0.2 + i*0.2, 0.8 - i*0.15, 0.3, 1],
                           free_joint=True,
                           mass=0.5)
            sim = MuJoCoSimulation.from_builder(builder)
            print(f"  ✓ Scale {scale} works")
            sim.close()
        except Exception as e:
            print(f"  ✗ Scale {scale} failed: {e}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test mesh loading and ghost setup")
    parser.add_argument("--test", type=int, default=0, 
                       help="Test number to run (0=all, 1=basic mesh, 2=ghost boxes, 3=mesh+ghost, 4=scales)")
    parser.add_argument("--run", action="store_true", help="Run simulation viewer")
    args = parser.parse_args()
    
    sim = None
    
    if args.test == 0 or args.test == 1:
        sim = test_basic_mesh()
        
    if args.test == 0 or args.test == 2:
        sim = test_ghost_setup()
        
    if args.test == 0 or args.test == 3:
        sim = test_mesh_with_ghost()
        
    if args.test == 0 or args.test == 4:
        test_mesh_scales()
    
    # Run the last successful simulation if requested
    if args.run and sim is not None:
        print("\n=== Running Simulation ===")
        print("  - The green object is the 'target' (physical)")
        print("  - The transparent object is the 'ghost' (no collision, follows target)")
        print("  - The red tendon connects ghost to the post")
        print("  - The ghost transmits tendon forces to the target indirectly")
        print("  - Gravity IS working - watch the object fall!")
        sim.run(passive=True, realtime_speed=1.0)  # 1.0 = real-time speed
    elif sim is not None:
        print("\nRun with --run flag to view simulation")
        sim.close()
