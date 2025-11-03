#!/usr/bin/env python3
"""
Quick test to verify ModelBuilder functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mujoco_sim_template import MuJoCoSimulation, ModelBuilder
import numpy as np


def test_basic_model_building():
    """Test basic model building operations."""
    print("Testing ModelBuilder...")
    
    # Minimal scene
    xml = """
    <mujoco>
        <option timestep="0.002"/>
        <worldbody>
            <light pos="0 0 3"/>
            <geom type="plane" size="2 2 0.1"/>
            <body name="boxA" pos="0 0 0.5">
                <joint type="free"/>
                <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
            </body>
            <body name="boxB" pos="0.5 0 0.5">
                <joint type="free"/>
                <geom type="box" size="0.1 0.1 0.1" rgba="0 1 0 1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    
    # Create builder
    builder = ModelBuilder()
    builder.from_xml_string(xml)
    
    # Add sites
    print("  ✓ Adding sites...")
    builder.add_site("siteA", "boxA", pos=[0.1, 0, 0])
    builder.add_site("siteB", "boxB", pos=[-0.1, 0, 0])
    
    # Add tendon
    print("  ✓ Adding tendon...")
    builder.add_tendon("spring", sites=["siteA", "siteB"], 
                      stiffness=100.0, damping=5.0)
    
    # Add new body
    print("  ✓ Adding new body...")
    builder.add_body("boxC", pos=[0.25, 0.25, 0.5],
                     geom_type="sphere", geom_size=[0.08],
                     geom_rgba=[0, 0, 1, 1])
    
    # Compile
    print("  ✓ Compiling...")
    sim = MuJoCoSimulation.from_builder(builder)
    
    # Register entities
    print("  ✓ Registering entities...")
    bodyA = sim.reg_body("A", "boxA")
    bodyB = sim.reg_body("B", "boxB")
    bodyC = sim.reg_body("C", "boxC")
    spring = sim.reg_tendon("spring", "spring")
    
    # Verify properties
    print("  ✓ Checking properties...")
    assert hasattr(bodyA, 'pos'), "Body should have pos property"
    assert hasattr(spring, 'length'), "Tendon should have length property"
    assert hasattr(spring, 'force'), "Tendon should have force property"
    
    # Run a few steps
    print("  ✓ Running simulation steps...")
    for _ in range(10):
        sim.step()
    
    # Check tendon properties
    print(f"  ✓ Tendon length: {spring.length:.3f}m")
    print(f"  ✓ Tendon force: {spring.force:.2f}N")
    print(f"  ✓ Body positions: A={bodyA.pos[2]:.3f}, B={bodyB.pos[2]:.3f}, C={bodyC.pos[2]:.3f}")
    
    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_basic_model_building()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
