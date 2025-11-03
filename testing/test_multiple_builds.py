#!/usr/bin/env python3
"""
Test that we can build multiple models in sequence without errors.
This verifies that ModelBuilder and MjSpec cleanup is working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mujoco_sim_template import MuJoCoSimulation, ModelBuilder
import gc


def test_multiple_builds():
    """Test building multiple models in sequence."""
    
    print("Testing multiple model builds...")
    print("=" * 60)
    
    for i in range(5):
        print(f"\nBuild {i+1}/5")
        print("-" * 40)
        
        try:
            # Create fresh builder
            builder = ModelBuilder(None)
            builder.from_xml_path("Sciurus17_mujoco_sim_example/URDFs/sciurus17_description/urdf/sciurus17.xml")
            
            # Add some bodies
            builder.add_body("test_box", pos=[0, 0, 0.5], 
                           geom_type="box", geom_size=[0.05, 0.05, 0.05],
                           geom_rgba=[1, 0, 0, 1], mass=0.1)
            
            # Build simulation
            sim = MuJoCoSimulation.from_builder(builder)
            
            print(f"  ✓ Model built successfully")
            print(f"  Model has {sim.model.nbody} bodies")
            
            # Clean up
            del builder
            sim.close()
            del sim
            gc.collect()
            
            print(f"  ✓ Cleanup complete")
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 60)
    print("✓ All builds successful! Cleanup is working correctly.")
    return True


if __name__ == "__main__":
    success = test_multiple_builds()
    sys.exit(0 if success else 1)
