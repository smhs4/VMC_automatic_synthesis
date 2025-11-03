#!/usr/bin/env python3
"""
Demo of ContactSensor functionality.

This script demonstrates how to use contact sensors to detect
contacts between bodies and measure contact forces.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mujoco_sim_template import MuJoCoSimulation, ModelBuilder, quat_from_euler
import numpy as np
import argparse


def create_demo_scene():
    """Create a simple scene with objects for contact detection."""
    
    # Start with empty scene
    builder = ModelBuilder()
    
    # Add floor
    builder.add_body(
        name="floor",
        pos=[0, 0, 0],
        geom_type="plane",
        geom_size=[2, 2, 0.1],
        geom_rgba=[0.9, 0.9, 0.9, 1],
        mass=0.0
    )
    
    # Add target block (green)
    builder.add_body(
        name="target_block",
        pos=[0.0, 0.0, 0.15],
        geom_type="box",
        geom_size=[0.05, 0.05, 0.05],
        geom_rgba=[0, 1, 0, 1],
        mass=0.1
    )
    
    # Add left manipulator (red)
    builder.add_body(
        name="left_arm",
        pos=[-0.1, 0.0, 0.2],
        geom_type="sphere",
        geom_size=[0.03],
        geom_rgba=[1, 0, 0, 1],
        mass=0.05
    )
    
    # Add right manipulator (blue)
    builder.add_body(
        name="right_arm",
        pos=[0.1, 0.0, 0.2],
        geom_type="sphere",
        geom_size=[0.03],
        geom_rgba=[0, 0, 1, 1],
        mass=0.05
    )
    
    # Add top pusher (yellow)
    builder.add_body(
        name="top_pusher",
        pos=[0.0, 0.0, 0.35],
        geom_type="cylinder",
        geom_size=[0.02, 0.05],
        geom_rgba=[1, 1, 0, 1],
        mass=0.05
    )
    
    return builder.build(timestep=0.01)


def run_demo(passive: bool = False, duration: float = 10.0):
    """Run the contact sensor demonstration."""
    
    # Create simulation
    print("Creating demo scene...")
    builder = ModelBuilder()
    sim = create_demo_scene()
    
    # Register bodies
    target = sim.reg_body("target", "target_block")
    left_arm = sim.reg_body("left", "left_arm")
    right_arm = sim.reg_body("right", "right_arm")
    top_pusher = sim.reg_body("top", "top_pusher")
    
    # Register contact sensor for the target block
    # Monitor contacts with all manipulators
    contact_sensor = sim.reg_contact_sensor(
        "target_contacts",
        "target_block",
        ["left_arm", "right_arm", "top_pusher"]
    )
    
    print("\nContact Sensor Demo")
    print("=" * 60)
    print("Target block (green) will be monitored for contacts with:")
    print("  - Left arm (red sphere)")
    print("  - Right arm (blue sphere)")
    print("  - Top pusher (yellow cylinder)")
    print("\nApplying forces to create contacts...")
    print("=" * 60)
    
    # Control callback to apply forces and print contact info
    step_count = [0]
    last_print = [0]
    
    def control(sim):
        step_count[0] += 1
        t = step_count[0] * sim.model.opt.timestep
        
        # Apply sinusoidal forces to manipulators to create varying contacts
        # Left arm moves right
        left_force = np.array([30.0 * np.sin(t * 0.5), 0, -5])
        left_arm.add_force(left_force)
        
        # Right arm moves left
        right_force = np.array([-30.0 * np.sin(t * 0.5), 0, -5])
        right_arm.add_force(right_force)
        
        # Top pusher pushes down
        top_force = np.array([0, 0, -20.0 * (1 + 0.5 * np.sin(t * 1.0))])
        top_pusher.add_force(top_force)
        
        # Print contact information every 0.5 seconds
        if step_count[0] - last_print[0] >= 50:
            last_print[0] = step_count[0]
            
            print(f"\nTime: {t:.2f}s")
            print("-" * 60)
            
            # Get all contacts
            contacts = contact_sensor.get_contacts()
            
            if contacts:
                print(f"Active contacts: {len(contacts)}")
                for contact in contacts:
                    print(f"  {contact.body_name:15s} - "
                          f"Normal: {contact.normal_force:6.2f} N, "
                          f"Total: {contact.total_force:6.2f} N")
                    print(f"    Position: [{contact.contact_pos[0]:6.3f}, "
                          f"{contact.contact_pos[1]:6.3f}, "
                          f"{contact.contact_pos[2]:6.3f}]")
                
                # Show total contact force
                total = contact_sensor.get_total_contact_force()
                print(f"  Total contact force: {total:.2f} N")
            else:
                print("No contacts detected")
            
            # Check specific bodies
            if contact_sensor.is_in_contact("left_arm"):
                force = contact_sensor.get_contact_force_by_body("left_arm")
                print(f"  ✓ Left arm in contact ({force:.2f} N)")
            
            if contact_sensor.is_in_contact("right_arm"):
                force = contact_sensor.get_contact_force_by_body("right_arm")
                print(f"  ✓ Right arm in contact ({force:.2f} N)")
            
            if contact_sensor.is_in_contact("top_pusher"):
                force = contact_sensor.get_contact_force_by_body("top_pusher")
                print(f"  ✓ Top pusher in contact ({force:.2f} N)")
    
    # Run simulation
    if passive:
        print("\nRunning in passive viewer mode...")
        sim.run(control_callback=control, passive=True, duration=duration)
    else:
        print("\nRunning in interactive viewer mode...")
        sim.visualize(control_callback=control, duration=duration)
    
    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contact Sensor Demo")
    parser.add_argument('--passive', action='store_true', 
                       help='Use passive viewer (non-interactive)')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Simulation duration in seconds')
    
    args = parser.parse_args()
    
    run_demo(passive=args.passive, duration=args.duration)
