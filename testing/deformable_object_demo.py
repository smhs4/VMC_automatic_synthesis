#!/usr/bin/env python3
"""
Demonstration of deformable objects in MuJoCo.

Shows how to create soft, squishy objects that can be grasped and deformed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mujoco_sim_template import MuJoCoSimulation, ModelBuilder
import numpy as np
import networkx as nx


def create_deformable_gripper_scene():
    """Create a scene with a deformable object and simple gripper."""
    
    # Create base model
    G = nx.Graph()  # Empty graph for ModelBuilder
    builder = ModelBuilder(G)
    
    # Define complete XML with deformable objects
    # Note: MuJoCo's MjSpec API doesn't support adding composites programmatically
    # They must be defined in the base XML
    base_xml = """
    <mujoco>
        <option timestep="0.002" gravity="0 0 -9.81"/>
        
        <visual>
            <headlight ambient="0.5 0.5 0.5"/>
        </visual>
        
        <asset>
            <texture name="grid" type="2d" builtin="checker" width="512" height="512" 
                     rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3"/>
            <material name="grid" texture="grid" texrepeat="1 1"/>
        </asset>
        
        <worldbody>
            <!-- Ground plane -->
            <geom name="floor" type="plane" size="2 2 0.1" material="grid"/>
            
            <!-- Light -->
            <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
            
            <!-- Deformable sponge-like object (moderately soft) -->
            <composite type="particle" count="5 5 5" spacing="0.015" offset="0 0 0.15">
                <geom size="0.008" rgba="1.0 0.8 0.2 0.7" mass="0.002"/>
                <tendon kind="main" stiffness="1000" damping="20"/>
                <skin texcoord="true" rgba="1.0 0.8 0.2 0.7" inflate="0.004"/>
            </composite>
            
            <!-- Deformable ball (stiffer) -->
            <composite type="particle" count="7 7 7" spacing="0.012" offset="0.3 0 0.1">
                <geom size="0.006" rgba="0.2 0.6 1.0 0.7" mass="0.001"/>
                <tendon kind="main" stiffness="1500" damping="15"/>
                <skin texcoord="true" rgba="0.2 0.6 1.0 0.7" inflate="0.003"/>
            </composite>
            
            <!-- Very soft gel-like object -->
            <composite type="particle" count="6 5 4" spacing="0.01" offset="-0.3 0 0.08">
                <geom size="0.005" rgba="0.2 1.0 0.4 0.6" mass="0.003"/>
                <tendon kind="main" stiffness="300" damping="30"/>
                <skin texcoord="true" rgba="0.2 1.0 0.4 0.6" inflate="0.002"/>
            </composite>
        </worldbody>
    </mujoco>
    """
    
    builder.from_xml_string(base_xml)
    
    # Add simple gripper jaws (rigid)
    builder.add_body(
        name="left_jaw",
        pos=[-0.1, -0.15, 0.15],
        geom_type="box",
        geom_size=[0.02, 0.01, 0.08],
        geom_rgba=[0.5, 0.5, 0.5, 1],
        free_joint=False,
        joints=[[0, 1, 0]]  # Can slide in Y direction
    )
    
    builder.add_body(
        name="right_jaw",
        pos=[-0.1, 0.15, 0.15],
        geom_type="box",
        geom_size=[0.02, 0.01, 0.08],
        geom_rgba=[0.5, 0.5, 0.5, 1],
        free_joint=False,
        joints=[[0, 1, 0]]  # Can slide in Y direction
    )
    
    return builder


def gripper_control(sim: MuJoCoSimulation):
    """Control callback to open/close gripper."""
    try:
        left_jaw = sim.joints.get('left_jaw')
        right_jaw = sim.joints.get('right_jaw')
        
        if left_jaw and right_jaw:
            t = sim.time
            
            # Gripper motion pattern
            if t < 2.0:
                # Open
                target_left = -0.15
                target_right = 0.15
            elif t < 4.0:
                # Close slowly
                progress = (t - 2.0) / 2.0
                target_left = -0.15 + progress * 0.12
                target_right = 0.15 - progress * 0.12
            elif t < 6.0:
                # Hold closed
                target_left = -0.03
                target_right = 0.03
            else:
                # Open again
                progress = min(1.0, (t - 6.0) / 2.0)
                target_left = -0.03 - progress * 0.12
                target_right = 0.03 + progress * 0.12
            
            # Simple position control with damping
            Kp = 500.0
            Kd = 50.0
            
            left_force = Kp * (target_left - left_jaw.qpos[0]) - Kd * left_jaw.qvel[0]
            right_force = Kp * (target_right - right_jaw.qpos[0]) - Kd * right_jaw.qvel[0]
            
            left_jaw.qfrc_applied = np.array([left_force])
            right_jaw.qfrc_applied = np.array([right_force])
    except Exception as e:
        pass  # Ignore errors if joints not registered yet


def main():
    """Run the deformable object demonstration."""
    
    print("Creating deformable gripper scene...")
    builder = create_deformable_gripper_scene()
    
    print("Compiling model...")
    sim = MuJoCoSimulation.from_builder(builder)
    
    # Register gripper joints
    try:
        sim.reg_joint('left_jaw', 'left_jaw_joint_0.01.00.0')
        sim.reg_joint('right_jaw', 'right_jaw_joint_0.01.00.0')
        print("Gripper joints registered")
    except Exception as e:
        print(f"Note: Gripper control disabled - {e}")
    
    print("\nRunning simulation...")
    print("You should see:")
    print("  - Yellow sponge (moderately deformable)")
    print("  - Blue ball (stiffer)")
    print("  - Green gel (very soft)")
    print("  - Grey gripper jaws that will squeeze the sponge")
    print("\nPress Ctrl+C to exit.")
    
    # Run simulation with gripper control
    sim.run(
        control_callback=gripper_control,
        passive=True,
        duration=None,  # Run until user closes
        realtime_speed=1.0,
        viewer_distance=1.0,
        viewer_lookat=np.array([0.0, 0.0, 0.15])
    )


if __name__ == "__main__":
    main()
