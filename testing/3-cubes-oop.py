#!/usr/bin/env python3
"""
Example: 3-Cubes simulation using the OOP template with ModelBuilder.

This demonstrates:
1. Using ModelBuilder to load and enhance an existing model
2. Adding custom sites dynamically
3. Creating additional tendons programmatically
4. Clean OOP control interface
"""

import sys
import os
# Add parent directory to path to import from tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mujoco_sim_template import (
    MuJoCoSimulation, ModelBuilder,
    quat_mul, quat_conj, quat_to_axis_angle
)
import numpy as np
import argparse
import networkx as nx


def main():
    G = nx.Graph()
    # Parse arguments
    parser = argparse.ArgumentParser(description="3-Cubes simulation with OOP template + ModelBuilder")
    parser.add_argument("--passive", action="store_true", 
                        help="Run with passive viewer")
    parser.add_argument("--add-features", action="store_true",
                        help="Add extra sites and tendons dynamically")
    args = parser.parse_args()
    
    # Option 1: Use ModelBuilder to enhance the model
    if args.add_features:
        print("Building enhanced model with additional features...")
        builder = ModelBuilder(G)
        builder.from_xml_path("testing/model.xml")
        
        # Add extra monitoring sites
        print("  Adding monitoring sites...")
        builder.add_site("red_center", "red_box", pos=[0, 0, 0], 
                        rgba=[1, 0, 0, 0.5], size=0.03)
        builder.add_site("green_center", "green_box", pos=[0, 0, 0],
                        rgba=[0, 1, 0, 0.5], size=0.03)
        builder.add_site("blue_center", "blue_box", pos=[0, 0, 0],
                        rgba=[0, 0, 1, 0.5], size=0.03)
        
        
        print(" Adding sites on box faces")
        # Find box sizes from model
        size = 0.4
        
        face_offsets = [
            [0, size, 0],
            [0, 0, size],
            [0, -size, 0],
            [0, 0, -size],
        ]
        colors = {"red_box": [1, 0, 0, 0.5],
                  "green_box": [0, 1, 0, 0.5]}
        for box_name in ["red_box", "green_box"]:
            for i, offset in enumerate(face_offsets):
                site_name = f"{box_name}_face_{i}"
                builder.add_site(site_name, box_name, pos=offset,
                                 rgba=colors[box_name], size=0.02)
        # blue box radius
        radius = 0.4
        points = [
            [radius, radius, 0],
            [radius, 0, radius],
            [radius, -radius, 0],
            [radius, 0, -radius],
            [-radius, radius, 0],
            [-radius, 0, radius],
            [-radius, -radius, 0],
            [-radius, 0, -radius],
        ]

        theta_pos = np.deg2rad(30.0)
        theta_neg = np.deg2rad(-30.0)

        def rotate_y_about_x(point, angle, cx):
            x, y, z = point
            xt = x - cx
            x_r = np.cos(angle) * xt + np.sin(angle) * z
            z_r = -np.sin(angle) * xt + np.cos(angle) * z
            return [x_r + cx, y, z_r]

        for i in range(4):
            points[i] = rotate_y_about_x(points[i], theta_pos, radius)

        for i in range(4, 8):
            points[i] = rotate_y_about_x(points[i], theta_neg, -radius)
        for i, point in enumerate(points):
            builder.add_site(f"blue_box_point_{i}", "dummy", pos=point,
                             rgba=[0, 0, 1, 0.5], size=0.02)
            if i < 4:
                face_site = f"green_box_face_{i}"
                tendon_name = f"tendon_green_blue_{i}"
            else:
                face_site = f"red_box_face_{i-4}"
                tendon_name = f"tendon_red_blue_{i-4}"
        
            G.add_edge(f'blue_box_point_{i}', f'{face_site}')
        # Add an extra cross-tendon for stability
        print("  Adding cross-stabilization tendons...")
        builder.add_tendons_from_graph(G, default_stiffness=7000.0,
                                       default_damping=50.0,
                                       default_springlength=[0, 0],
                                       default_rgba=[1, 1, 0, 0.5],)

        builder.add_tendon("tendon_red_green_down", sites=["red_box_face_3", "green_box_face_3"], 
                           stiffness=5000.0, damping=150.0, springlength=[0.75, 0.75])
        # builder.add_tendon("tendon_red_green_far", sites=["red_box_face_0", "green_box_face_0"], 
        #                    stiffness=4000.0, damping=150.0, springlength=[0.8, 0.8])
        # builder.add_tendon("tendon_red_green_near", sites=["red_box_face_2", "green_box_face_2"], 
        #                    stiffness=4000.0, damping=150.0, springlength=[0.8, 0.8])
        
        
        # add a non-intersecting object
        print("  Adding extra body...")
        builder.add_body("dummy_green", pos=[0.4, 0, 0],
                         geom_type="box", geom_size=[0.3, 0.02, 0.02],
                          intersection=False, joints=[[0, 0, 1],], free_joint=False, geom_rgba=[0,1,0,0.3], mass=100)
        builder.add_body("dummy_red", pos=[-0.4, 0, 0],
                         geom_type="box", geom_size=[0.3, 0.02, 0.02],
                          intersection=False, joints=[[0, 0, 1]], free_joint=False, geom_rgba=[1,0,0,0.3], mass=100)

        builder.add_body("dummy_dum_dum", pos=[0, 0, 1.0],
                         geom_type="box", geom_size=[0.02, 0.02, 0.02],
                         parent="crane_base",
                          intersection=False, joints=[], free_joint=False, geom_rgba=[0.5,0.5,0.5,0.3], mass=10)
        builder.add_site("dummy_green_site", "dummy_green", pos=[0, 0, 0], size=0.01)
        builder.add_site("dummy_red_site", "dummy_red", pos=[0, 0, 0], size=0.01)
        builder.add_tendon("tendon_dummy2_green", sites=["dummy_green_site", "green_center"],
                           stiffness=5000.0, damping=0, springlength=[0,0.05])
        builder.add_tendon("tendon_dummy2_red", sites=["dummy_red_site", "red_center"],
                           stiffness=5000.0, damping=0, springlength=[0,0.05])
        # Compile enhanced model
        print("  Compiling enhanced model...")
        sim = MuJoCoSimulation.from_builder(builder)
        
    else:
        # Option 2: Use model as-is
        sim = MuJoCoSimulation(xml_path="testing/model.xml")
    
    # Register entities (works with both standard and enhanced models)
    crane = sim.reg_joint("crane", "crane_base__up_down")
    lift = sim.reg_actuator("lift", "lift")

    dummy_x = sim.reg_joint("dummy_x", "dummy__tx")
    dummy_y = sim.reg_joint("dummy_y", "dummy__ty")
    dummy_z = sim.reg_joint("dummy_z", "dummy__tz")
    # dummy = sim.reg_joint("dummy", "dummy_joint")
    target = sim.reg_joint("target", "blue_box__blue_joint")
    dummy_red_z = sim.reg_joint("dummy_red", "dummy_red_joint_001") if args.add_features else None
    #dummy_red_y = sim.reg_joint("dummy_red", "dummy_red_joint_010") if args.add_features else None
    dummy_green_z = sim.reg_joint("dummy_green", "dummy_green_joint_001") if args.add_features else None
    #dummy_green_y = sim.reg_joint("dummy_green", "dummy_green_joint_010") if args.add_features else None


    red_box = sim.reg_body("red_box", "red_box")
    green_box = sim.reg_body("green_box", "green_box")
    blue_box = sim.reg_body("blue_box", "blue_box")
    
    # Control parameters
    target_height = 1.5
    t_rise = 3.0
    Kp = 20.0
    Kd = 50.0
    
    # Define control
    def control(sim: MuJoCoSimulation):
        # Dummy follows target (blue box)
        dummy_x.qpos = target.qpos[0]
        dummy_y.qpos = target.qpos[1]
        dummy_z.qpos = target.qpos[2]
        dummy_x.qvel = target.qvel[0]
        dummy_y.qvel = target.qvel[1]
        dummy_z.qvel = target.qvel[2]

        

        dummy_red_z.qpos = red_box.pos[2]
        # dummy_red_z.qvel = target.qvel[2]
        

        dummy_green_z.qpos = green_box.pos[2]
        # dummy_green_z.qvel = target.qvel[2]
        
        # Crane trajectory
        t = sim.time
        if 0.5 < t < 0.5 + t_rise:
            alpha = 0.5 - 0.5*np.cos(np.pi * (t - 0.5) / t_rise)
            crane_target = crane.qpos[0]*(1 - alpha) + target_height*alpha
        elif t <= 0.5:
            crane_target = 0.0
        else:
            crane_target = target_height
        
        lift.ctrl = crane_target


        
        # Orientation synchronization (optional - uncomment to enable)
        # qA = red_box.quat
        # qB = green_box.quat
        # q_err = quat_mul(quat_conj(qA), qB)
        # ang_err = quat_to_axis_angle(q_err)
        # wA = red_box.angular_vel
        # wB = green_box.angular_vel
        # w_err = wB + wA / 2
        # tau = -Kp*ang_err - Kd*w_err
        # red_box.add_torque(-tau)
        # green_box.add_torque(tau)
        
        # Print status periodically
        if sim.time % 2.0 < sim.model.opt.timestep:
            print(f"t={sim.time:.1f}s: "
                  f"Crane={crane.qpos[0]:.3f}m, "
                  f"Red z={red_box.pos[2]:.3f}m, "
                  f"Green z={green_box.pos[2]:.3f}m, "
                  f"Blue z={blue_box.pos[2]:.3f}m")
            
    # Run simulation
    print("\nStarting simulation...")
    if args.add_features:
        print("Running with enhanced model (extra sites and tendons)")
    else:
        print("Running with standard model")
    print("Press ESC to exit\n")
    
    sim.run(
        control_callback=control,
        passive=args.passive,
        viewer_distance=5.0,
        viewer_lookat=np.array([0.0, 0.0, 0.25]),
        realtime_speed=1
    )


if __name__ == "__main__":
    main()
