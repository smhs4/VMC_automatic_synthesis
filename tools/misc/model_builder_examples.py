#!/usr/bin/env python3
"""
Examples of using ModelBuilder for dynamic model manipulation.

This demonstrates how to:
1. Add sites to existing bodies
2. Create tendons between sites
3. Add new bodies/objects dynamically
4. Combine everything in a simulation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mujoco_sim_template import MuJoCoSimulation, ModelBuilder
import numpy as np


# ==============================================================================
# Example 1: Adding Sites and Tendons to Existing Model
# ==============================================================================

def example1_add_tendons_to_robot():
    """
    Take an existing robot model and add tendon connections between body parts.
    """
    print("=" * 70)
    print("Example 1: Adding Sites and Tendons to Sciurus17 Robot")
    print("=" * 70)
    
    # Start with base model
    builder = ModelBuilder()
    xml_path = os.path.join(
        os.path.dirname(__file__),
        "../Sciurus17_mujoco_sim_example/URDFs/sciurus17_description/urdf/sciurus17_pickup.xml"
    )
    builder.from_xml_path(xml_path)
    
    # Add sites to robot hands for tendon attachment
    print("Adding sites to robot hands...")
    builder.add_site("r_hand_attach", "r_handA_link", 
                     pos=[0, 0, 0.05], rgba=[1, 0, 0, 1])
    builder.add_site("l_hand_attach", "l_handA_link", 
                     pos=[0, 0, 0.05], rgba=[0, 0, 1, 1])
    
    # Create tendon between hands
    print("Creating tendon between hands...")
    builder.add_tendon("hand_connection",
                      sites=["r_hand_attach", "l_hand_attach"],
                      stiffness=50.0,
                      damping=2.0,
                      rgba=[0.9, 0.7, 0.3, 1])
    
    # Compile and create simulation
    print("Compiling model...")
    sim = MuJoCoSimulation.from_builder(builder)
    
    # Register entities
    print("Registering entities...")
    tendon = sim.reg_tendon("hand_tendon", "hand_connection")
    r_hand = sim.reg_body("r_hand", "r_handA_link")
    l_hand = sim.reg_body("l_hand", "l_handA_link")
    
    # Monitor tendon properties
    def control(sim):
        if sim.time % 1.0 < sim.model.opt.timestep:  # Print every second
            print(f"t={sim.time:.2f}s: "
                  f"Tendon length={tendon.length:.3f}m, "
                  f"force={tendon.force:.2f}N")
    
    print("\nRunning simulation with gravity compensation...")
    print("Watch the tendon connecting the robot's hands!")
    print("-" * 70)
    
    # Run with gravity compensation
    def full_control(sim):
        # Apply gravity compensation first
        import mujoco
        model = sim.model
        data = sim.data
        
        acc_zero = np.zeros(model.nv)
        qacc_old = data.qacc.copy()
        data.qacc[:] = acc_zero
        mujoco.mj_rne(model, data, 0, data.qfrc_bias)
        data.qacc[:] = qacc_old
        
        if model.nu > 0:
            data.ctrl[:] = 0.0
            for i in range(model.nu):
                jnt_id = model.actuator_trnid[i, 0]
                if jnt_id >= 0 and jnt_id < model.njnt:
                    dof_adr = model.jnt_dofadr[jnt_id]
                    gear = model.actuator_gear[i, 0]
                    data.ctrl[i] = (data.qfrc_bias[dof_adr] / gear) if gear != 0 else data.qfrc_bias[dof_adr]
        
        # Monitor tendon
        control(sim)
    
    sim.run(control_callback=full_control, passive=True, duration=10.0)
    print("\n✓ Example 1 complete!\n")


# ==============================================================================
# Example 2: Adding New Objects
# ==============================================================================

def example2_add_objects():
    """
    Add new objects to a simple scene and connect them with tendons.
    """
    print("=" * 70)
    print("Example 2: Creating Objects and Connecting with Tendons")
    print("=" * 70)
    
    # Start with minimal XML
    minimal_xml = """
    <mujoco>
        <option timestep="0.002" gravity="0 0 -9.81"/>
        <worldbody>
            <light pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
        </worldbody>
    </mujoco>
    """
    
    builder = ModelBuilder()
    builder.from_xml_string(minimal_xml)
    
    # Add three boxes
    print("Adding boxes...")
    builder.add_body("box1", pos=[0, 0, 1.0], mass=1.0,
                     geom_size=[0.1, 0.1, 0.1], geom_rgba=[1, 0, 0, 1])
    builder.add_body("box2", pos=[0.3, 0, 1.0], mass=1.0,
                     geom_size=[0.1, 0.1, 0.1], geom_rgba=[0, 1, 0, 1])
    builder.add_body("box3", pos=[0.15, 0.2, 1.0], mass=1.0,
                     geom_size=[0.1, 0.1, 0.1], geom_rgba=[0, 0, 1, 1])
    
    # Add sites to boxes
    print("Adding sites...")
    builder.add_site("box1_site", "box1", pos=[0.1, 0, 0])
    builder.add_site("box2_site", "box2", pos=[-0.1, 0, 0])
    builder.add_site("box3_site", "box3", pos=[0, -0.1, 0])
    
    # Connect boxes with tendons
    print("Creating tendons...")
    builder.add_tendon("tendon_12", 
                      sites=["box1_site", "box2_site"],
                      stiffness=100.0, damping=5.0)
    builder.add_tendon("tendon_23",
                      sites=["box2_site", "box3_site"],
                      stiffness=100.0, damping=5.0)
    builder.add_tendon("tendon_31",
                      sites=["box3_site", "box1_site"],
                      stiffness=100.0, damping=5.0)
    
    # Create simulation
    print("Compiling model...")
    sim = MuJoCoSimulation.from_builder(builder)
    
    # Register entities
    box1 = sim.reg_body("box1", "box1")
    box2 = sim.reg_body("box2", "box2")
    box3 = sim.reg_body("box3", "box3")
    
    tendon12 = sim.reg_tendon("t12", "tendon_12")
    tendon23 = sim.reg_tendon("t23", "tendon_23")
    tendon31 = sim.reg_tendon("t31", "tendon_31")
    
    def control(sim):
        # Print tendon info every second
        if sim.time % 1.0 < sim.model.opt.timestep:
            print(f"t={sim.time:.2f}s: "
                  f"T12={tendon12.length:.3f}m ({tendon12.force:.1f}N), "
                  f"T23={tendon23.length:.3f}m ({tendon23.force:.1f}N), "
                  f"T31={tendon31.length:.3f}m ({tendon31.force:.1f}N)")
    
    print("\nRunning simulation...")
    print("Three boxes connected in a triangle by tendons!")
    print("-" * 70)
    
    sim.run(control_callback=control, passive=True, duration=10.0)
    print("\n✓ Example 2 complete!\n")


# ==============================================================================
# Example 3: Complex Scene with Multiple Element Types
# ==============================================================================

def example3_complex_manipulation():
    """
    Build a manipulation scene with robot gripper, target object, and tendons.
    """
    print("=" * 70)
    print("Example 3: Building Complete Manipulation Scene")
    print("=" * 70)
    
    # Base scene
    base_xml = """
    <mujoco>
        <option timestep="0.002" gravity="0 0 -9.81"/>
        <worldbody>
            <light pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>
            
            <!-- Simple gripper -->
            <body name="gripper_base" pos="0 0 0.5">
                <geom type="box" size="0.05 0.05 0.1" rgba="0.3 0.3 0.3 1"/>
                <joint name="gripper_slide" type="slide" axis="0 0 1" range="0 2"/>
                
                <body name="left_finger" pos="-0.05 0 0.1">
                    <geom type="box" size="0.02 0.05 0.08" rgba="0.5 0.5 0.5 1"/>
                    <joint name="left_finger_slide" type="slide" axis="1 0 0" range="-0.1 0.1"/>
                </body>
                
                <body name="right_finger" pos="0.05 0 0.1">
                    <geom type="box" size="0.02 0.05 0.08" rgba="0.5 0.5 0.5 1"/>
                    <joint name="right_finger_slide" type="slide" axis="1 0 0" range="-0.1 0.1"/>
                </body>
            </body>
        </worldbody>
        
        <actuator>
            <position name="gripper_actuator" joint="gripper_slide" kp="100"/>
            <position name="left_finger_actuator" joint="left_finger_slide" kp="50"/>
            <position name="right_finger_actuator" joint="right_finger_slide" kp="50"/>
        </actuator>
    </mujoco>
    """
    
    builder = ModelBuilder()
    builder.from_xml_string(base_xml)
    
    # Add target object
    print("Adding target object...")
    builder.add_body("target", pos=[0, 0, 0.2], mass=0.5,
                     geom_type="sphere", geom_size=[0.08],
                     geom_rgba=[1, 0.5, 0, 1])
    
    # Add sites to fingers and target
    print("Adding attachment sites...")
    builder.add_site("left_tip", "left_finger", pos=[0, 0, 0.08])
    builder.add_site("right_tip", "right_finger", pos=[0, 0, 0.08])
    builder.add_site("target_left", "target", pos=[-0.08, 0, 0])
    builder.add_site("target_right", "target", pos=[0.08, 0, 0])
    
    # Add helper object (like a platform)
    print("Adding platform...")
    builder.add_body("platform", pos=[0.3, 0, 0.1], mass=2.0,
                     geom_type="box", geom_size=[0.15, 0.15, 0.02],
                     geom_rgba=[0.6, 0.4, 0.2, 1])
    
    # Create simulation
    print("Compiling model...")
    sim = MuJoCoSimulation.from_builder(builder)
    
    # Register entities
    gripper_joint = sim.reg_joint("gripper", "gripper_slide")
    left_joint = sim.reg_joint("left", "left_finger_slide")
    right_joint = sim.reg_joint("right", "right_finger_slide")
    
    gripper_act = sim.reg_actuator("gripper_ctrl", "gripper_actuator")
    left_act = sim.reg_actuator("left_ctrl", "left_finger_actuator")
    right_act = sim.reg_actuator("right_ctrl", "right_finger_actuator")
    
    target = sim.reg_body("target", "target")
    
    # Control sequence
    def control(sim):
        t = sim.time
        
        # Phase 1: Move gripper down
        if t < 2.0:
            gripper_act.ctrl = 0.0
            left_act.ctrl = 0.0
            right_act.ctrl = 0.0
        # Phase 2: Open gripper
        elif t < 3.0:
            gripper_act.ctrl = 0.2
            left_act.ctrl = -0.08
            right_act.ctrl = 0.08
        # Phase 3: Close gripper
        elif t < 5.0:
            gripper_act.ctrl = 0.2
            left_act.ctrl = 0.05
            right_act.ctrl = -0.05
        # Phase 4: Lift
        else:
            gripper_act.ctrl = 0.8
            left_act.ctrl = 0.05
            right_act.ctrl = -0.05
        
        # Print status
        if t % 1.0 < sim.model.opt.timestep:
            print(f"t={t:.2f}s: Gripper height={gripper_joint.qpos[0]:.3f}m, "
                  f"Target height={target.pos[2]:.3f}m")
    
    print("\nRunning manipulation sequence...")
    print("Gripper picks up the target object!")
    print("-" * 70)
    
    sim.run(control_callback=control, passive=True, duration=8.0)
    print("\n✓ Example 3 complete!\n")


# ==============================================================================
# Example 4: Quick Tendon Addition Workflow
# ==============================================================================

def example4_quick_tendon_workflow():
    """
    Show the typical workflow for adding tendons to an existing model.
    """
    print("=" * 70)
    print("Example 4: Quick Workflow for Adding Tendons")
    print("=" * 70)
    print()
    print("Typical workflow:")
    print("1. Load base model")
    print("2. Identify bodies you want to connect")
    print("3. Add sites to those bodies")
    print("4. Create tendons between sites")
    print("5. Compile and simulate")
    print()
    
    # Simple example
    simple_xml = """
    <mujoco>
        <worldbody>
            <body name="A" pos="0 0 1">
                <joint type="free"/>
                <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
            </body>
            <body name="B" pos="0.5 0 1">
                <joint type="free"/>
                <geom type="box" size="0.1 0.1 0.1" rgba="0 1 0 1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    
    # Step 1: Load model
    print("Step 1: Loading base model with bodies A and B...")
    builder = ModelBuilder().from_xml_string(simple_xml)
    
    # Step 2-3: Add sites
    print("Step 2-3: Adding sites to bodies A and B...")
    builder.add_site("A_attach", "A", pos=[0.1, 0, 0], rgba=[1, 0, 0, 1])
    builder.add_site("B_attach", "B", pos=[-0.1, 0, 0], rgba=[0, 1, 0, 1])
    
    # Step 4: Create tendon
    print("Step 4: Creating tendon between sites...")
    builder.add_tendon("AB_spring",
                      sites=["A_attach", "B_attach"],
                      stiffness=200.0,
                      damping=10.0)
    
    # Step 5: Compile and simulate
    print("Step 5: Compiling and running...")
    sim = MuJoCoSimulation.from_builder(builder)
    
    # Register
    bodyA = sim.reg_body("A", "A")
    bodyB = sim.reg_body("B", "B")
    spring = sim.reg_tendon("spring", "AB_spring")
    
    def control(sim):
        if sim.time % 0.5 < sim.model.opt.timestep:
            dist = np.linalg.norm(bodyA.pos - bodyB.pos)
            print(f"t={sim.time:.2f}s: Distance={dist:.3f}m, "
                  f"Tendon length={spring.length:.3f}m, force={spring.force:.1f}N")
    
    print("\nSimulating two boxes connected by a spring...")
    print("-" * 70)
    sim.run(control_callback=control, passive=True, duration=5.0)
    print("\n✓ Example 4 complete!\n")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    examples = {
        "1": ("Add tendons to robot", example1_add_tendons_to_robot),
        "2": ("Add objects and connect them", example2_add_objects),
        "3": ("Complex manipulation scene", example3_complex_manipulation),
        "4": ("Quick workflow demo", example4_quick_tendon_workflow),
    }
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            _, func = examples[choice]
            func()
        else:
            print(f"Unknown example: {choice}")
            print("Available examples:", ", ".join(examples.keys()))
    else:
        print("=" * 70)
        print("ModelBuilder Examples - Dynamic Model Manipulation")
        print("=" * 70)
        print()
        print("Usage: python model_builder_examples.py [example_number]")
        print()
        print("Available examples:")
        for num, (desc, _) in examples.items():
            print(f"  {num}: {desc}")
        print()
        print("Example: python model_builder_examples.py 2")
        print()
        print("Or run all examples:")
        
        # Run example 4 (quickest)
        example4_quick_tendon_workflow()
