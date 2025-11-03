import sys
import os
import mujoco
import numpy as np
# Add parent directory to path to import from tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mujoco_sim_template import MuJoCoSimulation

# Use relative path from project root
xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                        "Sciurus17_mujoco_sim_example/URDFs/sciurus17_description/urdf/sciurus17_pickup.xml")
sim = MuJoCoSimulation(xml_path=xml_path)

# Register robot parts
r_hand = sim.add_body("r_hand", "r_handA_link")
l_hand = sim.add_body("l_hand", "l_handA_link")
target = sim.add_body("target", "target_box")

# Optional: Register specific actuators if you want to control them directly
# For example, if you want to add additional control on top of gravity comp:
# waist_actuator = sim.add_actuator("waist", "waist_joint")


def control(sim):
    """
    Main control callback - combines gravity compensation with task control.
    This function is called every timestep.
    """
    model = sim.model
    data = sim.data
    
    # # === GRAVITY COMPENSATION ===
    # # Compute gravity and Coriolis forces using inverse dynamics
    # acc_zero = np.zeros(model.nv)
    # qacc_old = data.qacc.copy()
    # data.qacc[:] = acc_zero
    
    # # Call inverse dynamics to get bias forces
    # mujoco.mj_rne(model, data, 0, data.qfrc_bias)
    # data.qacc[:] = qacc_old
    
    # # Apply gravity compensation through actuator controls
    # if model.nu > 0:
    #     # Initialize with gravity compensation
    #     data.ctrl[:] = 0.0
        
    #     for i in range(model.nu):
    #         jnt_id = model.actuator_trnid[i, 0]
    #         if jnt_id >= 0 and jnt_id < model.njnt:
    #             dof_adr = model.jnt_dofadr[jnt_id]
    #             gear = model.actuator_gear[i, 0]
    #             if gear != 0:
    #                 data.ctrl[i] = data.qfrc_bias[dof_adr] / gear
    #             else:
    #                 data.ctrl[i] = data.qfrc_bias[dof_adr]
    
    # === TASK-SPECIFIC CONTROL ===
    # Now add your custom control logic using the OOP interface
    
    # Example: Apply upward force to right hand if target is low
    
    r_hand.add_force([1000, 1000, 1000.0])
    
    # Example: You can also modify actuator controls directly
    # if you registered them above:
    # waist_actuator.ctrl += additional_torque


# Run simulation with combined control
sim.run(control_callback=control, passive=True)