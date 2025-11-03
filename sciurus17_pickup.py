import mujoco
from mujoco import viewer
import numpy as np

xml_path = "Sciurus17_mujoco_sim_example/URDFs/sciurus17_description/urdf/sciurus17.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data  = mujoco.MjData(model)

def gravity_compensation(model, data):
    # Compute gravity and Coriolis forces using inverse dynamics
    # Set acceleration to zero to get just the bias forces (gravity + Coriolis)
    acc_zero = np.zeros(model.nv)
    
    # Store current acceleration and set to zero temporarily
    qacc_old = data.qacc.copy()
    data.qacc[:] = acc_zero
    
    # Call inverse dynamics to compute generalized forces needed
    # to achieve zero acceleration (which gives us gravity compensation)
    mujoco.mj_rne(model, data, 0, data.qfrc_bias)
    
    # Restore original acceleration
    data.qacc[:] = qacc_old
    
    # Apply gravity compensation through actuator controls
    # Map generalized forces to actuator space
    if model.nu > 0:
        # Clear existing controls
        data.ctrl[:] = 0.0
        
        # For each actuator, find the corresponding joint and apply compensation
        for i in range(model.nu):
            # Get the joint associated with this actuator
            jnt_id = model.actuator_trnid[i, 0]  # First transmission element
            
            if jnt_id >= 0 and jnt_id < model.njnt:
                # Get the DOF address for this joint
                dof_adr = model.jnt_dofadr[jnt_id]
                
                # Apply gravity compensation force for this DOF
                # Scale by gear ratio if needed
                gear = model.actuator_gear[i, 0]
                if gear != 0:
                    data.ctrl[i] = data.qfrc_bias[dof_adr] / gear
                else:
                    data.ctrl[i] = data.qfrc_bias[dof_adr]

mujoco.set_mjcb_control(gravity_compensation)

with viewer.launch_passive(model, data) as v:
    v.cam.distance = 2.5
    v.cam.lookat[:] = [0.0, 0.0, 0.25]

    while v.is_running():
        mujoco.mj_step(model, data)
        v.sync()
        # time.sleep(0.001)  # Optional: slow down the simulation