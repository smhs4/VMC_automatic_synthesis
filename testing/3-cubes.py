import mujoco
import argparse
import time
from mujoco import viewer
import numpy as np

with open('testing/model.xml', 'r') as file:
  xml = file.read()

parser = argparse.ArgumentParser(description="Friction test with passive/non-passive viewer")
parser.add_argument("--passive", action="store_true", help="Run with passive viewer (no built-in sim loop)")
args = parser.parse_args()


model = mujoco.MjModel.from_xml_string(xml)
data  = mujoco.MjData(model)



# Convenience handles
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "crane_base__up_down")
act_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift")
qpos_adr = model.jnt_qposadr[joint_id]

dummy_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "dummy__dummy_joint")
target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "blue_box__blue_joint")

dof_dummy = model.jnt_qposadr[dummy_id]
dof_target  = model.jnt_qposadr[target_id]

# Simple trajectory: ease from current height to 1.5 m over ~3 seconds, then hover
target_final = 1.5
t_rise = 3.0  # seconds

def quat_mul(q1, q2):
    # w x y z
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conj(q):
    w,x,y,z = q
    return np.array([w, -x, -y, -z])

def small_angle_from_quat(q):
    # For small errors: vector part ~ 0.5*theta*axis
    return 2.0 * q[1:]  # x,y,z

Kp = 20.0
Kd = 50.0

idxA = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'red_box')
idxB = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'green_box')

def control(model, data):
    # world-frame quaternions
    qA = data.xquat[idxA].copy()
    qB = data.xquat[idxB].copy()

    

    # orientation error from A to B: q_err = conj(qA) * qB
    q_err = quat_mul(quat_conj(qA), qB)
    ang_err = small_angle_from_quat(q_err)

    # angular velocities in world frame

    
    wA = data.cvel[idxA, 0:3]
    
    wB = data.cvel[idxB, 0:3]
    
    w_err = wB + wA / 2  # we want wB = -wA

    tau = -Kp*ang_err - Kd*w_err  # torque on B (apply opposite on A)

    # apply to the rotational DOFs of the free joints (last 3 dofs of each free joint)
    dofA = model.body_dofadr[idxA]
    dofB = model.body_dofadr[idxB]
    data.qfrc_applied[dofB+3:dofB+6] += tau
    data.qfrc_applied[dofA+3:dofA+6] -= tau

mujoco.set_mjcb_control(control)

if args.passive:
      x = 0
      with viewer.launch_passive(model, data) as v:
            v.cam.distance = 2.5
            v.cam.lookat[:] = [0.0, 0.0, 0.25]

            t0 = 0.0
            x = 0
            while v.is_running():

                  # dummy box follows the target box completely (no lag)
                  
                  data.qpos[dof_dummy:dof_dummy+7] = data.qpos[dof_target:dof_target+7]
                  #data.qpos[dof_dummy+3:dof_dummy+7] = [1, 0, 0, 0]  # no rotation
                  
                  

                  t = data.time - t0
                  
                  # Smooth step using a cosine ramp

                  if t-0.5 < t_rise and t > 0.5:
                        alpha = 0.5 - 0.5*np.cos(np.pi * (t-0.5) / t_rise)
                        target = data.qpos[qpos_adr]*(1 - alpha) + target_final*alpha
                  elif t <= 0.5:
                        target = 0.0
                  else:
                        target = target_final

                  # Position actuator: set desired joint position directly
                  data.ctrl[act_id] = target
                  # if x < 500:
                  mujoco.mj_step(model, data)
                  v.sync()
                  time.sleep(0.001)  # 10 ms delay to slow down the simulation
                  x += 1

                  
                  
else:
      viewer.launch(model, data)
