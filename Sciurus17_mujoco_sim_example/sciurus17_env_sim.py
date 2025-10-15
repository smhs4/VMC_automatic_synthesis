from sciurus17_env import Sciurus17Env
import numpy as np
from gymnasium.wrappers import RecordVideo

xml = r"URDFs/sciurus17_description/urdf/sciurus17.xml"

# Example: control only the 7-DoF right arm joints (names from your XML)
act_right = [
    "waist_yaw_joint", "neck_yaw_joint", "neck_pitch_joint",
    "r_arm_joint1","r_arm_joint2","r_arm_joint3","r_arm_joint4",
    "r_arm_joint5","r_arm_joint6","r_arm_joint7","r_hand_joint","r_hand_mimic_joint",
    "l_arm_joint1","l_arm_joint2","l_arm_joint3","l_arm_joint4",
    "l_arm_joint5","l_arm_joint6","l_arm_joint7","l_hand_joint","l_hand_mimic_joint"
]

env = Sciurus17Env(
    xml_path=xml,
    actuator_names=act_right,   # or None to control all actuators
    action_mode="position",     # "torque" or "position"
    gravity_comp=True,          # add qfrc_bias
    Kp=150.0, Kd=1.0, # Kp=150.0, Kd=1.0,
    render_mode="rgb_array",
    frame_skip=5
)

# Wrap for video (writes .mp4 into the folder)
env = RecordVideo(env, video_folder="videos", name_prefix="sciurus17")

obs, info = env.reset()

# Hold a comfortable pose
q_des = np.array([0.05062136600022616, -0.3436116964863836, -1.3959225169759335, 
                  0.10584467436410924, -1.5539225381281545, 0.09817477042468103, 2.741223667951641, 0.02761165418194154, -1.7916895602504288, -1.6122138080678088, 0.0, 0.0, 
                  0.1227184630308513, 1.4894953450369577, 0.4816699673960913, -2.739689687163755, -0.006135923151542565, 1.556990499703926, 0.06902913545485385, 0.0, 0.0], dtype=np.float32)

for _ in range(100):
    obs, rew, term, trunc, info = env.step(q_des)
    if term or trunc:
        break

env.close()