
import sys
import os
import math
# Add parent directory to path to import from tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mujoco_sim_template import (
    MuJoCoSimulation, ModelBuilder,
    quat_mul, quat_conj, quat_to_axis_angle
)
import numpy as np
import argparse
import networkx


xml_path = "Sciurus17_mujoco_sim_example/URDFs/sciurus17_description/urdf/sciurus17.xml"

def quat_from_euler(roll, pitch, yaw):
    """Convert Euler angles to quaternion."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w, x, y, z]


G = networkx.Graph()
builder = ModelBuilder(G)
builder.from_xml_path(xml_path)

# Set initial joint positions for arms
initial_joint_positions = np.array([0.05062136600022616, -0.3436116964863836, -1.3959225169759335, 
                0.10584467436410924, -1.5539225381281545, 0.09817477042468103, 2.741223667951641,
                0.02761165418194154, -1.7916895602504288, -1.6122138080678088, 0.0, 
                -0.0798, 1.53, 0.145, -2.26,
                0.0372, 1.07, -2, 0.5], dtype=np.float32)
joint_names = [
"waist_yaw_joint", "neck_yaw_joint", "neck_pitch_joint",
"r_arm_joint1","r_arm_joint2","r_arm_joint3","r_arm_joint4",
"r_arm_joint5","r_arm_joint6","r_arm_joint7","r_hand_mimic_joint",
"l_arm_joint1","l_arm_joint2","l_arm_joint3","l_arm_joint4",
"l_arm_joint5","l_arm_joint6","l_arm_joint7","l_hand_mimic_joint"
]




# colors for chopsticks
left_chopstick_color = [1, 0, 0, 1]
right_chopstick_color = [0, 0, 1, 1]

# Parameters for chopsticks
radius = 0.02
half_len = 0.1


# Replace hands with chopsticks
builder.remove_body("r_link7")
builder.remove_body("l_link7")
builder.add_body("left_chopstick", pos=[0, 0.1, 0],
                    quat=quat_from_euler(math.pi/2, 0, 0),
                    geom_type="capsule", geom_size=[radius, half_len],
                    parent="l_link6",
                    free_joint=False, geom_rgba=left_chopstick_color, mass=0.1)
builder.add_body("right_chopstick", pos=[0, -0.1, 0],
                    quat=quat_from_euler(math.pi/2, 0, 0),
                    geom_type="capsule", geom_size=[radius, half_len],
                    parent="r_link6",
                    free_joint=False, geom_rgba=right_chopstick_color, mass=0.1)

# Add a platform to place objects on
builder.add_body("platform", pos=[0.5, 0, 0.1],
                     geom_type="box", geom_size=[0.2, 0.2, 0.1],
                     free_joint=False, geom_rgba=[0.7,0.5,0.5,1], mass=10)

builder.add_body("target", pos=[0.5, 0, 0.3],
                     geom_type="box", geom_size=[0.05, 0.05, 0.05],
                     free_joint=True, geom_rgba=[0,1,0,1], mass=1)
builder.add_body("ghost_target", pos=[0.5, 0, 0.3],
                     geom_type="box", geom_size=[0.05, 0.05, 0.05],
                     free_joint=True, geom_rgba=[1,1,1,0.1], mass=1, intersection=False)


# Add a 3D grid of sites around each chopstick
chopsticks = [
    ("left_chopstick", left_chopstick_color),
    ("right_chopstick", right_chopstick_color),
]

margin_r = 0.1
margin_z = 0.2

nx, ny, nz = 3, 3, 5
xs = np.linspace(-(radius + margin_r), (radius + margin_r), nx)
ys = np.linspace(-(radius + margin_r), (radius + margin_r), ny)
zs = np.linspace(-(half_len + margin_z), (half_len + margin_z), nz)

for body_name, color in chopsticks:
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                site_name = f"{body_name}_site_{i}_{j}_{k}"
                builder.add_site(
                    name=site_name,
                    body_name=body_name,
                    pos=[float(x), float(y), float(z)],
                    size=0.003,
                    rgba=color
                )

margin_r = 0.3
margin_z = 0.3
nxg, nyg, nzg = 5, 5, 5
xs_g = np.linspace(-(radius + margin_r), (radius + margin_r), nxg)
ys_g = np.linspace(-(radius + margin_r), (radius + margin_r), nyg)
zs_g = np.linspace(-(half_len + margin_z), (half_len + margin_z), nzg)
for i, x in enumerate(xs_g):
    for j, y in enumerate(ys_g):
        for k, z in enumerate(zs_g):
            site_name = f"ghost_target_site_{i}_{j}_{k}"
            builder.add_site(
                name=site_name,
                body_name="ghost_target",
                pos=[float(x), float(y), float(z)],
                size=0.003,
                rgba=[1,1,1,0.5]
            )

labels = list(builder.graph.nodes())
n = len(labels)

# Mask allowing only cross-object connections (no self or intra-object)
eligible = np.zeros((n, n), dtype=bool)
for i, node_i in enumerate(labels):
    for j, node_j in enumerate(labels):
        if i != j:
            if ("left_chopstick" in node_i and "ghost_target" in node_j) or \
                ("ghost_target" in node_i and "right_chopstick" in node_j):
                eligible[i, j] = True

# Random symmetric adjacency (tune p as needed)
p = 0.001
rand = np.random.rand(n, n)
A = ((rand < p) & eligible).astype(np.uint8)

adjacency_matrix = A
# Add edges based on adjacency matrix
for i in range(n):
    for j in range(i + 1, n):
        springlen = 0
        if adjacency_matrix[i, j]:
            node_i = labels[i]
            node_j = labels[j]
            stiffness = np.random.uniform(10, 50)
            springlength = [springlen, springlen+0.001]
            rgba = [0, 1, 0, 0.5]  # greenish tendons
            builder.add_tendon(
                name=f"tendon_{node_i}_{node_j}",
                sites=[node_i, node_j],
                stiffness=stiffness,
                damping=5.0,
                springlength=springlength,
                rgba=rgba
            )

sim = MuJoCoSimulation.from_builder(builder)


for name, pos in zip(joint_names, initial_joint_positions):
    try:
        joint = sim.reg_joint(name, name)
        joint.qpos = pos
    except Exception as e:
        print(f"Error setting joint '{name}': {e}")

sim.forward()

target = sim.reg_joint("target", "target_freejoint")
ghost_target = sim.reg_joint("ghost_target", "ghost_target_freejoint")

def control(sim: MuJoCoSimulation):
    ghost_target.qpos = target.qpos
    ghost_target.qvel = target.qvel
    


sim.run(
    passive=True,
    viewer_distance=5,
    viewer_lookat=[0,0,0.25],
    realtime_speed=0.5,
    gravity_comp=True,
    control_callback=control,
    duration=2.0
    )



