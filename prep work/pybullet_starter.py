"""
Virtual Model Control demo for a KUKA iiwa arm in PyBullet.

The end-effector is tethered to a user-controlled target by a virtual
cartesian spring-damper. In addition, one of the intermediate links is
constrained to remain near an imaginary plane via another virtual
spring that only acts along the plane normal. This keeps the link on the
plane while allowing motion within it (virtual fixture).
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data

# End-effector virtual spring parameters
TIME_STEP = 1.0 / 240.0
STIFFNESS = 1500.0  # [N/m]
DAMPING = 120.0     # [N·s/m]
MAX_TORQUE = 200.0  # [N·m]

# Plane constraint parameters (xz plane at y = 0, centred ahead of the base)
PLANE_POINT = np.array([0.5, 0.1, 0.6])
PLANE_NORMAL = np.array([0.0, 1.0, 0.0])  # must remain unit length
PLANE_STIFFNESS = 10000.0  # [N/m]
PLANE_DAMPING = 160.0     # [N·s/m]


def setup_simulation() -> int:
    """Connect to PyBullet, configure the world, and load the robot."""
    p.connect(p.GUI)
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.8,
            cameraYaw=45,
            cameraPitch=-35,
            cameraTargetPosition=[0.0, 0.0, 0.8],
        )
    except p.error:
        # DIRECT connections do not expose the debug visualiser; ignore.
        pass

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0.0, 0.0, -9.81)
    p.setTimeStep(TIME_STEP)

    p.loadURDF("plane.urdf")
    robot_uid = p.loadURDF(
        "kuka_iiwa/model.urdf",
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )
    return robot_uid


def get_active_joint_indices(robot_uid: int) -> List[int]:
    joint_indices: List[int] = []
    for joint_id in range(p.getNumJoints(robot_uid)):
        joint_type = p.getJointInfo(robot_uid, joint_id)[2]
        if joint_type != p.JOINT_FIXED:
            joint_indices.append(joint_id)
    return joint_indices


def reset_initial_pose(robot_uid: int, joint_indices: List[int]) -> None:
    # Slightly bent pose keeps the arm away from singularities.
    initial_pose = [0.0, 0.3, 0.0, -1.5, 0.0, 1.2, 0.0]
    for joint_id, joint_angle in zip(joint_indices, initial_pose):
        p.resetJointState(robot_uid, joint_id, joint_angle)

    # Disable the default motor controllers so we can apply torques directly.
    p.setJointMotorControlArray(
        robot_uid,
        joint_indices,
        p.VELOCITY_CONTROL,
        forces=[0.0] * len(joint_indices),
    )


def create_target_marker(initial_position: np.ndarray) -> int:
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.03,
        rgbaColor=[1.0, 0.1, 0.1, 0.7],
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual_shape,
        basePosition=initial_position.tolist(),
    )


def create_plane_marker() -> int:
    half_extents = [0.45, 0.002, 0.45]
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=[0.1, 0.3, 1.0, 0.15],
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual_shape,
        basePosition=PLANE_POINT.tolist(),
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )


def add_target_sliders(initial_position: np.ndarray) -> Optional[Dict[str, int]]:
    ranges = {
        "x": (0.2, 0.8),
        "y": (-0.4, 0.4),
        "z": (0.3, 1.2),
    }
    sliders: Dict[str, int] = {}
    try:
        for axis, (low, high) in ranges.items():
            start_value = float(initial_position["xyz".index(axis)])
            sliders[axis] = p.addUserDebugParameter(f"target_{axis}", low, high, start_value)
    except p.error:
        return None
    return sliders


def remove_target_sliders(slider_ids: Optional[Dict[str, int]]) -> None:
    if not slider_ids:
        return
    for uid in slider_ids.values():
        try:
            p.removeUserDebugItem(uid)
        except p.error:
            pass


def read_target_from_sliders(slider_ids: Dict[str, int]) -> np.ndarray:
    return np.array([
        p.readUserDebugParameter(slider_ids["x"]),
        p.readUserDebugParameter(slider_ids["y"]),
        p.readUserDebugParameter(slider_ids["z"]),
    ])


def get_joint_state(robot_uid: int, joint_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    joint_states = p.getJointStates(robot_uid, joint_indices)
    positions = np.array([state[0] for state in joint_states])
    velocities = np.array([state[1] for state in joint_states])
    return positions, velocities


def compute_end_effector_torques(
    robot_uid: int,
    joint_indices: List[int],
    end_effector_link: int,
    joint_positions: np.ndarray,
    joint_velocities: np.ndarray,
    target_position: np.ndarray,
) -> np.ndarray:
    link_state = p.getLinkState(robot_uid, end_effector_link, computeLinkVelocity=1)
    current_position = np.array(link_state[0])
    current_linear_velocity = np.array(link_state[6])

    spring_force = STIFFNESS * (target_position - current_position)
    damping_force = -DAMPING * current_linear_velocity
    cartesian_force = spring_force + damping_force

    jac_trans, _ = p.calculateJacobian(
        bodyUniqueId=robot_uid,
        linkIndex=end_effector_link,
        localPosition=[0.0, 0.0, 0.0],
        objPositions=joint_positions.tolist(),
        objVelocities=joint_velocities.tolist(),
        objAccelerations=[0.0] * len(joint_indices),
    )
    return np.array(jac_trans).T @ cartesian_force


def compute_plane_constraint_torques(
    robot_uid: int,
    joint_indices: List[int],
    constrained_link: int,
    joint_positions: np.ndarray,
    joint_velocities: np.ndarray,
) -> np.ndarray:
    normal = PLANE_NORMAL / np.linalg.norm(PLANE_NORMAL)
    link_state = p.getLinkState(robot_uid, constrained_link, computeLinkVelocity=1)
    link_pos = np.array(link_state[0])
    link_vel = np.array(link_state[6])

    distance_along_normal = np.dot(link_pos - PLANE_POINT, normal)
    normal_velocity = np.dot(link_vel, normal)

    restoring_scalar = -PLANE_STIFFNESS * distance_along_normal - PLANE_DAMPING * normal_velocity
    constraint_force = restoring_scalar * normal

    jac_trans, _ = p.calculateJacobian(
        bodyUniqueId=robot_uid,
        linkIndex=constrained_link,
        localPosition=[0.0, 0.0, 0.0],
        objPositions=joint_positions.tolist(),
        objVelocities=joint_velocities.tolist(),
        objAccelerations=[0.0] * len(joint_indices),
    )
    return np.array(jac_trans).T @ constraint_force


def compute_total_torques(
    robot_uid: int,
    joint_indices: List[int],
    end_effector_link: int,
    constrained_link: int,
    target_position: np.ndarray,
) -> np.ndarray:
    joint_positions, joint_velocities = get_joint_state(robot_uid, joint_indices)

    ee_torques = compute_end_effector_torques(
        robot_uid,
        joint_indices,
        end_effector_link,
        joint_positions,
        joint_velocities,
        target_position,
    )
    plane_torques = compute_plane_constraint_torques(
        robot_uid,
        joint_indices,
        constrained_link,
        joint_positions,
        joint_velocities,
    )

    gravity_torques = np.array(
        p.calculateInverseDynamics(
            robot_uid,
            joint_positions.tolist(),
            joint_velocities.tolist(),
            [0.0] * len(joint_indices),
        )
    )

    total_torques = ee_torques + plane_torques + gravity_torques
    return np.clip(total_torques, -MAX_TORQUE, MAX_TORQUE)


def run_control_loop(robot_uid: int) -> None:
    joint_indices = get_active_joint_indices(robot_uid)
    end_effector_link = joint_indices[-1]
    constrained_link = joint_indices[2]  # Apply the plane constraint at joint 3

    reset_initial_pose(robot_uid, joint_indices)

    initial_target = np.array([0.5, 0.0, 0.8])
    target_marker = create_target_marker(initial_target)
    plane_marker = create_plane_marker()
    slider_ids = add_target_sliders(initial_target)

    target_position = initial_target.copy()

    while p.isConnected():
        if slider_ids is None:
            slider_ids = add_target_sliders(target_position)
            # Slider creation can fail when the GUI is hidden; try again next step.
            if slider_ids is None:
                p.stepSimulation()
                time.sleep(TIME_STEP)
                continue
        else:
            try:
                target_position = read_target_from_sliders(slider_ids)
            except p.error:
                # remove_target_sliders(slider_ids)
                # slider_ids = None
                continue

        p.resetBasePositionAndOrientation(
            target_marker,
            target_position.tolist(),
            [0.0, 0.0, 0.0, 1.0],
        )
        p.resetBasePositionAndOrientation(
            plane_marker,
            PLANE_POINT.tolist(),
            [0.0, 0.0, 0.0, 1.0],
        )

        torques = compute_total_torques(
            robot_uid,
            joint_indices,
            end_effector_link,
            constrained_link,
            target_position,
        )
        p.setJointMotorControlArray(robot_uid, joint_indices, p.TORQUE_CONTROL, forces=torques.tolist())

        p.stepSimulation()
        time.sleep(TIME_STEP)


def main() -> None:
    robot_uid = setup_simulation()
    try:
        run_control_loop(robot_uid)
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
