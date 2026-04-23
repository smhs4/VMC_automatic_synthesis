#!/usr/bin/env python3
"""
Genetic Algorithm for optimizing tendon network connections using DEAP.

This script evolves the adjacency matrix for tendon connections between chopsticks
to learn the best configuration for picking up a target block.

Fitness is based on:
- Contact between chopsticks and target
- Target height (lifting)
- Stability of grasp

Dependencies:
    pip install deap numpy networkx
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set matplotlib to non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt

from tools.mujoco_sim_template import Body, MuJoCoSimulation, ModelBuilder, quat_from_euler
import numpy as np
import networkx
import mujoco
import pickle
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import math
import random
import argparse
import networkx as nx
import gc  # For garbage collection
import multiprocessing
from functools import partial
import shutil
import json

# DEAP imports
from deap import base, creator, tools, algorithms
import time


# ============================================================================
# Configuration
# ============================================================================

class GAConfig:
    """Configuration for Genetic Algorithm using DEAP"""
    
    # GA Parameters
    POPULATION_SIZE = 70
    NUM_GENERATIONS = 750
    TOURNAMENT_SIZE = 2
    CROSSOVER_PROB = 0.7
    MUTATION_PROB = 0.1
    ELITE_SIZE = 3
    
    # Parallelization
    NUM_PROCESSES = None  # None = use all CPU cores, or set to specific number
    
    # Body and Simulation Parameters
    RADIUS = 0.01  # radius of chopstick
    HALF_LENGTH = 0.09  # half length of chopstick

    # Site parameters
    N_SITES_ARM_X = 3  # Number of connection sites per chopstick 
    N_SITES_ARM_Y = 3  # Number of connection sites per chopstick 
    N_SITES_ARM_Z = 6  # Number of connection sites per chopstick
    N_SITES_ARM = N_SITES_ARM_X * N_SITES_ARM_Y * N_SITES_ARM_Z  # Total sites per chopstick
    N_SITES_OBJECT_X = 5  # Number of connection sites on object 
    N_SITES_OBJECT_Y = 5  # Number of connection sites on object 
    N_SITES_OBJECT_Z = 5  # Number of connection sites on object 
    N_SITES_OBJECT = N_SITES_OBJECT_X * N_SITES_OBJECT_Y * N_SITES_OBJECT_Z  # Total sites on object
    N_SITES_LINK5_X = 2
    N_SITES_LINK5_Y = 2
    N_SITES_LINK5_Z = 2
    N_SITES_LINK4_X = 2
    N_SITES_LINK4_Y = 2
    N_SITES_LINK4_Z = 2
    N_SITES_BODY_X = 2
    N_SITES_BODY_Y = 2
    N_SITES_BODY_Z = 2
    MARGIN_R_ARM = 0.01
    MARGIN_Z_ARM = 0
    MARGIN_R_OBJECT = 0.01
    MARGIN_Z_OBJECT = 0.01
    BOX_DIM = 0.025

    # Tendon parameters
    MIN_STIFFNESS = 0.0   # N/m
    MAX_STIFFNESS = 20.0  # N/m
    MIN_DAMPING = 1.0      # N*s/m
    MAX_DAMPING = 5.0     # N*s/m
    MIN_SPRING_LENGTH = 0.0
    MAX_SPRING_LENGTH = 0.1
    TENDON_NUM = 4

    # Connection groups define what can connect to what, and how many tendons.
    # spring_mode:
    #   - "range": springlength=[0, length]
    #   - "fixed": springlength=[length, length]
    CONNECTION_GROUPS = [
        {"name": "left_to_target", "src": "left_chopstick", "dst": "target", "num_tendons": 4, "spring_mode": "fixed"},
        {"name": "right_to_target", "src": "right_chopstick", "dst": "target", "num_tendons": 4, "spring_mode": "fixed"},
        {"name": "right_to_left", "src": "right_chopstick", "dst": "left_chopstick", "num_tendons": 4, "spring_mode": "fixed"},
        {"name": "r_link6_to_target", "src": "r_link6", "dst": "target", "num_tendons": 0, "spring_mode": "fixed"},
        {"name": "l_link6_to_target", "src": "l_link6", "dst": "target", "num_tendons": 0, "spring_mode": "fixed"},
        {"name": "link5_to_link5", "src": "r_link5", "dst": "l_link5", "num_tendons": 0, "spring_mode": "fixed"},
        {"name": "left_link4_to_body", "src": "l_link4", "dst": "body_link", "num_tendons": 0, "spring_mode": "fixed"},
        {"name": "right_link4_to_body", "src": "r_link4", "dst": "body_link", "num_tendons": 0, "spring_mode": "fixed"},
    ]

    # Mutation parameters
    FLIP_PROB = 0.3        # Probability to flip each connection
    GAUSSIAN_MU = 0.0
    GAUSSIAN_SIGMA = 10.0  # For stiffness/damping mutation
    LENGTH_SIGMA = 0.002    # For spring length mutation
    
    # Simulation parameters
    SIM_DURATION = 4.0     # seconds
    SIM_DT = 0.01          # seconds
    
    # Fitness weights
    CONTACT_WEIGHT = 0
    HEIGHT_WEIGHT = 6
    HEIGHT_FINAL_RATIO = 1
    HEIGHT_INTEGRAL_RATIO = 0
    CENTRE_WEIGHT = 0
    FORCE_PENALTY_WEIGHT = 0
    CONTROL_EFFORT_WEIGHT = 0
    CONTROL_SMOOTHNESS_WEIGHT = 0
    ROBUSTNESS_WEIGHT = 0
    STABILITY_WEIGHT = 0
    EFFICIENCY_WEIGHT = 0

    # Evaluation scenarios (3 trials): vary target x, target yaw, and arm start pose.
    TARGET_BASE_X = 0.4
    TARGET_BASE_Y = 0.0
    TARGET_BASE_Z = 0.3
    TARGET_TRIAL_X_OFFSETS = (-0.05, 0.0, 0.05, -0.1, 0.1)
    TARGET_TRIAL_YAWS = (-np.pi/5, 0.0, np.pi/5)
    TARGET_TRIAL_Y_OFFSETS = (-0.05, 0.0, 0.05, -0.1, 0.1)
    # Per-step disturbance during rollout (applied directly to target freejoint pose).
    TARGET_STEP_JITTER_X = 0.0005      # +/- m per sim step
    TARGET_STEP_JITTER_Y = 0.0005      # +/- m per sim step
    TARGET_STEP_YAW_JITTER = np.deg2rad(0.2)  # +/- rad per sim step
    ARM_TRIAL_PERTURBATIONS = (
        # {"r_arm_joint1": -0.10, "r_arm_joint2": 0.08, "l_arm_joint1": 0.10, "l_arm_joint2": -0.08},
         {},{},{},{},{},{},{},
        # {"r_arm_joint1": 0.10, "r_arm_joint2": -0.08, "l_arm_joint1": -0.10, "l_arm_joint2": 0.08},
    )
    
    # Checkpoint
    CHECKPOINT_DIR = "ga_checkpoints"
    CHECKPOINT_INTERVAL = 5  # Save every N generations
    
    # Statistics
    STATS_DIR = "ga_stats"
    
    # Results archiving
    RESULTS_DIR = "training_results"

    # Chopstick lift control parameters
    LIFT_DELAY = 3         # seconds before applying the lift
    LIFT_DURATION = 0.1       # seconds to keep applying the lift (None = indefinite)
    LIFT_FORCE = 20.0         # nominal upward force/torque magnitude
    TORQUE_CLIP = 3.0         # absolute clip on generalized torque (Nm-equivalent)

    # Velocity-gain (damping) feedback delay.
    # Simulates real-world latency in the velocity feedback path: sensor sampling,
    # numerical differentiation of encoder position, communication, and actuator lag.
    # TendonTorqueController already bypasses MuJoCo's native damping and computes
    # forces from per-tendon (k, b, l0); enabling a non-zero delay here makes the
    # damping term -b*v respond to a *stale* velocity reading. Phase lag in this loop
    # is the mechanism that drives the real rig into self-excited oscillation when
    # damping is high (classical dead-time instability). Set to 0.0 to recover the
    # instantaneous-velocity behaviour of v2.
    VELOCITY_DELAY = 0.5     # seconds of delay applied to tendon velocity used for damping



# ============================================================================
# DEAP Setup - Define Fitness and Individual
# ============================================================================

# Maximize fitness
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)


# ============================================================================
# Genome Encoding/Decoding
# ============================================================================

def site_family_counts() -> Dict[str, Tuple[int, int, int]]:
    """Site grid sizes for each connection family."""
    return {
        "left_chopstick": (GAConfig.N_SITES_ARM_X, GAConfig.N_SITES_ARM_Y, GAConfig.N_SITES_ARM_Z),
        "right_chopstick": (GAConfig.N_SITES_ARM_X, GAConfig.N_SITES_ARM_Y, GAConfig.N_SITES_ARM_Z),
        "target": (GAConfig.N_SITES_OBJECT_X, GAConfig.N_SITES_OBJECT_Y, GAConfig.N_SITES_OBJECT_Z),
        "l_link6": (GAConfig.N_SITES_LINK5_X, GAConfig.N_SITES_LINK5_Y, GAConfig.N_SITES_LINK5_Z),
        "r_link6": (GAConfig.N_SITES_LINK5_X, GAConfig.N_SITES_LINK5_Y, GAConfig.N_SITES_LINK5_Z),
        "l_link5": (GAConfig.N_SITES_LINK5_X, GAConfig.N_SITES_LINK5_Y, GAConfig.N_SITES_LINK5_Z),
        "r_link5": (GAConfig.N_SITES_LINK5_X, GAConfig.N_SITES_LINK5_Y, GAConfig.N_SITES_LINK5_Z),
        "l_link4": (GAConfig.N_SITES_LINK4_X, GAConfig.N_SITES_LINK4_Y, GAConfig.N_SITES_LINK4_Z),
        "r_link4": (GAConfig.N_SITES_LINK4_X, GAConfig.N_SITES_LINK4_Y, GAConfig.N_SITES_LINK4_Z),
        "body_link": (GAConfig.N_SITES_BODY_X, GAConfig.N_SITES_BODY_Y, GAConfig.N_SITES_BODY_Z),
    }


def family_site_count(family: str) -> int:
    nx, ny, nz = site_family_counts()[family]
    return nx * ny * nz


def expanded_gene_specs() -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    for group in GAConfig.CONNECTION_GROUPS:
        specs.extend([group] * int(group["num_tendons"]))
    return specs


def site_name_from_index(family: str, flat_idx: int) -> Optional[str]:
    nx, ny, nz = site_family_counts()[family]
    total = nx * ny * nz
    if flat_idx < 0 or flat_idx >= total:
        return None
    i = int(flat_idx % nx)
    j = int((flat_idx // nx) % ny)
    k = int(flat_idx // (nx * ny))
    return f"{family}_site_{i}_{j}_{k}"

def create_individual():
    """
    Create a random individual.
    
    Genome encoding (flat list):
    - First N*(N-1)/2 values: adjacency matrix (upper triangle, binary 0/1)
    - Next N*(N-1)/2 values: stiffness values (only for connected tendons)
    - Last N*(N-1)/2 values: damping values (only for connected tendons)
    
    """
    
    

    gene_specs = expanded_gene_specs()

    genome = []
    for spec in gene_specs:
        src_idx = np.random.randint(-1, family_site_count(spec["src"]))
        dst_idx = np.random.randint(-1, family_site_count(spec["dst"]))
        genome.append((src_idx, dst_idx, 0, 0, 0))
    
    genome = np.array(genome, dtype=float)
    for i in range(len(genome)):
        stiffness = random.uniform(GAConfig.MIN_STIFFNESS, GAConfig.MAX_STIFFNESS)
        damping = random.uniform(GAConfig.MIN_DAMPING, GAConfig.MAX_DAMPING)
        # Generate length with higher probability for values closer to 0.
        length = np.random.exponential(scale=0.005)
        length = min(length, GAConfig.MAX_SPRING_LENGTH)  # Cap
        
        genome[i, 2:] = (stiffness, damping, length)
    return creator.Individual(genome)


def decode_genome(individual: creator.Individual) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode flat genome into matrices.
    
    Returns:
        (adjacency_matrix, stiffness_matrix, damping_matrix)
        Each is an 8x8 symmetric matrix
    """

    # write individual to a file for debugging and append to prevent optimizations
    with open("debug_individual.txt", "a") as f:
        f.write(str(individual) + "\n")

    gene_specs = expanded_gene_specs()

    G = nx.Graph()
    for index in range(min(individual.shape[0], len(gene_specs))):
        spec = gene_specs[index]
        src_i = int(individual[index][0])
        dst_i = int(individual[index][1])
        if src_i < 0 or dst_i < 0:
            continue

        src_name = site_name_from_index(spec["src"], src_i)
        dst_name = site_name_from_index(spec["dst"], dst_i)
        if src_name is None or dst_name is None:
            continue

        stiffness = float(individual[index][2])
        damping = float(individual[index][3])
        length = float(max(GAConfig.MIN_SPRING_LENGTH, min(GAConfig.MAX_SPRING_LENGTH, individual[index][4])))
        spring_mode = spec.get("spring_mode", "range")
        springlength = [length, length] if spring_mode == "fixed" else [0.0, length]

        G.add_edge(
            src_name,
            dst_name,
            type='spatial',
            stiffness=stiffness,
            damping=damping,
            springlength=springlength,
        )

    return G



# ============================================================================
# Torque Control Callback (tendon Jacobian -> actuator torques)
# ============================================================================
class TendonTorqueController:
    """Virtual tendon controller that drives robot via actuator torques."""

    def __init__(
        self,
        sim: MuJoCoSimulation,
        tendon_names: Optional[List[str]] = None,
        lift_tendon_names: Optional[List[str]] = None,
        lift_delay: float = GAConfig.LIFT_DELAY,
        lift_duration: float = GAConfig.LIFT_DURATION,
        torque_clip: float = GAConfig.TORQUE_CLIP,
        velocity_delay: float = GAConfig.VELOCITY_DELAY,
    ):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.lift_delay = lift_delay
        self.lift_duration = lift_duration
        self.torque_clip = torque_clip

        # Velocity-delay ring buffer.
        # delay_steps = round(velocity_delay / dt); buffer holds delay_steps+1 samples
        # so that index "now - delay_steps" is always a valid slot. Pre-filled with
        # zeros, which matches the physical initial condition where every tendon
        # starts at rest.
        dt = float(self.model.opt.timestep)
        self.velocity_delay = float(max(0.0, velocity_delay))
        self.delay_steps = int(round(self.velocity_delay / dt)) if dt > 0 else 0
        # Buffer is indexed later by the tendon's slot in self.tendon_ids, so its
        # second dimension is populated after tendon_ids is known (below).

        if tendon_names is None:
            self.tendon_ids = [
                i for i in range(self.model.ntendon)
                if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_TENDON, i) is not None
            ]
        else:
            self.tendon_ids = []
            for name in tendon_names:
                t_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TENDON, name)
                if t_id != -1:
                    self.tendon_ids.append(int(t_id))

        self.k = np.zeros(len(self.tendon_ids), dtype=np.float64)
        self.b = np.zeros(len(self.tendon_ids), dtype=np.float64)
        self.l0 = np.zeros(len(self.tendon_ids), dtype=np.float64)
        self.original_tendon_params: Dict[str, Dict[str, float]] = {}

        for idx, t_id in enumerate(self.tendon_ids):
            self.k[idx] = float(np.atleast_1d(self.model.tendon_stiffness[t_id])[0])
            self.b[idx] = float(np.atleast_1d(self.model.tendon_damping[t_id])[0])
            self.l0[idx] = float(np.atleast_1d(self.model.tendon_lengthspring[t_id])[-1])
            t_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_TENDON, t_id)
            if t_name is not None:
                self.original_tendon_params[t_name] = {
                    "stiffness": self.k[idx],
                    "damping": self.b[idx],
                    "length": self.l0[idx],
                }

            # Disable physical tendon forces; tendons remain visual/measurement elements.
            self.model.tendon_stiffness[t_id] = 0.0
            self.model.tendon_damping[t_id] = 0.0

        self.lift_tendon_ids = set()
        if lift_tendon_names is not None:
            for name in lift_tendon_names:
                t_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TENDON, name)
                if t_id != -1:
                    self.lift_tendon_ids.add(int(t_id))

        self.actuator_map: List[Tuple[int, float]] = []
        for i in range(self.model.nu):
            jnt_id = int(self.model.actuator_trnid[i, 0])
            if jnt_id < 0 or jnt_id >= self.model.njnt:
                self.actuator_map.append((-1, 1.0))
                continue
            dof_adr = int(self.model.jnt_dofadr[jnt_id])
            gear = float(self.model.actuator_gear[i, 0])
            if abs(gear) < 1e-9:
                gear = 1.0
            self.actuator_map.append((dof_adr, gear))

        # Allocate ring buffer now that the number of controlled tendons is known.
        # Shape: (delay_steps + 1, n_tendons). Index self._buf_idx is the slot we
        # will write the current sample into on the next callback; reading from
        # (idx - delay_steps) mod (delay_steps + 1) yields the velocity from
        # delay_steps samples ago.
        self._buf_len = self.delay_steps + 1
        self._velocity_buffer = np.zeros((self._buf_len, len(self.tendon_ids)), dtype=np.float64)
        self._buf_idx = 0

    def _lift_gain(self, current_time: float) -> float:
        if current_time < self.lift_delay:
            return 0.0
        if self.lift_duration <= 0:
            return 1.0
        if current_time < self.lift_delay + self.lift_duration:
            progress = (current_time - self.lift_delay) / self.lift_duration
            return progress * progress * (3.0 - 2.0 * progress)
        return 1.0

    def callback(self, _sim: MuJoCoSimulation):
        tau = np.zeros(self.model.nv, dtype=np.float64)
        ten_j = self.data.ten_J
        lift_gain = self._lift_gain(self.data.time)

        # Push the current per-tendon velocities into the ring buffer and fetch
        # the sample from delay_steps ago. When delay_steps == 0 the read and
        # write point at the same slot, so the damping term uses the current
        # velocity and we recover v2's behaviour exactly.
        current_velocities = np.asarray(
            [float(self.data.ten_velocity[t_id]) for t_id in self.tendon_ids],
            dtype=np.float64,
        )
        self._velocity_buffer[self._buf_idx, :] = current_velocities
        delayed_idx = (self._buf_idx - self.delay_steps) % self._buf_len
        delayed_velocities = self._velocity_buffer[delayed_idx]
        self._buf_idx = (self._buf_idx + 1) % self._buf_len

        for idx, t_id in enumerate(self.tendon_ids):
            length = float(self.data.ten_length[t_id])
            # Spring force uses instantaneous length (position feedback is
            # usually much faster than the velocity-estimation path); only the
            # damping term is delayed, mirroring the physical failure mode.
            velocity = float(delayed_velocities[idx])
            force = -self.k[idx] * (length - self.l0[idx]) - self.b[idx] * velocity
            if t_id in self.lift_tendon_ids:
                force = 0 if length < 0.01 else force
                force *= lift_gain
            tau += ten_j[t_id] * force

        tau = np.clip(tau, -self.torque_clip, self.torque_clip)

        # Gravity compensation (if enabled) is already placed in data.ctrl upstream.
        # We add torque commands on top.
        for act_id, (dof_adr, gear) in enumerate(self.actuator_map):
            if dof_adr < 0 or dof_adr >= self.model.nv:
                continue
            ctrl = float(self.data.ctrl[act_id] + tau[dof_adr] / gear)
            if int(self.model.actuator_ctrllimited[act_id]) == 1:
                lo = float(self.model.actuator_ctrlrange[act_id, 0])
                hi = float(self.model.actuator_ctrlrange[act_id, 1])
                ctrl = float(np.clip(ctrl, lo, hi))
            self.data.ctrl[act_id] = ctrl


# ============================================================================
# Simulation & Fitness Evaluation
# ============================================================================

def build_model_from_genome(
    G,
    angle=None,
    return_builder: bool = False,
    target_pos: Optional[List[float]] = None,
    joint_position_offsets: Optional[Dict[str, float]] = None,
):
    """
    Build MuJoCo simulation from genome matrices, with timing breakdown.
    
    IMPORTANT: This function creates a completely fresh ModelBuilder and MjSpec
    for each call. Do not reuse builders across multiple calls.
    """

    times = {}
    t_total_start = time.perf_counter()

    t0 = time.perf_counter()
    xml_path = "Sciurus17_mujoco_sim_example/URDFs/sciurus17_description/urdf/sciurus17.xml"
    # CRITICAL: Create a fresh ModelBuilder with fresh graph for each evaluation
    # MjSpec cannot be reused after compilation
    builder = ModelBuilder(G)
    builder.from_xml_path(xml_path)  # This creates a fresh MjSpec from XML
    times["builder_init_and_xml"] = time.perf_counter() - t0

    if angle is not None:
        orientation = quat_from_euler(0, 0, angle)
    else:
        orientation = quat_from_euler(0, 0, random.uniform(-math.pi/4, math.pi/4))

    if target_pos is not None:
        target_position = target_pos
    else:
        target_position = [random.uniform(GAConfig.TARGET_BASE_X-0.05, GAConfig.TARGET_BASE_X+0.05), random.uniform(GAConfig.TARGET_BASE_Y-0.05, GAConfig.TARGET_BASE_Y+0.05), GAConfig.TARGET_BASE_Z]
    # Add a platform and target object
    t0 = time.perf_counter()
    builder.add_body("platform", pos=[0.5, 0, 0.05],
                        geom_type="box", geom_size=[0.2, 0.2, 0.05],
                        free_joint=False, geom_rgba=[0.7,0.5,0.5,1], mass=10, friction=[0, 0.01, 0.001])

    builder.add_body("target", pos=target_position,
                     quat=orientation,
                        geom_type="box", geom_size=[GAConfig.BOX_DIM, GAConfig.BOX_DIM, GAConfig.BOX_DIM],
                        free_joint=True, geom_rgba=[0,1,0,1], mass=0.1, friction=[1, 0.01, 0.001]
                        )
    
    times["add_static_bodies"] = time.perf_counter() - t0

    # colors for chopsticks
    left_chopstick_color = [1, 0, 0, 1]
    right_chopstick_color = [0, 0, 1, 1]
    left_site_color = [0, 1, 1, 1]
    right_site_color = [0, 1, 1, 1]

    chopsticks = [
        ("left_chopstick", left_chopstick_color, left_site_color),
        ("right_chopstick", right_chopstick_color, right_site_color),
    ]

    t0 = time.perf_counter()
    # builder.remove_body("r_link7")
    # builder.remove_body("l_link7")
    times["remove_end_links"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    builder.add_body("left_chopstick", pos=[0, GAConfig.HALF_LENGTH, 0],
                    quat=quat_from_euler(math.pi/2, -math.pi/8, 0),
                    geom_type="box", geom_size=[GAConfig.RADIUS, GAConfig.RADIUS, GAConfig.HALF_LENGTH],
                    parent="l_link6",
                    free_joint=False, geom_rgba=left_chopstick_color, mass=0.1,
                    friction=[1.0, 0.01, 0.001])
    builder.add_body("right_chopstick", pos=[0, -GAConfig.HALF_LENGTH, 0],
                    quat=quat_from_euler(math.pi/2, 0, 0),
                    geom_type="box", geom_size=[GAConfig.RADIUS, GAConfig.RADIUS, GAConfig.HALF_LENGTH],
                    parent="r_link6",
                    free_joint=False, geom_rgba=right_chopstick_color, mass=0.1,
                    friction=[1.0, 0.01, 0.001])
    times["add_chopstick_bodies"] = time.perf_counter() - t0

    # Adding mechanism to lift chopsticks
    builder.add_site('left_hand_site', body_name='l_link6', pos=[0, 0, 0], size=0.01)
    builder.add_site('left_shoulder_site', body_name='root', pos=np.array(target_position) + [0, 0.025, 0.0], size=0.01)
    builder.add_site('left_chopstick_end', body_name='left_chopstick', pos=[0, 0, -0.09])
    builder.add_site('left_shoulder_end', body_name='body_link', pos=[0.29, 0.05, 0.4])
    builder.add_site('right_hand_site', body_name='r_link6', pos=[0, 0, 0], size=0.01)
    builder.add_site('right_shoulder_site', body_name='root', pos=np.array(target_position) + [0, -0.025, 0.0], size=0.01)
    builder.add_site('right_chopstick_end', body_name='right_chopstick', pos=[0, 0, 0.09])
    builder.add_site('right_shoulder_end', body_name='body_link', pos=[0.29, -0.05, 0.4])

    # Add lifting tendons (used by torque controller via tendon Jacobians)
    # builder.add_tendon('left_lifter', tendon_type='spatial',
    #                    sites=['left_hand_site', 'left_shoulder_site'],
    #                    stiffness=GAConfig.LIFT_FORCE * 20.0,
    #                    damping=5.0,
    #                    springlength=[0.0, 0.1],
    #                    rgba=[0, 0, 0, 0.5])
    
    # builder.add_tendon('right_lifter', tendon_type='spatial',
    #                    sites=['right_hand_site', 'right_shoulder_site'],
    #                    stiffness=GAConfig.LIFT_FORCE * 20.0,
    #                    damping=5.0,
    #                    springlength=[0.0, 0.1],
    #                    rgba=[0, 0, 0, 0.5])

    builder.add_tendon('left_end_lifter', tendon_type='spatial',
                       sites=['left_chopstick_end', 'left_shoulder_site'],
                       stiffness=GAConfig.LIFT_FORCE * 20.0,
                       damping=5.0,
                       springlength=[0.0, 0.1],
                       rgba=[0, 0, 0, 0.5])
    
    builder.add_tendon('right_end_lifter', tendon_type='spatial',
                       sites=['right_chopstick_end', 'right_shoulder_site'],
                       stiffness=GAConfig.LIFT_FORCE * 20.0,
                       damping=5.0,
                       springlength=[0.0, 0.1],
                       rgba=[0, 0, 0, 0.5])
    
    
    times["add_lifting_mechanism"] = time.perf_counter() - t0

    # Add sites for all families referenced by connection groups.
    family_cfg = {
        "left_chopstick": {"body": "left_chopstick", "counts": (GAConfig.N_SITES_ARM_X, GAConfig.N_SITES_ARM_Y, GAConfig.N_SITES_ARM_Z), "extents": (GAConfig.RADIUS + GAConfig.MARGIN_R_ARM, GAConfig.RADIUS + GAConfig.MARGIN_R_ARM, GAConfig.HALF_LENGTH + GAConfig.MARGIN_Z_ARM), "size": 0.003, "rgba": [0, 1, 1, 1]},
        "right_chopstick": {"body": "right_chopstick", "counts": (GAConfig.N_SITES_ARM_X, GAConfig.N_SITES_ARM_Y, GAConfig.N_SITES_ARM_Z), "extents": (GAConfig.RADIUS + GAConfig.MARGIN_R_ARM, GAConfig.RADIUS + GAConfig.MARGIN_R_ARM, GAConfig.HALF_LENGTH + GAConfig.MARGIN_Z_ARM), "size": 0.003, "rgba": [0, 1, 1, 1]},
        "target": {"body": "target", "counts": (GAConfig.N_SITES_OBJECT_X, GAConfig.N_SITES_OBJECT_Y, GAConfig.N_SITES_OBJECT_Z), "extents": (GAConfig.BOX_DIM + GAConfig.MARGIN_R_OBJECT, GAConfig.BOX_DIM + GAConfig.MARGIN_R_OBJECT, GAConfig.BOX_DIM + GAConfig.MARGIN_Z_OBJECT), "size": 0.003, "rgba": [0, 1, 0, 0.3]},
        "l_link6": {"body": "l_link6", "counts": (GAConfig.N_SITES_LINK5_X, GAConfig.N_SITES_LINK5_Y, GAConfig.N_SITES_LINK5_Z), "extents": (0.03, 0.03, 0.03), "size": 0.004, "rgba": [1, 0.8, 0, 0.8]},
        "r_link6": {"body": "r_link6", "counts": (GAConfig.N_SITES_LINK5_X, GAConfig.N_SITES_LINK5_Y, GAConfig.N_SITES_LINK5_Z), "extents": (0.03, 0.03, 0.03), "size": 0.004, "rgba": [1, 0.8, 0, 0.8]},
        "l_link5": {"body": "l_link5", "counts": (GAConfig.N_SITES_LINK5_X, GAConfig.N_SITES_LINK5_Y, GAConfig.N_SITES_LINK5_Z), "extents": (0.03, 0.03, 0.03), "size": 0.004, "rgba": [1, 0.8, 0, 0.8]},
        "r_link5": {"body": "r_link5", "counts": (GAConfig.N_SITES_LINK5_X, GAConfig.N_SITES_LINK5_Y, GAConfig.N_SITES_LINK5_Z), "extents": (0.03, 0.03, 0.03), "size": 0.004, "rgba": [1, 0.8, 0, 0.8]},
        "l_link4": {"body": "l_link4", "counts": (GAConfig.N_SITES_LINK4_X, GAConfig.N_SITES_LINK4_Y, GAConfig.N_SITES_LINK4_Z), "extents": (0.03, 0.03, 0.03), "size": 0.004, "rgba": [1, 0.4, 0.2, 0.8]},
        "r_link4": {"body": "r_link4", "counts": (GAConfig.N_SITES_LINK4_X, GAConfig.N_SITES_LINK4_Y, GAConfig.N_SITES_LINK4_Z), "extents": (0.03, 0.03, 0.03), "size": 0.004, "rgba": [1, 0.4, 0.2, 0.8]},
        "body_link": {"body": "body_link", "counts": (GAConfig.N_SITES_BODY_X, GAConfig.N_SITES_BODY_Y, GAConfig.N_SITES_BODY_Z), "extents": (0.08, 0.06, 0.08), "size": 0.005, "rgba": [1, 1, 0, 0.6]},
    }
    families_needed = set()
    for group in GAConfig.CONNECTION_GROUPS:
        families_needed.add(group["src"])
        families_needed.add(group["dst"])

    t0 = time.perf_counter()
    total_sites_added = 0
    for family in sorted(families_needed):
        cfg = family_cfg[family]
        nx, ny, nz = cfg["counts"]
        ex, ey, ez = cfg["extents"]
        xs = np.linspace(-ex, ex, nx)
        ys = np.linspace(-ey, ey, ny)
        zs = np.linspace(-ez, ez, nz)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                for k, z in enumerate(zs):
                    builder.add_site(
                        name=f"{family}_site_{i}_{j}_{k}",
                        body_name=cfg["body"],
                        pos=[float(x), float(y), float(z)],
                        size=cfg["size"],
                        rgba=cfg["rgba"],
                    )
                    total_sites_added += 1
    times["add_connection_family_sites"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    builder.add_tendons_from_graph(G)
    times["add_tendons_from_graph"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sim = MuJoCoSimulation.from_builder(builder, keep_builder_spec=return_builder)
    times["build_simulation"] = time.perf_counter() - t0
    
    # Clean up builder and graph copy if caller doesn't need builder for export.
    if not return_builder:
        del builder
        del G
        gc.collect()

    t0 = time.perf_counter()
    # Set initial joint positions for arms
    initial_joint_positions = np.array([0.05062136600022616, -0.3436116964863836, -1.3959225169759335, 
                    0.10584467436410924, -1.5539225381281545, 0.09817477042468103, 2.741223667951641,
                    0.02761165418194154, -1.7916895602504288, -1.6122138080678088, 0.3, 
                    -0.0798, 1.53, 0.145, -2.26,
                    0.0372, 1.07, -2, -0.3], dtype=np.float32)
    joint_names = [
    "waist_yaw_joint", "neck_yaw_joint", "neck_pitch_joint",
    "r_arm_joint1","r_arm_joint2","r_arm_joint3","r_arm_joint4",
    "r_arm_joint5","r_arm_joint6","r_arm_joint7","r_hand_mimic_joint",
    "l_arm_joint1","l_arm_joint2","l_arm_joint3","l_arm_joint4",
    "l_arm_joint5","l_arm_joint6","l_arm_joint7","l_hand_mimic_joint"
    ]
    if joint_position_offsets:
        for idx, name in enumerate(joint_names):
            initial_joint_positions[idx] += float(joint_position_offsets.get(name, 0.0))

    for name, pos in zip(joint_names, initial_joint_positions):
        try:
            joint = sim.reg_joint(name, name)
            joint.qpos = pos
        except Exception as e:
            if verbose := False:
                print(f"Error setting joint '{name}': {e}")

    sim.forward()

    # Set up torque controller from tendon Jacobians (no ghost object sync).
    tendon_names = [
        name for i in range(sim.model.ntendon)
        if (name := mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_TENDON, i)) is not None
    ]
    torque_controller = TendonTorqueController(
        sim,
        tendon_names=tendon_names,
        lift_tendon_names=["left_lifter", "right_lifter", "left_end_lifter", "right_end_lifter"],
        lift_delay=GAConfig.LIFT_DELAY,
        lift_duration=GAConfig.LIFT_DURATION,
        torque_clip=GAConfig.TORQUE_CLIP,
        velocity_delay=GAConfig.VELOCITY_DELAY,
    )
    # Keep original tendon values for export; model tendon stiffness/damping are zeroed for torque control.
    sim._tendon_export_params = torque_controller.original_tendon_params
    sim.set_control_callback(torque_controller.callback, gravity_comp=True)

    times["register_joints_and_controller"] = time.perf_counter() - t0

    times["total_build_model_from_genome"] = time.perf_counter() - t_total_start

    # Summary
    try:
        num_tendons = G.number_of_edges()
    except Exception:
        num_tendons = -1

    # print("Timing breakdown (build_model_from_genome):")
    # print(f"  builder_init_and_xml:       {times['builder_init_and_xml']:.4f}s")
    # print(f"  add_static_bodies:          {times['add_static_bodies']:.4f}s")
    # print(f"  remove_end_links:           {times['remove_end_links']:.4f}s")
    # print(f"  add_chopstick_bodies:       {times['add_chopstick_bodies']:.4f}s")
    # print(f"  compute_arm_site_grid:      {times['compute_arm_site_grid']:.4f}s")
    # print(f"  add_arm_sites ({arm_site_count}):   {times['add_arm_sites']:.4f}s")
    # print(f"  compute_object_site_grid:   {times['compute_object_site_grid']:.4f}s")
    # print(f"  add_object_sites ({object_site_count}): {times['add_object_sites']:.4f}s")
    # print(f"  add_tendons_from_graph ({num_tendons}): {times['add_tendons_from_graph']:.4f}s")
    # print(f"  build_simulation:           {times['build_simulation']:.4f}s")
    # print(f"  register_joints/controller: {times['register_joints_and_controller']:.4f}s")
    # print(f"  TOTAL:                      {times['total_build_model_from_genome']:.4f}s")

    if return_builder:
        return sim, builder
    return sim

def evaluate_fitness(individual: creator.Individual) -> Tuple[float,]:
    """
    Evaluate fitness of an individual.
    
    Returns:
        Tuple with single fitness value (DEAP requirement)
    """
    # try:
    # Decode genome
    G = decode_genome(individual)
    
    # print("Genome decoded")
    
    # Build simulation
    def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dtype=np.float64)

    trial_scores = []
    trial_height_gains = []
    trial_force_penalties = []
    trial_angles = list(GAConfig.TARGET_TRIAL_YAWS)
    trial_y_positions = [GAConfig.TARGET_BASE_Y + dy for dy in GAConfig.TARGET_TRIAL_Y_OFFSETS]
    trial_x_positions = [GAConfig.TARGET_BASE_X + dx for dx in GAConfig.TARGET_TRIAL_X_OFFSETS]
    arm_perturbations = list(GAConfig.ARM_TRIAL_PERTURBATIONS)
    trial_num = 3

    for i in range(trial_num):
        angle = random.choice(trial_angles)
        x_pos = random.choice(trial_x_positions)
        arm_offsets = random.choice(arm_perturbations)
        y_pos = random.choice(trial_y_positions)
        sim = build_model_from_genome(
            G,
            angle=angle,
            target_pos=[x_pos, y_pos, GAConfig.TARGET_BASE_Z],
            joint_position_offsets=arm_offsets,
        )
        # print("Model built from genome")
        
        # Register bodies
        target = sim.reg_body('target_block', 'target')
        left = sim.reg_body('left', 'left_chopstick')
        right = sim.reg_body('right', 'right_chopstick')
        
        # Track metrics
        initial_height = target.pos[2]
        total_contact = 0.0
        total_velocity = 0.0
        total_height_gain = 0.0
        total_ctrl_effort = 0.0
        total_ctrl_delta = 0.0
        prev_ctrl = None
        n_steps = 0

        # Detect contact between chopsticks and target
        contact_sensor = sim.reg_contact_sensor(
            "target_contact_sensor",
            "target",
            ["left_chopstick", "right_chopstick"]
        )
        
        
        # Run simulation
        num_steps = int(GAConfig.SIM_DURATION / sim.model.opt.timestep)
        target_jnt_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "target_freejoint")
        target_qadr = int(sim.model.jnt_qposadr[target_jnt_id]) if target_jnt_id != -1 else -1

        
        for _ in range(num_steps):
            sim.step()

            # Apply small random pose disturbance to target during rollout for robustness.
            if target_qadr != -1:
                sim.data.qpos[target_qadr + 0] += random.uniform(-GAConfig.TARGET_STEP_JITTER_X, GAConfig.TARGET_STEP_JITTER_X)
                sim.data.qpos[target_qadr + 1] += random.uniform(-GAConfig.TARGET_STEP_JITTER_Y, GAConfig.TARGET_STEP_JITTER_Y)

                dyaw = random.uniform(-GAConfig.TARGET_STEP_YAW_JITTER, GAConfig.TARGET_STEP_YAW_JITTER)
                dq = np.array([math.cos(dyaw * 0.5), 0.0, 0.0, math.sin(dyaw * 0.5)], dtype=np.float64)
                q = np.array(sim.data.qpos[target_qadr + 3: target_qadr + 7], dtype=np.float64)
                q_new = _quat_mul(dq, q)
                q_norm = np.linalg.norm(q_new)
                if q_norm > 1e-12:
                    q_new /= q_norm
                    sim.data.qpos[target_qadr + 3: target_qadr + 7] = q_new
                mujoco.mj_forward(sim.model, sim.data)
            
            # Get contacts and sum forces
            contacts = contact_sensor.get_contacts()
            if len(contacts) > 0:
                # Sum up all contact forces
                step_contact_force = sum(c.total_force for c in contacts)
                total_contact += step_contact_force
                
            # Track target velocity (for stability)
            vel = np.linalg.norm(target.linear_vel)
            total_velocity += vel

            # Penalize large torques and abrupt changes in control.
            ctrl = sim.data.ctrl.copy()
            total_ctrl_effort += float(np.mean(np.abs(ctrl)))
            if prev_ctrl is not None:
                total_ctrl_delta += float(np.mean(np.abs(ctrl - prev_ctrl)))
            prev_ctrl = ctrl

            # Track integrated height gain relative to initial pose
            step_height_gain = target.pos[2] - initial_height
            total_height_gain += step_height_gain
            n_steps += 1
        

        # Final height
        final_height = target.pos[2]
        final_height_gain = final_height - initial_height
        final_height_gain = final_height_gain if final_height_gain < 1.5 else 0
        avg_height_gain = total_height_gain / n_steps if n_steps > 0 else 0.0
        height_gain = (
            GAConfig.HEIGHT_FINAL_RATIO * final_height_gain +
            GAConfig.HEIGHT_INTEGRAL_RATIO * avg_height_gain
        )

        final_centre_dist = np.linalg.norm(target.pos[:2] - np.array([0.5, 0.0]))
        dist = final_centre_dist if final_centre_dist > 0.05 else 0
        
        # Average velocity (lower is more stable)
        avg_velocity = total_velocity / n_steps if n_steps > 0 else 0
        stability = 1.0 / (1.0 + avg_velocity)
        
        # Count active tendons (fewer is more efficient)
        num_tendons = G.number_of_edges()
        efficiency = 1.0 / (1.0 + num_tendons)

        centre_dist = 1.0 / (1.0 + dist)
        
        avg_contact_force = total_contact / n_steps if n_steps > 0 else 0.0
        # Smooth bounded penalty in [0, 1) that increases with contact force.
        contact_force_penalty = avg_contact_force / (1.0 + avg_contact_force)
        avg_ctrl_effort = total_ctrl_effort / n_steps if n_steps > 0 else 0.0
        avg_ctrl_delta = total_ctrl_delta / max(n_steps - 1, 1)
        control_effort_penalty = avg_ctrl_effort / (1.0 + avg_ctrl_effort)
        control_smoothness_penalty = avg_ctrl_delta / (1.0 + avg_ctrl_delta)

        # Weighted fitness
        trial_score = (
            GAConfig.CONTACT_WEIGHT * total_contact +
            GAConfig.HEIGHT_WEIGHT * height_gain +
            GAConfig.STABILITY_WEIGHT * stability +
            GAConfig.EFFICIENCY_WEIGHT * efficiency +
            GAConfig.CENTRE_WEIGHT * centre_dist -
            GAConfig.FORCE_PENALTY_WEIGHT * contact_force_penalty -
            GAConfig.CONTROL_EFFORT_WEIGHT * control_effort_penalty -
            GAConfig.CONTROL_SMOOTHNESS_WEIGHT * control_smoothness_penalty
        )
        trial_scores.append(float(trial_score))
        trial_height_gains.append(float(height_gain))
        trial_force_penalties.append(float(
            contact_force_penalty +
            control_effort_penalty +
            control_smoothness_penalty
        ))
        
        # Clean up simulation resources
        sim.close()
        del sim
        
        # Force garbage collection to free memory immediately
        gc.collect()
    
    avg_trial_score = float(np.mean(trial_scores)) if trial_scores else 0.0
    height_std = float(np.std(trial_height_gains)) if trial_height_gains else 0.0
    score_std = float(np.std(trial_scores)) if trial_scores else 0.0
    force_std = float(np.std(trial_force_penalties)) if trial_force_penalties else 0.0
    robustness_bonus = 1.0 / (1.0 + height_std + 0.5 * score_std + 0.5 * force_std)

    fitness = avg_trial_score + GAConfig.ROBUSTNESS_WEIGHT * robustness_bonus
    return (fitness,)  # DEAP requires tuple
    
    # except Exception as e:
    #     import traceback
    #     print(f"Error evaluating individual: {e}")
    #     traceback.print_exc()
        
    #     # Try to clean up if sim was created
    #     try:
    #         sim.close()
    #         del sim
    #         gc.collect()
    #     except:
    #         pass
        
    #     return (0.0,)  # Return zero fitness on error


def export_model_and_tendon_json(sim: MuJoCoSimulation, run_dir: str, builder: Optional[ModelBuilder] = None) -> Tuple[str, str]:
    """
    Export model XML and tendon metadata JSON for external software.

    Returns:
        (xml_path, json_path)
    """
    xml_path = os.path.join(run_dir, "best_model.xml")
    json_path = os.path.join(run_dir, "best_model_tendons.json")

    # Save model as MJCF XML.
    if builder is not None and builder.spec is not None:
        builder.spec.to_file(xml_path)
    else:
        # Fallback path for models loaded directly from XML.
        mujoco.mj_saveLastXML(xml_path, sim.model)

    export_params = getattr(sim, "_tendon_export_params", {})
    tendon_entries = []
    for t_id in range(sim.model.ntendon):
        name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_TENDON, t_id)
        if name is None:
            continue

        if name in export_params:
            k_val = float(export_params[name]["stiffness"])
            b_val = float(export_params[name]["damping"])
            l_val = float(export_params[name]["length"])
        else:
            k_arr = np.atleast_1d(sim.model.tendon_stiffness[t_id]).astype(float)
            b_arr = np.atleast_1d(sim.model.tendon_damping[t_id]).astype(float)
            l_arr = np.atleast_1d(sim.model.tendon_lengthspring[t_id]).astype(float)
            k_val = float(k_arr[0])
            b_val = float(b_arr[0])
            l_val = float(l_arr[-1])

        tendon_entries.append({
            "name": name,
            "cosntant": k_val,
            "damping": b_val,
            "rest_length": l_val,
        })

    with open(json_path, "w") as f:
        json.dump(tendon_entries, f, indent=2)

    return xml_path, json_path


# ============================================================================
# Genetic Operators
# ============================================================================

def mutate_individual(individual: creator.Individual) -> Tuple[creator.Individual,]:
    """
    Mutate an individual using:
    - Polynomial bounded mutation for continuous parameters (stiffness, damping, length)
    - Integer creep mutation for discrete connection indices (small local changes)
    - Occasional random resampling for exploration

    This balances exploitation (local integer moves) and exploration (random jumps).
    """
    gene_specs = expanded_gene_specs()

    # Mutate discrete connection indices.
    for i in range(individual.shape[0]):
        if i >= len(gene_specs):
            break
        spec = gene_specs[i]
        src_count = family_site_count(str(spec["src"]))
        dst_count = family_site_count(str(spec["dst"]))

        # Mutate source index.
        if random.random() < GAConfig.FLIP_PROB:
            if random.random() < 0.7:  # Creep mutation (local search)
                step = random.choice([-3, -2, -1, 1, 2, 3])
                new_val = int(individual[i][0]) + step
                individual[i][0] = max(-1, min(src_count - 1, new_val))
            else:  # Random resampling (exploration)
                individual[i][0] = np.random.randint(-1, src_count)

        # Mutate destination index.
        if random.random() < GAConfig.FLIP_PROB:
            if random.random() < 0.7:  # Creep mutation
                step = random.choice([-3, -2, -1, 1, 2, 3])
                new_val = int(individual[i][1]) + step
                individual[i][1] = max(-1, min(dst_count - 1, new_val))
            else:  # Random resampling
                individual[i][1] = np.random.randint(-1, dst_count)

    # Mutate continuous parameters (stiffness, damping, length) with polynomial bounded mutation.
    continuous_params = individual[:, 2:5].flatten()

    low_bounds = []
    up_bounds = []
    for _ in range(individual.shape[0]):
        low_bounds.extend([GAConfig.MIN_STIFFNESS, GAConfig.MIN_DAMPING, GAConfig.MIN_SPRING_LENGTH])
        up_bounds.extend([GAConfig.MAX_STIFFNESS, GAConfig.MAX_DAMPING, GAConfig.MAX_SPRING_LENGTH])

    mutated_params, = tools.mutPolynomialBounded(
        continuous_params,
        eta=20.0,
        low=low_bounds,
        up=up_bounds,
        indpb=0.3,
    )

    individual[:, 2:5] = np.array(mutated_params).reshape(-1, 3)
    
    return (individual,)


def crossover_individuals(ind1: creator.Individual, ind2: creator.Individual) -> Tuple[creator.Individual, creator.Individual]:
    """
    Crossover two individuals using:
    - Two-point crossover for discrete connection indices
    - Simulated Binary Crossover (SBX) for continuous parameters
    """
    num_tendons = ind1.shape[0]

    # Two-point crossover for discrete columns (0 and 1).
    if num_tendons > 2:
        point1 = random.randint(1, num_tendons - 1)
        point2 = random.randint(1, num_tendons - 1)
        if point1 > point2:
            point1, point2 = point2, point1

        for col in [0, 1]:
            seg1 = ind1[point1:point2, col].copy()
            seg2 = ind2[point1:point2, col].copy()
            ind1[point1:point2, col] = seg2
            ind2[point1:point2, col] = seg1
    else:
        point = random.randint(1, num_tendons)
        for col in [0, 1]:
            seg1 = ind1[point:, col].copy()
            seg2 = ind2[point:, col].copy()
            ind1[point:, col] = seg2
            ind2[point:, col] = seg1

    # SBX for continuous columns (2, 3, 4).
    continuous_params1 = ind1[:, 2:5].flatten()
    continuous_params2 = ind2[:, 2:5].flatten()

    low_bounds = []
    up_bounds = []
    for _ in range(ind1.shape[0]):
        low_bounds.extend([GAConfig.MIN_STIFFNESS, GAConfig.MIN_DAMPING, GAConfig.MIN_SPRING_LENGTH])
        up_bounds.extend([GAConfig.MAX_STIFFNESS, GAConfig.MAX_DAMPING, GAConfig.MAX_SPRING_LENGTH])

    tools.cxSimulatedBinaryBounded(
        continuous_params1,
        continuous_params2,
        eta=20.0,
        low=low_bounds,
        up=up_bounds,
    )

    ind1[:, 2:5] = np.array(continuous_params1).reshape(-1, 3)
    ind2[:, 2:5] = np.array(continuous_params2).reshape(-1, 3)
    
    return ind1, ind2

# ============================================================================
# Helper Functions for DEAP
# ============================================================================

def array_similar(ind1, ind2):
    """
    Check if two individuals are similar by comparing their array values.
    
    This is used by DEAP's HallOfFame to determine if individuals are duplicates.
    Must be a module-level function to be picklable for checkpointing.
    """
    return np.allclose(ind1, ind2, rtol=1e-5, atol=1e-8)


# ============================================================================
# Statistics Plotting
# ============================================================================

def plot_statistics(logbook: tools.Logbook, save_path: str = None):
    """
    Plot evolution statistics over generations.
    
    Args:
        logbook: DEAP logbook containing statistics
        save_path: Path to save the figure (if None, displays instead)
    """
    gen = logbook.select("gen")
    fit_maxs = logbook.select("max")
    fit_avgs = logbook.select("avg")
    fit_mins = logbook.select("min")
    fit_stds = logbook.select("std")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot fitness over generations
    ax1.plot(gen, fit_maxs, 'b-', label='Max Fitness', linewidth=2)
    ax1.plot(gen, fit_avgs, 'g-', label='Avg Fitness', linewidth=2)
    ax1.plot(gen, fit_mins, 'r-', label='Min Fitness', linewidth=2)
    ax1.fill_between(gen, 
                      np.array(fit_avgs) - np.array(fit_stds),
                      np.array(fit_avgs) + np.array(fit_stds),
                      alpha=0.2, color='g', label='±1 Std Dev')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Fitness Evolution Over Generations')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot standard deviation over generations
    ax2.plot(gen, fit_stds, 'purple', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness Std Dev')
    ax2.set_title('Population Diversity Over Generations')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Statistics plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Main Evolution Loop
# ============================================================================

def run_evolution(visualize: bool = False, resume: str = None, plot_stats: bool = False, 
                  num_processes: int = None):
    """
    Run the genetic algorithm using DEAP with optional parallelization.
    
    Args:
        visualize: Whether to visualize the best individual after evolution
        resume: Path to checkpoint file to resume from
        plot_stats: Whether to generate statistics plots
        num_processes: Number of parallel processes (None = all cores, 1 = serial)
    """
    
    # Track start time
    evolution_start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create checkpoint and stats directories
    os.makedirs(GAConfig.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(GAConfig.STATS_DIR, exist_ok=True)
    
    # Create results directory for this run
    run_dir = os.path.join(GAConfig.RESULTS_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Create log file
    log_path = os.path.join(run_dir, "training_log.txt")
    log_file = open(log_path, 'w')
    
    def log(message):
        """Write to both console and log file"""
        print(message)
        log_file.write(message + '\n')
        log_file.flush()
    
    log("=" * 80)
    log(f"GA Training Run - {timestamp}")
    log("=" * 80)
    log("")
    
    # Setup DEAP toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_fitness)
    toolbox.register("mate", crossover_individuals)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=GAConfig.TOURNAMENT_SIZE)
    
    # Setup parallel processing if requested
    if num_processes is None:
        num_processes = GAConfig.NUM_PROCESSES
    
    if num_processes is None or num_processes > 1:
        # Use all available cores if num_processes is None
        pool = multiprocessing.Pool(processes=num_processes)
        toolbox.register("map", pool.map)
        if num_processes is None:
            log(f"Using all available CPU cores for parallel evaluation")
        else:
            log(f"Using {num_processes} parallel processes for evaluation")
    else:
        log("Running in serial mode (no parallelization)")
    
    # Log configuration
    log("Configuration Parameters:")
    log("-" * 80)
    log(f"  Population Size:      {GAConfig.POPULATION_SIZE}")
    log(f"  Generations:          {GAConfig.NUM_GENERATIONS}")
    log(f"  Tournament Size:      {GAConfig.TOURNAMENT_SIZE}")
    log(f"  Crossover Prob:       {GAConfig.CROSSOVER_PROB}")
    log(f"  Mutation Prob:        {GAConfig.MUTATION_PROB}")
    log(f"  Elite Size:           {GAConfig.ELITE_SIZE}")
    log(f"  Parallel Processes:   {num_processes if num_processes else 'All cores'}")
    log("")
    log(f"  Sites per Arm:        {GAConfig.N_SITES_ARM} ({GAConfig.N_SITES_ARM_X}x{GAConfig.N_SITES_ARM_Y}x{GAConfig.N_SITES_ARM_Z})")
    log(f"  Sites per Object:     {GAConfig.N_SITES_OBJECT} ({GAConfig.N_SITES_OBJECT_X}x{GAConfig.N_SITES_OBJECT_Y}x{GAConfig.N_SITES_OBJECT_Z})")
    log(f"  Stiffness Range:      {GAConfig.MIN_STIFFNESS} - {GAConfig.MAX_STIFFNESS} N/m")
    log(f"  Damping Range:        {GAConfig.MIN_DAMPING} - {GAConfig.MAX_DAMPING} N*s/m")
    log("  Connection Groups:")
    for group in GAConfig.CONNECTION_GROUPS:
        log(
            f"    - {group['name']}: {group['src']} -> {group['dst']}, "
            f"count={group['num_tendons']}, spring_mode={group.get('spring_mode', 'range')}"
        )
    log("")
    log(f"  Simulation Duration:  {GAConfig.SIM_DURATION} s")
    log(f"  Simulation Timestep:  {GAConfig.SIM_DT} s")
    log(f"  Target X Trials:      {[GAConfig.TARGET_BASE_X + dx for dx in GAConfig.TARGET_TRIAL_X_OFFSETS]}")
    log(f"  Target Yaw Trials:    {list(GAConfig.TARGET_TRIAL_YAWS)}")
    log(f"  Target Step Jitter X: ±{GAConfig.TARGET_STEP_JITTER_X} m")
    log(f"  Target Step Jitter Y: ±{GAConfig.TARGET_STEP_JITTER_Y} m")
    log(f"  Target Step Yaw Jitter: ±{GAConfig.TARGET_STEP_YAW_JITTER:.6f} rad")
    log(f"  Arm Start Trials:     {len(GAConfig.ARM_TRIAL_PERTURBATIONS)}")
    log(f"  Lift Delay:           {GAConfig.LIFT_DELAY} s")
    log(f"  Lift Duration:        {GAConfig.LIFT_DURATION} s")
    log(f"  Lift Force:           {GAConfig.LIFT_FORCE} N/m")
    _vd_steps = int(round(GAConfig.VELOCITY_DELAY / GAConfig.SIM_DT)) if GAConfig.SIM_DT > 0 else 0
    log(f"  Velocity Delay:       {GAConfig.VELOCITY_DELAY} s ({_vd_steps} sim steps)")
    log("")
    log(f"  Contact Weight:       {GAConfig.CONTACT_WEIGHT}")
    log(f"  Height Weight:        {GAConfig.HEIGHT_WEIGHT}")
    log(f"  Centre Weight:        {GAConfig.CENTRE_WEIGHT}")
    log(f"  Force Penalty Weight: {GAConfig.FORCE_PENALTY_WEIGHT}")
    log(f"  Control Effort:       {GAConfig.CONTROL_EFFORT_WEIGHT}")
    log(f"  Control Smoothness:   {GAConfig.CONTROL_SMOOTHNESS_WEIGHT}")
    log(f"  Robustness Weight:    {GAConfig.ROBUSTNESS_WEIGHT}")
    log(f"  Stability Weight:     {GAConfig.STABILITY_WEIGHT}")
    log(f"  Efficiency Weight:    {GAConfig.EFFICIENCY_WEIGHT}")
    log("")
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Hall of fame (best individuals) with custom similarity function for array comparison
    # array_similar is defined at module level to be picklable for checkpointing
    hof = tools.HallOfFame(GAConfig.ELITE_SIZE, similar=array_similar)
    
    # Create or load population
    if resume:
        log(f"Resuming from checkpoint: {resume}")
        with open(resume, 'rb') as f:
            checkpoint = pickle.load(f)
        population = checkpoint['population']
        start_gen = checkpoint['generation'] + 1
        logbook = checkpoint['logbook']
        hof = checkpoint['halloffame']
    else:
        log(f"Creating initial population of {GAConfig.POPULATION_SIZE} individuals...")
        population = toolbox.population(n=GAConfig.POPULATION_SIZE)
        start_gen = 0
        logbook = tools.Logbook()
    
    # Evaluate initial population if starting fresh
    if start_gen == 0:
        log("Evaluating initial population...")
        fitnesses = list(toolbox.map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        hof.update(population)
    
    # Evolution loop
    log(f"\nStarting evolution from generation {start_gen}...")
    log("=" * 60)
    
    for gen in range(start_gen, GAConfig.NUM_GENERATIONS):
        log(f"\nGeneration {gen}/{GAConfig.NUM_GENERATIONS}")
        gen_start_time = time.time()
        
        # Select next generation
        offspring = toolbox.select(population, len(population) - GAConfig.ELITE_SIZE)
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < GAConfig.CROSSOVER_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Mutation
        for mutant in offspring:
            if random.random() < GAConfig.MUTATION_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate offspring with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population (elitism: keep best from previous gen)
        population[:] = offspring + list(hof)
        
        # Update hall of fame
        hof.update(population)
        
        # Record statistics
        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        
        # Print stats
        gen_time = time.time() - gen_start_time
        log(f"  Best:  {record['max']:.2f}")
        log(f"  Avg:   {record['avg']:.2f}")
        log(f"  Std:   {record['std']:.2f}")
        log(f"  Time:  {gen_time:.2f}s")
        
        # Save checkpoint
        if (gen + 1) % GAConfig.CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(GAConfig.CHECKPOINT_DIR, f"gen_{gen}.pkl")
            checkpoint = {
                'generation': gen,
                'population': population,
                'halloffame': hof,
                'logbook': logbook
            }
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            log(f"  Checkpoint saved: {checkpoint_path}")
    
    # Calculate total time
    total_time = time.time() - evolution_start_time
    
    # Clean up parallel processing pool if used
    if num_processes is None or num_processes > 1:
        pool.close()
        pool.join()
    
    # Get best individual
    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]
    
    # Save final results to checkpoint dir
    final_path = os.path.join(GAConfig.CHECKPOINT_DIR, "best_final.pkl")
    
    result = {
        'individual': best_individual,
        'fitness': best_fitness,
        'generation': GAConfig.NUM_GENERATIONS - 1,
        'logbook': logbook
    }
    
    with open(final_path, 'wb') as f:
        pickle.dump(result, f)
    
    log("\n" + "=" * 80)
    log("Evolution Complete!")
    log("=" * 80)
    log(f"Best Fitness: {best_fitness:.4f}")
    log(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    log(f"Time per Generation: {total_time/GAConfig.NUM_GENERATIONS:.2f}s")
    log("")
    
    # Get final statistics
    final_stats = logbook[-1]
    log("Final Generation Statistics:")
    log(f"  Max Fitness:  {final_stats['max']:.4f}")
    log(f"  Avg Fitness:  {final_stats['avg']:.4f}")
    log(f"  Min Fitness:  {final_stats['min']:.4f}")
    log(f"  Std Dev:      {final_stats['std']:.4f}")
    log("")
    
    # Decode best individual for analysis
    G = decode_genome(best_individual)
    num_tendons = G.number_of_edges()
    log(f"Best Individual Analysis:")
    log(f"  Number of Tendons: {num_tendons}")
    log("")
    
    # Copy checkpoint to run directory
    shutil.copy(final_path, os.path.join(run_dir, "best_individual.pkl"))

    # Export best model as XML + tendon metadata JSON for external tools.
    export_sim = None
    export_builder = None
    try:
        export_sim, export_builder = build_model_from_genome(G.copy(), angle=0.0, return_builder=True)
        model_xml_path, tendon_json_path = export_model_and_tendon_json(export_sim, run_dir, builder=export_builder)
        log(f"Best model XML saved: {model_xml_path}")
        log(f"Tendon metadata JSON saved: {tendon_json_path}")
    except Exception as e:
        log(f"Warning: Failed to export best model XML/JSON: {e}")
    finally:
        if export_sim is not None:
            export_sim.close()
        if export_builder is not None:
            del export_builder
    
    # Generate and save statistics plot
    stats_plot_path = os.path.join(run_dir, "evolution_statistics.png")
    plot_statistics(logbook, save_path=stats_plot_path)
    log(f"Statistics plot saved: {stats_plot_path}")
    
    # Generate and save video of best individual
    log("\nGenerating video of best individual...")
    try:
        sim = build_model_from_genome(G.copy())
        video_path = os.path.join(run_dir, "best_individual.mp4")
        sim.run(
            passive=True,
            viewer_distance=5,
            viewer_lookat=[0, 0, 0.25],
            realtime_speed=0.01,
            duration=GAConfig.SIM_DURATION,
            control_preset=True,
            record_video=True
        )
        
        # Move video to run directory (it's saved in videos/ by default)
        default_video_path = "videos/simulation_recording.mp4"
        if os.path.exists(default_video_path):
            shutil.move(default_video_path, video_path)
            log(f"Video saved: {video_path}")
        else:
            log("Warning: Video file not found at expected location")
    except Exception as e:
        log(f"Warning: Failed to generate video: {e}")
        log("Continuing with other outputs...")
    
    # Save code snapshot
    code_snapshot_path = os.path.join(run_dir, "ga_tendon_optimizer_deap.py")
    shutil.copy(__file__, code_snapshot_path)
    log(f"Code snapshot saved: {code_snapshot_path}")
    
    # Save configuration as JSON
    config_path = os.path.join(run_dir, "configuration.json")
    config_dict = {
        'population_size': GAConfig.POPULATION_SIZE,
        'num_generations': GAConfig.NUM_GENERATIONS,
        'tournament_size': GAConfig.TOURNAMENT_SIZE,
        'crossover_prob': GAConfig.CROSSOVER_PROB,
        'mutation_prob': GAConfig.MUTATION_PROB,
        'elite_size': GAConfig.ELITE_SIZE,
        'num_processes': num_processes if num_processes else 'all',
        'n_sites_arm': GAConfig.N_SITES_ARM,
        'n_sites_arm_x': GAConfig.N_SITES_ARM_X,
        'n_sites_arm_y': GAConfig.N_SITES_ARM_Y,
        'n_sites_arm_z': GAConfig.N_SITES_ARM_Z,
        'n_sites_object': GAConfig.N_SITES_OBJECT,
        'n_sites_object_x': GAConfig.N_SITES_OBJECT_X,
        'n_sites_object_y': GAConfig.N_SITES_OBJECT_Y,
        'n_sites_object_z': GAConfig.N_SITES_OBJECT_Z,
        'min_stiffness': GAConfig.MIN_STIFFNESS,
        'max_stiffness': GAConfig.MAX_STIFFNESS,
        'min_damping': GAConfig.MIN_DAMPING,
        'max_damping': GAConfig.MAX_DAMPING,
        'sim_duration': GAConfig.SIM_DURATION,
        'sim_dt': GAConfig.SIM_DT,
        'target_base_x': GAConfig.TARGET_BASE_X,
        'target_base_y': GAConfig.TARGET_BASE_Y,
        'target_base_z': GAConfig.TARGET_BASE_Z,
        'target_trial_x_offsets': list(GAConfig.TARGET_TRIAL_X_OFFSETS),
        'target_trial_yaws': list(GAConfig.TARGET_TRIAL_YAWS),
        'target_step_jitter_x': GAConfig.TARGET_STEP_JITTER_X,
        'target_step_jitter_y': GAConfig.TARGET_STEP_JITTER_Y,
        'target_step_yaw_jitter': GAConfig.TARGET_STEP_YAW_JITTER,
        'arm_trial_perturbations': list(GAConfig.ARM_TRIAL_PERTURBATIONS),
        'lift_delay': GAConfig.LIFT_DELAY,
        'lift_duration': GAConfig.LIFT_DURATION,
        'lift_force': GAConfig.LIFT_FORCE,
        'velocity_delay': GAConfig.VELOCITY_DELAY,
        'contact_weight': GAConfig.CONTACT_WEIGHT,
        'height_weight': GAConfig.HEIGHT_WEIGHT,
        'centre_weight': GAConfig.CENTRE_WEIGHT,
        'force_penalty_weight': GAConfig.FORCE_PENALTY_WEIGHT,
        'control_effort_weight': GAConfig.CONTROL_EFFORT_WEIGHT,
        'control_smoothness_weight': GAConfig.CONTROL_SMOOTHNESS_WEIGHT,
        'robustness_weight': GAConfig.ROBUSTNESS_WEIGHT,
        'stability_weight': GAConfig.STABILITY_WEIGHT,
        'efficiency_weight': GAConfig.EFFICIENCY_WEIGHT,
        'best_fitness': float(best_fitness),
        'num_tendons': int(num_tendons),
        'total_time_seconds': float(total_time),
        'time_per_generation_seconds': float(total_time / GAConfig.NUM_GENERATIONS),
        'timestamp': timestamp
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    log(f"Configuration saved: {config_path}")
    
    # Create README for this run
    readme_path = os.path.join(run_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(f"# GA Training Run - {timestamp}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Best Fitness:** {best_fitness:.4f}\n")
        f.write(f"- **Number of Tendons:** {num_tendons}\n")
        f.write(f"- **Total Training Time:** {total_time/60:.2f} minutes\n")
        f.write(f"- **Generations:** {GAConfig.NUM_GENERATIONS}\n")
        f.write(f"- **Population Size:** {GAConfig.POPULATION_SIZE}\n\n")
        
        f.write(f"## Files in This Directory\n\n")
        f.write(f"- `training_log.txt` - Complete training log with all console output\n")
        f.write(f"- `best_individual.pkl` - Saved best individual (can be loaded with --load)\n")
        f.write(f"- `best_model.xml` - Exported MuJoCo model of best individual\n")
        f.write(f"- `best_model_tendons.json` - Tendon name/stiffness/damping/length for interoperability\n")
        f.write(f"- `evolution_statistics.png` - Fitness evolution plots\n")
        f.write(f"- `best_individual.mp4` - Video of best individual performing task\n")
        f.write(f"- `configuration.json` - All configuration parameters in JSON format\n")
        f.write(f"- `ga_tendon_optimizer_deap.py` - Code snapshot from this run\n")
        f.write(f"- `README.md` - This file\n\n")
        
        f.write(f"## Configuration Highlights\n\n")
        f.write(f"### GA Parameters\n")
        f.write(f"- Tournament Size: {GAConfig.TOURNAMENT_SIZE}\n")
        f.write(f"- Crossover Probability: {GAConfig.CROSSOVER_PROB}\n")
        f.write(f"- Mutation Probability: {GAConfig.MUTATION_PROB}\n")
        f.write(f"- Elite Size: {GAConfig.ELITE_SIZE}\n")
        f.write(f"- Parallel Processes: {num_processes if num_processes else 'All cores'}\n\n")
        
        f.write(f"### Network Parameters\n")
        f.write(f"- Sites per Arm: {GAConfig.N_SITES_ARM} ({GAConfig.N_SITES_ARM_X}×{GAConfig.N_SITES_ARM_Y}×{GAConfig.N_SITES_ARM_Z})\n")
        f.write(f"- Sites per Object: {GAConfig.N_SITES_OBJECT} ({GAConfig.N_SITES_OBJECT_X}×{GAConfig.N_SITES_OBJECT_Y}×{GAConfig.N_SITES_OBJECT_Z})\n")
        f.write(f"- Stiffness Range: {GAConfig.MIN_STIFFNESS} - {GAConfig.MAX_STIFFNESS} N/m\n")
        f.write(f"- Damping Range: {GAConfig.MIN_DAMPING} - {GAConfig.MAX_DAMPING} N·s/m\n\n")
        
        f.write(f"### Fitness Weights\n")
        f.write(f"- Contact: {GAConfig.CONTACT_WEIGHT}\n")
        f.write(f"- Height: {GAConfig.HEIGHT_WEIGHT}\n")
        f.write(f"- Centre: {GAConfig.CENTRE_WEIGHT}\n")
        f.write(f"- Force Penalty: {GAConfig.FORCE_PENALTY_WEIGHT}\n")
        f.write(f"- Control Effort: {GAConfig.CONTROL_EFFORT_WEIGHT}\n")
        f.write(f"- Control Smoothness: {GAConfig.CONTROL_SMOOTHNESS_WEIGHT}\n")
        f.write(f"- Robustness: {GAConfig.ROBUSTNESS_WEIGHT}\n")
        f.write(f"- Stability: {GAConfig.STABILITY_WEIGHT}\n")
        f.write(f"- Efficiency: {GAConfig.EFFICIENCY_WEIGHT}\n\n")
        
        f.write(f"## How to Load This Run\n\n")
        f.write(f"```bash\n")
        f.write(f"# Visualize the best individual\n")
        f.write(f"mjpython training/ga_tendon_optimizer_deap.py --load {os.path.relpath(os.path.join(run_dir, 'best_individual.pkl'))}\n\n")
        f.write(f"# Visualize and record video\n")
        f.write(f"mjpython training/ga_tendon_optimizer_deap.py --load {os.path.relpath(os.path.join(run_dir, 'best_individual.pkl'))} --record\n\n")
        f.write(f"# Plot statistics\n")
        f.write(f"mjpython training/ga_tendon_optimizer_deap.py --load {os.path.relpath(os.path.join(run_dir, 'best_individual.pkl'))} --stats\n")
        f.write(f"```\n")
    
    log(f"README saved: {readme_path}")
    
    # Update runs summary file
    summary_path = os.path.join(GAConfig.RESULTS_DIR, "runs_summary.csv")
    summary_exists = os.path.exists(summary_path)
    
    with open(summary_path, 'a') as f:
        if not summary_exists:
            # Write header
            f.write("timestamp,run_dir,best_fitness,num_tendons,generations,population_size,")
            f.write("total_time_minutes,time_per_gen_seconds,num_processes\n")
        
        # Write this run's data
        f.write(f"{timestamp},{os.path.basename(run_dir)},{best_fitness:.4f},{num_tendons},")
        f.write(f"{GAConfig.NUM_GENERATIONS},{GAConfig.POPULATION_SIZE},")
        f.write(f"{total_time/60:.2f},{total_time/GAConfig.NUM_GENERATIONS:.2f},")
        f.write(f"{num_processes if num_processes else 'all'}\n")
    
    log(f"Run summary updated: {summary_path}")
    
    log("")
    log("=" * 80)
    log(f"All results saved to: {run_dir}")
    log("=" * 80)
    log("")
    log("Directory Contents:")
    log("  - README.md                 : Quick summary and usage guide")
    log("  - training_log.txt          : This log file")
    log("  - best_individual.pkl       : Best evolved individual")
    log("  - evolution_statistics.png  : Fitness plots over generations")
    log("  - best_individual.mp4       : Video of best individual")
    log("  - configuration.json        : All configuration parameters")
    log("  - ga_tendon_optimizer_deap.py : Code snapshot")
    
    # Close log file
    log_file.close()
    
    # Visualize best individual if requested
    if visualize:
        print("\nVisualizing best individual...")
        sim = build_model_from_genome(G.copy())
        sim.run(
            passive=True,
            viewer_distance=5,
            viewer_lookat=[0, 0, 0.25],
            realtime_speed=1,
            duration=GAConfig.SIM_DURATION,
            control_preset=True
        )
    
    return best_individual, logbook


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Tendon Network Optimization")
    parser.add_argument('--visualize', action='store_true', help='Visualize best individual after evolution')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint file')
    parser.add_argument('--load', type=str, help='Load and visualize a saved individual')
    parser.add_argument('--export-pkl', type=str, help='Load a saved individual/checkpoint pickle and export XML + tendon JSON')
    parser.add_argument('--export-dir', type=str, default=None, help='Output directory for --export-pkl (default: same directory as pickle)')
    parser.add_argument('--record', action='store_true', help='Record simulation video when visualizing')
    parser.add_argument('--stats', action='store_true', help='Generate and save statistics plots')
    parser.add_argument('--parallel', type=int, default=None, metavar='N',
                        help='Number of parallel processes (default: all cores, 1=serial)')
    
    args = parser.parse_args()
    
    if args.export_pkl:
        print(f"Loading individual for export from: {args.export_pkl}")
        with open(args.export_pkl, 'rb') as f:
            result = pickle.load(f)

        if 'individual' in result:
            individual = result['individual']
            if 'fitness' in result:
                print(f"Fitness: {result['fitness']:.4f}")
        elif 'halloffame' in result and len(result['halloffame']) > 0:
            individual = result['halloffame'][0]
            print("Loaded best individual from Hall of Fame")
        else:
            raise ValueError("Unsupported pickle format: expected keys 'individual' or non-empty 'halloffame'")

        G = decode_genome(individual)
        out_dir = args.export_dir if args.export_dir else os.path.dirname(os.path.abspath(args.export_pkl))
        os.makedirs(out_dir, exist_ok=True)

        export_sim = None
        export_builder = None
        try:
            export_sim, export_builder = build_model_from_genome(G.copy(), angle=0.0, return_builder=True)
            xml_path, json_path = export_model_and_tendon_json(export_sim, out_dir, builder=export_builder)
            print(f"Exported model XML: {xml_path}")
            print(f"Exported tendon JSON: {json_path}")
        finally:
            if export_sim is not None:
                export_sim.close()
            if export_builder is not None:
                del export_builder

    elif args.load:
        # Load and visualize saved individual
        print(f"Loading individual from: {args.load}")
        with open(args.load, 'rb') as f:
            result = pickle.load(f)
        
        if 'individual' in result:
            individual = result['individual']
            print(f"Fitness: {result['fitness']:.2f}")
            
            # Plot stats if requested and logbook exists
            if args.stats and 'logbook' in result:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                stats_path = os.path.join(GAConfig.STATS_DIR, f"loaded_stats_{timestamp}.png")
                os.makedirs(GAConfig.STATS_DIR, exist_ok=True)
                plot_statistics(result['logbook'], save_path=stats_path)
        else:
            individual = result['halloffame'][1]
            print("Loaded individual from Hall of Fame")

        G = decode_genome(individual)
        num_tendons = G.number_of_edges()
        print(f"Number of tendons: {num_tendons}")

        sim = build_model_from_genome(G)
        sim.run(
            passive=True,
            viewer_distance=5,
            viewer_lookat=[0,0,0.25],
            realtime_speed=1,
            duration=4.0,
            control_preset=True,
            record_video=args.record
            )
    else:
        # Run evolution
        run_evolution(visualize=args.visualize, resume=args.resume, 
                     plot_stats=args.stats, num_processes=args.parallel)
        
