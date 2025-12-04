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
    NUM_GENERATIONS = 15
    TOURNAMENT_SIZE = 3
    CROSSOVER_PROB = 0.7
    MUTATION_PROB = 0.1
    ELITE_SIZE = 3
    
    # Parallelization
    NUM_PROCESSES = None  # None = use all CPU cores, or set to specific number
    
    # Body and Simulation Parameters
    RADIUS = 0.01  # radius of chopstick
    HALF_LENGTH = 0.15  # half length of chopstick

    # Site parameters
    N_SITES_ARM_X = 3  # Number of connection sites per chopstick 
    N_SITES_ARM_Y = 3  # Number of connection sites per chopstick 
    N_SITES_ARM_Z = 5  # Number of connection sites per chopstick
    N_SITES_ARM = N_SITES_ARM_X * N_SITES_ARM_Y * N_SITES_ARM_Z  # Total sites per chopstick
    N_SITES_OBJECT_X = 4  # Number of connection sites on object 
    N_SITES_OBJECT_Y = 4  # Number of connection sites on object 
    N_SITES_OBJECT_Z = 4  # Number of connection sites on object 
    N_SITES_OBJECT = N_SITES_OBJECT_X * N_SITES_OBJECT_Y * N_SITES_OBJECT_Z  # Total sites on object
    MARGIN_R_ARM = 0
    MARGIN_Z_ARM = 0
    MARGIN_R_OBJECT = 0
    MARGIN_Z_OBJECT = 0

    # Tendon parameters
    MIN_STIFFNESS = 10.0   # N/m
    MAX_STIFFNESS = 100.0  # N/m
    MIN_DAMPING = 1.0      # N*s/m
    MAX_DAMPING = 50.0     # N*s/m
    TENDON_NUM = 3
    
    # Mutation parameters
    FLIP_PROB = 0.3        # Probability to flip each connection
    GAUSSIAN_MU = 0.0
    GAUSSIAN_SIGMA = 10.0  # For stiffness/damping mutation
    
    # Simulation parameters
    SIM_DURATION = 4.0     # seconds
    SIM_DT = 0.01          # seconds
    
    # Fitness weights
    CONTACT_WEIGHT = 0
    HEIGHT_WEIGHT = 5
    CENTRE_WEIGHT = 5
    STABILITY_WEIGHT = 2
    EFFICIENCY_WEIGHT = 2
    
    # Checkpoint
    CHECKPOINT_DIR = "ga_checkpoints"
    CHECKPOINT_INTERVAL = 5  # Save every N generations
    
    # Statistics
    STATS_DIR = "ga_stats"
    
    # Results archiving
    RESULTS_DIR = "training_results"

    # Chopstick lift control parameters
    LIFT_DELAY = 4         # seconds before applying the lift
    LIFT_DURATION = 1.5       # seconds to keep applying the lift (None = indefinite)
    LIFT_FORCE = 20.0         # nominal upward force/torque magnitude
    


# ============================================================================
# DEAP Setup - Define Fitness and Individual
# ============================================================================

# Maximize fitness
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)


# ============================================================================
# Genome Encoding/Decoding
# ============================================================================

def create_individual():
    """
    Create a random individual.
    
    Genome encoding (flat list):
    - First N*(N-1)/2 values: adjacency matrix (upper triangle, binary 0/1)
    - Next N*(N-1)/2 values: stiffness values (only for connected tendons)
    - Last N*(N-1)/2 values: damping values (only for connected tendons)
    
    """
    
    

    genome = []
    
    # Adjacency left to obj
    for _ in range(GAConfig.TENDON_NUM):
        pairing_left = np.random.randint(-1, GAConfig.N_SITES_ARM)
        pairing_obj = np.random.randint(-1, GAConfig.N_SITES_OBJECT)
        genome.append((pairing_left, pairing_obj, 0, 0, 0))
    # Adjacency right to obj
    for _ in range(GAConfig.TENDON_NUM):
        pairing_right = np.random.randint(-1, GAConfig.N_SITES_ARM)
        pairing_obj = np.random.randint(-1, GAConfig.N_SITES_OBJECT)
        genome.append((pairing_right, pairing_obj, 0, 0, 0))
    # Adjacency right to left
    for _ in range(GAConfig.TENDON_NUM):
        pairing_right = np.random.randint(-1, GAConfig.N_SITES_ARM)
        pairing_left = np.random.randint(-1, GAConfig.N_SITES_ARM)
        genome.append((pairing_right, pairing_left, 0, 0, 0))
    
    genome = np.array(genome, dtype=float)
    for i in range(len(genome)):
        stiffness = random.uniform(GAConfig.MIN_STIFFNESS, GAConfig.MAX_STIFFNESS)
        damping = random.uniform(GAConfig.MIN_DAMPING, GAConfig.MAX_DAMPING)
        # Generate length with higher probability for values closer to 0
        # Using exponential distribution for bias towards 0
        length = np.random.exponential(scale=0.05)
        length = min(length, 0.2)  # Cap
        
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

    G = nx.Graph()

    for index in range(individual.shape[0]):
        if index < GAConfig.TENDON_NUM * 2:
        
            obj_i = individual[index][1]
            arm_i = individual[index][0]

            if obj_i < 0 or arm_i < 0:
                continue
            stiffness = individual[index][2]
            damping = individual[index][3]

            obj_x = ( obj_i % (GAConfig.N_SITES_OBJECT_X * GAConfig.N_SITES_OBJECT_Y) ) % GAConfig.N_SITES_OBJECT_X
            obj_y = ( obj_i % (GAConfig.N_SITES_OBJECT_X * GAConfig.N_SITES_OBJECT_Y) ) // GAConfig.N_SITES_OBJECT_X
            obj_z = ( obj_i // (GAConfig.N_SITES_OBJECT_X * GAConfig.N_SITES_OBJECT_Y) )

            arm_x = ( arm_i % (GAConfig.N_SITES_ARM_X * GAConfig.N_SITES_ARM_Y) ) % GAConfig.N_SITES_ARM_X
            arm_y = ( arm_i % (GAConfig.N_SITES_ARM_X * GAConfig.N_SITES_ARM_Y) ) // GAConfig.N_SITES_ARM_X
            arm_z = ( arm_i // (GAConfig.N_SITES_ARM_X * GAConfig.N_SITES_ARM_Y) )

            arm_x = int(arm_x)
            arm_y = int(arm_y)
            arm_z = int(arm_z)
            obj_x = int(obj_x)
            obj_y = int(obj_y)
            obj_z = int(obj_z)

            if index < GAConfig.TENDON_NUM:
                # Left chopstick
                G.add_edge(f"left_chopstick_site_{arm_x}_{arm_y}_{arm_z}", f"ghost_target_site_{obj_x}_{obj_y}_{obj_z}", 
                           type='spatial',
                           stiffness=stiffness,
                           damping=damping,
                           springlength=[0, 0.01])
            else:
                # Right chopstick
                G.add_edge(f"right_chopstick_site_{arm_x}_{arm_y}_{arm_z}", f"ghost_target_site_{obj_x}_{obj_y}_{obj_z}", 
                           type='spatial',
                           stiffness=stiffness,
                           damping=damping,
                           springlength=[0, 0.01])
            
            # print(individual[index], " : ", f"{'left' if index < GAConfig.TENDON_NUM else 'right'}_chopstick_site_{arm_x}_{arm_y}_{arm_z} <-> ghost_target_site_{obj_x}_{obj_y}_{obj_z}")
            
        else:
        
            arm_i = individual[index][0]
            arm_j = individual[index][1]

            if arm_i < 0 or arm_j < 0:
                continue
            stiffness = individual[index][2]
            damping = individual[index][3]
            length = individual[index][4]

            arm_i_x = ( arm_i % (GAConfig.N_SITES_ARM_X * GAConfig.N_SITES_ARM_Y) ) % GAConfig.N_SITES_ARM_X
            arm_i_y = ( arm_i % (GAConfig.N_SITES_ARM_X * GAConfig.N_SITES_ARM_Y) ) // GAConfig.N_SITES_ARM_X
            arm_i_z = ( arm_i // (GAConfig.N_SITES_ARM_X * GAConfig.N_SITES_ARM_Y) )

            arm_j_x = ( arm_j % (GAConfig.N_SITES_ARM_X * GAConfig.N_SITES_ARM_Y) ) % GAConfig.N_SITES_ARM_X
            arm_j_y = ( arm_j % (GAConfig.N_SITES_ARM_X * GAConfig.N_SITES_ARM_Y) ) // GAConfig.N_SITES_ARM_X
            arm_j_z = ( arm_j // (GAConfig.N_SITES_ARM_X * GAConfig.N_SITES_ARM_Y) )

            arm_i_x = int(arm_i_x)
            arm_i_y = int(arm_i_y)
            arm_i_z = int(arm_i_z)
            arm_j_x = int(arm_j_x)
            arm_j_y = int(arm_j_y)
            arm_j_z = int(arm_j_z)

            G.add_edge(f"right_chopstick_site_{arm_i_x}_{arm_i_y}_{arm_i_z}", f"left_chopstick_site_{arm_j_x}_{arm_j_y}_{arm_j_z}", 
                       type='spatial',
                       stiffness=stiffness,
                       damping=damping,
                       springlength=[length, length]
                       )

    return G



# ============================================================================
# Ghost Control Callback
# ============================================================================
def control(target, ghost_target, 
            lift_delay: float = GAConfig.LIFT_DELAY,
            lift_force: float = GAConfig.LIFT_FORCE,
            lift_duration: float = GAConfig.LIFT_DURATION,
            ):
    """
    Control ghost target to follow target position and gradually increase lifting tendon stiffness.
    
    The lifting mechanism works by:
    1. Keeping tendons slack (stiffness=0) before lift_delay
    2. Gradually increasing tendon stiffness during the lift period
    3. This pulls the arms upward as the tendons contract
    
    Args:
        target: Joint wrapper for the tracked target.
        ghost_target: Joint wrapper for the ghost target.
        lift_delay: Seconds before applying the lift.
        lift_force: Maximum tendon stiffness (N/m) to reach during lift.
        lift_duration: Duration over which stiffness increases.
    """

    max_stiffness = lift_force * 20  # Reuse lift_force parameter as max stiffness

    def callback(self: MuJoCoSimulation):
        # Keep ghost target synced with real target
        ghost_target.qpos = target.qpos
        ghost_target.qvel = target.qvel
        
        current_time = self.time
        
        # Calculate target stiffness based on time
        if current_time < lift_delay:
            # Before lift: tendons are slack
            target_stiffness = 0.0
        elif current_time < lift_delay + lift_duration:
            # During lift: gradually increase stiffness
            progress = (current_time - lift_delay) / lift_duration
            # Use smooth interpolation (ease-in-out cubic)
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            target_stiffness = smooth_progress * max_stiffness
        else:
            # After lift: maintain maximum stiffness
            target_stiffness = max_stiffness
        
        # Apply stiffness to lifting tendons
        try:
            # Register tendons if not already registered
            if 'left_lifter' not in self.tendons:
                left_lifter = self.reg_tendon('left_lifter', 'left_lifter')
                right_lifter = self.reg_tendon('right_lifter', 'right_lifter')
            else:
                left_lifter = self.tendons['left_lifter']
                right_lifter = self.tendons['right_lifter']
            
            # Update tendon stiffness dynamically
            # Note: We modify the model's tendon_stiffness array directly
            self.model.tendon_stiffness[left_lifter.id] = target_stiffness
            self.model.tendon_stiffness[right_lifter.id] = target_stiffness
            
        except Exception as e:
            # Tendons might not exist in all models
            pass

    return callback


# ============================================================================
# Simulation & Fitness Evaluation
# ============================================================================

def build_model_from_genome(G, angle=None) -> MuJoCoSimulation:
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
    # Add a platform, target and object to place objects on
    t0 = time.perf_counter()
    builder.add_body("platform", pos=[0.7, 0, 0.15],
                        geom_type="box", geom_size=[0.4, 0.5, 0.1],
                        free_joint=False, geom_rgba=[0.7,0.5,0.5,1], mass=10)

    builder.add_body("target", pos=[0.5, 0, 0.31],
                     quat=orientation,
                        geom_type="box", geom_size=[0.05, 0.05, 0.05],
                        free_joint=True, geom_rgba=[0,1,0,1], mass=0.3,
                        )
    
    builder.add_body("ghost_target", pos=[0.5, 0, 0.3],
                        quat=orientation,
                     geom_type="box", geom_size=[0.05, 0.05, 0.05],
                     free_joint=True, geom_rgba=[1,1,1,0.1], mass=0.3, intersection=False,
                     )
    times["add_static_bodies"] = time.perf_counter() - t0

    # colors for chopsticks
    left_chopstick_color = [1, 0, 0, 1]
    right_chopstick_color = [0, 0, 1, 1]

    chopsticks = [
        ("left_chopstick", left_chopstick_color),
        ("right_chopstick", right_chopstick_color),
    ]

    t0 = time.perf_counter()
    builder.remove_body("r_link7")
    builder.remove_body("l_link7")
    times["remove_end_links"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    builder.add_body("left_chopstick", pos=[0, 0.1, 0],
                    quat=quat_from_euler(math.pi/2, 0, 0),
                    geom_type="box", geom_size=[GAConfig.RADIUS, GAConfig.RADIUS, GAConfig.HALF_LENGTH],
                    parent="l_link6",
                    free_joint=False, geom_rgba=left_chopstick_color, mass=0.1,
                    friction=[2.0, 0.01, 0.001])
    builder.add_body("right_chopstick", pos=[0, -0.1, 0],
                    quat=quat_from_euler(math.pi/2, 0, 0),
                    geom_type="box", geom_size=[GAConfig.RADIUS, GAConfig.RADIUS, GAConfig.HALF_LENGTH],
                    parent="r_link6",
                    free_joint=False, geom_rgba=right_chopstick_color, mass=0.1,
                    friction=[2.0, 0.01, 0.001])
    times["add_chopstick_bodies"] = time.perf_counter() - t0

    # Adding mechanism to lift chopsticks
    builder.add_site('left_hand_site', body_name='l_link6', pos=[0, 0, 0], size=0.01)
    builder.add_site('left_shoulder_site', body_name='body_link', pos=[0.2, 0.05, 0.4], size=0.01)
    builder.add_site('right_hand_site', body_name='r_link6', pos=[0, 0, 0], size=0.01)
    builder.add_site('right_shoulder_site', body_name='body_link', pos=[0.2, -0.05, 0.4], size=0.01)

    # Add lifting tendons (will be controlled dynamically)
    builder.add_tendon('left_lifter', tendon_type='spatial',
                       sites=['left_hand_site', 'left_shoulder_site'],
                       stiffness=0.0,  # Will be increased dynamically
                       damping=5.0,
                       springlength=[0.0, 0.1],
                       rgba=[1, 1, 0, 0.5])
    
    builder.add_tendon('right_lifter', tendon_type='spatial',
                       sites=['right_hand_site', 'right_shoulder_site'],
                       stiffness=0.0,  # Will be increased dynamically
                       damping=5.0,
                       springlength=[0.0, 0.1],
                       rgba=[0, 1, 1, 0.5])
    
    times["add_lifting_mechanism"] = time.perf_counter() - t0

    # Add sites for tendons
    site_labels = []

    t0 = time.perf_counter()
    xs = np.linspace(-(GAConfig.RADIUS + GAConfig.MARGIN_R_ARM), (GAConfig.RADIUS + GAConfig.MARGIN_R_ARM), GAConfig.N_SITES_ARM_X)
    ys = np.linspace(-(GAConfig.RADIUS + GAConfig.MARGIN_R_ARM), (GAConfig.RADIUS + GAConfig.MARGIN_R_ARM), GAConfig.N_SITES_ARM_Y)
    zs = np.linspace(-(GAConfig.HALF_LENGTH + GAConfig.MARGIN_Z_ARM), (GAConfig.HALF_LENGTH + GAConfig.MARGIN_Z_ARM), GAConfig.N_SITES_ARM_Z)
    times["compute_arm_site_grid"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    arm_site_count = 0
    for body_name, color in chopsticks:
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                for k, z in enumerate(zs):
                    site_name = f"{body_name}_site_{i}_{j}_{k}"
                    site_labels.append(site_name)
                    builder.add_site(
                        name=site_name,
                        body_name=body_name,
                        pos=[float(x), float(y), float(z)],
                        size=0.003,
                        rgba=color
                    )
                    arm_site_count += 1
    times["add_arm_sites"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    xs = np.linspace(-(0.05 + GAConfig.MARGIN_R_OBJECT), (0.05 + GAConfig.MARGIN_R_OBJECT), GAConfig.N_SITES_OBJECT_X)
    ys = np.linspace(-(0.05 + GAConfig.MARGIN_R_OBJECT), (0.05 + GAConfig.MARGIN_R_OBJECT), GAConfig.N_SITES_OBJECT_Y)
    zs = np.linspace(-(0.05 + GAConfig.MARGIN_Z_OBJECT), (0.05 + GAConfig.MARGIN_Z_OBJECT), GAConfig.N_SITES_OBJECT_Z)
    times["compute_object_site_grid"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    object_site_count = 0
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                site_name = f"ghost_target_site_{i}_{j}_{k}"
                site_labels.append(site_name)
                builder.add_site(
                    name=site_name,
                    body_name="ghost_target",
                    pos=[float(x), float(y), float(z)],
                    size=0.003,
                    rgba=[0,1,0,0.3]
                )
                object_site_count += 1
    times["add_object_sites"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    builder.add_tendons_from_graph(G)
    times["add_tendons_from_graph"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sim = MuJoCoSimulation.from_builder(builder)
    times["build_simulation"] = time.perf_counter() - t0
    
    # Clean up builder and graph copy - we don't need them anymore
    del builder
    del G
    gc.collect()

    t0 = time.perf_counter()
    target_joint = sim.reg_joint('target_joint', 'target_freejoint')
    ghost_target_joint = sim.reg_joint('ghost_target_joint', 'ghost_target_freejoint')
    
    # Set up controller with lifting tendon control
    controller = control(
        target_joint,
        ghost_target_joint,
        lift_delay=GAConfig.LIFT_DELAY,
        lift_force=GAConfig.LIFT_FORCE,
        lift_duration=GAConfig.LIFT_DURATION,
    )
    sim.set_control_callback(controller, gravity_comp=True)
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
    for name, pos in zip(joint_names, initial_joint_positions):
        try:
            joint = sim.reg_joint(name, name)
            joint.qpos = pos
        except Exception as e:
            if verbose := False:
                print(f"Error setting joint '{name}': {e}")

    sim.forward()

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

    return sim

def evaluate_fitness(individual: creator.Individual) -> Tuple[float,]:
    """
    Evaluate fitness of an individual.
    
    Fitness rewards:
    - Flipping the cube by 180 degrees (rotation around any horizontal axis)
    - Keeping the cube near the center
    - Stability (low velocity)
    - Efficiency (fewer tendons)
    
    Returns:
        Tuple with single fitness value (DEAP requirement)
    """
    # Decode genome
    G = decode_genome(individual)
    
    # Build simulation
    fitness = 0.0
    trial_num = 3
    for trial in range(trial_num):  # Multiple trials with different initial orientations
        sim = build_model_from_genome(G, angle=np.pi/5 * (trial - 1))
        
        # Register bodies
        target = sim.reg_body('target_block', 'target')
        left = sim.reg_body('left', 'left_chopstick')
        right = sim.reg_body('right', 'right_chopstick')
        
        # Store initial orientation quaternion
        initial_quat = target.quat.copy()
        
        # Track metrics
        total_velocity = 0.0
        n_steps = 0
        
        # Run simulation
        num_steps = int(GAConfig.SIM_DURATION / sim.model.opt.timestep)
        
        for _ in range(num_steps):
            sim.step()
            
            # Track target velocity (for stability)
            vel = np.linalg.norm(target.linear_vel)
            total_velocity += vel
            n_steps += 1
        
        # Get final orientation
        final_quat = target.quat.copy()
        
        # Extract pitch angle (rotation around y-axis) from quaternions
        # Convert quaternion to Euler angles
        # For quaternion [w, x, y, z], pitch (y-axis rotation) is:
        # pitch = arcsin(2*(w*y - z*x))
        # But more robust: pitch = atan2(2*(w*x + y*z), 1 - 2*(x^2 + y^2))
        
        def quat_to_pitch(q):
            """Extract pitch angle (rotation around y-axis) from quaternion [w,x,y,z]"""
            w, x, y, z = q
            # Pitch (rotation around y-axis)
            pitch = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
            return pitch
        
        initial_pitch = quat_to_pitch(initial_quat)
        final_pitch = quat_to_pitch(final_quat)
        
        # Calculate rotation around y-axis
        pitch_rotation = final_pitch - initial_pitch
        # Normalize to [-pi, pi]
        pitch_rotation = np.arctan2(np.sin(pitch_rotation), np.cos(pitch_rotation))
        
        # Reward for flipping close to 180 degrees (pi radians) around y-axis
        # Maximum reward when pitch_rotation = ±pi
        flip_error = abs(abs(pitch_rotation) - np.pi)
        flip_reward = np.exp(-flip_error)  # Exponential reward, peaks at 180 degrees
        
        # Penalize if cube goes too far from center
        final_centre_dist = np.linalg.norm(target.pos - np.array([0.5, 0, 0.5]))
        # Strong penalty if further than 0.15m from center
        
        centre_penalty = final_centre_dist 
        centre_score = 1.0 / (1.0 + centre_penalty)
        
        
        # Weighted fitness for this trial
        trial_fitness = (
            10.0 * flip_reward +           # Main goal: flip 180 degrees
            15.0 * centre_score            # Keep cube centered
        )
        
        fitness += trial_fitness
        
        # Clean up simulation resources
        sim.close()
        del sim
        gc.collect()
    
    # Average over trials
    fitness /= trial_num
    return (fitness,)
    
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


# ============================================================================
# Genetic Operators
# ============================================================================

def mutate_individual(individual: creator.Individual) -> Tuple[creator.Individual,]:
    """
    Mutate an individual using:
    - Polynomial bounded mutation for continuous parameters (stiffness, damping, length)
    - Integer creep mutation for discrete connection indices (small local changes)
    - Occasional random resampling for exploration
    
    This approach balances exploitation (small changes) with exploration (random jumps).
    """
    original_shape = individual.shape
    
    # Mutate discrete connection indices with improved strategies
    for i in range(individual.shape[0]):
        # Mutate first connection index (left/right arm site)
        if random.random() < GAConfig.FLIP_PROB:
            if random.random() < 0.7:  # 70% chance: creep mutation (local search)
                # Small integer step: ±1, ±2, or ±3
                step = random.choice([-3, -2, -1, 1, 2, 3])
                new_val = int(individual[i][0]) + step
                # Clamp to valid range
                individual[i][0] = max(-1, min(GAConfig.N_SITES_ARM - 1, new_val))
            else:  # 30% chance: random resampling (exploration)
                individual[i][0] = np.random.randint(-1, GAConfig.N_SITES_ARM)
        
        # Mutate second connection index (object or other arm site)
        if random.random() < GAConfig.FLIP_PROB:
            max_idx = GAConfig.N_SITES_OBJECT if i < GAConfig.TENDON_NUM * 2 else GAConfig.N_SITES_ARM
            if random.random() < 0.7:  # 70% chance: creep mutation
                step = random.choice([-3, -2, -1, 1, 2, 3])
                new_val = int(individual[i][1]) + step
                individual[i][1] = max(-1, min(max_idx - 1, new_val))
            else:  # 30% chance: random resampling
                individual[i][1] = np.random.randint(-1, max_idx)
    
    # For continuous parameters (columns 2, 3, 4), use polynomial bounded mutation
    # Extract continuous parameters: stiffness (col 2), damping (col 3), length (col 4)
    continuous_params = individual[:, 2:5].flatten()
    
    # Define bounds for each parameter type
    # [stiffness_1, damping_1, length_1, stiffness_2, damping_2, length_2, ...]
    low_bounds = []
    up_bounds = []
    for i in range(individual.shape[0]):
        low_bounds.extend([GAConfig.MIN_STIFFNESS, GAConfig.MIN_DAMPING, 0.0])
        up_bounds.extend([GAConfig.MAX_STIFFNESS, GAConfig.MAX_DAMPING, 0.2])
    
    # Apply polynomial bounded mutation (eta=20 is standard, indpb is mutation probability per gene)
    mutated_params, = tools.mutPolynomialBounded(
        continuous_params, 
        eta=20.0,  # Crowding degree of the mutation (higher = more local)
        low=low_bounds, 
        up=up_bounds, 
        indpb=0.3  # Probability to mutate each continuous parameter
    )
    
    # Reshape back and update individual
    individual[:, 2:5] = mutated_params.reshape(-1, 3)
    
    return (individual,)


def crossover_individuals(ind1: creator.Individual, ind2: creator.Individual) -> Tuple[creator.Individual, creator.Individual]:
    """
    Crossover two individuals using:
    - Simulated Binary Crossover (SBX) for continuous parameters (stiffness, damping, length)
    - Two-point crossover for discrete connection indices (preserves building blocks)
    
    Two-point crossover is better than uniform for preserving groups of related genes.
    """
    original_shape = ind1.shape
    num_tendons = ind1.shape[0]
    # Two-point crossover for discrete connection indices (columns 0 and 1)
    # This preserves contiguous blocks of tendons that work well together
    if num_tendons > 2:  # Need at least 3 tendons for two-point crossover
        # Choose two random crossover points
        point1 = random.randint(1, num_tendons - 1)
        point2 = random.randint(1, num_tendons - 1)
        if point1 > point2:
            point1, point2 = point2, point1
        
        # Swap the segment between point1 and point2 for both connection columns
        for col in [0, 1]:  # Both connection index columns
            ind1[point1:point2, col], ind2[point1:point2, col] = \
                ind2[point1:point2, col].copy(), ind1[point1:point2, col].copy()
    else:
        # Fallback to single-point crossover for very small genomes
        point = random.randint(1, num_tendons)
        for col in [0, 1]:
            ind1[point:, col], ind2[point:, col] = \
                ind2[point:, col].copy(), ind1[point:, col].copy()
    
    # For continuous parameters (columns 2, 3, 4), use bounded SBX
    continuous_params1 = ind1[:, 2:5].flatten()
    continuous_params2 = ind2[:, 2:5].flatten()
    
    # Define bounds for each parameter type
    low_bounds = []
    up_bounds = []
    for i in range(ind1.shape[0]):
        low_bounds.extend([GAConfig.MIN_STIFFNESS, GAConfig.MIN_DAMPING, 0.0])
        up_bounds.extend([GAConfig.MAX_STIFFNESS, GAConfig.MAX_DAMPING, 0.2])
    
    # Apply bounded SBX (eta=20 is standard for real-coded GAs)
    tools.cxSimulatedBinaryBounded(
        continuous_params1,
        continuous_params2,
        eta=20.0,  # Crowding degree (higher = children closer to parents)
        low=low_bounds,
        up=up_bounds
    )
    
    # Reshape back and update individuals
    ind1[:, 2:5] = continuous_params1.reshape(-1, 3)
    ind2[:, 2:5] = continuous_params2.reshape(-1, 3)
    
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
    log("")
    log(f"  Simulation Duration:  {GAConfig.SIM_DURATION} s")
    log(f"  Simulation Timestep:  {GAConfig.SIM_DT} s")
    log(f"  Lift Delay:           {GAConfig.LIFT_DELAY} s")
    log(f"  Lift Duration:        {GAConfig.LIFT_DURATION} s")
    log(f"  Lift Force:           {GAConfig.LIFT_FORCE} N/m")
    log("")
    log(f"  Contact Weight:       {GAConfig.CONTACT_WEIGHT}")
    log(f"  Height Weight:        {GAConfig.HEIGHT_WEIGHT}")
    log(f"  Centre Weight:        {GAConfig.CENTRE_WEIGHT}")
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
        'lift_delay': GAConfig.LIFT_DELAY,
        'lift_duration': GAConfig.LIFT_DURATION,
        'lift_force': GAConfig.LIFT_FORCE,
        'contact_weight': GAConfig.CONTACT_WEIGHT,
        'height_weight': GAConfig.HEIGHT_WEIGHT,
        'centre_weight': GAConfig.CENTRE_WEIGHT,
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
            realtime_speed=0.5,
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
    parser.add_argument('--record', action='store_true', help='Record simulation video when visualizing')
    parser.add_argument('--stats', action='store_true', help='Generate and save statistics plots')
    parser.add_argument('--parallel', type=int, default=None, metavar='N',
                        help='Number of parallel processes (default: all cores, 1=serial)')
    
    args = parser.parse_args()
    
    if args.load:
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
        
