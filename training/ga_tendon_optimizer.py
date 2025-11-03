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

from tools.mujoco_sim_template import MuJoCoSimulation, ModelBuilder, quat_from_euler
import numpy as np
import networkx
import mujoco
import pickle
from datetime import datetime
from typing import List, Tuple, Dict
import math
import random

# DEAP imports
from deap import base, creator, tools, algorithms


# ============================================================================
# Configuration
# ============================================================================

class GAConfig:
    """Configuration for Genetic Algorithm using DEAP"""
    def __init__(self):
        # GA Parameters
        self.population_size = 20
        self.num_generations = 50
        self.tournament_size = 3
        self.crossover_prob = 0.7
        self.mutation_prob = 0.2
        self.elite_size = 2
        
        # Tendon parameters
        self.n_sites = 4  # Number of connection sites per chopstick
        self.stiffness_range = (10.0, 100.0)  # N/m
        self.damping_range = (1.0, 10.0)      # N*s/m
        self.connection_density = 0.3         # Initial probability of connection
        
        # Mutation parameters
        self.mutate_flip_prob = 0.1           # Probability to flip each connection
        self.mutate_gaussian_mu = 0.0
        self.mutate_gaussian_sigma = 10.0     # For stiffness/damping mutation
        
        # Simulation parameters
        self.sim_duration = 3.0               # seconds
        self.sim_dt = 0.01                    # seconds
        
        # Fitness weights
        self.contact_weight = 10.0
        self.height_weight = 5.0
        self.stability_weight = 3.0
        self.efficiency_weight = 1.0
        
        # Checkpoint
        self.checkpoint_dir = "ga_checkpoints"
        self.checkpoint_interval = 5  # Save every N generations


# ============================================================================
# Individual (Genome)
# ============================================================================

class Individual:
    """
    Represents one solution (adjacency matrix + tendon properties).
    
    Genome structure:
    - adjacency_matrix: n×n binary matrix of connections
    - stiffness_matrix: n×n matrix of stiffness values
    - damping_matrix: n×n matrix of damping values
    """
    
    def __init__(self, n_sites: int, labels: List[str], eligible: np.ndarray):
        self.n_sites = n_sites
        self.labels = labels
        self.eligible = eligible  # Mask for valid connections
        
        # Genome
        self.adjacency_matrix = np.zeros((n_sites, n_sites), dtype=np.uint8)
        self.stiffness_matrix = np.zeros((n_sites, n_sites), dtype=np.float32)
        self.damping_matrix = np.zeros((n_sites, n_sites), dtype=np.float32)
        
        # Fitness
        self.fitness = 0.0
        self.fitness_components = {}
        
    def randomize(self, connection_prob: float = None):
        """Initialize with random genome."""
        if connection_prob is None:
            connection_prob = GAConfig.BASE_CONNECTION_PROB
        
        # Random connections (only where eligible)
        rand = np.random.rand(self.n_sites, self.n_sites)
        self.adjacency_matrix = ((rand < connection_prob) & self.eligible).astype(np.uint8)
        
        # Random stiffness and damping for connected tendons
        for i in range(self.n_sites):
            for j in range(i + 1, self.n_sites):
                if self.adjacency_matrix[i, j]:
                    self.stiffness_matrix[i, j] = np.random.uniform(
                        GAConfig.MIN_STIFFNESS, GAConfig.MAX_STIFFNESS)
                    self.damping_matrix[i, j] = np.random.uniform(
                        GAConfig.MIN_DAMPING, GAConfig.MAX_DAMPING)
    
    def get_num_connections(self) -> int:
        """Count number of connections."""
        return np.sum(self.adjacency_matrix) // 2  # Symmetric, so divide by 2
    
    def copy(self) -> 'Individual':
        """Create a deep copy."""
        new_ind = Individual(self.n_sites, self.labels, self.eligible)
        new_ind.adjacency_matrix = self.adjacency_matrix.copy()
        new_ind.stiffness_matrix = self.stiffness_matrix.copy()
        new_ind.damping_matrix = self.damping_matrix.copy()
        new_ind.fitness = self.fitness
        new_ind.fitness_components = self.fitness_components.copy()
        return new_ind


# ============================================================================
# Simulation & Fitness Evaluation
# ============================================================================

def build_model_from_individual(individual: Individual, 
                                initial_joint_positions: np.ndarray,
                                joint_names: List[str]) -> Tuple[MuJoCoSimulation, object]:
    """
    Build MuJoCo simulation from an individual's genome.
    
    Returns:
        (simulation, target_body)
    """
    xml_path = "Sciurus17_mujoco_sim_example/URDFs/sciurus17_description/urdf/sciurus17.xml"
    
    # Create builder
    G = networkx.Graph()
    builder = ModelBuilder(G)
    builder.from_xml_path(xml_path)
    
    # Chopstick parameters
    left_chopstick_color = [1, 0, 0, 1]
    right_chopstick_color = [0, 0, 1, 1]
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
    
    # Add platform and target
    builder.add_body("platform", pos=[0.5, 0, 0.1],
                     geom_type="box", geom_size=[0.2, 0.2, 0.1],
                     free_joint=False, geom_rgba=[0.7, 0.5, 0.5, 1], mass=10)
    builder.add_body("target", pos=[0.5, 0, 0.3],
                     geom_type="box", geom_size=[0.05, 0.05, 0.05],
                     free_joint=True, geom_rgba=[0, 1, 0, 1], mass=0.5)
    
    # Add 3D grid of sites around each chopstick
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
    
    # Add tendons based on individual's genome
    for i in range(individual.n_sites):
        for j in range(i + 1, individual.n_sites):
            if individual.adjacency_matrix[i, j]:
                node_i = individual.labels[i]
                node_j = individual.labels[j]
                stiffness = float(individual.stiffness_matrix[i, j])
                damping = float(individual.damping_matrix[i, j])
                # Use [L, L] format for spatial tendon springlength
                # Setting to [0, 0] means no rest length (pure force-based tendon)
                springlength = [0.0, 0.0]
                rgba = [0, 1, 0, 0.5]
                
                builder.add_tendon(
                    name=f"tendon_{node_i}_{node_j}",
                    sites=[node_i, node_j],
                    stiffness=stiffness,
                    damping=damping,
                    springlength=springlength,
                    rgba=rgba
                )
    
    # Create simulation
    sim = MuJoCoSimulation.from_builder(builder)
    
    # Set initial joint positions
    for name, pos in zip(joint_names, initial_joint_positions):
        joint = sim.reg_joint(name, name)
        joint.qpos = pos
    
    # Update physics
    mujoco.mj_forward(sim.model, sim.data)
    
    # Register target body
    target = sim.reg_body("target", "target")
    
    return sim, target


def evaluate_fitness(individual: Individual,
                     initial_joint_positions: np.ndarray,
                     joint_names: List[str],
                     visualize: bool = False) -> float:
    """
    Evaluate fitness of an individual by running simulation.
    
    Fitness components:
    - Contact: Amount of contact between chopsticks and target
    - Height: Final height of target
    - Stability: Inverse of target velocity (stable grasp)
    - Efficiency: Penalize too many tendons
    """
    try:
        # Build simulation
        sim, target = build_model_from_individual(individual, initial_joint_positions, joint_names)
        
        # Track metrics during simulation
        contact_sum = 0.0
        max_height = 0.0
        final_velocity = 0.0
        steps = 0
        
        # Get geom IDs for contact detection
        left_chopstick_geom_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "left_chopstick_geom")
        right_chopstick_geom_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "right_chopstick_geom")
        target_geom_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        
        def control(sim_obj):
            nonlocal contact_sum, max_height, final_velocity, steps
            
            # Measure contact
            for i in range(sim_obj.data.ncon):
                contact = sim_obj.data.contact[i]
                geom1 = contact.geom1
                geom2 = contact.geom2
                
                # Check if contact involves chopstick and target
                if ((geom1 == left_chopstick_geom_id or geom1 == right_chopstick_geom_id) and geom2 == target_geom_id) or \
                   ((geom2 == left_chopstick_geom_id or geom2 == right_chopstick_geom_id) and geom1 == target_geom_id):
                    # Sum contact force magnitude
                    contact_sum += np.linalg.norm(contact.frame[:3])  # Normal force
            
            # Track height
            height = target.pos[2]
            max_height = max(max_height, height)
            
            # Track final velocity
            final_velocity = np.linalg.norm(target.linear_vel)
            
            steps += 1
        
        # Run simulation
        if visualize:
            sim.run(control_callback=control, passive=True, 
                   duration=GAConfig.SIM_DURATION, 
                   realtime_speed=1.0,
                   gravity_comp=True,
                   viewer_distance=2.5,
                   viewer_lookat=[0, 0, 0.25])
        else:
            # Run without visualization (faster)
            sim.set_control_callback(control, gravity_comp=True)
            
            t0 = sim.data.time
            while (sim.data.time - t0) < GAConfig.SIM_DURATION:
                sim.step()
        
        # Compute fitness components
        avg_contact = contact_sum / max(steps, 1)
        height_gain = max(0, max_height - 0.3)  # Initial height is 0.3
        stability = 1.0 / (1.0 + final_velocity)  # Lower velocity = more stable
        num_connections = individual.get_num_connections()
        efficiency = 1.0 / (1.0 + num_connections * 0.01)  # Small penalty for many tendons
        
        # Weighted fitness
        fitness = (GAConfig.WEIGHT_CONTACT * avg_contact +
                  GAConfig.WEIGHT_HEIGHT * height_gain +
                  GAConfig.WEIGHT_STABILITY * stability +
                  GAConfig.WEIGHT_EFFICIENCY * efficiency)
        
        # Store components for analysis
        individual.fitness_components = {
            'contact': avg_contact,
            'height': height_gain,
            'stability': stability,
            'efficiency': efficiency,
            'num_connections': num_connections
        }
        
        return fitness
        
    except Exception as e:
        print(f"Error evaluating individual: {e}")
        return 0.0


# ============================================================================
# Genetic Operators
# ============================================================================

def tournament_selection(population: List[Individual], k: int = GAConfig.TOURNAMENT_SIZE) -> Individual:
    """Select individual using tournament selection."""
    tournament = np.random.choice(population, size=k, replace=False)
    return max(tournament, key=lambda ind: ind.fitness)


def crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
    """
    Crossover two parents to create two offspring.
    Uses uniform crossover for adjacency and average for properties.
    """
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    if np.random.random() < GAConfig.CROSSOVER_RATE:
        # Uniform crossover for adjacency
        mask = np.random.rand(parent1.n_sites, parent1.n_sites) > 0.5
        child1.adjacency_matrix = np.where(mask, parent1.adjacency_matrix, parent2.adjacency_matrix)
        child2.adjacency_matrix = np.where(mask, parent2.adjacency_matrix, parent1.adjacency_matrix)
        
        # Average crossover for properties (where connections exist)
        for i in range(parent1.n_sites):
            for j in range(i + 1, parent1.n_sites):
                if child1.adjacency_matrix[i, j]:
                    alpha = np.random.random()
                    child1.stiffness_matrix[i, j] = (alpha * parent1.stiffness_matrix[i, j] + 
                                                     (1-alpha) * parent2.stiffness_matrix[i, j])
                    child1.damping_matrix[i, j] = (alpha * parent1.damping_matrix[i, j] + 
                                                   (1-alpha) * parent2.damping_matrix[i, j])
                
                if child2.adjacency_matrix[i, j]:
                    alpha = np.random.random()
                    child2.stiffness_matrix[i, j] = (alpha * parent2.stiffness_matrix[i, j] + 
                                                     (1-alpha) * parent1.stiffness_matrix[i, j])
                    child2.damping_matrix[i, j] = (alpha * parent2.damping_matrix[i, j] + 
                                                   (1-alpha) * parent1.damping_matrix[i, j])
    
    return child1, child2


def mutate(individual: Individual):
    """
    Mutate an individual.
    - Add/remove connections
    - Modify stiffness/damping values
    """
    # Mutation: flip connections
    for i in range(individual.n_sites):
        for j in range(i + 1, individual.n_sites):
            if individual.eligible[i, j] and np.random.random() < GAConfig.MUTATION_RATE:
                individual.adjacency_matrix[i, j] = 1 - individual.adjacency_matrix[i, j]
                individual.adjacency_matrix[j, i] = individual.adjacency_matrix[i, j]
                
                # If newly connected, assign random properties
                if individual.adjacency_matrix[i, j]:
                    individual.stiffness_matrix[i, j] = np.random.uniform(
                        GAConfig.MIN_STIFFNESS, GAConfig.MAX_STIFFNESS)
                    individual.damping_matrix[i, j] = np.random.uniform(
                        GAConfig.MIN_DAMPING, GAConfig.MAX_DAMPING)
    
    # Mutation: modify properties
    for i in range(individual.n_sites):
        for j in range(i + 1, individual.n_sites):
            if individual.adjacency_matrix[i, j]:
                if np.random.random() < GAConfig.MUTATION_RATE:
                    # Gaussian mutation
                    individual.stiffness_matrix[i, j] += np.random.normal(0, 10)
                    individual.stiffness_matrix[i, j] = np.clip(
                        individual.stiffness_matrix[i, j], 
                        GAConfig.MIN_STIFFNESS, GAConfig.MAX_STIFFNESS)
                
                if np.random.random() < GAConfig.MUTATION_RATE:
                    individual.damping_matrix[i, j] += np.random.normal(0, 1)
                    individual.damping_matrix[i, j] = np.clip(
                        individual.damping_matrix[i, j],
                        GAConfig.MIN_DAMPING, GAConfig.MAX_DAMPING)


# ============================================================================
# Main GA Loop
# ============================================================================

def run_genetic_algorithm(initial_joint_positions: np.ndarray,
                         joint_names: List[str],
                         visualize_best: bool = False):
    """
    Main genetic algorithm loop.
    """
    # Setup
    os.makedirs(GAConfig.CHECKPOINT_DIR, exist_ok=True)
    
    # Create site labels and eligibility mask
    chopsticks = ["left_chopstick", "right_chopstick"]
    radius = 0.02
    half_len = 0.1
    margin_r = 0.1
    margin_z = 0.2
    nx, ny, nz = 3, 3, 5
    
    labels = []
    for body_name in chopsticks:
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    labels.append(f"{body_name}_site_{i}_{j}_{k}")
    
    n = len(labels)
    
    # Eligibility mask (only cross-chopstick connections)
    eligible = np.zeros((n, n), dtype=bool)
    for i, node_i in enumerate(labels):
        for j, node_j in enumerate(labels):
            if i <= j:
                if ("left_chopstick" in node_i and "right_chopstick" in node_j) or \
                   ("right_chopstick" in node_i and "left_chopstick" in node_j):
                    eligible[i, j] = True
    
    # Initialize population
    print(f"Initializing population of {GAConfig.POPULATION_SIZE}...")
    population = []
    for _ in range(GAConfig.POPULATION_SIZE):
        ind = Individual(n, labels, eligible)
        ind.randomize()
        population.append(ind)
    
    # Track best individual
    best_ever = None
    best_fitness_history = []
    avg_fitness_history = []
    
    # Evolution loop
    for generation in range(GAConfig.GENERATIONS):
        print(f"\n{'='*60}")
        print(f"Generation {generation + 1}/{GAConfig.GENERATIONS}")
        print(f"{'='*60}")
        
        # Evaluate fitness
        print("Evaluating population...")
        for i, individual in enumerate(population):
            fitness = evaluate_fitness(individual, initial_joint_positions, joint_names)
            individual.fitness = fitness
            
            if (i + 1) % 5 == 0:
                print(f"  Evaluated {i+1}/{len(population)} individuals")
        
        # Sort by fitness
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        
        # Statistics
        best_fitness = population[0].fitness
        avg_fitness = np.mean([ind.fitness for ind in population])
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        # Update best ever
        if best_ever is None or best_fitness > best_ever.fitness:
            best_ever = population[0].copy()
        
        # Print statistics
        print(f"\nBest Fitness: {best_fitness:.4f}")
        print(f"Avg Fitness:  {avg_fitness:.4f}")
        print(f"Best Ever:    {best_ever.fitness:.4f}")
        print(f"\nBest Individual Components:")
        for key, value in population[0].fitness_components.items():
            print(f"  {key}: {value:.4f}")
        
        # Save checkpoint
        if (generation + 1) % GAConfig.SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(
                GAConfig.CHECKPOINT_DIR,
                f"best_gen_{generation+1}.pkl"
            )
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(best_ever, f)
            print(f"\nSaved checkpoint: {checkpoint_path}")
        
        # Create next generation
        if generation < GAConfig.GENERATIONS - 1:
            print("\nCreating next generation...")
            new_population = []
            
            # Elitism: keep best individuals
            for i in range(GAConfig.ELITE_SIZE):
                new_population.append(population[i].copy())
            
            # Generate offspring
            while len(new_population) < GAConfig.POPULATION_SIZE:
                parent1 = tournament_selection(population)
                parent2 = tournament_selection(population)
                
                child1, child2 = crossover(parent1, parent2)
                
                mutate(child1)
                mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < GAConfig.POPULATION_SIZE:
                    new_population.append(child2)
            
            population = new_population
    
    # Final results
    print(f"\n{'='*60}")
    print("EVOLUTION COMPLETE")
    print(f"{'='*60}")
    print(f"Best Fitness: {best_ever.fitness:.4f}")
    print(f"Connections: {best_ever.get_num_connections()}")
    print("\nFitness Components:")
    for key, value in best_ever.fitness_components.items():
        print(f"  {key}: {value:.4f}")
    
    # Save final best
    final_path = os.path.join(GAConfig.CHECKPOINT_DIR, "best_final.pkl")
    with open(final_path, 'wb') as f:
        pickle.dump(best_ever, f)
    print(f"\nSaved final best: {final_path}")
    
    # Save history
    history_path = os.path.join(GAConfig.CHECKPOINT_DIR, "fitness_history.npz")
    np.savez(history_path, 
             best=best_fitness_history,
             avg=avg_fitness_history)
    print(f"Saved fitness history: {history_path}")
    
    # Visualize best if requested
    if visualize_best:
        print("\nVisualizing best individual...")
        evaluate_fitness(best_ever, initial_joint_positions, joint_names, visualize=True)
    
    return best_ever, best_fitness_history, avg_fitness_history


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GA Tendon Network Optimizer")
    parser.add_argument('--visualize', action='store_true', 
                       help='Visualize best individual at end')
    parser.add_argument('--load', type=str,
                       help='Load and visualize a checkpoint')
    parser.add_argument('--generations', type=int,
                       help='Number of generations')
    parser.add_argument('--population', type=int,
                       help='Population size')
    args = parser.parse_args()
    
    # Override config if specified
    if args.generations:
        GAConfig.GENERATIONS = args.generations
    if args.population:
        GAConfig.POPULATION_SIZE = args.population
    
    # Initial joint positions
    initial_joint_positions = np.array([
        0.05062136600022616, -0.3436116964863836, -1.3959225169759335, 
        0.10584467436410924, -1.5539225381281545, 0.09817477042468103, 2.741223667951641,
        0.02761165418194154, -1.7916895602504288, -1.6122138080678088, 0.0, 
        -0.0798, 1.53, 0.145, -2.26,
        0.0372, 1.07, -2, 0.5
    ], dtype=np.float32)
    
    joint_names = [
        "waist_yaw_joint", "neck_yaw_joint", "neck_pitch_joint",
        "r_arm_joint1", "r_arm_joint2", "r_arm_joint3", "r_arm_joint4",
        "r_arm_joint5", "r_arm_joint6", "r_arm_joint7", "r_hand_mimic_joint",
        "l_arm_joint1", "l_arm_joint2", "l_arm_joint3", "l_arm_joint4",
        "l_arm_joint5", "l_arm_joint6", "l_arm_joint7", "l_hand_mimic_joint"
    ]
    
    if args.load:
        # Load and visualize checkpoint
        print(f"Loading checkpoint: {args.load}")
        with open(args.load, 'rb') as f:
            individual = pickle.load(f)
        
        print(f"Fitness: {individual.fitness:.4f}")
        print(f"Connections: {individual.get_num_connections()}")
        print("\nVisualizing...")
        evaluate_fitness(individual, initial_joint_positions, joint_names, visualize=True)
    else:
        # Run GA
        run_genetic_algorithm(
            initial_joint_positions,
            joint_names,
            visualize_best=args.visualize
        )
