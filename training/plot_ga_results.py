#!/usr/bin/env python3
"""
Visualize GA optimization results.

Usage:
    python plot_ga_results.py
    python plot_ga_results.py --checkpoint ga_checkpoints/best_final.pkl
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse


def plot_fitness_history(history_path='ga_checkpoints/fitness_history.npz'):
    """Plot fitness evolution over generations."""
    if not os.path.exists(history_path):
        print(f"History file not found: {history_path}")
        return
    
    data = np.load(history_path)
    best_fitness = data['best']
    avg_fitness = data['avg']
    
    generations = np.arange(1, len(best_fitness) + 1)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
    plt.plot(generations, avg_fitness, 'r--', linewidth=2, label='Average Fitness')
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('Genetic Algorithm Fitness Evolution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ga_checkpoints/fitness_evolution.png', dpi=150)
    print("Saved: ga_checkpoints/fitness_evolution.png")
    plt.show()


def visualize_individual(checkpoint_path):
    """Visualize an individual's genome structure."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    with open(checkpoint_path, 'rb') as f:
        individual = pickle.load(f)
    
    # fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Adjacency matrix
    # ax = axes[0]
    # im1 = ax.imshow(individual.adjacency_matrix, cmap='Greys', aspect='auto')
    # ax.set_title('Adjacency Matrix\n(Connections)', fontsize=12, fontweight='bold')
    # ax.set_xlabel('Site Index')
    # ax.set_ylabel('Site Index')
    # plt.colorbar(im1, ax=ax, label='Connected')
    
    # Stiffness matrix
    # ax = axes[1]
    # stiffness_display = np.where(individual.adjacency_matrix, individual.stiffness_matrix, np.nan)
    # im2 = ax.imshow(stiffness_display, cmap='hot', aspect='auto')
    # ax.set_title('Stiffness Matrix', fontsize=12, fontweight='bold')
    # ax.set_xlabel('Site Index')
    # ax.set_ylabel('Site Index')
    # plt.colorbar(im2, ax=ax, label='Stiffness (N/m)')
    
    # # Damping matrix
    # ax = axes[2]
    # damping_display = np.where(individual.adjacency_matrix, individual.damping_matrix, np.nan)
    # im3 = ax.imshow(damping_display, cmap='viridis', aspect='auto')
    # ax.set_title('Damping Matrix', fontsize=12, fontweight='bold')
    # ax.set_xlabel('Site Index')
    # ax.set_ylabel('Site Index')
    # plt.colorbar(im3, ax=ax, label='Damping (N·s/m)')
    
    plt.suptitle(f'Individual Genome (Fitness: , ' +
                 f'Connections: )',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    save_path = checkpoint_path.replace('.pkl', '_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()
    
    # Print fitness components
    if hasattr(individual, 'fitness_components') and individual.fitness_components:
        print("\nFitness Components:")
        for key, value in individual.fitness_components.items():
            print(f"  {key:15s}: {value:.4f}")


def compare_generations(checkpoint_dir='ga_checkpoints'):
    """Compare individuals across generations."""
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('best_gen_')])
    
    if not checkpoint_files:
        print(f"No generation checkpoints found in {checkpoint_dir}")
        return
    
    generations = []
    fitnesses = []
    connections = []
    
    for filename in checkpoint_files:
        gen_num = int(filename.split('_')[2].split('.')[0])
        filepath = os.path.join(checkpoint_dir, filename)
        
        with open(filepath, 'rb') as f:
            individual = pickle.load(f)
        
        generations.append(gen_num)
        fitnesses.append(individual.fitness)
        connections.append(individual.get_num_connections())
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Fitness
    ax1.plot(generations, fitnesses, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness')
    ax1.set_title('Best Fitness by Generation', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Connections
    ax2.plot(generations, connections, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Number of Connections')
    ax2.set_title('Network Complexity by Generation', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'generation_comparison.png'), dpi=150)
    print(f"Saved: {os.path.join(checkpoint_dir, 'generation_comparison.png')}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize GA results")
    parser.add_argument('--checkpoint', type=str, 
                       default='ga_checkpoints/best_final.pkl',
                       help='Path to checkpoint to visualize')
    parser.add_argument('--history', action='store_true',
                       help='Plot fitness history')
    parser.add_argument('--compare', action='store_true',
                       help='Compare across generations')
    args = parser.parse_args()
    
    if args.history:
        plot_fitness_history()
    
    if args.compare:
        compare_generations()
    
    if os.path.exists(args.checkpoint):
        visualize_individual(args.checkpoint)
    else:
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Run with --history to plot fitness evolution")
