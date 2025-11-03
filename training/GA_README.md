# Genetic Algorithm for Tendon Network Optimization

This system uses a genetic algorithm to evolve optimal tendon network configurations for picking up objects with chopstick grippers.

## Overview

The GA optimizes:
- **Adjacency matrix**: Which sites connect with tendons
- **Stiffness values**: Spring constants for each tendon
- **Damping values**: Damping coefficients for each tendon

### Fitness Function

The fitness is a weighted combination of:

1. **Contact** (weight: 10.0): Sum of contact forces between chopsticks and target
2. **Height** (weight: 5.0): How much the target is lifted
3. **Stability** (weight: 3.0): Inverse of target velocity (stable grasp preferred)
4. **Efficiency** (weight: 1.0): Penalty for too many tendons

## Quick Start

### Basic Run
```bash
# Run with default settings (20 pop, 50 generations)
mjpython training/ga_tendon_optimizer.py

# Visualize best at the end
mjpython training/ga_tendon_optimizer.py --visualize

# Custom settings
mjpython training/ga_tendon_optimizer.py --generations 100 --population 30
```

### Load and Visualize a Checkpoint
```bash
# View best from specific generation
mjpython training/ga_tendon_optimizer.py --load ga_checkpoints/best_gen_10.pkl

# View final best
mjpython training/ga_tendon_optimizer.py --load ga_checkpoints/best_final.pkl
```

### Plot Results
```bash
# Plot fitness evolution
python training/plot_ga_results.py --history

# Visualize genome structure
python training/plot_ga_results.py --checkpoint ga_checkpoints/best_final.pkl

# Compare across generations
python training/plot_ga_results.py --compare
```

## Configuration

Edit `GAConfig` class in `ga_tendon_optimizer.py`:

```python
class GAConfig:
    # GA Parameters
    POPULATION_SIZE = 20
    GENERATIONS = 50
    MUTATION_RATE = 0.15
    CROSSOVER_RATE = 0.7
    ELITE_SIZE = 2
    TOURNAMENT_SIZE = 3
    
    # Simulation Parameters
    SIM_DURATION = 5.0  # seconds
    
    # Tendon Parameters
    MIN_STIFFNESS = 10.0
    MAX_STIFFNESS = 100.0
    MIN_DAMPING = 1.0
    MAX_DAMPING = 10.0
    
    # Fitness Weights
    WEIGHT_CONTACT = 10.0
    WEIGHT_HEIGHT = 5.0
    WEIGHT_STABILITY = 3.0
    WEIGHT_EFFICIENCY = 1.0
```

## How It Works

### 1. Genome Representation

Each individual has three matrices (n×n, where n = number of sites):
- **adjacency_matrix**: Binary (0/1) indicating connections
- **stiffness_matrix**: Float values for connected tendons
- **damping_matrix**: Float values for connected tendons

### 2. Genetic Operators

**Selection**: Tournament selection (k=3)
- Pick k random individuals
- Select the best one

**Crossover**: Uniform + Average
- Uniform crossover for adjacency (randomly pick from parents)
- Average crossover for stiffness/damping

**Mutation**: 
- Flip connections (add/remove tendons)
- Gaussian mutation on stiffness/damping values

**Elitism**: Keep top 2 individuals each generation

### 3. Evaluation Process

For each individual:
1. Build MuJoCo model with specified tendon configuration
2. Run 5-second simulation with gravity compensation
3. Track contact forces, target height, target velocity
4. Compute weighted fitness score

### 4. Evolution Loop

```
Initialize random population
For each generation:
    Evaluate all individuals
    Select parents via tournament
    Create offspring via crossover
    Mutate offspring
    Keep elite individuals
    Replace population
Save best individual
```

## Output Files

All outputs saved to `training/ga_checkpoints/`:

- `best_gen_X.pkl`: Best individual from generation X (saved every 5 generations)
- `best_final.pkl`: Best individual from final generation
- `fitness_history.npz`: Arrays of best and average fitness per generation
- `fitness_evolution.png`: Plot of fitness over generations
- `*_visualization.png`: Genome visualizations for loaded checkpoints

## Tips for Tuning

### Increase Contact with Target
- Increase `WEIGHT_CONTACT`
- Increase simulation duration to allow more time for contact
- Increase `MAX_STIFFNESS` for stronger tendons

### Improve Lifting
- Increase `WEIGHT_HEIGHT`
- Adjust initial joint positions to position chopsticks better
- Increase population size for more exploration

### Reduce Computation Time
- Decrease `SIM_DURATION`
- Decrease `POPULATION_SIZE`
- Decrease `GENERATIONS`

### Encourage Simpler Solutions
- Increase `WEIGHT_EFFICIENCY`
- Decrease `BASE_CONNECTION_PROB` for sparser initial networks

## Advanced Usage

### Custom Fitness Function

Modify `evaluate_fitness()` in `ga_tendon_optimizer.py`:

```python
def evaluate_fitness(individual, ...):
    # ... run simulation ...
    
    # Custom fitness components
    my_metric = compute_my_metric(sim, target)
    
    fitness = (WEIGHT_CONTACT * contact +
               WEIGHT_CUSTOM * my_metric +
               ...)
    return fitness
```

### Different Site Configurations

Modify the site grid in `build_model_from_individual()`:

```python
# Current: 3×3×5 grid
nx, ny, nz = 3, 3, 5

# Denser grid for more options
nx, ny, nz = 5, 5, 7
```

### Multi-Objective Optimization

Track Pareto front of solutions:
- Contact vs. Efficiency
- Height vs. Stability
- etc.

## Troubleshooting

**"Import error: networkx"**
- Install: `pip install networkx`

**"Simulation hangs"**
- Reduce `SIM_DURATION`
- Check for unstable tendon configurations (very high stiffness)

**"All fitnesses are zero"**
- Check geom IDs are correct
- Verify contact detection is working
- Try increasing `WEIGHT_CONTACT`

**"No improvement over generations"**
- Increase `MUTATION_RATE`
- Increase `POPULATION_SIZE`
- Check if fitness function is too restrictive

## Example Workflow

```bash
# 1. Quick test run
mjpython training/ga_tendon_optimizer.py --generations 10 --population 10

# 2. Check results
python training/plot_ga_results.py --history

# 3. Full run
mjpython training/ga_tendon_optimizer.py --generations 100 --population 40

# 4. Visualize best
python training/plot_ga_results.py --checkpoint ga_checkpoints/best_final.pkl

# 5. Test in simulation
mjpython training/ga_tendon_optimizer.py --load ga_checkpoints/best_final.pkl
```

## Architecture

```
ga_tendon_optimizer.py
├── GAConfig              # Configuration parameters
├── Individual            # Genome representation
│   ├── adjacency_matrix
│   ├── stiffness_matrix
│   └── damping_matrix
├── build_model_from_individual()  # MuJoCo model builder
├── evaluate_fitness()    # Fitness evaluation
├── tournament_selection() # Selection operator
├── crossover()           # Crossover operator
├── mutate()              # Mutation operator
└── run_genetic_algorithm() # Main GA loop

plot_ga_results.py
├── plot_fitness_history()
├── visualize_individual()
└── compare_generations()
```

## Future Enhancements

- [ ] Parallel fitness evaluation (multiprocessing)
- [ ] Adaptive mutation rates
- [ ] Island model (multiple sub-populations)
- [ ] Co-evolution of joint positions and tendon networks
- [ ] Real-world transfer learning
- [ ] Interactive visualization during evolution

## References

- MuJoCo documentation: https://mujoco.readthedocs.io/
- Genetic Algorithms: Holland, J. H. (1992)
- Tendon-driven robots: Pratt & Williamson (1995)
