# Genetic Algorithm for Tendon Network Optimization (DEAP Version)

This system uses **DEAP** (Distributed Evolutionary Algorithms in Python) to evolve optimal tendon network configurations for robotic grasping tasks.

## Why DEAP?

- **Professional & Well-Tested**: Industry-standard GA library used in research and production
- **Efficient**: Optimized C implementations for genetic operators
- **Flexible**: Easy to customize selection, crossover, and mutation strategies
- **Statistics**: Built-in tools for tracking evolution progress
- **Hall of Fame**: Automatically tracks best individuals (elitism)

## Installation

```bash
# Install DEAP library
pip install deap

# Or using python3
python3 -m pip install deap
```

## Overview

The genetic algorithm evolves:
- **Adjacency Matrix**: Which sites are connected by tendons (binary)
- **Stiffness Values**: Spring constant for each tendon (10-100 N/m)
- **Damping Values**: Damping coefficient for each tendon (1-10 N·s/m)

### Genome Encoding

The genome is encoded as a flat list (for compatibility with DEAP):
1. First 16 values: Adjacency (0/1 for each cross-chopstick connection)
2. Next 16 values: Stiffness values (float)
3. Last 16 values: Damping values (float)

Total genome length: 48 genes

### Fitness Function

Fitness is a weighted combination of:

1. **Contact** (weight: 10.0): Total contact forces between chopsticks and target
2. **Height** (weight: 5.0): How much the target block is lifted
3. **Stability** (weight: 3.0): Inverse of target velocity (stable grasp)
4. **Efficiency** (weight: 1.0): Penalty for too many tendons

## Quick Start

```bash
# Run evolution with visualization of best result
mjpython training/ga_tendon_optimizer_deap.py --visualize

# Resume from checkpoint
mjpython training/ga_tendon_optimizer_deap.py --resume ga_checkpoints/gen_20.pkl

# Load and visualize saved best individual
mjpython training/ga_tendon_optimizer_deap.py --load ga_checkpoints/best_final.pkl
```

## Configuration

Edit the `GAConfig` class in the script to customize:

```python
class GAConfig:
    # GA Parameters
    POPULATION_SIZE = 20          # Number of individuals
    NUM_GENERATIONS = 50          # Evolution iterations
    TOURNAMENT_SIZE = 3           # Selection pressure
    CROSSOVER_PROB = 0.7          # Probability of crossover
    MUTATION_PROB = 0.2           # Probability of mutation
    ELITE_SIZE = 2                # Best individuals preserved
    
    # Tendon parameters
    N_SITES = 4                   # Sites per chopstick
    MIN_STIFFNESS = 10.0          # N/m
    MAX_STIFFNESS = 100.0         # N/m
    MIN_DAMPING = 1.0             # N*s/m
    MAX_DAMPING = 10.0            # N*s/m
    
    # Mutation
    FLIP_PROB = 0.1               # Connection flip probability
    GAUSSIAN_SIGMA = 10.0         # Stiffness/damping noise
    
    # Fitness weights
    CONTACT_WEIGHT = 10.0
    HEIGHT_WEIGHT = 5.0
    STABILITY_WEIGHT = 3.0
    EFFICIENCY_WEIGHT = 1.0
```

## DEAP Features Used

### Selection
- **Tournament Selection**: Randomly picks K individuals, selects best
- Controlled by `TOURNAMENT_SIZE` parameter

### Crossover
- **Uniform Crossover**: For adjacency matrix (topology)
- **Average Crossover**: For stiffness/damping (continuous values)

### Mutation
- **Bit Flip**: Randomly flip connections
- **Gaussian Noise**: Add noise to stiffness/damping values
- **Bounded**: Values clamped to valid ranges

### Elitism
- Best N individuals automatically preserved each generation
- Prevents losing good solutions

### Statistics
- Tracks min, max, avg, std of fitness each generation
- Logged and saved for analysis

## How It Works

1. **Initialization**: Create random population of tendon networks
2. **Evaluation**: Simulate each individual and measure fitness
3. **Selection**: Tournament selection picks parents
4. **Crossover**: Combine parent genomes to create offspring
5. **Mutation**: Randomly modify offspring
6. **Replacement**: New generation = offspring + elite from previous
7. **Repeat**: Continue for N generations

## Output Files

```
ga_checkpoints/
├── gen_5.pkl       # Checkpoint at generation 5
├── gen_10.pkl      # Checkpoint at generation 10
├── ...
└── best_final.pkl  # Best individual + evolution statistics
```

Each checkpoint contains:
- Population state
- Hall of fame (best individuals)
- Logbook (evolution statistics)
- Generation number

## Visualizing Results

After evolution completes:

```python
# The script will print:
# - Best fitness achieved
# - Number of tendons in best solution
# - Path to saved results

# If --visualize flag is used:
# - MuJoCo viewer will show best individual in action
# - Watch the evolved tendon network grasp the target
```

## Analyzing Evolution

Use the `plot_ga_results.py` script to visualize evolution:

```bash
# Plot fitness over generations
python training/plot_ga_results.py --history

# Visualize best individual's genome
python training/plot_ga_results.py --individual ga_checkpoints/best_final.pkl

# Compare generations
python training/plot_ga_results.py --compare
```

## Troubleshooting

### Error: "Import deap could not be resolved"
```bash
pip install deap
# or
python3 -m pip install deap
```

### Evolution stuck at low fitness
- Try increasing `POPULATION_SIZE` (more diversity)
- Increase `MUTATION_PROB` (more exploration)
- Adjust fitness weights to emphasize different objectives

### Checkpoints too large
- Reduce `POPULATION_SIZE`
- Increase `CHECKPOINT_INTERVAL` to save less frequently

### Simulation crashes
- Check that MuJoCo is properly installed
- Verify genome decoding produces valid matrices
- Add error handling in fitness evaluation

## Comparison: Custom GA vs DEAP

| Feature | Custom GA | DEAP |
|---------|-----------|------|
| Code complexity | Higher | Lower |
| Performance | Good | Excellent (C optimized) |
| Flexibility | Custom operators | Standard + custom |
| Statistics | Manual tracking | Built-in |
| Documentation | Self-written | Extensive online |
| Debugging | More effort | Easier with tools |
| Community | None | Large community |

## Advanced Usage

### Custom Genetic Operators

```python
# Define custom crossover
def my_crossover(ind1, ind2):
    # Your custom logic
    return ind1, ind2

# Register with toolbox
toolbox.register("mate", my_crossover)
```

### Multi-Objective Optimization

DEAP supports multi-objective GA (NSGA-II, SPEA2):

```python
# Change to multi-objective
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))

# Return multiple objectives
def evaluate(individual):
    return (contact_score, height_score, efficiency_score)
```

### Parallel Evaluation

```python
from multiprocessing import Pool

# Use multiprocessing
pool = Pool()
toolbox.register("map", pool.map)

# Evaluation will run in parallel
```

## References

- DEAP Documentation: https://deap.readthedocs.io/
- DEAP GitHub: https://github.com/DEAP/deap
- Paper: Fortin et al. (2012). "DEAP: Evolutionary Algorithms Made Easy"

## Next Steps

1. **Run evolution**: `mjpython training/ga_tendon_optimizer_deap.py --visualize`
2. **Analyze results**: Check fitness curves, best individual
3. **Tune parameters**: Adjust population size, mutation rates, fitness weights
4. **Experiment**: Try different selection methods, crossover strategies
5. **Deploy**: Use best individual in real robotic system

## License

DEAP is licensed under LGPL-3.0.
