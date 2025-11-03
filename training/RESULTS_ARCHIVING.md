# GA Training Results Archiving System

## Overview

Every training run now automatically creates a complete archive with all relevant data, ensuring no results are ever lost.

## What Gets Saved

Each training run creates a timestamped directory in `training_results/run_YYYYMMDD_HHMMSS/` containing:

### 1. **training_log.txt**
Complete console output including:
- All configuration parameters
- Per-generation statistics (best, avg, min, std, time)
- Final results summary
- File locations

### 2. **best_individual.pkl**
Saved best evolved individual that can be reloaded with:
```bash
mjpython training/ga_tendon_optimizer_deap.py --load training_results/run_XXX/best_individual.pkl
```

### 3. **evolution_statistics.png**
High-resolution plots showing:
- Max/Avg/Min fitness over generations
- ±1 standard deviation shaded region
- Population diversity (std dev) over time

### 4. **best_individual.mp4**
Video recording (1920x1080, 30fps) of the best individual performing the task

### 5. **configuration.json**
All parameters in JSON format:
- GA parameters (population, generations, crossover/mutation rates, etc.)
- Network parameters (sites, stiffness/damping ranges)
- Simulation parameters (duration, timestep, lift timing)
- Fitness weights
- Results (best fitness, num tendons, timing)

### 6. **ga_tendon_optimizer_deap.py**
Complete snapshot of the code used for this run, preserving:
- Algorithm implementation
- Fitness function
- All parameters
- Enables exact reproduction of results

### 7. **README.md**
Quick summary with:
- Best fitness and key metrics
- List of all files
- Configuration highlights
- Commands to load and test this run

## Additional Features

### runs_summary.csv
A CSV file in `training_results/` tracks all runs:
```csv
timestamp,run_dir,best_fitness,num_tendons,generations,population_size,total_time_minutes,time_per_gen_seconds,num_processes
20251031_143022,run_20251031_143022,125.43,47,30,50,15.2,30.4,all
```

Easy to import into spreadsheets or pandas for analysis.

## Usage

### Run Training (Automatically Archives)
```bash
# Standard run - creates archive automatically
mjpython training/ga_tendon_optimizer_deap.py

# With parallelization
mjpython training/ga_tendon_optimizer_deap.py --parallel 8

# Resume from checkpoint
mjpython training/ga_tendon_optimizer_deap.py --resume ga_checkpoints/gen_10.pkl
```

### Access Previous Results
```bash
# List all runs
ls -lh training_results/

# View specific run
cd training_results/run_20251031_143022/
cat README.md
less training_log.txt
open best_individual.mp4

# Load and test
mjpython training/ga_tendon_optimizer_deap.py --load training_results/run_20251031_143022/best_individual.pkl --visualize
```

### Compare Multiple Runs
```bash
# Compare fitness evolution
open training_results/run_*/evolution_statistics.png

# Compare configurations
cat training_results/run_*/configuration.json | jq '.best_fitness'

# View summary
cat training_results/runs_summary.csv
```

## Benefits

✅ **No Data Loss** - Every run is completely archived  
✅ **Reproducibility** - Code snapshots enable exact reproduction  
✅ **Traceability** - Track what parameters led to which results  
✅ **Comparison** - Easy to compare multiple runs  
✅ **Documentation** - Comprehensive logs for analysis  
✅ **Portability** - Share entire run directory with others  

## Storage

- Each run: ~100-500 MB (mostly video)
- Videos can be compressed or removed if space limited
- Old runs can be archived to external storage
- `.gitignore` prevents accidental commits of large files

## Example Workflow

```bash
# Day 1: Initial training
mjpython training/ga_tendon_optimizer_deap.py --parallel
# Creates: training_results/run_20251031_100000/

# Day 2: Try different weights
# Edit GAConfig.HEIGHT_WEIGHT = 10
mjpython training/ga_tendon_optimizer_deap.py --parallel
# Creates: training_results/run_20251101_140000/

# Day 3: Compare results
cd training_results/
cat run_*/configuration.json | jq '{run: .timestamp, fitness: .best_fitness, height_weight: .height_weight}'

# Load best one
mjpython training/ga_tendon_optimizer_deap.py --load run_20251101_140000/best_individual.pkl --visualize
```
