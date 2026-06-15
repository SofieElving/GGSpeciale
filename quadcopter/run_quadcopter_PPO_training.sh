#!/bin/bash

#SBATCH --account GGSPeciale
#SBATCH --mem 8g

# Exit on error
set -e

echo "======================================"
echo "Starting quadcopter training"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "======================================"

# Activate environment (edit if needed)
source ~/.bashrc
conda activate thesis-rl

# Move to project directory
cd /home/ashc/GGSpeciale/gym-pybullet-drones/gym_pybullet_drones/examples/
echo "Working dir: $(pwd)"

# Run the script
python learn.py  --multiagent false --gui false --record_video true --output_folder "/home/ashc/GGSpeciale/ashc_repo/GGSpeciale/quadcopter-suite/results"

echo "======================================"
echo "Finished run"
echo "Date: $(date)"
echo "======================================"