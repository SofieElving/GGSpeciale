#!/bin/bash

#SBATCH --account GGSPeciale
#SBATCH --mem 4g
#SBATCH --time 08:00:00

# Exit on error
set -e

echo "======================================"
echo "Starting Swimmer baselines"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "======================================"

# Activate environment (edit if needed)
source ~/.bashrc
conda activate thesis-rl

# Move to project directory
cd /home/ashc/GGSpeciale/ashc_repo/GGSpeciale/code/baseline_code/baseline_runs
echo "Working dir: $(pwd)"

# Run the script
python run_Swimmer.py

echo "======================================"
echo "Finished run"
echo "Date: $(date)"
echo "======================================"