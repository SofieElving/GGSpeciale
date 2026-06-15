#!/bin/bash

#SBATCH --account GGSPeciale
#SBATCH --mem 8g

# Exit on error
set -e

echo "======================================"
echo "Starting SPID CartPole experiment"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"
echo "======================================"

# Activate environment (edit if needed)
source ~/.bashrc
conda activate thesis-rl

# Move to project directory
cd /home/ashc/GGSpeciale/ashc_repo/GGSpeciale/code
echo "Working dir: $(pwd)"

# Run the script
python SPID_code/run_SPID_on_cartpole.py

echo "======================================"
echo "Finished run"
echo "Date: $(date)"
echo "======================================"