#!/bin/bash
#SBATCH --account=GGSpeciale
#SBATCH --job-name=spid
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -Eeuo pipefail

mkdir -p logs

PYTHON_BIN=/home/elisalaegs/miniforge3/envs/thesis/bin/python

PYTHONPATH=.. \
$PYTHON_BIN -u run_Acrobot.py