#!/bin/bash
#SBATCH --account=GGSpeciale
#SBATCH --job-name=spid
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/faststorage/project/GGSpeciale/GGSpeciale/bash_scripts/logs/%x_%j.out
#SBATCH --error=/faststorage/project/GGSpeciale/GGSpeciale/bash_scripts/logs/%x_%j.err

set -Eeuo pipefail
set -x

PROJECT_ROOT="/faststorage/project/GGSpeciale/GGSpeciale"
PYTHON_BIN="/home/elisalaegs/miniforge3/envs/thesis/bin/python"

mkdir -p "$PROJECT_ROOT/bash_scripts/logs"
mkdir -p "$PROJECT_ROOT/bash_scripts/results"

ENV_NAME="${ENV_NAME:-CartPole-v1}"
EXPERT_ALGO="${EXPERT_ALGO:-TD3}"
N_ITER="${N_ITER:-10}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500}"
EXPERT_MODEL="${EXPERT_MODEL:-$PROJECT_ROOT/code/baseline_code/baseline_models/cartpole/TD3_cartpole.zip}"
VERBOSE="${VERBOSE:-2}"

RUN_DIR="$PROJECT_ROOT/bash_scripts/results/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

cd "$PROJECT_ROOT"

PYTHONPATH="$PROJECT_ROOT/code" \
"$PYTHON_BIN" -u "$PROJECT_ROOT/bash_scripts/run_spid_job.py" \
  --teacher_path "$EXPERT_MODEL" \
  --teacher_algo "$EXPERT_ALGO" \
  --environment "$ENV_NAME" \
  --n_iter "$N_ITER" \
  --total_timesteps "$TOTAL_TIMESTEPS" \
  --verbose "$VERBOSE" \
  --output_dir "$RUN_DIR"