#!/bin/bash
#SBATCH --account=GGSpeciale
#SBATCH --job-name=simglucose_distill
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=output_distill/distill_%x_%A_%a.out
#SBATCH --error=output_distill/distill_%x_%A_%a.err

set -Eeuo pipefail
set -x

mkdir -p output

# Safer PySR / Julia / torch setup
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes
export PYTHONFAULTHANDLER=1
export TF_ENABLE_ONEDNN_OPTS=0

export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PATIENTS=(
"adult#001"
"adult#002"
"adult#003"
"adult#004"
"adult#005"
"adult#006"
"adult#007"
"adult#008"
"adult#009"
"adult#010"
)

PATIENT="${PATIENTS[$SLURM_ARRAY_TASK_ID]}"

TEACHER_ROOT="./teacher_models"
SAVE_ROOT="./distil_results_w_history"
REWARD_TYPE="smooth"

echo "Distilling patient: ${PATIENT}"

python distill.py \
    --teacher-root "${TEACHER_ROOT}" \
    --save-root "${SAVE_ROOT}" \
    --patients "${PATIENT}" \
    --reward-type "${REWARD_TYPE}" \
    --teacher-model-name final_model.zip \
    --scenario-mode semi_random_hb \
    --time-std-multiplier 0.5 \
    --include-snacks \
    --max-insulin-action 5.0 \
    --n-iter 12 \
    --total-timesteps 12000 \
    --n-eval-episodes 10 \
    --verbose 2