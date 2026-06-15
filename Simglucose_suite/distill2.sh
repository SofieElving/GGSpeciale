#!/bin/bash
#SBATCH --account=GGSpeciale
#SBATCH --job-name=simglucose_distill
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=output_distill_multi/distill_%x_%A_%a.out
#SBATCH --error=output_distill_multi/distill_%x_%A_%a.err

set -Eeuo pipefail
set -x

mkdir -p output_distill

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
)

PYSR_CONFIGS=(
    #"linear_10"
    #"square_10"
    #"square_threshold_10"
    "square_threshold_15"
    #"square_threshold_20"
)

N_REPEATS=5

N_PATIENTS=${#PATIENTS[@]}
N_CONFIGS=${#PYSR_CONFIGS[@]}
N_TOTAL=$((N_PATIENTS * N_CONFIGS * N_REPEATS))

if (( SLURM_ARRAY_TASK_ID >= N_TOTAL )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is outside N_TOTAL=${N_TOTAL}"
    exit 1
fi

PATIENT_INDEX=$((SLURM_ARRAY_TASK_ID / (N_CONFIGS * N_REPEATS)))
REM=$((SLURM_ARRAY_TASK_ID % (N_CONFIGS * N_REPEATS)))
CONFIG_INDEX=$((REM / N_REPEATS))
REPEAT_INDEX=$((REM % N_REPEATS))

PATIENT="${PATIENTS[$PATIENT_INDEX]}"
CONFIG="${PYSR_CONFIGS[$CONFIG_INDEX]}"
RUN_LABEL=$(printf "run_%02d" "${REPEAT_INDEX}")

TEACHER_ROOT="./models/closed_hist_optuna_optimal_2"
BASE_SAVE_ROOT="./distilled_model_multi_day/closed"
REWARD_TYPE="clarke_risk"
ENV_NAME="env_closed_action_history"

# Separate output folder per config and repeat.
SAVE_ROOT="${BASE_SAVE_ROOT}/${CONFIG}/${RUN_LABEL}"

echo "============================================================"
echo "SLURM task id: ${SLURM_ARRAY_TASK_ID}"
echo "Patient index: ${PATIENT_INDEX}"
echo "Config index: ${CONFIG_INDEX}"
echo "Repeat index: ${REPEAT_INDEX}"
echo "Patient: ${PATIENT}"
echo "PySR config: ${CONFIG}"
echo "Run label: ${RUN_LABEL}"
echo "Environment: ${ENV_NAME}"
echo "Teacher root: ${TEACHER_ROOT}"
echo "Save root: ${SAVE_ROOT}"
echo "============================================================"

python distill.py \
    --env "${ENV_NAME}" \
    --teacher-root "${TEACHER_ROOT}" \
    --save-root "${SAVE_ROOT}" \
    --patients "${PATIENT}" \
    --reward-type "${REWARD_TYPE}" \
    --teacher-model-name best_model.zip \
    --scenario-mode semi_random_hb \
    --include-snacks \
    --time-std-multiplier 1.5 \
    --amount-noise-std-fraction 0.4 \
    --actual-time-noise-std-min 5.0 \
    --actual-time-noise-clip-min 15.0 \
    --max-insulin-action 5 \
    --shield-bg-threshold 10 \
    --n-iter 12 \
    --total-timesteps 5760 \
    --n-eval-episodes 300 \
    --save-history \
    --max-sampling-episodes 300 \
    --max-episode-steps 1440 \
    --pysr-configs "${CONFIG}" \
    --sample-episodes 0 \
    --bb-warmup \
    --keep-early-terminal-episodes

echo "Finished patient=${PATIENT}, config=${CONFIG}, repeat=${REPEAT_INDEX}"