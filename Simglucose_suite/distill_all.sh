#!/bin/bash
#SBATCH --account=GGSpeciale
#SBATCH --job-name=simglucose_distill
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-49
#SBATCH --output=output_distill_un/distill_%x_%A_%a.out
#SBATCH --error=output_distill_un/distill_%x_%A_%a.err

set -Eeuo pipefail
set -x

mkdir -p output_distill

export PYTHON_JULIACALL_HANDLE_SIGNALS=yes
export PYTHONFAULTHANDLER=1
export TF_ENABLE_ONEDNN_OPTS=0

export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1
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

PYSR_CONFIG="square_threshold_15" # square

N_REPEATS=5

N_PATIENTS=${#PATIENTS[@]}
N_TOTAL=$((N_PATIENTS * N_REPEATS))

if (( SLURM_ARRAY_TASK_ID >= N_TOTAL )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is outside N_TOTAL=${N_TOTAL}"
    exit 1
fi

PATIENT_INDEX=$((SLURM_ARRAY_TASK_ID / N_REPEATS))
REPEAT_INDEX=$((SLURM_ARRAY_TASK_ID % N_REPEATS))

PATIENT="${PATIENTS[$PATIENT_INDEX]}"
SAFE_PATIENT="${PATIENT//#/-}"
RUN_LABEL=$(printf "run_%02d" "${REPEAT_INDEX}")

TEACHER_ROOT="./models/open_optuna_optimal"
BASE_SAVE_ROOT="./distilled_unnormalized/open"
REWARD_TYPE="clarke_risk"
ENV_NAME="env_open"

# Separate output folder per config, patient, and repeat.
SAVE_ROOT="${BASE_SAVE_ROOT}/${PYSR_CONFIG}/${SAFE_PATIENT}/${RUN_LABEL}"

echo "============================================================"
echo "SLURM task id: ${SLURM_ARRAY_TASK_ID}"
echo "Patient index: ${PATIENT_INDEX}"
echo "Repeat index: ${REPEAT_INDEX}"
echo "Patient: ${PATIENT}"
echo "Safe patient: ${SAFE_PATIENT}"
echo "PySR config: ${PYSR_CONFIG}"
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
    --time-std-multiplier 0.5 \
    --amount-noise-std-fraction 0.1 \
    --actual-time-noise-std-min 5.0 \
    --actual-time-noise-clip-min 15.0 \
    --max-insulin-action 5 \
    --shield-bg-threshold 10 \
    --n-iter 10 \
    --total-timesteps 4800 \
    --n-eval-episodes 300 \
    --save-history \
    --max-sampling-episodes 300 \
    --pysr-configs "${PYSR_CONFIG}" \
    --sample-episodes 5 \
    --bb-warmup \
    --keep-early-terminal-episodes

echo "Finished patient=${PATIENT}, config=${PYSR_CONFIG}, repeat=${REPEAT_INDEX}"