#!/bin/bash
#SBATCH --account=GGSpeciale
#SBATCH --job-name=simglucose_distill_progress
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-49
#SBATCH --output=output_distill_progress_2/distill_%x_%A_%a.out
#SBATCH --error=output_distill_progress_2/distill_%x_%A_%a.err

set -Eeuo pipefail
set -x

mkdir -p output_distill_progress

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
"square_threshold_15"
# "square_10"
# "square_threshold_10"
#"square_threshold_15"
#"square_threshold_20"
# "square_threshold_30"
)

# CHECKPOINT_STEPS=($(seq 300000 300000 1500000))
CHECKPOINT_STEPS=($(seq 100000 10000 200000))

N_REPEATS=5

N_PATIENTS=${#PATIENTS[@]}
N_CONFIGS=${#PYSR_CONFIGS[@]}
N_STEPS=${#CHECKPOINT_STEPS[@]}
N_TOTAL=$((N_PATIENTS * N_CONFIGS * N_STEPS * N_REPEATS))

if (( SLURM_ARRAY_TASK_ID >= N_TOTAL )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is outside N_TOTAL=${N_TOTAL}"
    exit 1
fi

PATIENT_INDEX=$((SLURM_ARRAY_TASK_ID / (N_CONFIGS * N_STEPS * N_REPEATS)))
REM=$((SLURM_ARRAY_TASK_ID % (N_CONFIGS * N_STEPS * N_REPEATS)))

CONFIG_INDEX=$((REM / (N_STEPS * N_REPEATS)))
REM=$((REM % (N_STEPS * N_REPEATS)))

STEP_INDEX=$((REM / N_REPEATS))
REPEAT_INDEX=$((REM % N_REPEATS))

PATIENT="${PATIENTS[$PATIENT_INDEX]}"
CONFIG="${PYSR_CONFIGS[$CONFIG_INDEX]}"
STEP="${CHECKPOINT_STEPS[$STEP_INDEX]}"
STEP_LABEL=$(printf "step_%07d" "${STEP}")
RUN_LABEL=$(printf "run_%02d" "${REPEAT_INDEX}")

SAFE_PATIENT="${PATIENT//#/-}"

TEACHER_BASE="./models/fine_grained/closed_hist_optuna_optimal_2"
BASE_SAVE_ROOT="./distilled_progress_finer_granularity_2"
REWARD_TYPE="clarke_risk"
ENV_NAME="env_closed_action_history"

REAL_PATIENT_DIR="${TEACHER_BASE}/${REWARD_TYPE}/${SAFE_PATIENT}"
REAL_MODEL_PATH="${REAL_PATIENT_DIR}/models/ppo_simglucose_${STEP}_steps.zip"

if [[ ! -f "${REAL_MODEL_PATH}" ]]; then
    echo "Missing checkpoint: ${REAL_MODEL_PATH}"
    exit 0
fi

REAL_MODEL_PATH="$(readlink -f "${REAL_MODEL_PATH}")"

STAGING_ROOT="./output_distill_progress/teacher_staging/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
STAGING_PATIENT_DIR="${STAGING_ROOT}/${REWARD_TYPE}/${SAFE_PATIENT}"
STAGING_BEST_DIR="${STAGING_PATIENT_DIR}/models/best"

mkdir -p "${STAGING_BEST_DIR}"
ln -sfn "${REAL_MODEL_PATH}" "${STAGING_BEST_DIR}/best_model.zip"

if [[ -f "${REAL_PATIENT_DIR}/train_config.json" ]]; then
    ln -sfn "$(readlink -f "${REAL_PATIENT_DIR}/train_config.json")" "${STAGING_PATIENT_DIR}/train_config.json"
    ln -sfn "$(readlink -f "${REAL_PATIENT_DIR}/train_config.json")" "${STAGING_PATIENT_DIR}/training_config.json"
elif [[ -f "${REAL_PATIENT_DIR}/training_config.json" ]]; then
    ln -sfn "$(readlink -f "${REAL_PATIENT_DIR}/training_config.json")" "${STAGING_PATIENT_DIR}/train_config.json"
    ln -sfn "$(readlink -f "${REAL_PATIENT_DIR}/training_config.json")" "${STAGING_PATIENT_DIR}/training_config.json"
fi

# Separate output by config, checkpoint, and repeat.
SAVE_ROOT="${BASE_SAVE_ROOT}/${CONFIG}/${STEP_LABEL}/${RUN_LABEL}"

echo "============================================================"
echo "SLURM task id: ${SLURM_ARRAY_TASK_ID}"
echo "Patient index: ${PATIENT_INDEX}"
echo "Config index: ${CONFIG_INDEX}"
echo "Step index: ${STEP_INDEX}"
echo "Repeat index: ${REPEAT_INDEX}"
echo "Patient: ${PATIENT}"
echo "Safe patient: ${SAFE_PATIENT}"
echo "PySR config: ${CONFIG}"
echo "Teacher checkpoint step: ${STEP}"
echo "Step label: ${STEP_LABEL}"
echo "Run label: ${RUN_LABEL}"
echo "Teacher model: ${REAL_MODEL_PATH}"
echo "Staging root: ${STAGING_ROOT}"
echo "Save root: ${SAVE_ROOT}"
echo "Environment: ${ENV_NAME}"
echo "============================================================"

python distill.py \
    --env "${ENV_NAME}" \
    --teacher-root "${STAGING_ROOT}" \
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
    --bb-warmup \
    --n-iter 10 \
    --total-timesteps 4800 \
    --n-eval-episodes 100 \
    --save-history \
    --max-sampling-episodes 500 \
    --pysr-configs "${CONFIG}" \
    --sample-episodes 0 \
    --keep-early-terminal-episodes

echo "Finished patient=${PATIENT}, config=${CONFIG}, checkpoint=${STEP}, repeat=${REPEAT_INDEX}"