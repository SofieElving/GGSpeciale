#!/bin/bash
#SBATCH --account=GGSpeciale
#SBATCH --job-name=simglucose_eval
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=output/slurm_%x_%A_%a.out
#SBATCH --error=output/slurm_%x_%A_%a.err

set -Eeuo pipefail
set -x

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

REWARD_TYPE="steps"
BASE_DIR="./teacher_models"
N_EVAL_EPISODES=10

PATIENT=${PATIENTS[$SLURM_ARRAY_TASK_ID]}
SAFE_PATIENT="${PATIENT//#/-}"

echo "Evaluating patient: $PATIENT"
echo "Safe patient folder: $SAFE_PATIENT"

python eval_patient.py \
    --patient "$PATIENT" \
    --model-dir "${BASE_DIR}/${REWARD_TYPE}/${SAFE_PATIENT}" \
    --reward-type "$REWARD_TYPE" \
    --scenario-mode "fixed_hb" \
    --n-eval-episodes 100 \
    --shield-bg-threshold 10