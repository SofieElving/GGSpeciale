#!/bin/bash
#SBATCH --account=GGSpeciale
#SBATCH --job-name=simglucose_train
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=output/slurm_%x_%A_%a.out
#SBATCH --error=output/slurm_%x_%A_%a.err

set -Eeuo pipefail
set -x
# source ~/miniforge3/envs/thesis-env/bin/activate

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

REWARD_TYPE="smooth"
TIMESTEPS=3000000
BASE_DIR="./teacher_models"

PATIENT=${PATIENTS[$SLURM_ARRAY_TASK_ID]}
SAFE_PATIENT="${PATIENT//#/-}"

echo "Running patient: $PATIENT"

python train.py \
    --patient "$PATIENT" \
    --reward-type "$REWARD_TYPE" \
    --timesteps $TIMESTEPS \
    --seed 42 \
    --outdir "${BASE_DIR}/${REWARD_TYPE}/${SAFE_PATIENT}" \
    --scenario-mode "fixed_hb" \
    --learning-rate 3e-4 \
    --n-steps 480 \
    --batch-size 240 \
    --n-epochs 10 \
    --gamma 0.995 \
    --gae-lambda 0.95 \
    --clip-range 0.15 \
    --ent-coef 0.05 \
    --vf-coef 0.5 \
    --max-grad-norm 0.5