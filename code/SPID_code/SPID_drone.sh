#!/bin/bash
#SBATCH --account=GGSpeciale
#SBATCH --job-name=drone_distill
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-9
#SBATCH --output=/home/sofelving/GGSpeciale/GGSpeciale/quadcopter-suite/output_distil/distill_%x_%A_%a.out
#SBATCH --error=/home/sofelving/GGSpeciale/GGSpeciale/quadcopter-suite/output_distil/distill_%x_%A_%a.err


set -Eeuo pipefail
set -x

cd /home/sofelving/GGSpeciale/GGSpeciale/code/SPID_code

source ~/miniforge3/etc/profile.d/conda.sh
conda activate thesis-env

TEACHER_PATH="/home/sofelving/GGSpeciale/GGSpeciale/quadcopter-suite/results/save-04.10.2026_15.39.13/best_model.zip"
BASE_SAVE="/home/sofelving/GGSpeciale/GGSpeciale/quadcopter-suite/distil_timesteps"

NESTED='{"sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0}, "exp": {"exp": 0, "log": 0}, "log": {"exp": 0, "log": 0}, "sqrt": {"sqrt": 0}}'
UNARY_OPERATORS=(sin cos exp log sqrt)

# Define experiments as: name total_timesteps maxsize n_iter maxdepth
experiments=(
    "timesteps_1250_maxsize30_1  1250  30 10"
    "timesteps_1250_maxsize30_2  1250  30 10"
    "timesteps_1250_maxsize30_3  1250  30 10"
    "timesteps_1250_maxsize30_4  1250  30 10"
    "timesteps_1250_maxsize30_5  1250  30 10"
    "timesteps_1250_maxsize30_6  1250  30 10"
    "timesteps_1250_maxsize30_7  1250  30 10"
    "timesteps_1250_maxsize30_8  1250  30 10"
    "timesteps_1250_maxsize30_9  1250  30 10"
    "timesteps_1250_maxsize30_10  1250  30 10"
)

# Pick the experiment for this array task
read -r name timesteps maxsize n_iter <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

echo "Running experiment: $name"

python SPID_drone.py \
    --teacher_path "$TEACHER_PATH" \
    --save_folder "$BASE_SAVE/$name" \
    --total_timesteps $timesteps \
    --maxsize $maxsize \
    --n_iter $n_iter \
    --n_eval_episodes 10 \
    --verbose 2
    # --nested_constraints "$NESTED" \
    # --unary_operators "${UNARY_OPERATORS[@]}"

echo "Finished experiment: $name"