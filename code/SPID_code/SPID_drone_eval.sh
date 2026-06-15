#!/bin/bash
#SBATCH --account=GGSpeciale
#SBATCH --job-name=drone_analyze
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/home/sofelving/GGSpeciale/GGSpeciale/quadcopter-suite/output_eval/analyze_%x_%j.out
#SBATCH --error=/home/sofelving/GGSpeciale/GGSpeciale/quadcopter-suite/output_eval/analyze_%x_%j.err

set -Eeuo pipefail
set -x

cd /home/sofelving/GGSpeciale/GGSpeciale/code/SPID_code

source ~/miniforge3/etc/profile.d/conda.sh
conda activate thesis-env

TEACHER_PATH="/home/sofelving/GGSpeciale/GGSpeciale/quadcopter-suite/results/save-04.10.2026_15.39.13/best_model.zip"
DISTIL_DIR="/home/sofelving/GGSpeciale/GGSpeciale/quadcopter-suite/distil_first_policy"

for folder in "$DISTIL_DIR"/*/; do
    policy_path="$folder/best_student_policy.joblib"
    
    if [ ! -f "$policy_path" ]; then
        echo "Skipping $folder — no joblib found"
        continue
    fi
    
    echo "Analyzing: $folder"
    
    python SPID_drone_eval.py \
        --policy_path "$policy_path" \
        --teacher_path "$TEACHER_PATH" \
        --output_dir "$folder/plots"
    
    echo "Done: $folder"
done