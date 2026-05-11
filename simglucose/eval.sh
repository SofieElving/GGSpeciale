#!/bin/bash
#SBATCH --job-name=eval_spid_simglucose
#SBATCH --output=logs/eval_spid_simglucose_%j.out
#SBATCH --error=logs/eval_spid_simglucose_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

mkdir -p logs


python eval2.py \
  --train-dir "./output_changed_mealnorm" \
  --distill-dir "./distil_single" \
  --outdir "./eval_single" \
  --patients "adult#001,adult#002,adult#003,adult#004,adult#005,adult#006,adult#007,adult#008,adult#009,adult#010" \
  --teacher-model-path "./output_changed_mealnorm/models/best/best_model.zip" \
  --student-path "./distil_single/best_student_policy.joblib" \
  --scenario-mode "semi_random_hb" \
  --use-custom-reward \
  --warning-window-min 20 \
  --max-insulin-action 6