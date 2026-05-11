#!/bin/bash
#SBATCH --job-name=spid_distill
#SBATCH --output=spid_distill.out
#SBATCH --error=spid_distill.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G


echo "Starting SPID distillation"

python distill.py \
  --teacher-model-path "output_changed_mealnorm/models/best/best_model.zip" \
  --save-folder-path "distil_single" \
  --train-patients "adult#008" \
  --scenario-mode "semi_random_hb" \
  --time-std-multiplier 0.5 \
  --use-custom-reward \
  --max-insulin-action 6 \
  --n-iter 30 \
  --total-timesteps 60000 \
  --n-eval-episodes 25

echo "Done"