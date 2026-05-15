#!/bin/bash
#SBATCH --job-name=spid_distill
#SBATCH --output=spid_distill.out
#SBATCH --error=spid_distill.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=output/slurm_%x_%A_%a.out
#SBATCH --error=output/slurm_%x_%A_%a.err


echo "Starting SPID distillation"

python distill.py \
    --teacher-root "./teacher_models" \
    --save-root "./distil_results" \
    --patients "adult#001,adult#002,adult#003,adult#004,adult#005,adult#006,adult#007,adult#008,adult#009,adult#010" \
    --reward-type smooth \
    --teacher-model-name final_model.zip \
    --scenario-mode fixed_hb \
    --time-std-multiplier 0.5 \
    --include-snacks \
    --max-insulin-action 5.0 \
    --n-iter 12 \
    --total-timesteps 12000 \
    --n-eval-episodes 10
echo "Done"