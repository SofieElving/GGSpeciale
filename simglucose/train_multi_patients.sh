#!/bin/bash
#SBATCH --account=GGSpeciale
#SBATCH --job-name=simglucose_multi_train
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=output/slurm_%x_%j.out
#SBATCH --error=output/slurm_%x_%j.err

set -Eeuo pipefail
set -x

mkdir -p output

python train_sb3_simglucose.py \
    --patient "adult#010" \
    --train-patients "adult#001,adult#002,adult#003,adult#004,adult#005,adult#006,adult#007" \
    --timesteps 3000000 \
    --seed 42 \
    --max-episode-steps 480 \
    --outdir "./output_test2" \
    --meals "7:45,12:70,16:15,18:80,23:10" \
    --scenario-mode "fixed_hb" \
    --time-std-multiplier 0.5 \
    --learning-rate 3e-4 \
    --n-steps 480 \
    --batch-size 240 \
    --n-epochs 10 \
    --gamma 0.995 \
    --gae-lambda 0.95 \
    --clip-range 0.15 \
    --ent-coef 0.05 \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --net-arch "128,128"