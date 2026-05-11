from __future__ import annotations

import argparse
import os
import warnings
import sys
from pathlib import Path

SPID_PATH = Path(__file__).resolve().parents[1] / "code" / "SPID_code"
sys.path.insert(0, str(SPID_PATH))

from gmDAGGER import train_spid

from stable_baselines3 import PPO

from simglucose_env import MultiPatientSimglucoseEnv, parse_meal_schedule


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--teacher-model-path", type=str, required=True)
    parser.add_argument("--save-folder-path", type=str, default="./distil_results")

    parser.add_argument(
        "--train-patients",
        type=str,
        default="adult#001,adult#002,adult#003,adult#004,adult#005,adult#006,adult#007",
    )

    parser.add_argument("--meals", type=str, default="7:45,12:70,16:15,18:80,23:10")
    parser.add_argument("--scenario-mode", type=str, default="fixed_hb")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episode-steps", type=int, default=480)
    parser.add_argument("--warning-window-min", type=float, default=20.0)
    parser.add_argument("--time-std-multiplier", type=float, default=0.5)
    parser.add_argument("--max-insulin-action", type=float, default=6.0)
    parser.add_argument("--use-custom-reward", action="store_true")

    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--total-timesteps", type=int, default=12000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--verbose", type=int, default=2)

    return parser


def main() -> None:
    args = build_argparser().parse_args()

    warnings.filterwarnings("ignore")

    patient_names = [
        p.strip() for p in args.train_patients.split(",") if p.strip()
    ]

    meals = parse_meal_schedule(args.meals)

    def environment():
        return MultiPatientSimglucoseEnv(
            patient_names=patient_names,
            env_id="simglucose-distill",
            max_episode_steps=args.max_episode_steps,
            normalize=True,
            meal_schedule=meals,
            scenario_mode=args.scenario_mode,
            seed=args.seed,
            warning_window_min=args.warning_window_min,
            time_std_multiplier=args.time_std_multiplier,
            use_custom_reward=args.use_custom_reward,
            max_insulin_action=args.max_insulin_action,  # <-- clipping happens here
        )

    print("\n=== DISTILL CONFIG ===")
    print(f"teacher_model_path: {args.teacher_model_path}")
    print(f"patients: {patient_names}")
    print(f"scenario_mode: {args.scenario_mode}")
    print(f"max_insulin_action (CLIP): {args.max_insulin_action}")
    print(f"timesteps: {args.total_timesteps}")
    print("=====================\n")

    rewards, best_policy, wrapper, run_dir = train_spid(
        teacher_path=args.teacher_model_path,
        teacher_model=PPO,
        save_folder_path=args.save_folder_path,
        save_results=True,
        environment=environment,
        n_iter=args.n_iter,
        total_timesteps=args.total_timesteps,
        n_eval_episodes=args.n_eval_episodes,
        verbose=args.verbose,
    )

    print("\nDistillation complete.")
    print(f"Saved to: {run_dir}")


if __name__ == "__main__":
    main()