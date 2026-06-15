import argparse
import json
from pathlib import Path

import sys
import os

print(os.getcwd())

ROOT = Path.cwd()
sys.path.append(str(ROOT))

print(os.getcwd())

from baseline_controllers.BBControllerWrapper import BBPolicy
from baseline_controllers.PIDControllerWrapper import PIDPolicy
from evaluate2 import evaluate_insulin_policy
from envs.env_open import make_simglucose_spid_env
from train import TrainConfig


parser = argparse.ArgumentParser(
    description="Evaluate a baseline controller across multiple simglucose patients."
)

parser.add_argument(
    "--config-path",
    type=str,
    required=True,
    help="Path to the train_config.json file to use for all patients.",
)

parser.add_argument(
    "--controller",
    type=str,
    choices=["bb", "pid"],
    default="bb",
    help="Controller to evaluate: 'bb' for BBPolicy or 'pid' for PIDPolicy.",
)

parser.add_argument(
    "--patients",
    type=str,
    nargs="+",
    default=[f"adult#{i:03d}" for i in range(1, 11)],
    help="Patient names to evaluate, e.g. adult#001 adult#002 ...",
)

parser.add_argument(
    "--n-eval-episodes",
    type=int,
    default=25,
    help="Number of evaluation episodes per patient.",
)

parser.add_argument(
    "--save-root",
    type=str,
    default="./logs/controller_eval",
    help="Root folder where evaluation logs are saved.",
)

parser.add_argument(
    "--scenario-mode",
    type=str,
    default="semi_random_hb",
    help="Scenario mode used during evaluation.",
)

parser.add_argument(
    "--max-insulin-action",
    type=float,
    default=10.0,
    help="Maximum insulin action used by the environment and controller.",
)

args = parser.parse_args()

config_path = Path(args.config_path)

controller_name = args.controller.lower()
controller_cls = {
    "bb": BBPolicy,
    "pid": PIDPolicy,
}[controller_name]

save_root = Path(args.save_root) / controller_name
save_root.mkdir(parents=True, exist_ok=True)

with config_path.open("r", encoding="utf-8") as f:
    config_dict = json.load(f)

config = TrainConfig(**config_dict)

print("=" * 80)
print(f"Using config: {config_path}")
print(f"Controller: {controller_cls.__name__}")
print(f"Patients: {args.patients}")
print(f"n_eval_episodes: {args.n_eval_episodes}")
print(f"Save root: {save_root}")
print("=" * 80)

results = {}

for patient in args.patients:
    safe_patient = patient.replace("#", "-")
    patient_save_path = save_root / safe_patient
    patient_save_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Evaluating patient: {patient}")
    print(f"Controller: {controller_cls.__name__}")
    print(f"Saving to: {patient_save_path}")
    print("=" * 80)

    env = make_simglucose_spid_env(
        patient_name=patient,
        meal_schedule=config.meals,
        env_id="simglucose-spid-eval-v0",
        max_episode_steps=config.max_episode_steps,
        normalize=False,
        scenario_mode=args.scenario_mode,
        seed=None,
        warning_window_min=1,
        insulin_tau_min=config.insulin_tau_min,
        sample_time_min=config.sample_time_min,
        time_std_multiplier=config.time_std_multiplier,
        include_snacks=config.include_snacks,
        amount_noise_std_fraction=config.amount_noise_std_fraction,
        actual_time_noise_std_min=config.actual_time_noise_std_min,
        actual_time_noise_clip_min=config.actual_time_noise_clip_min,
        reward_type=config.reward_type,
        max_insulin_action=args.max_insulin_action,
        shield_bg_threshold=config.shield_bg_threshold,
        use_bb_warmup=config.use_bb_warmup,
    )

    policy = controller_cls(
        env=env,
        max_insulin_action=args.max_insulin_action,
    )

    eval_result = evaluate_insulin_policy(
        policy,
        env,
        n_eval_episodes=args.n_eval_episodes,
        save_path=str(patient_save_path),
    )

    results[patient] = eval_result

    if hasattr(env, "close"):
        env.close()

summary_path = save_root / f"{controller_name}_eval_summary.json"
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "=" * 80)
print(f"Finished {controller_cls.__name__} evaluation.")
print(f"Summary saved to: {summary_path}")
print("=" * 80)