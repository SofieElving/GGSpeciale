from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from train3 import make_env_fn
from evaluate import evaluate_insulin_policy


CONFIG_KEYS = [
    "max_episode_steps",
    "time_std_multiplier",
    "amount_noise_std_fraction",
    "actual_time_noise_std_min",
    "actual_time_noise_clip_min",
]


def load_eval_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "train_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config file: {config_path}")

    with config_path.open("r") as f:
        config = json.load(f)

    missing = [key for key in CONFIG_KEYS if key not in config]
    if missing:
        raise KeyError(
            f"Missing required config keys in {config_path}: {missing}"
        )

    return {
        "max_episode_steps": int(config["max_episode_steps"]),
        "time_std_multiplier": float(config["time_std_multiplier"]),
        "amount_noise_std_fraction": float(config["amount_noise_std_fraction"]),
        "actual_time_noise_std_min": float(config["actual_time_noise_std_min"]),
        "actual_time_noise_clip_min": float(config["actual_time_noise_clip_min"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--patient", type=str, required=True)  # e.g. adult#010
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--reward-type", type=str, required=True)

    parser.add_argument("--scenario-mode", type=str, default="fixed_hb")
    parser.add_argument("--n-eval-episodes", type=int, default=100)

    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--shield-bg-threshold", type=float, default=10.0)
    parser.add_argument("--max-insulin-action", type=float, default=5.0)

    args = parser.parse_args()

    patient = args.patient
    safe_patient = patient.replace("#", "-")

    model_dir = Path(args.model_dir)
    model_path = model_dir / "models" / "best" / "best_model.zip"
    save_path = model_dir / "logs" / "eval"

    if not model_path.exists():
        raise FileNotFoundError(f"Could not find model: {model_path}")

    eval_config = load_eval_config(model_dir)

    max_episode_steps = eval_config["max_episode_steps"]
    time_std_multiplier = eval_config["time_std_multiplier"]
    amount_noise_std_fraction = eval_config["amount_noise_std_fraction"]
    actual_time_noise_std_min = eval_config["actual_time_noise_std_min"]
    actual_time_noise_clip_min = eval_config["actual_time_noise_clip_min"]

    save_path.mkdir(parents=True, exist_ok=True)

    meals = [
        (7 * 60, 45.0),
        (12 * 60, 70.0),
        (16 * 60, 15.0),
        (18 * 60, 80.0),
        (23 * 60, 10.0),
    ]

    eval_env = DummyVecEnv([
        make_env_fn(
            env_id=f"simglucose-spid-eval-{safe_patient}-v0",
            patient=patient,
            seed=123,
            meals=meals,
            max_episode_steps=max_episode_steps,
            scenario_mode=args.scenario_mode,

            # Loaded from train_config.json
            time_std_multiplier=time_std_multiplier,
            amount_noise_std_fraction=amount_noise_std_fraction,
            actual_time_noise_clip_min=actual_time_noise_clip_min,
            actual_time_noise_std_min=actual_time_noise_std_min,

            # Fixed/eval settings
            include_snacks=True,

            # Must match training setup where relevant
            reward_type=args.reward_type,
            warning_window_min=20,
            insulin_tau_min=55,
            sample_time_min=3,
            max_insulin_action=args.max_insulin_action,
            use_bb_warmup=True,
            shield_bg_threshold=args.shield_bg_threshold,
        )
    ])

    eval_env = VecMonitor(eval_env)

    print("=" * 80)
    print(f"Evaluating patient: {patient}")
    print(f"Model: {model_path}")
    print(f"Config: {model_dir / 'train_config.json'}")
    print(f"Save path: {save_path}")
    print(f"Reward type: {args.reward_type}")
    print(f"Scenario mode: {args.scenario_mode}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Shield threshold: {args.shield_bg_threshold}")
    print(f"Max insulin action: {args.max_insulin_action}")
    print(f"Max episode steps: {max_episode_steps}")
    print(f"Time std multiplier: {time_std_multiplier}")
    print(f"Amount noise std fraction: {amount_noise_std_fraction}")
    print(f"Actual time noise std min: {actual_time_noise_std_min}")
    print(f"Actual time noise clip min: {actual_time_noise_clip_min}")
    print("=" * 80)

    model = PPO.load(str(model_path), env=eval_env)

    evaluate_insulin_policy(
        model,
        eval_env,
        save_path=str(save_path),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=args.deterministic,
        save_history=True,
        generate_report=True,
        verbose=1,
        clear_history_before=True,
        clear_history_after=True,
        max_steps=max_episode_steps,
    )

    eval_env.close()


if __name__ == "__main__":
    main()