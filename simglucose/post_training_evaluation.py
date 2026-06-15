from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from joblib import load

from train import make_env_fn
from evaluate import evaluate_insulin_policy


ENV_CHOICES = ("env_fully_closed", "env_hybrid_closed")

CONFIG_KEYS = [
    "max_episode_steps",
    "time_std_multiplier",
    "amount_noise_std_fraction",
    "actual_time_noise_std_min",
    "actual_time_noise_clip_min",
]

def add_spid_path(spid_path: str | None) -> None:
    """
    Makes PySRWrapper importable when loading joblib policies.
    """
    if spid_path is None:
        candidate = Path(__file__).resolve().parents[1] / "code" / "SPID_code"
    else:
        candidate = Path(spid_path).resolve()

    if candidate.exists():
        sys.path.insert(0, str(candidate))
        print(f"Using SPID path: {candidate}")
    else:
        print(f"WARNING: SPID path does not exist: {candidate}")


def load_env_module(env_name: str):
    """
    Load environment module from envs/.
    """
    if env_name not in ENV_CHOICES:
        raise ValueError(
            f"Unknown environment {env_name!r}. Expected one of: {list(ENV_CHOICES)}"
        )

    module = importlib.import_module(f"envs.{env_name}")

    required_attrs = ("REWARD_FNS", "make_simglucose_spid_env")
    missing = [name for name in required_attrs if not hasattr(module, name)]

    if missing:
        raise AttributeError(
            f"envs.{env_name} is missing required attribute(s): {missing}"
        )

    return module


def load_train_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "train_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config file: {config_path}")

    with config_path.open("r") as f:
        return json.load(f)


def load_eval_config(model_dir: Path) -> dict[str, Any]:
    """
     Loads relevant evaluation values from train_config.json:

        max_episode_steps
        time_std_multiplier
        amount_noise_std_fraction
        actual_time_noise_std_min
        actual_time_noise_clip_min
    """
    config_path = model_dir / "train_config.json"
    config = load_train_config(model_dir)

    missing = [key for key in CONFIG_KEYS if key not in config]
    if missing:
        raise KeyError(f"Missing required config keys in {config_path}: {missing}")

    return {
        "max_episode_steps": int(config["max_episode_steps"]),
        "time_std_multiplier": float(config["time_std_multiplier"]),
        "amount_noise_std_fraction": float(config["amount_noise_std_fraction"]),
        "actual_time_noise_std_min": float(config["actual_time_noise_std_min"]),
        "actual_time_noise_clip_min": float(config["actual_time_noise_clip_min"]),
    }


def infer_env_name(model_dir: Path, cli_env: str | None) -> str:
    """
    Prefer --env if passed.
    Otherwise read env_name from train_config.json.
    Otherwise default to env_fully_closed.
    """
    if cli_env is not None:
        return cli_env

    config = load_train_config(model_dir)
    return str(config.get("env_name", "env_fully_closed"))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--patient", type=str, required=True)  
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="PPO", choices=["PPO", "PySR"])
    parser.add_argument("--reward-type", type=str, required=True)
    parser.add_argument("--config-path", type=str, default=None)

    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=ENV_CHOICES,
        help=(
            "Environment module to use from envs/. "
            "If omitted, env_name is read from train_config.json."
        ),
    )

    parser.add_argument("--scenario-mode", type=str, default="fixed_hb")
    parser.add_argument("--n-eval-episodes", type=int, default=100)

    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--shield-bg-threshold", type=float, default=10.0)
    parser.add_argument("--max-insulin-action", type=float, default=5.0)

    args = parser.parse_args()

    patient = args.patient
    safe_patient = patient.replace("#", "-")
    add_spid_path(None)

    model_dir = Path(args.model_dir)
    if args.model_type == "PPO":
        model_path = model_dir / "models" / "best" / "best_model.zip"
    elif args.model_type == "PySR":
        model_path = model_dir / "best_student_policy.joblib"
    else:
        raise Exception(f"Not valid mode type {args.model_type}. Must be PPO or PySR.")

    save_path = model_dir / "logs" / "eval_stress_test"

    if not model_path.exists():
        raise FileNotFoundError(f"Could not find model: {model_path}")
    
    if args.model_type == "PPO":
        eval_config = load_eval_config(model_dir)
        max_episode_steps = eval_config["max_episode_steps"]
        time_std_multiplier = eval_config["time_std_multiplier"]
        amount_noise_std_fraction = eval_config["amount_noise_std_fraction"]
        actual_time_noise_std_min = eval_config["actual_time_noise_std_min"]
        actual_time_noise_clip_min = eval_config["actual_time_noise_clip_min"]


    elif args.model_type == "PySR":
        max_episode_steps = 480*3
        time_std_multiplier = 2.5
        amount_noise_std_fraction=0.4 
        actual_time_noise_std_min = 5.0
        actual_time_noise_clip_min = 15.0

    # Overriding values
    max_episode_steps = 480*3
    time_std_multiplier = 2.5
    amount_noise_std_fraction=0.4 


    env_name = infer_env_name(model_dir=model_dir, cli_env=args.env)
    env_module = load_env_module(env_name)
    env_factory = env_module.make_simglucose_spid_env
    reward_fns = env_module.REWARD_FNS

    if args.reward_type not in reward_fns:
        raise ValueError(
            f"Unknown --reward-type={args.reward_type!r} for env={env_name!r}. "
            f"Expected one of: {list(reward_fns)}"
        )

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
            env_factory=env_factory,
            env_id=f"simglucose-spid-eval-{safe_patient}-v0",
            patient=patient,
            seed=123,
            meals=meals,
            max_episode_steps=480*3, #max_episode_steps,
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
    print(f"Environment: {env_name}")
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

    if args.model_type == "PPO":
        model = PPO.load(str(model_path), env=eval_env)
    elif args.model_type == "PySR":
        model = load(str(model_path))
    else: 
        raise Exception("Unknow model type")

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