from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from train3 import make_env_fn
from evaluate import evaluate_insulin_policy


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--patient", type=str, required=True)  # e.g. adult#010
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--reward-type", type=str, required=True)

    parser.add_argument("--scenario-mode", type=str, default="fixed_hb")
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=480)

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
            meals=meals,
            max_episode_steps=args.max_episode_steps,
            seed=123,
            scenario_mode=args.scenario_mode,

            # Fixed/eval settings
            time_std_multiplier=0.0,
            include_snacks=False,
            amount_noise_std_fraction=0.0,
            actual_time_noise_clip_min=0.0,
            actual_time_noise_std_min=0.0,

            # Must match training setup where relevant
            reward_type=args.reward_type,
            warning_window_min=20,
            insulin_tau_min=55,
            sample_time_min=3,
            max_insulin_action=args.max_insulin_action,
            use_bb_warmup=False,
            shield_bg_threshold=args.shield_bg_threshold,
        )
    ])

    eval_env = VecMonitor(eval_env)

    print("=" * 80)
    print(f"Evaluating patient: {patient}")
    print(f"Model: {model_path}")
    print(f"Save path: {save_path}")
    print(f"Reward type: {args.reward_type}")
    print(f"Scenario mode: {args.scenario_mode}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Shield threshold: {args.shield_bg_threshold}")
    print("=" * 80)

    model = PPO.load(str(model_path), env=eval_env)

    evaluate_insulin_policy(
        model,
        eval_env,
        save_path=str(save_path),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=args.deterministic,
    )

    eval_env.close()


if __name__ == "__main__":
    main()