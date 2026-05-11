from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Keep this if your symbolic student / PySR objects require Julia.
try:
    import juliacall  # noqa: F401
except ImportError:
    pass

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from simglucose_env import make_simglucose_spid_env


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_env_fn(
    env_id: str,
    patient: str,
    meals,
    max_episode_steps: int,
    seed: int,
    scenario_mode: str,
    use_custom_reward: bool,
    warning_window_min: float,
    max_insulin_action: float,
):
    def _init():
        env = make_simglucose_spid_env(
            patient_name=patient,
            meal_schedule=meals,
            env_id=env_id,
            max_episode_steps=max_episode_steps,
            normalize=True,
            scenario_mode=scenario_mode,
            seed=seed,
            warning_window_min=warning_window_min,
            use_custom_reward=use_custom_reward,
            max_insulin_action=max_insulin_action,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return _init


def create_env(
    patient: str,
    meals,
    max_episode_steps: int,
    seed: int,
    env_id: str,
    scenario_mode: str,
    use_custom_reward: bool,
    warning_window_min: float,
    max_insulin_action: float,
):
    env = DummyVecEnv([
        make_env_fn(
            env_id=env_id,
            patient=patient,
            meals=meals,
            max_episode_steps=max_episode_steps,
            seed=seed,
            scenario_mode=scenario_mode,
            use_custom_reward=use_custom_reward,
            warning_window_min=warning_window_min,
            max_insulin_action=max_insulin_action,
        )
    ])
    env = VecMonitor(env)
    return env


def ensure_vec_action(action):
    action = np.asarray(action, dtype=np.float32)

    if action.ndim == 0:
        action = action.reshape(1, 1)
    elif action.ndim == 1:
        action = action.reshape(1, -1)

    return action


def compute_metrics(cgms, rewards, insulin):
    cgms = np.asarray(cgms, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    insulin = np.asarray(insulin, dtype=float)

    valid = np.isfinite(cgms)
    cgms_valid = cgms[valid]

    if len(cgms_valid) == 0:
        return {
            "mean_reward": np.nan,
            "sum_reward": np.nan,
            "mean_cgm": np.nan,
            "tir_70_180": np.nan,
            "below_70": np.nan,
            "below_54": np.nan,
            "above_180": np.nan,
            "above_250": np.nan,
            "mean_insulin": np.nan,
            "max_insulin": np.nan,
            "total_insulin": np.nan,
        }

    return {
        "mean_reward": float(np.mean(rewards)) if len(rewards) > 0 else np.nan,
        "sum_reward": float(np.sum(rewards)) if len(rewards) > 0 else np.nan,
        "mean_cgm": float(np.mean(cgms_valid)),
        "tir_70_180": float(np.mean((cgms_valid >= 70) & (cgms_valid <= 180)) * 100.0),
        "below_70": float(np.mean(cgms_valid < 70) * 100.0),
        "below_54": float(np.mean(cgms_valid < 54) * 100.0),
        "above_180": float(np.mean(cgms_valid > 180) * 100.0),
        "above_250": float(np.mean(cgms_valid > 250) * 100.0),
        "mean_insulin": float(np.nanmean(insulin)) if len(insulin) > 0 else np.nan,
        "max_insulin": float(np.nanmax(insulin)) if len(insulin) > 0 else np.nan,
        "total_insulin": float(np.nansum(insulin)) if len(insulin) > 0 else np.nan,
    }


def rollout_policy(policy, env, max_steps: int):
    obs = env.reset()

    times = []
    cgms = []
    meals = []
    insulin = []
    raw_actions = []
    rewards = []
    meal_warning = []
    meal_size = []
    iob = []

    for step in range(max_steps):
        action, _ = policy.predict(obs, deterministic=True)
        action = ensure_vec_action(action)

        obs, reward, done, info = env.step(action)
        info0 = info[0]

        dt = float(info0.get("sample_time", 3.0))
        times.append(step * dt)

        cgms.append(float(info0.get("plot_cgm_raw", np.nan)))
        meals.append(float(info0.get("plot_meal", 0.0)))
        insulin.append(float(info0.get("plot_insulin_action", np.nan)))
        raw_actions.append(float(info0.get("raw_policy_action", np.nan)))
        meal_warning.append(float(info0.get("plot_meal_warning", 0.0)))
        meal_size.append(float(info0.get("plot_meal_size", 0.0)))
        iob.append(float(info0.get("plot_iob", 0.0)))
        rewards.append(float(np.asarray(reward).reshape(-1)[0]))

        if done[0]:
            break

    return {
        "times": np.asarray(times),
        "cgms": np.asarray(cgms),
        "meals": np.asarray(meals),
        "insulin": np.asarray(insulin),
        "raw_actions": np.asarray(raw_actions),
        "meal_warning": np.asarray(meal_warning),
        "meal_size": np.asarray(meal_size),
        "iob": np.asarray(iob),
        "rewards": np.asarray(rewards),
    }


def plot_patient_comparison(
    patient: str,
    teacher_rollout: dict,
    student_rollout: dict,
    outpath: str | Path,
):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # CGM
    axes[0].plot(
        teacher_rollout["times"],
        teacher_rollout["cgms"],
        label="Teacher",
        linewidth=2,
    )
    axes[0].plot(
        student_rollout["times"],
        student_rollout["cgms"],
        label="Student",
        linewidth=2,
    )

    axes[0].axhspan(0, 54, color="red", alpha=0.20)
    axes[0].axhspan(54, 70, color="orange", alpha=0.20)
    axes[0].axhspan(70, 180, color="green", alpha=0.15)
    axes[0].axhspan(180, 250, color="orange", alpha=0.20)
    axes[0].axhspan(250, 600, color="red", alpha=0.20)

    axes[0].axhline(70, color="black", linestyle="--", linewidth=1)
    axes[0].axhline(180, color="black", linestyle="--", linewidth=1)

    axes[0].set_ylabel("CGM")
    axes[0].set_ylim(40, 400)
    axes[0].set_title(f"Patient {patient}: Teacher vs Symbolic Student")
    axes[0].legend(loc="upper right")

    # Meal
    axes[1].plot(
        teacher_rollout["times"],
        teacher_rollout["meals"],
        label="Meal",
        linewidth=2,
    )
    axes[1].set_ylabel("Meal")
    axes[1].legend(loc="upper right")

    # Insulin
    axes[2].plot(
        teacher_rollout["times"],
        teacher_rollout["insulin"],
        label="Teacher insulin",
        linewidth=2,
    )
    axes[2].plot(
        student_rollout["times"],
        student_rollout["insulin"],
        label="Student insulin",
        linewidth=2,
    )
    axes[2].set_ylabel("Insulin")
    axes[2].legend(loc="upper right")

    # IOB
    axes[3].plot(
        teacher_rollout["times"],
        teacher_rollout["iob"],
        label="Teacher IOB",
        linewidth=2,
    )
    axes[3].plot(
        student_rollout["times"],
        student_rollout["iob"],
        label="Student IOB",
        linewidth=2,
    )
    axes[3].set_ylabel("IOB")
    axes[3].set_xlabel("Time (min)")
    axes[3].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-dir", type=str, default="./output")
    parser.add_argument("--distill-dir", type=str, default="./distill_output")
    parser.add_argument("--outdir", type=str, default="./multi_patient_eval")

    parser.add_argument(
        "--patients",
        type=str,
        required=True,
        help='Comma-separated, e.g. "adult#001,adult#005,adult#010"',
    )

    parser.add_argument("--teacher-model-path", type=str, default=None)
    parser.add_argument("--student-path", type=str, default=None)

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=None)

    parser.add_argument("--scenario-mode", type=str, default=None)
    parser.add_argument("--use-custom-reward", action="store_true")
    parser.add_argument("--no-custom-reward", action="store_true")
    parser.add_argument("--warning-window-min", type=float, default=None)
    parser.add_argument("--max-insulin-action", type=float, default=None)

    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    train_cfg = load_json(Path(args.train_dir) / "train_config.json")

    teacher_model_path = (
        Path(args.teacher_model_path)
        if args.teacher_model_path is not None
        else Path(args.train_dir) / "models" / "best" / "best_model.zip"
    )

    if not teacher_model_path.exists():
        teacher_model_path = Path(args.train_dir) / "models" / "final_model.zip"

    student_path = (
        Path(args.student_path)
        if args.student_path is not None
        else Path(args.distill_dir) / "best_student_policy.joblib"
    )

    meals = train_cfg["meals"]
    max_steps = int(args.max_steps or train_cfg["max_episode_steps"])
    patients = [p.strip() for p in args.patients.split(",") if p.strip()]

    scenario_mode = args.scenario_mode or train_cfg.get("scenario_mode", "fixed")
    warning_window_min = float(
        args.warning_window_min
        if args.warning_window_min is not None
        else train_cfg.get("warning_window_min", 60.0)
    )
    max_insulin_action = float(
        args.max_insulin_action
        if args.max_insulin_action is not None
        else train_cfg.get("max_insulin_action", 6.0)
    )

    if args.no_custom_reward:
        use_custom_reward = False
    elif args.use_custom_reward:
        use_custom_reward = True
    else:
        use_custom_reward = bool(train_cfg.get("use_custom_reward", True))

    print("Evaluation config:")
    print(f"  teacher_model_path: {teacher_model_path}")
    print(f"  student_path:       {student_path}")
    print(f"  scenario_mode:      {scenario_mode}")
    print(f"  use_custom_reward:  {use_custom_reward}")
    print(f"  warning_window_min: {warning_window_min}")
    print(f"  max_insulin_action: {max_insulin_action}")
    print(f"  patients:           {patients}")

    student = joblib.load(student_path)
    rows = []

    for i, patient in enumerate(patients):
        print(f"\nEvaluating patient: {patient}")

        seed_i = args.seed + 1000 * i

        teacher_env = create_env(
            patient=patient,
            meals=meals,
            max_episode_steps=max_steps,
            seed=seed_i,
            env_id=f"teacher-eval-{patient.replace('#', '-')}",
            scenario_mode=scenario_mode,
            use_custom_reward=use_custom_reward,
            warning_window_min=warning_window_min,
            max_insulin_action=max_insulin_action,
        )

        student_env = create_env(
            patient=patient,
            meals=meals,
            max_episode_steps=max_steps,
            seed=seed_i,
            env_id=f"student-eval-{patient.replace('#', '-')}",
            scenario_mode=scenario_mode,
            use_custom_reward=use_custom_reward,
            warning_window_min=warning_window_min,
            max_insulin_action=max_insulin_action,
        )

        teacher = PPO.load(str(teacher_model_path), env=teacher_env)

        teacher_rollout = rollout_policy(
            teacher,
            teacher_env,
            max_steps=max_steps,
        )

        student_rollout = rollout_policy(
            student,
            student_env,
            max_steps=max_steps,
        )

        teacher_metrics = compute_metrics(
            teacher_rollout["cgms"],
            teacher_rollout["rewards"],
            teacher_rollout["insulin"],
        )
        student_metrics = compute_metrics(
            student_rollout["cgms"],
            student_rollout["rewards"],
            student_rollout["insulin"],
        )

        row = {"patient": patient}

        for k, v in teacher_metrics.items():
            row[f"teacher_{k}"] = v

        for k, v in student_metrics.items():
            row[f"student_{k}"] = v

        rows.append(row)

        plot_patient_comparison(
            patient=patient,
            teacher_rollout=teacher_rollout,
            student_rollout=student_rollout,
            outpath=Path(outdir) / f"{patient.replace('#', '_')}_comparison.png",
        )

        teacher_env.close()
        student_env.close()

    df = pd.DataFrame(rows)
    summary_path = Path(outdir) / "multi_patient_summary.csv"
    df.to_csv(summary_path, index=False)



if __name__ == "__main__":
    main()