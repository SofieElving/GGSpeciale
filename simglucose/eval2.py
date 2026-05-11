from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make SPID / PySRWrapper importable when loading joblib student.
# Assumes this script is in: .../GGSpeciale/simglucose/eval.py
SPID_PATH = Path(__file__).resolve().parents[1] / "code" / "SPID_code"
sys.path.insert(0, str(SPID_PATH))

# Keep this if your symbolic student / PySR objects require Julia.
try:
    import juliacall  # noqa: F401
except Exception:
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
    time_std_multiplier: float,
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
            time_std_multiplier=time_std_multiplier,
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
    time_std_multiplier: float,
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
            time_std_multiplier=time_std_multiplier,
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


def predict_action(policy, obs, deterministic: bool = True):
    """
    Supports both SB3 policies and joblib-loaded symbolic SPID/PySR policies.
    """
    try:
        result = policy.predict(obs, deterministic=deterministic)
    except TypeError:
        result = policy.predict(obs)
    except AttributeError:
        result = policy(obs)

    # SB3 returns (action, state). Some symbolic policies may return action only.
    if isinstance(result, tuple):
        return result[0]
    return result


def time_in_range(cgms, low: float = 70.0, high: float = 180.0) -> float:
    cgms = np.asarray(cgms, dtype=float)
    valid = np.isfinite(cgms)

    if not np.any(valid):
        return np.nan

    cgms = cgms[valid]
    return float(np.mean((cgms >= low) & (cgms <= high)) * 100.0)


def get_symbolic_equation(student) -> str:
    """
    Extract symbolic equations from a PySRPolicy/SPID student.

    Expected structure:
        student.policy_list[i].sr.get_best()["equation"]

    Returns one formatted string containing all action dimensions.
    """

    # Main expected case: PySRPolicy with one policy per action dimension
    if hasattr(student, "policy_list"):
        equations = []

        for i, policy in enumerate(student.policy_list):
            try:
                sr = policy.sr
                best = sr.get_best()
                eq = best["equation"]
                equations.append(f"Action dimension {i}: {eq}")
            except Exception as e:
                equations.append(f"Action dimension {i}: could not extract equation ({e})")

        return "\n".join(equations)

    # Single-policy fallback: student.sr.get_best()["equation"]
    if hasattr(student, "sr"):
        try:
            best = student.sr.get_best()
            eq = best["equation"]
            return f"Action dimension 0: {eq}"
        except Exception as e:
            return f"Could not extract equation from student.sr.get_best(): {e}"

    return "Symbolic equation not found."


def wrap_equation_text(text: str, width: int = 120) -> str:
    text = str(text).replace("\n", " ")
    return "\n".join(textwrap.wrap(text, width=width))


def compute_metrics(cgms, rewards, insulin, fidelity_gaps=None):
    cgms = np.asarray(cgms, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    insulin = np.asarray(insulin, dtype=float)

    valid = np.isfinite(cgms)
    cgms_valid = cgms[valid]

    if fidelity_gaps is None:
        fidelity_gaps = np.asarray([], dtype=float)
    else:
        fidelity_gaps = np.asarray(fidelity_gaps, dtype=float)

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
            "mean_fidelity_gap": np.nan,
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
        "mean_fidelity_gap": (
            float(np.nanmean(fidelity_gaps))
            if len(fidelity_gaps) > 0
            else np.nan
        ),
    }


def rollout_policy(
    policy,
    env,
    max_steps: int,
    teacher_policy=None,
    teacher_clip_value: float | None = 6.0,
):
    """
    Roll out one policy.

    If teacher_policy is provided, this also computes the teacher action at the
    exact states visited by the rolled-out policy. This gives a fidelity gap:
        || student_action - clipped_teacher_action ||^2
    """
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

    teacher_actions_same_state = []
    fidelity_gaps = []

    for step in range(max_steps):
        action = predict_action(policy, obs, deterministic=True)
        action = ensure_vec_action(action)

        if teacher_policy is not None:
            teacher_action = predict_action(teacher_policy, obs, deterministic=True)
            teacher_action = ensure_vec_action(teacher_action)

            if teacher_clip_value is not None:
                teacher_action = np.clip(teacher_action, 0.0, teacher_clip_value)

            teacher_actions_same_state.append(
                float(np.asarray(teacher_action).reshape(-1)[0])
            )
            fidelity_gaps.append(float(np.sum((action - teacher_action) ** 2)))

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
        "teacher_actions_same_state": np.asarray(teacher_actions_same_state),
        "fidelity_gaps": np.asarray(fidelity_gaps),
    }


def plot_patient_comparison(
    patient: str,
    teacher_rollout: dict,
    student_rollout: dict,
    outpath: str | Path,
    symbolic_equation: str | None = None,
):
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    teacher_tir = time_in_range(teacher_rollout["cgms"])
    student_tir = time_in_range(student_rollout["cgms"])

    fidelity_gaps = student_rollout.get("fidelity_gaps", np.asarray([]))
    mean_fidelity_gap = (
        float(np.nanmean(fidelity_gaps))
        if len(fidelity_gaps) > 0
        else np.nan
    )

    # CGM
    axes[0].plot(
        teacher_rollout["times"],
        teacher_rollout["cgms"],
        label=f"Teacher",
        linewidth=2,
    )
    axes[0].plot(
        student_rollout["times"],
        student_rollout["cgms"],
        label=f"Student",
        linewidth=2,
    )

    axes[0].axhspan(0, 54, color="red", alpha=0.20)
    axes[0].axhspan(54, 70, color="orange", alpha=0.20)
    axes[0].axhspan(70, 180, color="green", alpha=0.15, label="Time in range")
    axes[0].axhspan(180, 250, color="orange", alpha=0.20)
    axes[0].axhspan(250, 600, color="red", alpha=0.20)

    axes[0].axhline(70, color="black", linestyle="--", linewidth=1)
    axes[0].axhline(180, color="black", linestyle="--", linewidth=1)

    axes[0].set_ylabel("CGM")
    axes[0].set_ylim(40, 400)
    axes[0].set_title(f"Patient {patient}: Teacher vs Symbolic Student")
    axes[0].legend(loc="upper right")

    info_text = (
        f"Teacher TIR: {teacher_tir:.1f}%\n"
        f"Student TIR: {student_tir:.1f}%"
    )

    if np.isfinite(mean_fidelity_gap):
        info_text += f"\nMean fidelity gap: {mean_fidelity_gap:.4g}"

    axes[0].text(
        0.01,
        0.97,
        info_text,
        transform=axes[0].transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

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

    if symbolic_equation is not None:
        equation_text = wrap_equation_text(symbolic_equation, width=120)

        fig.text(
            0.01,
            0.005,
            f"Symbolic policy: {equation_text}",
            ha="left",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.90),
        )

        fig.tight_layout(rect=[0, 0.08, 1, 1])
    else:
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
    parser.add_argument("--time-std-multiplier", type=float, default=0.5)

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
        else train_cfg.get("warning_window_min", 20.0)
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
    print(f"  teacher_model_path:   {teacher_model_path}")
    print(f"  student_path:         {student_path}")
    print(f"  scenario_mode:        {scenario_mode}")
    print(f"  use_custom_reward:    {use_custom_reward}")
    print(f"  warning_window_min:   {warning_window_min}")
    print(f"  max_insulin_action:   {max_insulin_action}")
    print(f"  time_std_multiplier:  {args.time_std_multiplier}")
    print(f"  patients:             {patients}")

    teacher = PPO.load(str(teacher_model_path))
    student = joblib.load(student_path)

    symbolic_equation = get_symbolic_equation(student)
    print("\nSymbolic equation:")
    print(symbolic_equation)

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
            time_std_multiplier=args.time_std_multiplier,
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
            time_std_multiplier=args.time_std_multiplier,
        )

        teacher.set_env(teacher_env)

        teacher_rollout = rollout_policy(
            teacher,
            teacher_env,
            max_steps=max_steps,
        )

        student_rollout = rollout_policy(
            student,
            student_env,
            max_steps=max_steps,
            teacher_policy=teacher,
            teacher_clip_value=max_insulin_action,
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
            fidelity_gaps=student_rollout["fidelity_gaps"],
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
            symbolic_equation=symbolic_equation,
        )

        teacher_env.close()
        student_env.close()

    df = pd.DataFrame(rows)
    summary_path = Path(outdir) / "multi_patient_summary.csv"
    df.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(f"- {summary_path}")
    print("- one comparison PNG per patient")


if __name__ == "__main__":
    main()