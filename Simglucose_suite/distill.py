from __future__ import annotations
import json
import argparse
import importlib
import os
import sys
import warnings
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# IMPORTANT for PySR / JuliaCall / gmDAGGER
# Must be set before gmDAGGER / juliacall is imported.
# ---------------------------------------------------------------------

os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
os.environ.setdefault("PYTHONFAULTHANDLER", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

os.environ.setdefault("JULIA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from juliacall import Main as jl  # noqa: F401


# ---------------------------------------------------------------------
# Path to SPID / gmDAGGER code
# ---------------------------------------------------------------------

SPID_PATH = Path(__file__).resolve().parents[1] / "code" / "SPID_code"
sys.path.insert(0, str(SPID_PATH))

from gmDAGGER_safe import train_spid

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from evaluate2 import evaluate_insulin_policy
from meal_scenarios import DEFAULT_MEALS, parse_meal_schedule


ENV_CHOICES = (
    "env_closed",
    "env_open",
    "env_closed_action_history",
)


NO_NESTING = {
    "square": {"square": 0},
}


PYSR_CONFIGS = {
    "linear_10": {
        "binary_operators": ["+", "*", "-"],
        "unary_operators": [],
        "maxsize": 10,
        "maxdepth": 3,
    },

    "square_10": {
        "binary_operators": ["+", "*", "-"],
        "unary_operators": ["square"],
        "maxsize": 10,
        "maxdepth": 3,
        "nested_constraints": NO_NESTING,
    },

    "square_threshold_10": {
        "binary_operators": ["+", "*", "-", "<", ">"],
        "unary_operators": ["square"],
        "maxsize": 10,
        "maxdepth": 3,
        "nested_constraints": NO_NESTING,
    },

    "square_threshold_15": {
        "binary_operators": ["+", "*", "-", "<", ">"],
        "unary_operators": ["square"],
        "maxsize": 15,
        "maxdepth": 3,
        "nested_constraints": NO_NESTING,
    },

    "square_threshold_20": {
        "binary_operators": ["+", "*", "-", "<", ">"],
        "unary_operators": ["square"],
        "maxsize": 20,
        "maxdepth": 3,
        "nested_constraints": NO_NESTING,
    },

}

def parse_patient_list(text: str) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def safe_patient_name(patient: str) -> str:
    return patient.replace("#", "-")


def parse_pysr_config_list(text: str) -> list[str]:
    configs = [c.strip() for c in text.split(",") if c.strip()]
    unknown = [c for c in configs if c not in PYSR_CONFIGS]

    if unknown:
        raise ValueError(
            f"Unknown PySR config(s): {unknown}. "
            f"Available: {list(PYSR_CONFIGS.keys())}"
        )

    return configs


def load_env_module(env_name: str):
    """
    Load environment module.

    First tries:
        envs.<env_name>

    Then falls back to:
        <env_name>

    This works whether your files are in envs/ or directly in the project root.
    """
    if env_name not in ENV_CHOICES:
        raise ValueError(
            f"Unknown --env={env_name!r}. Expected one of: {list(ENV_CHOICES)}"
        )

    module_candidates = [
        f"envs.{env_name}",
        env_name,
    ]

    errors = []

    for module_path in module_candidates:
        try:
            module = importlib.import_module(module_path)
            break

        except ModuleNotFoundError as exc:
            # Only suppress if the requested module itself is missing.
            # If a dependency inside the module is missing, re-raise.
            top_level = module_path.split(".")[0]
            if exc.name not in {module_path, top_level}:
                raise

            errors.append(f"{module_path}: {exc}")

    else:
        raise ModuleNotFoundError(
            f"Could not import environment {env_name!r}. Tried: {module_candidates}. "
            f"Errors: {errors}"
        )

    required_attrs = ("REWARD_FNS", "make_simglucose_spid_env")
    missing = [attr for attr in required_attrs if not hasattr(module, attr)]

    if missing:
        raise AttributeError(
            f"Environment module {module.__name__!r} is missing required "
            f"attribute(s): {missing}"
        )

    return module


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        type=str,
        default="env_closed",
        choices=ENV_CHOICES,
        help="Environment module to use.",
    )

    parser.add_argument(
        "--pysr-configs",
        type=str,
        default="medium_default_10",
        help=(
            "Comma-separated PySR configs to test. "
            f"Available: {list(PYSR_CONFIGS.keys())}"
        ),
    )

    parser.add_argument(
        "--teacher-root",
        type=str,
        required=True,
        help="Root folder containing patient-specific teacher models.",
    )

    parser.add_argument(
        "--save-root",
        type=str,
        default="./distil_results",
        help="Root folder where distilled models/results are saved.",
    )

    parser.add_argument(
        "--patients",
        type=str,
        default=(
            "adult#001,adult#002,adult#003,adult#004,adult#005,"
            "adult#006,adult#007,adult#008,adult#009,adult#010"
        ),
    )

    parser.add_argument(
        "--reward-type",
        type=str,
        required=True,
        help="Reward type. Valid options depend on the selected --env.",
    )

    parser.add_argument(
        "--teacher-model-name",
        type=str,
        default="final_model.zip",
        choices=["final_model.zip", "best_model.zip"],
    )

    parser.add_argument("--meals", type=str, default="7:45,12:70,16:15,18:80,23:10")

    parser.add_argument(
        "--scenario-mode",
        type=str,
        default="semi_random_hb",
        choices=["fixed", "fixed_hb", "semi_random_hb"],
    )

    parser.add_argument(
        "--use-shared-teacher",
        action="store_true",
        help=(
            "Use one shared mixed-patient PPO teacher for all patient-specific "
            "distillations."
        ),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-seed", type=int, default=None)

    parser.add_argument("--max-episode-steps", type=int, default=480)
    parser.add_argument("--warning-window-min", type=float, default=20.0)
    parser.add_argument("--insulin-tau-min", type=float, default=55.0)
    parser.add_argument("--sample-time-min", type=float, default=3.0)

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Use raw observations instead of normalized observations.",
    )

    parser.add_argument("--time-std-multiplier", type=float, default=0.5)
    parser.add_argument("--include-snacks", action="store_true")
    parser.add_argument("--amount-noise-std-fraction", type=float, default=0.15)
    parser.add_argument("--actual-time-noise-std-min", type=float, default=0.0)
    parser.add_argument("--actual-time-noise-clip-min", type=float, default=30.0)

    parser.add_argument("--max-insulin-action", type=float, default=5.0)
    parser.add_argument("--shield-bg-threshold", type=float, default=10.0)

    parser.add_argument("--bb-warmup", action="store_true")

    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--total-timesteps", type=int, default=12_000)
    parser.add_argument("--n-eval-episodes", type=int, default=100)
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument("--skip-initial-steps", type=int, default=10)
    parser.add_argument("--sample-episodes", type=int, default=5)
    parser.add_argument("--keep-terminal-transitions", action="store_true")
    parser.add_argument("--keep-early-terminal-episodes", action="store_true")
    parser.add_argument("--max-sampling-episodes", type=int, default=200)

    parser.add_argument("--no-distilled-eval", action="store_true")
    parser.add_argument("--distilled-eval-episodes", type=int, default=None)

    # Intentionally no deterministic eval flag.
    # Distilled evaluation is always stochastic / deterministic=False.
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--save-history", action="store_true")

    parser.add_argument("--skip-missing", action="store_true")

    return parser


def find_teacher_model(
    teacher_root: Path,
    reward_type: str,
    patient: str,
    teacher_model_name: str,
) -> Path:
    safe_patient = safe_patient_name(patient)

    if teacher_model_name == "best_model.zip":
        return (
            teacher_root
            / reward_type
            / safe_patient
            / "models"
            / "best"
            / "best_model.zip"
        )

    return (
        teacher_root
        / reward_type
        / safe_patient
        / "models"
        / "final_model.zip"
    )


def find_shared_teacher_model(
    teacher_root: Path,
    reward_type: str,
) -> Path:
    return teacher_root / reward_type / "adult-all" / "models" / "best" / "best_model.zip"


def _format_space(space) -> str:
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    shape = getattr(space, "shape", None)

    if low is not None and high is not None:
        return (
            f"{space.__class__.__name__}("
            f"shape={shape}, "
            f"low={np.asarray(low).tolist()}, "
            f"high={np.asarray(high).tolist()}, "
            f"dtype={getattr(space, 'dtype', None)}"
            ")"
        )

    return repr(space)


def _assert_box_spaces_match(teacher_space, env_space, name: str) -> None:
    teacher_shape = getattr(teacher_space, "shape", None)
    env_shape = getattr(env_space, "shape", None)

    if teacher_shape != env_shape:
        raise ValueError(
            f"{name} shape mismatch: "
            f"teacher={_format_space(teacher_space)} vs "
            f"env={_format_space(env_space)}"
        )

    teacher_low = getattr(teacher_space, "low", None)
    teacher_high = getattr(teacher_space, "high", None)
    env_low = getattr(env_space, "low", None)
    env_high = getattr(env_space, "high", None)

    if (
        teacher_low is not None
        and teacher_high is not None
        and env_low is not None
        and env_high is not None
    ):
        if not np.allclose(teacher_low, env_low) or not np.allclose(teacher_high, env_high):
            raise ValueError(
                f"{name} bounds mismatch: "
                f"teacher={_format_space(teacher_space)} vs "
                f"env={_format_space(env_space)}. "
                "Use the same environment, state space, normalization, and action scaling."
            )


def check_teacher_matches_env(teacher_model_path: Path, env) -> None:
    teacher = PPO.load(str(teacher_model_path), device="cpu")

    print("Teacher observation space:", _format_space(teacher.observation_space))
    print("Distill env observation space:", _format_space(env.observation_space))
    print("Teacher action space:", _format_space(teacher.action_space))
    print("Distill env action space:", _format_space(env.action_space))

    _assert_box_spaces_match(
        teacher_space=teacher.observation_space,
        env_space=env.observation_space,
        name="Observation space",
    )

    _assert_box_spaces_match(
        teacher_space=teacher.action_space,
        env_space=env.action_space,
        name="Action space",
    )

    obs, _info = env.reset()
    action, _state = teacher.predict(obs, deterministic=False)
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)

    if not np.all(np.isfinite(action_arr)):
        raise ValueError(f"Teacher produced non-finite action during smoke test: {action_arr}")

    print("Teacher/env smoke test OK. First teacher action:", action_arr.tolist())


def make_eval_env_for_patient(
    *,
    env_factory,
    env_name: str,
    patient: str,
    safe_patient: str,
    meals: list[tuple[int, float]],
    args: argparse.Namespace,
):
    env_id_prefix = env_name.replace("_", "-")
    eval_env_id = f"simglucose-spid-{env_id_prefix}-distilled-eval-{safe_patient}-v0"

    eval_env = DummyVecEnv([
        lambda: env_factory(
            patient_name=patient,
            meal_schedule=meals,
            env_id=eval_env_id,
            max_episode_steps=args.max_episode_steps,
            normalize=not args.no_normalize,
            scenario_mode=args.scenario_mode,
            seed=args.scenario_seed,
            warning_window_min=args.warning_window_min,
            insulin_tau_min=args.insulin_tau_min,
            sample_time_min=args.sample_time_min,
            time_std_multiplier=args.time_std_multiplier,
            include_snacks=args.include_snacks,
            amount_noise_std_fraction=args.amount_noise_std_fraction,
            actual_time_noise_std_min=args.actual_time_noise_std_min,
            actual_time_noise_clip_min=args.actual_time_noise_clip_min,
            reward_type=args.reward_type,
            max_insulin_action=args.max_insulin_action,
            shield_bg_threshold=args.shield_bg_threshold,
            use_bb_warmup=args.bb_warmup,
        )
    ])

    return VecMonitor(eval_env)


def run_distillation_for_patient(
    *,
    patient: str,
    teacher_model_path: Path,
    save_root: Path,
    env_factory,
    env_name: str,
    args: argparse.Namespace,
) -> None:
    safe_patient = safe_patient_name(patient)
    meals = parse_meal_schedule(args.meals, DEFAULT_MEALS)

    env_counter = 0
    env_id_prefix = env_name.replace("_", "-")

    def environment():
        nonlocal env_counter
        env_counter += 1

        env_id = f"simglucose-spid-{env_id_prefix}-distill-{safe_patient}-{env_counter}-v0"

        return env_factory(
            patient_name=patient,
            meal_schedule=meals,
            env_id=env_id,
            max_episode_steps=args.max_episode_steps,
            normalize=not args.no_normalize,
            scenario_mode=args.scenario_mode,
            seed=args.scenario_seed,
            warning_window_min=args.warning_window_min,
            insulin_tau_min=args.insulin_tau_min,
            sample_time_min=args.sample_time_min,
            time_std_multiplier=args.time_std_multiplier,
            include_snacks=args.include_snacks,
            amount_noise_std_fraction=args.amount_noise_std_fraction,
            actual_time_noise_std_min=args.actual_time_noise_std_min,
            actual_time_noise_clip_min=args.actual_time_noise_clip_min,
            reward_type=args.reward_type,
            max_insulin_action=args.max_insulin_action,
            shield_bg_threshold=args.shield_bg_threshold,
            use_bb_warmup=args.bb_warmup,
        )

    save_folder_path = save_root / args.reward_type / safe_patient

    print("\n" + "=" * 80)
    print(f"Distilling patient: {patient}")
    print(f"Environment: {env_name}")
    print(f"Teacher: {teacher_model_path}")
    print(f"Save folder: {save_folder_path}")
    print(f"PySR configs: {args.pysr_configs}")
    print(f"Scenario mode: {args.scenario_mode}")
    print(f"Scenario seed: {args.scenario_seed}")
    print(f"Normalize observations: {not args.no_normalize}")
    print(f"Reward type: {args.reward_type}")
    print(f"Max insulin action / I_max: {args.max_insulin_action}")
    print(f"Shield BG threshold: {args.shield_bg_threshold}")
    print(f"Use BB warmup: {args.bb_warmup}")
    print(f"Max episode steps: {args.max_episode_steps}")
    print(f"Skip first SPID samples after reset: {args.skip_initial_steps}")
    print(f"Accepted rollout samples per gmDAGGER iteration: {args.sample_episodes}")
    print(f"Drop terminal transition from training: {not args.keep_terminal_transitions}")
    print(f"Discard early-terminal episodes: {not args.keep_early_terminal_episodes}")
    print("=" * 80 + "\n")

    preflight_env = environment()
    try:
        check_teacher_matches_env(teacher_model_path, preflight_env)
    finally:
        preflight_env.close()
    
    selected_config_names = parse_pysr_config_list(args.pysr_configs)

    if len(selected_config_names) != 1:
        raise ValueError(
            "This script expects exactly one PySR config per run. "
            f"Got: {selected_config_names}"
        )

    selected_config_name = selected_config_names[0]
    selected_pysr_config = PYSR_CONFIGS[selected_config_name]

    print("=" * 80)
    print(f"Using PySR config: {selected_config_name}")
    print(json.dumps(selected_pysr_config, indent=2))
    print("=" * 80)

    rewards, best_policy, wrapper, run_dir = train_spid(
        teacher_path=str(teacher_model_path),
        teacher_model=PPO,
        save_folder_path=str(save_folder_path),
        save_results=True,
        environment=environment,
        n_iter=args.n_iter,
        total_timesteps=args.total_timesteps,
        n_eval_episodes=args.n_eval_episodes,
        verbose=args.verbose,
        skip_initial_steps=args.skip_initial_steps,
        sample_episodes=args.sample_episodes,
        drop_terminal_transitions=not args.keep_terminal_transitions,
        discard_early_terminal_episodes=not args.keep_early_terminal_episodes,
        max_episode_steps=args.max_episode_steps,
        max_sampling_episodes=args.max_sampling_episodes,
        pysr_config=selected_pysr_config,
        pysr_config_name=selected_config_name,
        max_insulin_action=args.max_insulin_action,
    )

    print(f"\nFinished distillation for {patient}.")
    print(f"Saved to: {run_dir}")

    if args.no_distilled_eval:
        print("Skipping distilled-policy evaluation because --no-distilled-eval was used.")
        return

    eval_episodes = (
        args.distilled_eval_episodes
        if args.distilled_eval_episodes is not None
        else args.n_eval_episodes
    )

    eval_deterministic = False

    print("\nEvaluating distilled policy...")
    print(f"Eval episodes: {eval_episodes}")
    print(f"Eval deterministic: {eval_deterministic}")

    eval_env = make_eval_env_for_patient(
        env_factory=env_factory,
        env_name=env_name,
        patient=patient,
        safe_patient=safe_patient,
        meals=meals,
        args=args,
    )

    eval_save_path = Path(run_dir) / "distilled_eval"

    try:
        eval_results = evaluate_insulin_policy(
            model=best_policy,
            eval_env=eval_env,
            n_eval_episodes=eval_episodes,
            deterministic=eval_deterministic,
            save_path=eval_save_path,
            save_history=args.save_history,
            generate_report=not args.no_report,
            verbose=1,
            clear_history_before=True,
            clear_history_after=True,
        )

        print("\nDistilled evaluation metrics:")
        print(eval_results["metrics"])
        print(f"Distilled evaluation saved to: {eval_save_path}")

    finally:
        eval_env.close()


def validate_args(args: argparse.Namespace, reward_fns: dict[str, object]) -> None:
    parse_pysr_config_list(args.pysr_configs)

    if args.reward_type not in reward_fns:
        raise ValueError(
            f"Unknown --reward-type={args.reward_type!r} for --env={args.env!r}. "
            f"Expected one of: {list(reward_fns.keys())}"
        )

    if args.skip_initial_steps < 0:
        raise ValueError("--skip-initial-steps must be >= 0.")

    if args.sample_episodes is not None and args.sample_episodes < 0:
        raise ValueError("--sample-episodes must be >= 0.")

    if args.max_sampling_episodes <= 0:
        raise ValueError("--max-sampling-episodes must be > 0.")

    if args.max_insulin_action <= 0:
        raise ValueError("--max-insulin-action must be > 0.")

    if args.shield_bg_threshold <= 0:
        raise ValueError("--shield-bg-threshold must be > 0.")

    if args.amount_noise_std_fraction < 0:
        raise ValueError("--amount-noise-std-fraction must be >= 0.")

    if args.actual_time_noise_std_min < 0:
        raise ValueError("--actual-time-noise-std-min must be >= 0.")

    if args.actual_time_noise_clip_min < 0:
        raise ValueError("--actual-time-noise-clip-min must be >= 0.")

    if args.max_episode_steps <= 0:
        raise ValueError("--max-episode-steps must be > 0.")

    if args.n_iter <= 0:
        raise ValueError("--n-iter must be > 0.")

    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be > 0.")


def main() -> None:
    warnings.filterwarnings("ignore")

    args = build_argparser().parse_args()

    env_module = load_env_module(args.env)
    reward_fns = env_module.REWARD_FNS
    env_factory = env_module.make_simglucose_spid_env

    validate_args(args, reward_fns)

    teacher_root = Path(args.teacher_root)
    save_root = Path(args.save_root)
    patients = parse_patient_list(args.patients)

    if len(patients) == 0:
        raise ValueError("No patients provided.")

    if not teacher_root.exists():
        raise FileNotFoundError(f"Teacher root does not exist: {teacher_root}")

    save_root.mkdir(parents=True, exist_ok=True)

    print("\n=== DISTILL PATIENTS CONFIG ===")
    print(f"environment: {args.env}")
    print(f"teacher_root: {teacher_root}")
    print(f"save_root: {save_root}")
    print(f"patients: {patients}")
    print(f"pysr_configs: {args.pysr_configs}")
    print(f"reward_type: {args.reward_type}")
    print(f"teacher_model_name: {args.teacher_model_name}")
    print(f"use_shared_teacher: {args.use_shared_teacher}")

    if args.use_shared_teacher:
        print(f"shared_teacher_path: {find_shared_teacher_model(teacher_root, args.reward_type)}")

    print(f"scenario_mode: {args.scenario_mode}")
    print(f"scenario_seed: {args.scenario_seed}")
    print(f"normalize observations: {not args.no_normalize}")
    print(f"max_insulin_action / I_max: {args.max_insulin_action}")
    print(f"shield_bg_threshold: {args.shield_bg_threshold}")
    print(f"bb_warmup: {args.bb_warmup}")
    print(f"n_iter: {args.n_iter}")
    print(f"total_timesteps: {args.total_timesteps}")
    print(f"skip_initial_steps: {args.skip_initial_steps}")
    print(f"sample_episodes per gmDAGGER iteration: {args.sample_episodes}")
    print(f"drop_terminal_transitions: {not args.keep_terminal_transitions}")
    print(f"discard_early_terminal_episodes: {not args.keep_early_terminal_episodes}")
    print(f"max_sampling_episodes: {args.max_sampling_episodes}")
    print(f"n_eval_episodes during distillation: {args.n_eval_episodes}")
    print(f"distilled eval enabled: {not args.no_distilled_eval}")
    print("Distilled evaluation deterministic: False")
    print("================================\n")

    for patient in patients:
        if args.use_shared_teacher:
            teacher_model_path = find_shared_teacher_model(
                teacher_root=teacher_root,
                reward_type=args.reward_type,
            )
        else:
            teacher_model_path = find_teacher_model(
                teacher_root=teacher_root,
                reward_type=args.reward_type,
                patient=patient,
                teacher_model_name=args.teacher_model_name,
            )

        if not teacher_model_path.exists():
            msg = f"Missing teacher model for {patient}: {teacher_model_path}"

            if args.skip_missing:
                print(f"WARNING: {msg}. Skipping.")
                continue

            raise FileNotFoundError(msg)

        run_distillation_for_patient(
            patient=patient,
            teacher_model_path=teacher_model_path,
            save_root=save_root,
            env_factory=env_factory,
            env_name=args.env,
            args=args,
        )

    print("\nAll requested distillations complete.")


if __name__ == "__main__":
    main()