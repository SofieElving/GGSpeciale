from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path


# ---------------------------------------------------------------------
# IMPORTANT for PySR / JuliaCall / gmDAGGER
# Must be set before gmDAGGER / juliacall is imported.
# Still preferably set these in the SLURM script too.
# ---------------------------------------------------------------------

os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
os.environ.setdefault("PYTHONFAULTHANDLER", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Conservative thread settings reduce Julia/PyTorch/BLAS conflicts on SLURM.
os.environ.setdefault("JULIA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ---------------------------------------------------------------------
# Path to SPID / gmDAGGER code
# ---------------------------------------------------------------------

SPID_PATH = Path(__file__).resolve().parents[1] / "code" / "SPID_code"
sys.path.insert(0, str(SPID_PATH))

# Keep gmDAGGER before stable_baselines3.
# stable_baselines3 imports torch; gmDAGGER/PySR may import juliacall.
from gmDAGGER import train_spid

from stable_baselines3 import PPO

from env3 import make_simglucose_spid_env
from meal_scenarios import DEFAULT_MEALS, parse_meal_schedule


def parse_patient_list(text: str) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def safe_patient_name(patient: str) -> str:
    return patient.replace("#", "-")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

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
        default="smooth",
        choices=["default", "smooth", "strict", "steps", "positive"],
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

    # Used for reproducible bookkeeping / optional deterministic scenarios.
    # For stochastic semi_random_hb distillation, leave --scenario-seed unset.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-seed", type=int, default=None)

    parser.add_argument("--max-episode-steps", type=int, default=480)
    parser.add_argument("--warning-window-min", type=float, default=20.0)
    parser.add_argument("--insulin-tau-min", type=float, default=55.0)
    parser.add_argument("--sample-time-min", type=float, default=3.0)
    parser.add_argument("--time-std-multiplier", type=float, default=0.5)
    parser.add_argument("--include-snacks", action="store_true")

    # Must match teacher training exactly.
    # If the PPO teacher was trained with I_max=5.0, use 5.0.
    # If it was trained with I_max=1.0, use 1.0.
    parser.add_argument("--max-insulin-action", type=float, default=5.0)

    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--total-timesteps", type=int, default=12_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip patients where teacher model is missing instead of raising an error.",
    )

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


def run_distillation_for_patient(
    patient: str,
    teacher_model_path: Path,
    save_root: Path,
    args: argparse.Namespace,
) -> None:
    safe_patient = safe_patient_name(patient)
    meals = parse_meal_schedule(args.meals, DEFAULT_MEALS)

    env_counter = 0

    def environment():
        nonlocal env_counter
        env_counter += 1

        # Unique env_id avoids Gymnasium registry collisions during repeated
        # gmDAGGER environment construction.
        env_id = f"simglucose-spid-distill-{safe_patient}-{env_counter}-v0"

        return make_simglucose_spid_env(
            patient_name=patient,
            meal_schedule=meals,
            env_id=env_id,
            max_episode_steps=args.max_episode_steps,
            normalize=True,
            scenario_mode=args.scenario_mode,
            seed=args.scenario_seed,
            warning_window_min=args.warning_window_min,
            insulin_tau_min=args.insulin_tau_min,
            sample_time_min=args.sample_time_min,
            time_std_multiplier=args.time_std_multiplier,
            include_snacks=args.include_snacks,
            reward_type=args.reward_type,
            max_insulin_action=args.max_insulin_action,
        )

    save_folder_path = save_root / args.reward_type / safe_patient

    print("\n" + "=" * 80)
    print(f"Distilling patient: {patient}")
    print(f"Teacher: {teacher_model_path}")
    print(f"Save folder: {save_folder_path}")
    print(f"Scenario mode: {args.scenario_mode}")
    print(f"Scenario seed: {args.scenario_seed}")
    print(f"Reward type: {args.reward_type}")
    print(f"Max insulin action / I_max: {args.max_insulin_action}")
    print(f"Max episode steps: {args.max_episode_steps}")
    print("=" * 80 + "\n")

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
    )

    print(f"\nFinished distillation for {patient}.")
    print(f"Saved to: {run_dir}")


def main() -> None:
    warnings.filterwarnings("ignore")
    args = build_argparser().parse_args()

    teacher_root = Path(args.teacher_root)
    save_root = Path(args.save_root)
    patients = parse_patient_list(args.patients)

    if len(patients) == 0:
        raise ValueError("No patients provided.")

    if not teacher_root.exists():
        raise FileNotFoundError(f"Teacher root does not exist: {teacher_root}")

    save_root.mkdir(parents=True, exist_ok=True)

    print("\n=== DISTILL ALL PATIENTS CONFIG ===")
    print(f"teacher_root: {teacher_root}")
    print(f"save_root: {save_root}")
    print(f"patients: {patients}")
    print(f"reward_type: {args.reward_type}")
    print(f"teacher_model_name: {args.teacher_model_name}")
    print(f"scenario_mode: {args.scenario_mode}")
    print(f"scenario_seed: {args.scenario_seed}")
    print(f"max_insulin_action / I_max: {args.max_insulin_action}")
    print(f"n_iter: {args.n_iter}")
    print(f"total_timesteps: {args.total_timesteps}")
    print(f"n_eval_episodes: {args.n_eval_episodes}")
    print("===================================\n")

    for patient in patients:
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
            args=args,
        )

    print("\nAll requested distillations complete.")


if __name__ == "__main__":
    main()