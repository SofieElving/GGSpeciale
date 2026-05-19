from __future__ import annotations

"""
- env_w_history.py: SimGlucose-wrapper + miljøfabrik
- plotting.py: progress-plots + checkpoint-metrics
- train.py: CLI, konfiguration og PPO-træning

Eksempel på kørsel:
    python train.py \
        --patient adult#010 \
        --reward-type smooth \
        --scenario-mode semi_random_hb \
        --time-std-multiplier 0.5 \
        --amount-noise-std-fraction 0.10 \
        --actual-time-noise-std-min 5 \
        --actual-time-noise-clip-min 15 \
        --include-snacks \
        --max-insulin-action 5.0 \
        --shield-bg-threshold 90 \
        --bb-warmup \
        --timesteps 3000000 \
        --seed 42 \
        --outdir ./output/smooth/adult-010
"""

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from evaluate import EvalInsulinPolicy

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from env3 import REWARD_FNS, make_simglucose_spid_env
from meal_scenarios import DEFAULT_MEALS, parse_meal_schedule
from plotting import SimglucoseProgressPlotCallback


@dataclass
class TrainConfig:
    patient: str
    reward_type: str
    meals: list[tuple[int, float]]
    scenario_mode: str

    time_std_multiplier: float
    include_snacks: bool
    amount_noise_std_fraction: float
    actual_time_noise_std_min: float
    actual_time_noise_clip_min: float

    timesteps: int
    seed: int | None
    max_episode_steps: int
    outdir: str

    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    net_arch: list[int]

    warning_window_min: float
    insulin_tau_min: float
    sample_time_min: float
    max_insulin_action: float
    shield_bg_threshold: float
    use_bb_warmup: bool


def make_env_fn(
    env_id: str,
    patient: str,
    meals: list[tuple[int, float]],
    max_episode_steps: int,
    seed: int | None,
    scenario_mode: str,
    time_std_multiplier: float,
    include_snacks: bool,
    amount_noise_std_fraction: float,
    actual_time_noise_std_min: float,
    actual_time_noise_clip_min: float,
    reward_type: str,
    warning_window_min: float,
    insulin_tau_min: float,
    sample_time_min: float,
    max_insulin_action: float,
    shield_bg_threshold: float,
    use_bb_warmup: bool,
):
    def _init():
        env = make_simglucose_spid_env(
            patient_name=patient,
            meal_schedule=meals,
            env_id=env_id,
            max_episode_steps=max_episode_steps,
            normalize=True,
            scenario_mode=scenario_mode,
            seed=None,#seed,
            warning_window_min=warning_window_min,
            insulin_tau_min=insulin_tau_min,
            sample_time_min=sample_time_min,
            time_std_multiplier=time_std_multiplier,
            include_snacks=include_snacks,
            amount_noise_std_fraction=amount_noise_std_fraction,
            actual_time_noise_std_min=actual_time_noise_std_min,
            actual_time_noise_clip_min=actual_time_noise_clip_min,
            reward_type=reward_type,
            max_insulin_action=max_insulin_action,
            shield_bg_threshold=shield_bg_threshold,
            use_bb_warmup=use_bb_warmup,
        )

        return env

    return _init


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--patient", type=str, default="adult#010")

    parser.add_argument(
        "--reward-type",
        type=str,
        default="default",
        choices=list(REWARD_FNS.keys()),
    )

    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=480)
    parser.add_argument("--outdir", type=str, default="./output")
    parser.add_argument("--meals", type=str, default="7:45,12:70,16:15,18:80,23:10")

    parser.add_argument(
        "--scenario-mode",
        type=str,
        default="fixed",
        choices=["fixed", "fixed_hb", "semi_random_hb"],
    )

    parser.add_argument(
        "--time-std-multiplier",
        type=float,
        default=1.0,
        help=(
            "Controls semi-random announced meal-time variability. "
            "0.5 gives less variable meal times than 1.0."
        ),
    )
    parser.add_argument("--include-snacks", action="store_true")

    parser.add_argument(
        "--amount-noise-std-fraction",
        type=float,
        default=0.15,
        help=(
            "Relative std for actual meal-size noise in semi_random_hb. "
            "Example: 0.15 = 15 percent std. Use 0 for deterministic sizes."
        ),
    )
    parser.add_argument(
        "--actual-time-noise-std-min",
        type=float,
        default=0.0,
        help=(
            "Std in minutes for mismatch between announced meal time and actual "
            "delivered meal time in semi_random_hb. Use 0 for no mismatch."
        ),
    )
    parser.add_argument(
        "--actual-time-noise-clip-min",
        type=float,
        default=30.0,
        help=(
            "Maximum absolute mismatch in minutes between announced and actual "
            "meal time in semi_random_hb."
        ),
    )

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=240)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--net-arch", type=str, default="128,128")

    parser.add_argument("--warning-window-min", type=float, default=20.0)
    parser.add_argument("--insulin-tau-min", type=float, default=55.0)
    parser.add_argument("--sample-time-min", type=float, default=3.0)
    parser.add_argument("--max-insulin-action", type=float, default=5.0)

    parser.add_argument(
        "--shield-bg-threshold",
        type=float,
        default=90.0,
        help="Suspend insulin when current observed CGM is below this threshold.",
    )

    parser.add_argument(
        "--bb-warmup",
        action="store_true",
        help="Run hardcoded BBController warm-up before PPO starts.",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.max_insulin_action <= 0:
        raise ValueError("--max-insulin-action skal være > 0.")

    if args.shield_bg_threshold <= 0:
        raise ValueError("--shield-bg-threshold skal være > 0.")

    if args.warning_window_min <= 0:
        raise ValueError("--warning-window-min skal være > 0.")

    if args.insulin_tau_min <= 0:
        raise ValueError("--insulin-tau-min skal være > 0.")

    if args.sample_time_min <= 0:
        raise ValueError("--sample-time-min skal være > 0.")

    if args.max_episode_steps <= 0:
        raise ValueError("--max-episode-steps skal være > 0.")

    if args.timesteps <= 0:
        raise ValueError("--timesteps skal være > 0.")

    if args.n_steps <= 0:
        raise ValueError("--n-steps skal være > 0.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size skal være > 0.")

    if args.amount_noise_std_fraction < 0:
        raise ValueError("--amount-noise-std-fraction skal være >= 0.")

    if args.actual_time_noise_std_min < 0:
        raise ValueError("--actual-time-noise-std-min skal være >= 0.")

    if args.actual_time_noise_clip_min < 0:
        raise ValueError("--actual-time-noise-clip-min skal være >= 0.")

    if args.max_insulin_action > 30:
        warnings.warn(
            "max_insulin_action er større end SimGlucose' klassiske action-område. "
            "Dobbelttjek at action-skaleringen er tilsigtet.",
            stacklevel=2,
        )

    if args.scenario_mode != "semi_random_hb":
        if args.actual_time_noise_std_min > 0 or args.amount_noise_std_fraction != 0.15:
            warnings.warn(
                "Meal noise arguments only affect scenario_mode='semi_random_hb'.",
                stacklevel=2,
            )


def main() -> None:
    args = build_argparser().parse_args()
    validate_args(args)

    meals = parse_meal_schedule(args.meals, DEFAULT_MEALS)

    net_arch = [int(part) for part in args.net_arch.split(",") if part.strip()]
    if not net_arch:
        raise ValueError("--net-arch skal indeholde mindst ét lag, fx '128,128'.")

    config = TrainConfig(
        patient=args.patient,
        reward_type=args.reward_type,
        meals=meals,
        scenario_mode=args.scenario_mode,
        time_std_multiplier=args.time_std_multiplier,
        include_snacks=args.include_snacks,
        amount_noise_std_fraction=args.amount_noise_std_fraction,
        actual_time_noise_std_min=args.actual_time_noise_std_min,
        actual_time_noise_clip_min=args.actual_time_noise_clip_min,
        timesteps=args.timesteps,
        seed=None,#args.seed,
        max_episode_steps=args.max_episode_steps,
        outdir=args.outdir,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        net_arch=net_arch,
        warning_window_min=args.warning_window_min,
        insulin_tau_min=args.insulin_tau_min,
        sample_time_min=args.sample_time_min,
        max_insulin_action=args.max_insulin_action,
        shield_bg_threshold=args.shield_bg_threshold,
        use_bb_warmup=args.bb_warmup,
    )

    outdir = Path(config.outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(parents=True, exist_ok=True)
    (outdir / "eval").mkdir(parents=True, exist_ok=True)
    (outdir / "progress").mkdir(parents=True, exist_ok=True)

    config_path = outdir / "train_config.json"
    tmp_config_path = config_path.with_name(config_path.name + ".tmp")
    with open(tmp_config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    tmp_config_path.replace(config_path)

    train_seed = None#config.seed
    eval_seed = None# if config.seed is None else config.seed + 10_000
    progress_seed = None #if config.seed is None else config.seed + 20_000

    train_env = VecMonitor(
        DummyVecEnv(
            [
                make_env_fn(
                    env_id="simglucose-spid-train-v0",
                    patient=config.patient,
                    meals=config.meals,
                    max_episode_steps=config.max_episode_steps,
                    seed=None,#train_seed,
                    scenario_mode=config.scenario_mode,
                    time_std_multiplier=config.time_std_multiplier,
                    include_snacks=config.include_snacks,
                    amount_noise_std_fraction=config.amount_noise_std_fraction,
                    actual_time_noise_std_min=config.actual_time_noise_std_min,
                    actual_time_noise_clip_min=config.actual_time_noise_clip_min,
                    reward_type=config.reward_type,
                    warning_window_min=config.warning_window_min,
                    insulin_tau_min=config.insulin_tau_min,
                    sample_time_min=config.sample_time_min,
                    max_insulin_action=config.max_insulin_action,
                    shield_bg_threshold=config.shield_bg_threshold,
                    use_bb_warmup=config.use_bb_warmup,
                )
            ]
        )
    )

    eval_env = VecMonitor(
        DummyVecEnv(
            [
                make_env_fn(
                    env_id="simglucose-spid-eval-v0",
                    patient=config.patient,
                    meals=config.meals,
                    max_episode_steps=config.max_episode_steps,
                    seed=None,#eval_seed,
                    scenario_mode=config.scenario_mode,
                    time_std_multiplier=config.time_std_multiplier,
                    include_snacks=config.include_snacks,
                    amount_noise_std_fraction=config.amount_noise_std_fraction,
                    actual_time_noise_std_min=config.actual_time_noise_std_min,
                    actual_time_noise_clip_min=config.actual_time_noise_clip_min,
                    reward_type=config.reward_type,
                    warning_window_min=config.warning_window_min,
                    insulin_tau_min=config.insulin_tau_min,
                    sample_time_min=config.sample_time_min,
                    max_insulin_action=config.max_insulin_action,
                    shield_bg_threshold=config.shield_bg_threshold,
                    use_bb_warmup=config.use_bb_warmup,
                )
            ]
        )
    )

    progress_callback = SimglucoseProgressPlotCallback(
        eval_env_fn=lambda: VecMonitor(
            DummyVecEnv(
                [
                    make_env_fn(
                        env_id="simglucose-spid-progress-v0",
                        patient=config.patient,
                        meals=config.meals,
                        max_episode_steps=config.max_episode_steps,
                        seed=None,#progress_seed,
                        scenario_mode=config.scenario_mode,
                        time_std_multiplier=config.time_std_multiplier,
                        include_snacks=config.include_snacks,
                        amount_noise_std_fraction=config.amount_noise_std_fraction,
                        actual_time_noise_std_min=config.actual_time_noise_std_min,
                        actual_time_noise_clip_min=config.actual_time_noise_clip_min,
                        reward_type=config.reward_type,
                        warning_window_min=config.warning_window_min,
                        insulin_tau_min=config.insulin_tau_min,
                        sample_time_min=config.sample_time_min,
                        max_insulin_action=config.max_insulin_action,
                        shield_bg_threshold=config.shield_bg_threshold,
                        use_bb_warmup=config.use_bb_warmup,
                    )
                ]
            )
        ),
        save_dir=outdir / "progress",
        save_freq=100_000,
        max_steps=config.max_episode_steps,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(outdir / "models"),
        name_prefix="ppo_simglucose",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(outdir / "models" / "best"),
        log_path=str(outdir / "eval"),
        eval_freq=10_000,
        deterministic=False,
        render=False,
        n_eval_episodes=5,
        verbose=1,
    )
    insulin_eval_callback = EvalInsulinPolicy(
        eval_env=eval_env,
        eval_freq=100_000,
        n_eval_episodes=5,
        save_path=str(outdir / "insulin_eval"),
        save_history=True,
        generate_report=True,
        deterministic=False,
        render=False,
        verbose=1,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs={"net_arch": config.net_arch},
        tensorboard_log=str(outdir / "logs"),
        seed=None,#config.seed,
        verbose=1,
    )

    model.learn(
        total_timesteps=config.timesteps,
        callback=[
            checkpoint_callback,
            eval_callback,
            progress_callback,
            insulin_eval_callback,
        ],
        progress_bar=True,
        tb_log_name="ppo_simglucose",
    )

    final_path = outdir / "models" / "final_model"
    model.save(final_path)

    train_env.close()
    eval_env.close()

    print(f"Gemte slutmodel: {final_path}.zip")


if __name__ == "__main__":
    main()