from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from simglucose.simglucose_env3 import make_simglucose_spid_env, parse_meal_schedule, DEFAULT_MEALS


@dataclass
class TrainConfig:
    patient: str
    meals: list[tuple[int, float]]
    scenario_mode: str
    time_std_multiplier: float
    include_snacks: bool
    timesteps: int
    seed: int
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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", type=str, default="adult#010")
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episode-steps", type=int, default=480)
    parser.add_argument("--outdir", type=str, default="./output")
    parser.add_argument("--meals", type=str, default="7:45,12:70,16:15,18:80,23:10")

    parser.add_argument(
        "--scenario-mode",
        type=str,
        default="fixed",
        choices=["fixed", "fixed_hb", "semi_random_hb"],
    )
    parser.add_argument("--time-std-multiplier", type=float, default=1.0)
    parser.add_argument("--include-snacks", action="store_true")

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
    return parser


def make_env_fn(
    env_id: str,
    patient: str,
    meals: list[tuple[int, float]],
    max_episode_steps: int,
    seed: int,
    scenario_mode: str,
    time_std_multiplier: float,
    include_snacks: bool,
):
    def _init():
        env = make_simglucose_spid_env(
            patient_name=patient,
            meal_schedule=meals,
            scenario_mode=scenario_mode,
            time_std_multiplier=time_std_multiplier,
            include_snacks=include_snacks,
            env_id=env_id,
            max_episode_steps=max_episode_steps,
            normalize=True,
            seed=seed,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return _init


class SimglucoseProgressPlotCallback(BaseCallback):
    def __init__(
        self,
        eval_env_fn,
        save_dir: str,
        save_freq: int = 100_000,
        max_steps: int = 480,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.save_dir = Path(save_dir)
        self.save_freq = int(save_freq)
        self.max_steps = int(max_steps)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self._run_rollout_and_save()
        return True

    def _run_rollout_and_save(self) -> None:
        env = self.eval_env_fn()
        obs = env.reset()

        times: list[float] = []
        cgms: list[float] = []
        insulin_actions: list[float] = []
        meals: list[float] = []

        for step in range(self.max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            info0 = info[0]
            cgm = float(info0.get("plot_cgm_raw", float("nan")))
            meal = float(info0.get("plot_meal", 0.0))
            insulin = float(info0.get("plot_insulin_action", float("nan")))

            dt = float(info0.get("sample_time", 3.0))
            times.append(step * dt)
            cgms.append(cgm)
            insulin_actions.append(insulin)
            meals.append(meal)

            if done[0]:
                break

        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

        axes[0].plot(times, cgms, label="CGM", linewidth=2)
        axes[0].axhspan(0, 54, color="red", alpha=0.20)
        axes[0].axhspan(54, 70, color="orange", alpha=0.20)
        axes[0].axhspan(70, 180, color="green", alpha=0.15)
        axes[0].axhspan(180, 250, color="orange", alpha=0.20)
        axes[0].axhspan(250, 600, color="red", alpha=0.20)
        axes[0].axhline(70, color="black", linestyle="--", linewidth=1)
        axes[0].axhline(180, color="black", linestyle="--", linewidth=1)
        axes[0].set_ylabel("CGM (mg/dL)")
        axes[0].set_ylim(40, 400)
        axes[0].set_title(f"Simglucose progress at {self.num_timesteps:,} steps")
        axes[0].legend(loc="upper right")

        axes[1].plot(times, meals, label="Meal signal")
        axes[1].set_ylabel("Meal")
        axes[1].legend(loc="upper right")

        axes[2].plot(times, insulin_actions, label="Insulin action")
        axes[2].set_ylabel("Insulin")
        axes[2].set_xlabel("Time (min)")
        axes[2].legend(loc="upper right")

        png_path = self.save_dir / f"progress_{self.num_timesteps:08d}.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        env.close()

        if self.verbose > 0:
            print(f"Saved progress plot to: {png_path}")


def make_progress_env_fn(
    patient: str,
    meals: list[tuple[int, float]],
    max_episode_steps: int,
    seed: int,
    scenario_mode: str,
    time_std_multiplier: float,
    include_snacks: bool,
):
    def _init():
        env = DummyVecEnv([
            make_env_fn(
                env_id="simglucose-spid-progress-v0",
                patient=patient,
                meals=meals,
                max_episode_steps=max_episode_steps,
                seed=seed,
                scenario_mode=scenario_mode,
                time_std_multiplier=time_std_multiplier,
                include_snacks=include_snacks,
            )
        ])
        env = VecMonitor(env)
        return env

    return _init


def main() -> None:
    args = build_argparser().parse_args()

    meals = parse_meal_schedule(args.meals, DEFAULT_MEALS)
    net_arch = [int(x) for x in args.net_arch.split(",") if x.strip()]

    config = TrainConfig(
        patient=args.patient,
        meals=meals,
        scenario_mode=args.scenario_mode,
        time_std_multiplier=args.time_std_multiplier,
        include_snacks=args.include_snacks,
        timesteps=args.timesteps,
        seed=args.seed,
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
    )

    outdir = Path(config.outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(parents=True, exist_ok=True)
    (outdir / "eval").mkdir(parents=True, exist_ok=True)
    (outdir / "progress").mkdir(parents=True, exist_ok=True)

    with open(outdir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    train_env = DummyVecEnv([
        make_env_fn(
            env_id="simglucose-spid-train-v0",
            patient=config.patient,
            meals=config.meals,
            max_episode_steps=config.max_episode_steps,
            seed=config.seed,
            scenario_mode=config.scenario_mode,
            time_std_multiplier=config.time_std_multiplier,
            include_snacks=config.include_snacks,
        )
    ])
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv([
        make_env_fn(
            env_id="simglucose-spid-eval-v0",
            patient=config.patient,
            meals=config.meals,
            max_episode_steps=config.max_episode_steps,
            seed=config.seed + 10_000,
            scenario_mode=config.scenario_mode,
            time_std_multiplier=config.time_std_multiplier,
            include_snacks=config.include_snacks,
        )
    ])
    eval_env = VecMonitor(eval_env)

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
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
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1,
    )

    progress_callback = SimglucoseProgressPlotCallback(
        eval_env_fn=make_progress_env_fn(
            patient=config.patient,
            meals=config.meals,
            max_episode_steps=config.max_episode_steps,
            seed=config.seed + 20_000,
            scenario_mode=config.scenario_mode,
            time_std_multiplier=config.time_std_multiplier,
            include_snacks=config.include_snacks,
        ),
        save_dir=str(outdir / "progress"),
        save_freq=100_000,
        max_steps=config.max_episode_steps,
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
        seed=config.seed,
        verbose=1,
    )

    model.learn(
        total_timesteps=config.timesteps,
        callback=[
            checkpoint_callback,
            eval_callback,
            progress_callback,
        ],
        progress_bar=True,
        tb_log_name="ppo_simglucose",
    )

    final_path = outdir / "models" / "final_model"
    model.save(final_path)

    print(f"Saved final model to: {final_path}.zip")


if __name__ == "__main__":
    main()