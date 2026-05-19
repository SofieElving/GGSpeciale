from __future__ import annotations

"""
Plot-callback til SB3.

Denne fil antager SB3 VecEnv-API:
- reset() returnerer kun obs
- step() returnerer (obs, rewards, dones, infos)

For n_envs=1 ligger alle info-felter derfor i infos[0].
"""

import csv
import json
import os
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class SimglucoseProgressPlotCallback(BaseCallback):
    def __init__(
        self,
        eval_env_fn: Callable[[], VecEnv],
        save_dir: str | Path,
        save_freq: int = 100_000,
        max_steps: int = 480,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.save_dir = Path(save_dir)
        self.save_freq = int(save_freq)
        self.max_steps = int(max_steps)
        self._next_save_timestep = int(save_freq)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.save_dir / "checkpoint_metrics.csv"

        self.fieldnames = [
            "timesteps",
            "episode_steps",
            "mean_cgm",
            "mean_action",
            "mean_insulin",
            "mean_proposed_insulin",
            "mean_shielded_insulin",
            "total_insulin",
            "total_proposed_insulin",
            "total_shielded_insulin",
            "mean_iob",
            "pct_tir_70_180",
            "pct_above_250",
            "pct_below_70",
            "pct_shield_active",
            "shield_count",
            "total_reward",
            "mean_reward",
        ]

        if not self.metrics_path.exists():
            tmp_path = self.metrics_path.with_name(self.metrics_path.name + ".tmp")
            with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(self.metrics_path)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_save_timestep:
            self._run_rollout_and_save()
            self._next_save_timestep += self.save_freq
        return True

    def _run_rollout_and_save(self) -> None:
        env = self.eval_env_fn()
        obs = env.reset()

        times: list[float] = []
        cgms: list[float] = []
        meals: list[float] = []
        insulin_actions: list[float] = []
        proposed_insulins: list[float] = []
        shielded_insulins: list[float] = []
        shield_active: list[float] = []
        shield_cgms: list[float] = []
        iobs: list[float] = []
        policy_actions: list[float] = []
        rewards: list[float] = []

        for step in range(self.max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            info0 = info[0]
            dt = float(info0.get("sample_time", 3.0))

            delivered = float(info0.get("plot_insulin_action", float("nan")))
            proposed = float(info0.get("plot_proposed_insulin", delivered))
            shielded = float(info0.get("plot_shielded_insulin", delivered))
            shield = float(info0.get("plot_shield_active", 0.0))

            times.append(step * dt)
            cgms.append(float(info0.get("plot_cgm_raw", float("nan"))))
            meals.append(float(info0.get("plot_meal", 0.0)))
            insulin_actions.append(delivered)
            proposed_insulins.append(proposed)
            shielded_insulins.append(shielded)
            shield_active.append(shield)
            shield_cgms.append(float(info0.get("plot_shield_cgm", float("nan"))))
            iobs.append(float(info0.get("plot_iob", float("nan"))))
            policy_actions.append(float(info0.get("policy_action", float("nan"))))
            rewards.append(float(reward[0]))

            if bool(done[0]):
                break

        cgm_arr = np.asarray(cgms, dtype=np.float32)
        action_arr = np.asarray(policy_actions, dtype=np.float32)
        insulin_arr = np.asarray(insulin_actions, dtype=np.float32)
        proposed_arr = np.asarray(proposed_insulins, dtype=np.float32)
        shielded_arr = np.asarray(shielded_insulins, dtype=np.float32)
        shield_arr = np.asarray(shield_active, dtype=np.float32)
        iob_arr = np.asarray(iobs, dtype=np.float32)
        reward_arr = np.asarray(rewards, dtype=np.float32)

        valid_cgm = np.isfinite(cgm_arr)
        if valid_cgm.any():
            cgm_valid = cgm_arr[valid_cgm]
            mean_cgm = float(np.mean(cgm_valid))
            pct_tir = 100.0 * float(np.mean((cgm_valid >= 70.0) & (cgm_valid <= 180.0)))
            pct_above_250 = 100.0 * float(np.mean(cgm_valid > 250.0))
            pct_below_70 = 100.0 * float(np.mean(cgm_valid < 70.0))
        else:
            mean_cgm = float("nan")
            pct_tir = float("nan")
            pct_above_250 = float("nan")
            pct_below_70 = float("nan")

        valid_action = np.isfinite(action_arr)
        valid_insulin = np.isfinite(insulin_arr)
        valid_proposed = np.isfinite(proposed_arr)
        valid_shielded = np.isfinite(shielded_arr)
        valid_iob = np.isfinite(iob_arr)
        valid_reward = np.isfinite(reward_arr)
        valid_shield = np.isfinite(shield_arr)

        shield_count = int(np.sum(shield_arr[valid_shield] > 0.5)) if valid_shield.any() else 0
        pct_shield_active = (
            100.0 * float(np.mean(shield_arr[valid_shield] > 0.5))
            if valid_shield.any()
            else float("nan")
        )

        metrics = {
            "timesteps": int(self.num_timesteps),
            "episode_steps": int(len(cgms)),
            "mean_cgm": mean_cgm,
            "mean_action": float(np.mean(action_arr[valid_action])) if valid_action.any() else float("nan"),
            "mean_insulin": float(np.mean(insulin_arr[valid_insulin])) if valid_insulin.any() else float("nan"),
            "mean_proposed_insulin": float(np.mean(proposed_arr[valid_proposed])) if valid_proposed.any() else float("nan"),
            "mean_shielded_insulin": float(np.mean(shielded_arr[valid_shielded])) if valid_shielded.any() else float("nan"),
            "total_insulin": float(np.sum(insulin_arr[valid_insulin])) if valid_insulin.any() else float("nan"),
            "total_proposed_insulin": float(np.sum(proposed_arr[valid_proposed])) if valid_proposed.any() else float("nan"),
            "total_shielded_insulin": float(np.sum(shielded_arr[valid_shielded])) if valid_shielded.any() else float("nan"),
            "mean_iob": float(np.mean(iob_arr[valid_iob])) if valid_iob.any() else float("nan"),
            "pct_tir_70_180": pct_tir,
            "pct_above_250": pct_above_250,
            "pct_below_70": pct_below_70,
            "pct_shield_active": pct_shield_active,
            "shield_count": shield_count,
            "total_reward": float(np.sum(reward_arr[valid_reward])) if valid_reward.any() else float("nan"),
            "mean_reward": float(np.mean(reward_arr[valid_reward])) if valid_reward.any() else float("nan"),
        }

        with open(self.metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(metrics)
            f.flush()
            os.fsync(f.fileno())

        json_path = self.save_dir / f"metrics_{self.num_timesteps:08d}.json"
        tmp_json_path = json_path.with_name(json_path.name + ".tmp")
        with open(tmp_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp_json_path.replace(json_path)

        # 4 panels: CGM, meal, insulin/proposed/shield, IOB.
        fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

        axes[0].plot(times, cgms, linewidth=2, label="CGM")

        axes[0].axhspan(0, 54, color="red", alpha=0.20)
        axes[0].axhspan(54, 70, color="orange", alpha=0.20)
        axes[0].axhspan(70, 180, color="green", alpha=0.15)
        axes[0].axhspan(180, 250, color="orange", alpha=0.20)
        axes[0].axhspan(250, 600, color="red", alpha=0.20)

        axes[0].axhline(70, color="black", linestyle="--", linewidth=1)
        axes[0].axhline(180, color="black", linestyle="--", linewidth=1)
        axes[0].axhline(75, color="blue", linestyle=":", linewidth=1, label="Shield threshold")

        # Mark shield activations on CGM panel.
        shield_times = [
            t for t, active in zip(times, shield_active)
            if active > 0.5
        ]
        shield_cgm_values = [
            c for c, active in zip(cgms, shield_active)
            if active > 0.5
        ]

        if len(shield_times) > 0:
            axes[0].scatter(
                shield_times,
                shield_cgm_values,
                label="Shield",
                marker="x",
                s=40,
            )

        axes[0].set_ylabel("CGM (mg/dL)")
        axes[0].set_ylim(40, 400)
        axes[0].set_title(
            #f"Simglucose progress ved {self.num_timesteps:,} steps | "
            f"TIR={metrics['pct_tir_70_180']:.1f}% | "
            f">250={metrics['pct_above_250']:.1f}% | "
            f"<70={metrics['pct_below_70']:.1f}% | "
            f"Shield={metrics['pct_shield_active']:.1f}% | "
            f"R={metrics['total_reward']:.1f}"
        )
        axes[0].legend(loc="upper right")

        axes[1].plot(times, meals, label="Måltid")
        axes[1].set_ylabel("Måltid")
        axes[1].legend(loc="upper right")

        axes[2].plot(times, proposed_insulins, label="Foreslået insulin", linestyle="--")
        axes[2].plot(times, shielded_insulins, label="Leveret insulin")

        shield_insulin_values = [
            y for y, active in zip(shielded_insulins, shield_active)
            if active > 0.5
        ]

        if len(shield_times) > 0:
            axes[2].scatter(
                shield_times,
                shield_insulin_values,
                label="Shield aktiv",
                marker="x",
                s=40,
            )

        axes[2].set_ylabel("Insulin")
        axes[2].legend(loc="upper right")

        axes[3].plot(times, iobs, label="IOB")
        axes[3].set_ylabel("IOB")
        axes[3].set_xlabel("Tid (min)")
        axes[3].legend(loc="upper right")

        png_path = self.save_dir / f"progress_{self.num_timesteps:08d}.png"
        tmp_png_path = png_path.with_name(png_path.name + ".tmp")
        fig.tight_layout()
        fig.savefig(tmp_png_path, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        tmp_png_path.replace(png_path)

        env.close()

        if self.verbose > 0:
            print(f"Gemte progress-plot: {png_path}")
            print(f"Opdaterede metrics-fil: {self.metrics_path}")
            print(json.dumps(metrics, indent=2))