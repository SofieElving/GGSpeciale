'''
Evaluate any sbs3 compatible policy trained on simglucose environment 
'''
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from simglucose.analysis.report import report

import json
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd


def risk_index():
    pass 

class EvalInsulinPolicy(EvalCallback):
    '''
    Evaluation wrapper. Takes compatible policy and env, 
    where env must have a get_history() method.
    '''
    def __init__(
        self,
        eval_env,
        eval_freq=10_000,
        n_eval_episodes=5,
        save_path="./logs/test/",
        save_history=False,
        generate_report=True, 
        best_model_save_path="./logs/test/best_model",
        deterministic=True,
        render=False,
        verbose=1,
    ):
        super().__init__(
            eval_env=eval_env,
            best_model_save_path=None,   # disables inherited model saving
            log_path=None,               # disables inherited eval .npz saving
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=render,
            verbose=1,                   # disables inherited printing
        )

        self.eval_index = 0
        self.save_history = save_history
        self.generate_report = generate_report

        self.save_path = Path(save_path)
        self.eval_log_path = self.save_path / "eval_log"
        self.eval_log_path.mkdir(parents=True, exist_ok=True)

        self.history_path = self.eval_log_path / "cgm_history.csv"
        self.metrics_path = self.eval_log_path / "metrics_log.csv"
        self.figures_path = self.eval_log_path / "figures"

        self.figures_path.mkdir(parents=True, exist_ok=True)

        if best_model_save_path is not None:
            Path(best_model_save_path).mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                render=self.render,
                return_episode_rewards=True,
                warn=False,
            )

            self.latest_rewards = episode_rewards
            self.latest_lengths = episode_lengths

            self.after_eval()
            self.eval_env.env_method("clear_history")

        return True
    
    def after_eval(self):
        if self.latest_rewards is None:
            return
        
        if self.verbose > 0:
            print("Evaluating model...")
             
        history_list = self.eval_env.get_attr("history")[0]
        history_df = pd.concat(history_list, axis=0, keys=range(len(history_list)))
        history_df.index.names = ["episode", "step"]

        if history_df.empty:
            if self.verbose > 0:
                print("[EvalInsulinPolicy] Warning: get_history() returned empty history.")
            return
        
        history_df.insert(0, "eval_index", self.eval_index)

        if self.save_history:
            history_df.to_csv(
                self.history_path,
                mode="a",
                header=not self.history_path.exists(),
                index=True,
            )

        metrics = compute_scores(history_df)

        max_steps = self.eval_env.get_attr("spec")[0].max_episode_steps
        episode_steps = [len(episode) for episode in history_list]
        critical_failure_rate = np.mean(np.array(episode_steps) < max_steps)*100

        latest_rewards = self.latest_rewards
        rewards = {
            "critical_failure_rate" : float(critical_failure_rate),
            "num_timesteps": int(self.num_timesteps),
            "mean_reward": float(np.mean(latest_rewards)),
            "std_reward": float(np.std(latest_rewards)),
            "n_eval_episodes": int(self.n_eval_episodes)
        }

        new_row = {"eval_index": int(self.eval_index)} | metrics | rewards

        pd.DataFrame([new_row]).to_csv(
            self.metrics_path,
            mode="a",
            header=not self.metrics_path.exists(),
            index=False,
        )

        if self.generate_report:
            if self.verbose > 0:
                print("generating report...")

            report_dir = self.figures_path / f"eval_{self.eval_index:04d}"
            report_dir.mkdir(parents=True, exist_ok=True)

            report_df = history_df.copy()
            report_df = report_df.reset_index(level="step", drop=True)
            report_df = report_df.set_index("Time", append=True)
            report_df.index.names = ["episode", "Time"]

            report(report_df, save_path=report_dir)

        self.eval_index += 1
        
        
def compute_scores(df: pd.DataFrame) -> dict:
    cf_bounds = [0, 54, 250, 999]
    tir_bounds = [0, 70, 180, 999]

    df = df.dropna().copy()
    n = len(df)

    cf = ((df.BG.value_counts(bins=cf_bounds).sort_index() / n) * 100)
    tir = ((df.BG.value_counts(bins=tir_bounds).sort_index() / n) * 100)

    TBR_II, TIR_II, TAR_II = cf
    TBR_I, TIR_I, TAR_I = tir

    df["insulin2"] = df.insulin.astype(float)
    df["D"] = df["Time"].dt.date

    daily_insulin = df.groupby(["eval_index", "episode", "D"])["insulin2"].sum()

    metrics = {
        "TBR_II": float(TBR_II),
        "TBR_I": float(TBR_I),
        "TIR": float(TIR_I),
        "TAR_I": float(TAR_I),
        "TAR_II": float(TAR_II),
        "total_daily_insulin": float(daily_insulin.mean()),
        "average_insulin": float(df["insulin2"].mean()),
    }
    
    return metrics





def _get_env_attr(env, name: str, default: Any = None):
    """
    Works for both SB3 VecEnv and normal Gymnasium envs.
    """
    if hasattr(env, "get_attr"):
        values = env.get_attr(name)
        return values[0] if len(values) > 0 else default

    return getattr(env, name, default)


def _call_env_method(env, name: str):
    """
    Works for both SB3 VecEnv and normal Gymnasium envs.
    """
    if hasattr(env, "env_method"):
        return env.env_method(name)

    method = getattr(env, name, None)
    if callable(method):
        return method()

    return None


def evaluate_insulin_policy(
    model,
    eval_env,
    n_eval_episodes: int = 5,
    deterministic: bool = True,
    render: bool = False,
    save_path: Optional[str | Path] = "./logs/final_eval/",
    save_history: bool = False,
    generate_report: bool = True,
    eval_index: int = 0,
    num_timesteps: Optional[int] = None,
    verbose: int = 1,
    warn: bool = False,
    clear_history_before: bool = True,
    clear_history_after: bool = True,
) -> dict:
    """
    Evaluate a finished trained insulin policy/model.

    Assumptions
    -----------
    - `model` is SB3-compatible, i.e. has `predict(obs, deterministic=...)`.
    - `eval_env` is compatible with SB3 `evaluate_policy`.
    - The underlying environment stores episode histories in `history`.
    - The underlying environment has `clear_history()`.
    - The history DataFrames contain at least:
        - BG
        - insulin
        - Time

    Returns
    -------
    dict
        Dictionary containing metrics, rewards, episode lengths, and optionally history.
    """

    if clear_history_before:
        _call_env_method(eval_env, "clear_history")

    if verbose > 0:
        print("Evaluating insulin policy...")

    episode_rewards, episode_lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=render,
        return_episode_rewards=True,
        warn=warn,
    )

    history_list = _get_env_attr(eval_env, "history", default=None)

    if history_list is None or len(history_list) == 0:
        raise ValueError(
            "No evaluation history found. Make sure the environment records history "
            "and exposes it as `env.history`."
        )

    history_df = pd.concat(history_list, axis=0, keys=range(len(history_list)))
    history_df.index.names = ["episode", "step"]

    if history_df.empty:
        raise ValueError("Evaluation history is empty.")

    history_df.insert(0, "eval_index", eval_index)

    metrics = compute_scores(history_df)

    spec = _get_env_attr(eval_env, "spec", default=None)
    max_steps = getattr(spec, "max_episode_steps", None)

    if max_steps is not None:
        episode_steps = np.array([len(episode) for episode in history_list])
        critical_failure_rate = np.mean(episode_steps < max_steps) * 100
    else:
        critical_failure_rate = np.nan

    reward_metrics = {
        "critical_failure_rate": float(critical_failure_rate),
        "num_timesteps": int(num_timesteps) if num_timesteps is not None else None,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "n_eval_episodes": int(n_eval_episodes),
    }

    results_row = {
        "eval_index": int(eval_index),
        **metrics,
        **reward_metrics,
    }

    if save_path is not None:
        save_path = Path(save_path)
        eval_log_path = save_path / "eval_log"
        figures_path = eval_log_path / "figures"

        eval_log_path.mkdir(parents=True, exist_ok=True)
        figures_path.mkdir(parents=True, exist_ok=True)

        history_path = eval_log_path / "cgm_history.csv"
        metrics_path = eval_log_path / "metrics_log.csv"

        if save_history:
            history_df.to_csv(
                history_path,
                mode="a",
                header=not history_path.exists(),
                index=True,
            )

        pd.DataFrame([results_row]).to_csv(
            metrics_path,
            mode="a",
            header=not metrics_path.exists(),
            index=False,
        )

        if generate_report:
            if verbose > 0:
                print("Generating simglucose report...")

            report_dir = figures_path / f"eval_{eval_index:04d}"
            report_dir.mkdir(parents=True, exist_ok=True)

            report_df = history_df.copy()

            # simglucose report expects a MultiIndex where inner level is Time
            report_df = report_df.reset_index(level="step", drop=True)
            report_df = report_df.set_index("Time", append=True)
            report_df.index.names = ["episode", "Time"]

            report(report_df, save_path=report_dir)

    if clear_history_after:
        _call_env_method(eval_env, "clear_history")

    return {
        "metrics": results_row,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "history": history_df,
    }
