"""
Evaluate any SB3-compatible policy trained on a SimGlucose environment.

Adds:
- episode-level tolerance intervals for TIR/TBR/TAR/insulin metrics
- exact binomial confidence interval for critical failure rate
- one shared tolerance_episode_n column
- no saved per-episode metrics file
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from scipy.stats import beta as beta_dist
from scipy.stats import chi2, norm

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from simglucose.analysis.report import report


# ============================================================
# Metric definitions
# ============================================================

CLINICAL_METRICS = [
    "TBR_II",
    "TBR_I",
    "TIR",
    "TAR_I",
    "TAR_II",
    "total_daily_insulin",
    "average_insulin",
]

PERCENT_METRICS = {
    "TBR_II",
    "TBR_I",
    "TIR",
    "TAR_I",
    "TAR_II",
}

NONNEGATIVE_METRICS = {
    "total_daily_insulin",
    "average_insulin",
}


def risk_index():
    pass


# ============================================================
# NaN helpers
# ============================================================

def _nan_clinical_metrics() -> dict:
    return {metric: np.nan for metric in CLINICAL_METRICS}


def _nan_tolerance_metrics(
    content: float = 0.95,
    confidence: float = 0.95,
) -> dict:
    label = f"ti{int(content * 100)}_{int(confidence * 100)}"

    out = {}

    for metric in CLINICAL_METRICS:
        out[f"{metric}_episode_mean"] = np.nan
        out[f"{metric}_episode_std"] = np.nan
        out[f"{metric}_{label}_lower"] = np.nan
        out[f"{metric}_{label}_upper"] = np.nan

    out["tolerance_episode_n"] = 0

    return out


# ============================================================
# Clinical score functions
# ============================================================

def _zone_metrics_from_bg(bg_values: pd.Series | np.ndarray) -> dict:
    """
    Computes glucose-zone metrics from BG values.

    TBR_II: BG < 54
    TBR_I:  BG < 70
    TIR:    70 <= BG <= 180
    TAR_I:  BG > 180
    TAR_II: BG > 250
    """
    bg = pd.to_numeric(pd.Series(bg_values), errors="coerce").dropna().to_numpy()

    if len(bg) == 0:
        return {
            "TBR_II": np.nan,
            "TBR_I": np.nan,
            "TIR": np.nan,
            "TAR_I": np.nan,
            "TAR_II": np.nan,
        }

    return {
        "TBR_II": float(np.mean(bg < 54.0) * 100.0),
        "TBR_I": float(np.mean(bg < 70.0) * 100.0),
        "TIR": float(np.mean((bg >= 70.0) & (bg <= 180.0)) * 100.0),
        "TAR_I": float(np.mean(bg > 180.0) * 100.0),
        "TAR_II": float(np.mean(bg > 250.0) * 100.0),
    }


def compute_scores(df: pd.DataFrame) -> dict:
    """
    Pooled clinical metrics over all completed episode histories.

    This gives the same kind of summary as your original compute_scores().
    Tolerance intervals are computed separately from episode-level values.
    """
    if df is None or df.empty:
        return _nan_clinical_metrics()

    if "BG" not in df.columns:
        raise ValueError("history dataframe must contain a 'BG' column.")

    if "insulin" not in df.columns:
        raise ValueError("history dataframe must contain an 'insulin' column.")

    work = df.copy()

    zone_metrics = _zone_metrics_from_bg(work["BG"])

    work["insulin2"] = pd.to_numeric(work["insulin"], errors="coerce")

    if "Time" in work.columns:
        work["Time"] = pd.to_datetime(work["Time"], errors="coerce")
        work = work.dropna(subset=["Time"])
        work["D"] = work["Time"].dt.date

        reset_work = work.reset_index()

        group_cols = []

        if "eval_index" in reset_work.columns:
            group_cols.append("eval_index")

        if "episode" in reset_work.columns:
            group_cols.append("episode")

        group_cols.append("D")

        daily_insulin = reset_work.groupby(group_cols)["insulin2"].sum()
        total_daily_insulin = float(daily_insulin.mean()) if len(daily_insulin) > 0 else np.nan
    else:
        total_daily_insulin = float(work["insulin2"].sum())

    average_insulin = float(work["insulin2"].mean())

    return {
        **zone_metrics,
        "total_daily_insulin": total_daily_insulin,
        "average_insulin": average_insulin,
    }


def compute_scores_one_episode(df: pd.DataFrame) -> dict:
    """
    Clinical metrics for one completed episode.
    Used internally for tolerance intervals.
    """
    if df is None or df.empty:
        return _nan_clinical_metrics()

    if "BG" not in df.columns:
        raise ValueError("episode dataframe must contain a 'BG' column.")

    if "insulin" not in df.columns:
        raise ValueError("episode dataframe must contain an 'insulin' column.")

    work = df.copy()

    zone_metrics = _zone_metrics_from_bg(work["BG"])

    work["insulin2"] = pd.to_numeric(work["insulin"], errors="coerce")

    if "Time" in work.columns:
        work["Time"] = pd.to_datetime(work["Time"], errors="coerce")
        work = work.dropna(subset=["Time"])
        work["D"] = work["Time"].dt.date

        daily_insulin = work.groupby("D")["insulin2"].sum()
        total_daily_insulin = float(daily_insulin.mean()) if len(daily_insulin) > 0 else np.nan
    else:
        total_daily_insulin = float(work["insulin2"].sum())

    average_insulin = float(work["insulin2"].mean())

    return {
        **zone_metrics,
        "total_daily_insulin": total_daily_insulin,
        "average_insulin": average_insulin,
    }


def compute_episode_scores(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes one row per completed episode.

    This dataframe is used only in memory.
    It is not saved to disk.
    """
    if history_df is None or history_df.empty:
        return pd.DataFrame(columns=["eval_index", "episode", *CLINICAL_METRICS])

    work = history_df.copy().reset_index()

    if "eval_index" not in work.columns:
        work["eval_index"] = 0

    if "episode" not in work.columns:
        raise ValueError("history_df must contain an episode index level or episode column.")

    rows = []

    for (eval_index, episode), ep_df in work.groupby(["eval_index", "episode"]):
        row = {
            "eval_index": int(eval_index),
            "episode": int(episode),
        }
        row.update(compute_scores_one_episode(ep_df))
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Tolerance intervals
# ============================================================

def normal_tolerance_interval(
    values,
    content: float = 0.95,
    confidence: float = 0.95,
) -> tuple[float, float, float, float, int]:
    """
    Approximate two-sided normal tolerance interval.

    content:
        Proportion of future episode-level outcomes the interval should cover.
        Example: 0.95 means the interval targets 95% population coverage.

    confidence:
        Confidence in the coverage.
        Example: 0.95 means 95% confidence.

    Returns:
        lower, upper, mean, sd, n_valid
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)

    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0

    if n == 1:
        return np.nan, np.nan, float(x[0]), np.nan, 1

    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))

    if not np.isfinite(sd):
        return np.nan, np.nan, mean, sd, n

    nu = n - 1
    z = norm.ppf((1.0 + content) / 2.0)
    chi2_crit = chi2.ppf(1.0 - confidence, nu)

    if chi2_crit <= 0 or not np.isfinite(chi2_crit):
        return np.nan, np.nan, mean, sd, n

    k = z * np.sqrt((nu * (1.0 + 1.0 / n)) / chi2_crit)

    lower = mean - k * sd
    upper = mean + k * sd

    return float(lower), float(upper), mean, sd, n


def compute_tolerance_intervals(
    episode_metrics: pd.DataFrame,
    content: float = 0.95,
    confidence: float = 0.95,
) -> dict:
    """
    Computes tolerance intervals across episode-level clinical metrics.

    Only summary columns are returned.
    Per-episode metrics are not saved.
    """
    if episode_metrics is None or episode_metrics.empty:
        return _nan_tolerance_metrics(content=content, confidence=confidence)

    label = f"ti{int(content * 100)}_{int(confidence * 100)}"

    out = {}

    for metric in CLINICAL_METRICS:
        lower, upper, mean, sd, _ = normal_tolerance_interval(
            episode_metrics[metric],
            content=content,
            confidence=confidence,
        )

        if metric in PERCENT_METRICS:
            lower = max(0.0, lower) if np.isfinite(lower) else np.nan
            upper = min(100.0, upper) if np.isfinite(upper) else np.nan

        if metric in NONNEGATIVE_METRICS:
            lower = max(0.0, lower) if np.isfinite(lower) else np.nan

        out[f"{metric}_episode_mean"] = mean
        out[f"{metric}_episode_std"] = sd
        out[f"{metric}_{label}_lower"] = lower
        out[f"{metric}_{label}_upper"] = upper

    out["tolerance_episode_n"] = int(
        episode_metrics[CLINICAL_METRICS].dropna(how="all").shape[0]
    )

    return out


# ============================================================
# Binomial confidence interval for critical failure
# ============================================================

def binomial_confidence_interval(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Exact Clopper-Pearson confidence interval for a binomial percentage.

    Used for critical failure rate because each episode is either:
        failed = 1
        completed = 0
    """
    if n <= 0:
        return np.nan, np.nan

    alpha = 1.0 - confidence

    if successes == 0:
        lower = 0.0
    else:
        lower = beta_dist.ppf(alpha / 2.0, successes, n - successes + 1)

    if successes == n:
        upper = 1.0
    else:
        upper = beta_dist.ppf(1.0 - alpha / 2.0, successes + 1, n - successes)

    return float(lower * 100.0), float(upper * 100.0)


def compute_critical_failure_stats(
    history_list: list[pd.DataFrame],
    max_steps: Optional[int] = 480,
    confidence: float = 0.95,
) -> dict:
    """
    Computes critical failure rate and exact binomial confidence interval.

    A critical failure is defined as:
        len(episode_history) < max_steps
    """
    n_total = len(history_list)

    ci_label = int(confidence * 100)

    if n_total == 0 or max_steps is None:
        return {
            "critical_failure_rate": np.nan,
            f"critical_failure_rate_ci{ci_label}_lower": np.nan,
            f"critical_failure_rate_ci{ci_label}_upper": np.nan,
            "critical_failure_count": 0,
            "completed_episode_count": 0,
        }

    episode_steps = np.array([len(ep) for ep in history_list], dtype=int)
    failed = episode_steps < max_steps

    n_failures = int(np.sum(failed))
    n_completed = int(n_total - n_failures)

    critical_failure_rate = float(np.mean(failed) * 100.0)

    ci_low, ci_high = binomial_confidence_interval(
        successes=n_failures,
        n=n_total,
        confidence=confidence,
    )

    return {
        "critical_failure_rate": critical_failure_rate,
        f"critical_failure_rate_ci{ci_label}_lower": ci_low,
        f"critical_failure_rate_ci{ci_label}_upper": ci_high,
        "critical_failure_count": n_failures,
        "completed_episode_count": n_completed,
    }


# ============================================================
# History helpers
# ============================================================

def histories_to_dataframe(
    history_list: list[pd.DataFrame],
    eval_index: int,
) -> pd.DataFrame:
    """
    Converts a list of episode histories into one MultiIndex dataframe.
    """
    if history_list is None or len(history_list) == 0:
        return pd.DataFrame()

    history_df = pd.concat(
        history_list,
        axis=0,
        keys=range(len(history_list)),
    )

    history_df.index.names = ["episode", "step"]

    if "eval_index" not in history_df.columns:
        history_df.insert(0, "eval_index", int(eval_index))

    return history_df


def filter_surviving_histories(
    history_list: list[pd.DataFrame],
    max_steps: Optional[int] = 480,
) -> list[pd.DataFrame]:
    """
    Keeps only completed episodes for TIR/TBR/TAR/insulin metrics.

    Failed episodes are still counted in critical_failure_rate.
    """
    if max_steps is None:
        return history_list

    return [hist for hist in history_list if len(hist) >= max_steps]


# ============================================================
# Report helper
# ============================================================

def save_simglucose_report(
    history_df: pd.DataFrame,
    report_dir: Path,
):
    """
    Generates and saves the SimGlucose report with colored BG regions.
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    report_df = history_df.copy()
    report_df = report_df.reset_index(level="step", drop=True)
    report_df = report_df.set_index("Time", append=True)
    report_df.index.names = ["episode", "Time"]

    results, ri_per_hour, zone_stats, figs, axes = report(
        report_df,
        save_path=report_dir,
    )

    fig_ensemble = figs[0]
    ax1, ax2, ax3 = axes[:3]

    for ax in [ax1, ax2]:
        old_ylim = ax.get_ylim()

        ax.axhspan(0, 54, color="red", alpha=0.20, zorder=0, label="_nolegend_")
        ax.axhspan(54, 70, color="orange", alpha=0.20, zorder=0, label="_nolegend_")
        ax.axhspan(70, 180, color="green", alpha=0.15, zorder=0, label="_nolegend_")
        ax.axhspan(180, 250, color="orange", alpha=0.20, zorder=0, label="_nolegend_")
        ax.axhspan(250, 600, color="red", alpha=0.20, zorder=0, label="_nolegend_")

        ax.set_ylim(old_ylim)

        for line in ax.lines:
            line.set_zorder(3)

    fig_ensemble.set_size_inches(8, 6)

    for ax in [ax1, ax2]:
        legend = ax.get_legend()

        if legend is not None:
            legend.set_bbox_to_anchor((1.02, 1.0))
            legend._loc = 2

            for text in legend.get_texts():
                text.set_fontsize(7)

    fig_ensemble.subplots_adjust(
        top=0.9,
        right=0.78,
        hspace=0.20,
    )

    if ax1.get_title():
        ax1.set_title(ax1.get_title(), pad=12)

    fig_ensemble.savefig(
        report_dir / "BG_trace.png",
        dpi=300,
        bbox_inches="tight",
    )

    return results, ri_per_hour, zone_stats, figs, axes


# ============================================================
# Environment helper functions
# ============================================================

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


# ============================================================
# Callback evaluator
# ============================================================

class EvalInsulinPolicy(EvalCallback):
    """
    Evaluation callback for SB3-compatible insulin policies.

    Requirements:
    - eval_env must expose `history`
    - eval_env must expose `clear_history()`
    - each history dataframe must contain BG, insulin, and Time
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        save_path: str | Path = "./100_eval_best",
        save_history: bool = False,
        generate_report: bool = True,
        best_model_save_path: str | Path | None = "./logs/test/best_model",
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        max_steps: int = 480,
        tolerance_content: float = 0.95,
        tolerance_confidence: float = 0.95,
    ):
        super().__init__(
            eval_env=eval_env,
            best_model_save_path=None,
            log_path=None,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
        )

        self.eval_index = 0
        self.save_history = save_history
        self.generate_report = generate_report
        self.max_steps = max_steps
        self.tolerance_content = tolerance_content
        self.tolerance_confidence = tolerance_confidence

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

        failure_stats = compute_critical_failure_stats(
            history_list=history_list,
            max_steps=self.max_steps,
            confidence=self.tolerance_confidence,
        )

        history_list_survivors = filter_surviving_histories(
            history_list=history_list,
            max_steps=self.max_steps,
        )

        n_survivors = len(history_list_survivors)

        history_df = histories_to_dataframe(
            history_list=history_list_survivors,
            eval_index=self.eval_index,
        )

        if history_df.empty:
            if self.verbose > 0:
                print(
                    "[EvalInsulinPolicy] Warning: no completed episodes. "
                    "Clinical metrics and tolerance intervals set to NaN."
                )

            metrics = _nan_clinical_metrics()
            tolerance_metrics = _nan_tolerance_metrics(
                content=self.tolerance_content,
                confidence=self.tolerance_confidence,
            )
        else:
            if self.save_history:
                history_df.to_csv(
                    self.history_path,
                    mode="a",
                    header=not self.history_path.exists(),
                    index=True,
                )

            metrics = compute_scores(history_df)

            episode_metrics = compute_episode_scores(history_df)
            tolerance_metrics = compute_tolerance_intervals(
                episode_metrics,
                content=self.tolerance_content,
                confidence=self.tolerance_confidence,
            )

        rewards = {
            **failure_stats,
            "survivors": int(n_survivors),
            "num_timesteps": int(self.num_timesteps),
            "mean_reward": float(np.mean(self.latest_rewards)),
            "std_reward": float(np.std(self.latest_rewards)),
            "n_eval_episodes": int(self.n_eval_episodes),
        }

        new_row = (
            {"eval_index": int(self.eval_index)}
            | metrics
            | tolerance_metrics
            | rewards
        )

        pd.DataFrame([new_row]).to_csv(
            self.metrics_path,
            mode="a",
            header=not self.metrics_path.exists(),
            index=False,
        )

        if self.generate_report and not history_df.empty:
            if self.verbose > 0:
                print("Generating report...")

            report_dir = self.figures_path / f"eval_{self.eval_index:04d}"

            save_simglucose_report(
                history_df=history_df,
                report_dir=report_dir,
            )

        self.eval_index += 1


# ============================================================
# Final evaluation function
# ============================================================

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
    max_steps: int = 480,
    tolerance_content: float = 0.95,
    tolerance_confidence: float = 0.95,
) -> dict:
    """
    Evaluate a finished trained insulin policy/model.

    Clinical metrics and tolerance intervals are computed from completed episodes.
    Critical failure rate is computed from all episodes.

    Per-episode metrics are used internally but are not saved.
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

    failure_stats = compute_critical_failure_stats(
        history_list=history_list,
        max_steps=max_steps,
        confidence=tolerance_confidence,
    )

    history_list_survivors = filter_surviving_histories(
        history_list=history_list,
        max_steps=max_steps,
    )

    n_survivors = len(history_list_survivors)

    history_df = histories_to_dataframe(
        history_list=history_list_survivors,
        eval_index=eval_index,
    )

    if history_df.empty:
        if verbose > 0:
            print(
                "[evaluate_insulin_policy] Warning: no completed episodes. "
                "Clinical metrics and tolerance intervals set to NaN."
            )

        metrics = _nan_clinical_metrics()
        tolerance_metrics = _nan_tolerance_metrics(
            content=tolerance_content,
            confidence=tolerance_confidence,
        )
    else:
        metrics = compute_scores(history_df)

        episode_metrics = compute_episode_scores(history_df)
        tolerance_metrics = compute_tolerance_intervals(
            episode_metrics,
            content=tolerance_content,
            confidence=tolerance_confidence,
        )

    reward_metrics = {
        **failure_stats,
        "survivors": int(n_survivors),
        "num_timesteps": int(num_timesteps) if num_timesteps is not None else None,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "n_eval_episodes": int(n_eval_episodes),
    }

    results_row = {
        "eval_index": int(eval_index),
        **metrics,
        **tolerance_metrics,
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

        if save_history and not history_df.empty:
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

        if generate_report and not history_df.empty:
            if verbose > 0:
                print("Generating simglucose report...")

            report_dir = figures_path / f"eval_{eval_index:04d}"

            save_simglucose_report(
                history_df=history_df,
                report_dir=report_dir,
            )

    if clear_history_after:
        _call_env_method(eval_env, "clear_history")

    return {
        "metrics": results_row,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "history": history_df,
    }