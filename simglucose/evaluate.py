from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import binom

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
    "average_insulin"
]


# ============================================================
# NaN helpers
# ============================================================

def _nan_clinical_metrics() -> dict:
    return {metric: np.nan for metric in CLINICAL_METRICS}


def _nan_episode_summary_metrics() -> dict:
    out = {}

    for metric in CLINICAL_METRICS:
        out[f"{metric}_episode_mean"] = np.nan
        out[f"{metric}_episode_std"] = np.nan

    out["episode_metric_n"] = 0

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

    This gives the same kind of summary as the original compute_scores().
    Episode-level standard deviations are computed separately.
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
    Used internally for episode-level summary statistics.
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
# Plot tolerance interval and episode summary helpers
# ============================================================

def _finite_mean_std(values) -> tuple[float, float, int]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)

    if n == 0:
        return np.nan, np.nan, 0

    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else np.nan

    return mean, sd, n


def compute_episode_summary_metrics(episode_metrics: pd.DataFrame) -> dict:
    """
    Computes episode-level mean and standard deviation for clinical metrics.

    This replaces non-plot tolerance interval summaries. It preserves the
    episode-level mean/std outputs while removing unused tolerance bounds.
    """
    if episode_metrics is None or episode_metrics.empty:
        return _nan_episode_summary_metrics()

    out = {}

    for metric in CLINICAL_METRICS:
        mean, sd, _ = _finite_mean_std(episode_metrics[metric])
        out[f"{metric}_episode_mean"] = mean
        out[f"{metric}_episode_std"] = sd

    out["episode_metric_n"] = int(
        episode_metrics[CLINICAL_METRICS].dropna(how="all").shape[0]
    )

    return out


def nonparametric_tolerance_interval(
    values,
    content: float = 0.90,
    confidence: float = 0.95,
) -> tuple[float, float, float, float, int]:
    """
    Two-sided distribution-free non-parametric tolerance interval.

    Used only for BG/CGM tolerance bands in the report plots.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)

    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0

    if n == 1:
        return np.nan, np.nan, float(x[0]), np.nan, 1

    if not (0.0 < content < 1.0):
        raise ValueError("content must be between 0 and 1")

    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1")

    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))

    x_sorted = np.sort(x)

    best_lower_idx = None
    best_upper_idx = None

    max_symmetric_trim = (n - 2) // 2

    for trim in range(max_symmetric_trim + 1):
        lower_idx = trim
        upper_idx = n - trim - 1
        m = upper_idx - lower_idx

        achieved_confidence = binom.cdf(m - 1, n, content)

        if achieved_confidence >= confidence:
            best_lower_idx = lower_idx
            best_upper_idx = upper_idx
        else:
            break

    if best_lower_idx is None:
        return np.nan, np.nan, mean, sd, n

    lower = float(x_sorted[best_lower_idx])
    upper = float(x_sorted[best_upper_idx])

    return lower, upper, mean, sd, n


# ============================================================
# Critical failure statistics
# ============================================================

def compute_critical_failure_stats(
    history_list: list[pd.DataFrame],
    max_steps: Optional[int] = 480,
    confidence: float = 0.95,
    n_eval_episodes: Optional[int] = None,
) -> dict:
    """
    Computes critical failure rate and episode counts.

    A critical failure is defined as:
        len(episode_history) < max_steps

    The `confidence` argument is kept for backward compatibility with older
    evaluation calls, but no confidence interval is computed.
    """
    _ = confidence

    if n_eval_episodes is not None:
        history_list = history_list[:n_eval_episodes]

    n_total = len(history_list)

    if n_total == 0 or max_steps is None:
        return {
            "critical_failure_rate": np.nan,
            "critical_failure_count": 0,
            "completed_episode_count": 0,
        }

    episode_steps = np.array([len(ep) for ep in history_list], dtype=int)
    failed = episode_steps < max_steps

    n_failures = int(np.sum(failed))
    n_completed = int(n_total - n_failures)

    critical_failure_rate = float(np.mean(failed) * 100.0)

    return {
        "critical_failure_rate": critical_failure_rate,
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

def ensemble_BG(
    BG,
    ax=None,
    plot_tolerance=True,
    content=0.90,
    confidence=0.95,
    max_individual_lines=100,
    random_state=42,
):
    """
    Plot ensemble blood glucose curves with non-parametric tolerance bands.

    Parameters
    ----------
    BG:
        DataFrame where rows are time points and columns are individual curves.

    ax:
        Optional matplotlib axis.

    plot_tolerance:
        If True, plot non-parametric tolerance bands at each timestamp.

    content:
        Target population coverage of the tolerance interval.
        Example: 0.95 means the interval targets 95% population coverage.

    confidence:
        Confidence that the interval covers at least `content` of the population.
        Example: 0.95 means 95% confidence.

    max_individual_lines:
        Maximum number of individual background curves to plot.

    random_state:
        Random seed used when subsampling individual curves.

    Returns
    -------
    ax:
        Matplotlib axis.
    """
    BG = BG.copy()

    t_values = pd.to_datetime(BG.index)

    mean_curve = BG.mean(axis=1)

    lower_tol = []
    upper_tol = []

    for _, row in BG.iterrows():
        lower, upper, _, _, _ = nonparametric_tolerance_interval(
            row.values,
            content=content,
            confidence=confidence,
        )
        lower_tol.append(lower)
        upper_tol.append(upper)

    lower_tol = pd.Series(lower_tol, index=BG.index, dtype=float)
    upper_tol = pd.Series(upper_tol, index=BG.index, dtype=float)

    if ax is None:
        _, ax = plt.subplots(1)

    if plot_tolerance and not lower_tol.isnull().all():
        ax.fill_between(
            t_values,
            lower_tol,
            upper_tol,
            alpha=0.5,
            label=f"{content:.0%} / {confidence:.0%} TI",
        )

    # Plot at most max_individual_lines background curves
    columns = list(BG.columns)

    if len(columns) > max_individual_lines:
        rng = np.random.default_rng(random_state)
        columns_to_plot = rng.choice(
            columns,
            size=max_individual_lines,
            replace=False,
        )
    else:
        columns_to_plot = columns

    for p in columns_to_plot:
        ax.plot_date(
            t_values,
            BG[p],
            "-",
            color="grey",
            alpha=0.5,
            lw=0.5,
            label="_nolegend_",
        )

    ax.plot(
        t_values,
        mean_curve,
        lw=2,
        label="Mean Curve",
    )

    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M\n"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%b %d"))

    ax.axhline(
        70,
        c="green",
        linestyle="--",
        label="Hypoglycemia",
        lw=1,
    )
    ax.axhline(
        180,
        c="red",
        linestyle="--",
        label="Hyperglycemia",
        lw=1,
    )

    ax.set_xlim([t_values[0], t_values[-1]])
    ax.set_ylim([BG.min().min() - 10, BG.max().max() + 10])
    ax.legend()
    ax.set_ylabel("Blood Glucose (mg/dl)")

    return ax


def ensemblePlot(
    df,
    content=0.90,
    confidence=0.95,
    max_individual_lines=100,
    random_state=42,
):
    df_BG = df.unstack(level=0).BG
    df_CGM = df.unstack(level=0).CGM
    df_CHO = df.unstack(level=0).CHO

    fig_ensemble, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        sharex=True,
    )

    ax1 = ensemble_BG(
        df_BG,
        ax=ax1,
        plot_tolerance=True,
        content=content,
        confidence=confidence,
        max_individual_lines=max_individual_lines,
        random_state=random_state,
    )

    ax2 = ensemble_BG(
        df_CGM,
        ax=ax2,
        plot_tolerance=True,
        content=content,
        confidence=confidence,
        max_individual_lines=max_individual_lines,
        random_state=random_state,
    )

    t = pd.to_datetime(df_CHO.index)
    ax3.plot(t, df_CHO)

    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)

    ax3.xaxis.set_minor_locator(mdates.AutoDateLocator())
    ax3.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M\n"))
    ax3.xaxis.set_major_locator(mdates.DayLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("\n%b %d"))
    ax3.set_xlim([t[0], t[-1]])

    ax1.set_xlim(ax3.get_xlim())
    ax2.set_xlim(ax3.get_xlim())

    ax1.set_ylabel("Blood Glucose (mg/dl)")
    ax2.set_ylabel("CGM (mg/dl)")
    ax3.set_ylabel("CHO (g)")

    # Add glucose range background bands to BG and CGM panels
    for ax in [ax1, ax2]:
        old_ylim = ax.get_ylim()

        ax.axhspan(
            0,
            54,
            color="red",
            alpha=0.20,
            zorder=0,
            label="_nolegend_",
        )
        ax.axhspan(
            54,
            70,
            color="orange",
            alpha=0.20,
            zorder=0,
            label="_nolegend_",
        )
        ax.axhspan(
            70,
            180,
            color="green",
            alpha=0.15,
            zorder=0,
            label="_nolegend_",
        )
        ax.axhspan(
            180,
            250,
            color="orange",
            alpha=0.20,
            zorder=0,
            label="_nolegend_",
        )
        ax.axhspan(
            250,
            600,
            color="red",
            alpha=0.20,
            zorder=0,
            label="_nolegend_",
        )

        ax.set_ylim(old_ylim)

        # Keep plotted curves, mean lines, threshold lines, and tolerance bands
        # visually above the background regions.
        for line in ax.lines:
            line.set_zorder(3)

        for collection in ax.collections:
            collection.set_zorder(2)

    width = min(1, len(df_BG) // 480)

    fig_ensemble.set_size_inches(8 * width, 6)

    # Move legends outside the first two panels
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

    return fig_ensemble, ax1, ax2, ax3


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

    fig_ensemble_tolerance_bands, _, _, _ = ensemblePlot(report_df)
    fig_ensemble_tolerance_bands.savefig(
        report_dir / "BG_trace_tolerance.png",
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
        deterministic: bool = False,
        render: bool = False,
        verbose: int = 1,
        max_steps: int = 480,
        tolerance_content: float = 0.90,
        tolerance_confidence: float = 0.95,
    ):
        super().__init__(
            eval_env=eval_env,
            best_model_save_path=None,
            log_path=None,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=False,
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
                deterministic=False,
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

        # Keep exactly the requested evaluation episodes. Some environments append
        # an additional reset-created partial history after evaluation; slicing avoids
        # counting that extra history in CFR or plotting.
        history_list_eval = history_list[: self.n_eval_episodes]

        failure_stats = compute_critical_failure_stats(
            history_list=history_list_eval,
            max_steps=self.max_steps,
            confidence=self.tolerance_confidence,
            n_eval_episodes=self.n_eval_episodes,
        )

        # All evaluated episodes, including early terminations, are retained for
        # history saving and BG trace/report visualization.
        history_df_report = histories_to_dataframe(
            history_list=history_list_eval,
            eval_index=self.eval_index,
        )

        # Completed episodes only are used for TIR/TBR/TAR and insulin metrics.
        history_list_survivors = filter_surviving_histories(
            history_list=history_list_eval,
            max_steps=self.max_steps,
        )

        n_survivors = len(history_list_survivors)

        history_df_metrics = histories_to_dataframe(
            history_list=history_list_survivors,
            eval_index=self.eval_index,
        )

        print(" ==== metrics ====")
        print(history_df_metrics.head(5))

        if history_df_metrics.empty:
            if self.verbose > 0:
                print(
                    "[EvalInsulinPolicy] Warning: no completed episodes. "
                    "Clinical metrics and episode summaries set to NaN."
                )

            metrics = _nan_clinical_metrics()
            episode_summary_metrics = _nan_episode_summary_metrics()
        else:
            metrics = compute_scores(history_df_metrics)

            episode_metrics = compute_episode_scores(history_df_metrics)
            episode_summary_metrics = compute_episode_summary_metrics(episode_metrics)

        reward_mean, reward_sd, _ = _finite_mean_std(self.latest_rewards)

        if self.save_history and not history_df_report.empty:
            history_df_report.to_csv(
                self.history_path,
                mode="a",
                header=not self.history_path.exists(),
                index=True,
            )

        rewards = {
            **failure_stats,
            "survivors": int(n_survivors),
            "num_timesteps": int(self.num_timesteps),
            "mean_reward": float(reward_mean),
            "std_reward": float(reward_sd),
            "n_eval_episodes": int(self.n_eval_episodes),
        }

        new_row = (
            {"eval_index": int(self.eval_index)}
            | metrics
            | episode_summary_metrics
            | rewards
        )

        pd.DataFrame([new_row]).to_csv(
            self.metrics_path,
            mode="a",
            header=not self.metrics_path.exists(),
            index=False,
        )

        if self.generate_report and not history_df_report.empty:
            if self.verbose > 0:
                print("Generating report...")

            report_dir = self.figures_path / f"eval_{self.eval_index:04d}"

            save_simglucose_report(
                history_df=history_df_report,
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
    deterministic: bool = False,
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
    tolerance_content: float = 0.90,
    tolerance_confidence: float = 0.95,
) -> dict:
    """
    Evaluate a finished trained insulin policy/model.

    Clinical metrics and episode-level standard deviations are computed from
    completed episodes. Critical failure rate is computed from all episodes.

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
        deterministic=False,
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

    # Keep exactly the requested evaluation episodes. Some environments append
    # an additional reset-created partial history after evaluation; slicing avoids
    # counting that extra history in CFR or plotting.
    history_list_eval = history_list[:n_eval_episodes]

    failure_stats = compute_critical_failure_stats(
        history_list=history_list_eval,
        max_steps=max_steps,
        confidence=tolerance_confidence,
        n_eval_episodes=n_eval_episodes,
    )

    # All evaluated episodes, including early terminations, are retained for
    # history saving and BG trace/report visualization.
    history_df_report = histories_to_dataframe(
        history_list=history_list_eval,
        eval_index=eval_index,
    )

    # Completed episodes only are used for TIR/TBR/TAR and insulin metrics.
    history_list_survivors = filter_surviving_histories(
        history_list=history_list_eval,
        max_steps=max_steps,
    )

    n_survivors = len(history_list_survivors)

    history_df_metrics = histories_to_dataframe(
        history_list=history_list_survivors,
        eval_index=eval_index,
    )

    if history_df_metrics.empty:
        if verbose > 0:
            print(
                "[evaluate_insulin_policy] Warning: no completed episodes. "
                "Clinical metrics and episode summaries set to NaN."
            )

        metrics = _nan_clinical_metrics()
        episode_summary_metrics = _nan_episode_summary_metrics()
    else:
        metrics = compute_scores(history_df_metrics)
        episode_metrics = compute_episode_scores(history_df_metrics)
        episode_summary_metrics = compute_episode_summary_metrics(episode_metrics)

    reward_mean, reward_sd, _ = _finite_mean_std(episode_rewards)

    reward_metrics = {
        **failure_stats,
        "survivors": int(n_survivors),
        "num_timesteps": int(num_timesteps) if num_timesteps is not None else None,
        "mean_reward": float(reward_mean),
        "std_reward": float(reward_sd),
        "n_eval_episodes": int(n_eval_episodes),
    }

    results_row = {
        "eval_index": int(eval_index),
        **metrics,
        **episode_summary_metrics,
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

        if save_history and not history_df_report.empty:
            history_df_report.to_csv(
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

        if generate_report and not history_df_report.empty:
            if verbose > 0:
                print("Generating simglucose report...")

            report_dir = figures_path / f"eval_{eval_index:04d}"

            save_simglucose_report(
                history_df=history_df_report,
                report_dir=report_dir,
            )

    if clear_history_after:
        _call_env_method(eval_env, "clear_history")

    return {
        "metrics": results_row,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "history_report": history_df_report,
        "history_metrics": history_df_metrics,
    }