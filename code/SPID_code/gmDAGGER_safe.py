from pathlib import Path
import csv
import json
from datetime import datetime

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import TRPO, TQC, ARS, CrossQ

from huggingface_hub import hf_hub_download
from huggingface_sb3 import load_from_hub

from PySRWrapper_safe import PySRPolicy

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch
import warnings



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


warnings.filterwarnings(
    "ignore",
    message="Could not deserialize object learning_rate",
)

warnings.filterwarnings(
    "ignore",
    message="You loaded a model that was trained using OpenAI Gym",
)

warnings.filterwarnings(
    "ignore",
    message="Could not deserialize object clip_range",
    module="stable_baselines3.common.save_util",
)

warnings.filterwarnings(
    "ignore",
    message="Could not deserialize object lr_schedule",
)

warnings.filterwarnings(
    "ignore",
    message="You are trying to run PPO on the GPU",
)

warnings.filterwarnings(
    "ignore",
    message="Note: it looks like you are running in Jupyter",
)


DEFAULT_MAX_INSULIN_ACTION = 5.0


def raw_action_to_proposed_insulin(
    raw_action,
    max_insulin_action: float = DEFAULT_MAX_INSULIN_ACTION,
):
    """
    Convert PPO/env raw action [-1, 1] to proposed insulin dose
    [0, max_insulin_action].

    This must match env_closed.py:
        insulin = max_insulin_action * exp(4 * (raw_action - 1))
    """
    a = np.asarray(raw_action, dtype=np.float32)
    a = np.clip(a, -1.0, 1.0)

    max_insulin_action = float(max_insulin_action)
    insulin = max_insulin_action * np.exp(4.0 * (a - 1.0))
    insulin = np.clip(insulin, 0.0, max_insulin_action)

    return insulin.astype(np.float32)


def proposed_insulin_to_raw_action(
    insulin,
    max_insulin_action: float = DEFAULT_MAX_INSULIN_ACTION,
):
    """
    Convert proposed insulin dose [0, max_insulin_action] back to raw env
    action [-1, 1].

    This is the inverse of raw_action_to_proposed_insulin(...).
    It is useful if the symbolic policy predicts insulin dose, but the
    environment expects raw actions.
    """
    u = np.asarray(insulin, dtype=np.float32)

    max_insulin_action = float(max_insulin_action)
    min_insulin = max_insulin_action * np.exp(-8.0)
    u = np.clip(u, min_insulin, max_insulin_action)

    raw_action = 1.0 + np.log(u / max_insulin_action) / 4.0
    raw_action = np.clip(raw_action, -1.0, 1.0)

    return raw_action.astype(np.float32)


def make_env(environment, env_kwargs=None):
    env_kwargs = env_kwargs or {}

    def _init():
        return Monitor(gym.make(environment, **env_kwargs))

    return _init


def create_env(environment, hf_repo_id=None, vecnormalize_path=None, env_kwargs=None):
    env_kwargs = env_kwargs or {}

    try:
        if hf_repo_id is not None and vecnormalize_path is not None:
            print("Loading normalized env from Hugging Face")
            vecnorm_file = hf_hub_download(
                repo_id=hf_repo_id,
                filename=vecnormalize_path,
            )

            env = DummyVecEnv([make_env(environment, env_kwargs)])
            env = VecNormalize.load(vecnorm_file, env)
            env.training = False
            env.norm_reward = False

        else:
            env = make_vec_env(
                environment,
                n_envs=1,
                env_kwargs=env_kwargs,
                monitor_dir=None,
            )

        return env

    except Exception as e:
        print(e)
        raise ValueError(
            f"Could not create environment '{environment}'. "
            f"Original error: {e}"
        )


def get_algo_class(algo_name: str):
    algo_name = algo_name.lower()
    algo_map = {
        "ppo": PPO,
        "sac": SAC,
        "td3": TD3,
        "a2c": A2C,
        "ddpg": DDPG,
        "trpo": TRPO,
        "tqc": TQC,
        "ars": ARS,
        "crossq": CrossQ,
    }
    if algo_name not in algo_map:
        raise ValueError(f"Unsupported algorithm: {algo_name}")
    return algo_map[algo_name]


def create_run_folder(save_folder_path):
    run_dir = Path(save_folder_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_rewards_csv(rewards, path):
    path = Path(path)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "mean_reward"])
        for i, reward in enumerate(rewards):
            writer.writerow([i, float(reward)])


def save_iteration_summary_csv(rewards, path):
    path = Path(path)
    best_idx = int(np.argmax(rewards))
    best_reward = float(np.max(rewards))

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["best_iteration", "best_mean_reward", "n_iterations"])
        writer.writerow([best_idx, best_reward, len(rewards)])


def save_final_results_json(
    path,
    teacher_metrics,
    student_metrics,
    best_iteration,
    dataset_size,
    pysr_config_name=None,
):
    payload = {
        "best_student_iteration": int(best_iteration),
        "dataset_size": int(dataset_size),
        "teacher": {
            "mean_reward": float(teacher_metrics[0]),
            "std_reward": float(teacher_metrics[1]),
        },
        "student": {
            "mean_reward": float(student_metrics[0]),
            "std_reward": float(student_metrics[1]),
        },
    }

    if pysr_config_name is not None:
        payload["pysr_config_name"] = str(pysr_config_name)

    with Path(path).open("w") as f:
        json.dump(payload, f, indent=2)


def append_rows_to_csv(path, rows):
    """
    Append list[dict] rows to one CSV file.
    Header is written only once.
    """
    if rows is None or len(rows) == 0:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    write_header = not path.exists()

    df.to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
    )


def summarize_diagnostic_rows(rows, iteration, group):
    """
    Option A diagnostics:
    one aggregated row per iteration per group.

    group should usually be:
        "used"      = samples actually used for symbolic regression
        "terminal"  = samples from discarded early-terminal episodes
    """
    if rows is None or len(rows) == 0:
        return {
            "iteration": int(iteration),
            "group": str(group),
            "n_rows": 0,
            "n_episodes": 0,
            "mean_V_teacher_s": np.nan,
            "median_V_teacher_s": np.nan,
            "p05_V_teacher_s": np.nan,
            "p95_V_teacher_s": np.nan,
            "mean_Q_TD": np.nan,
            "median_Q_TD": np.nan,
            "mean_A_TD": np.nan,
            "median_A_TD": np.nan,
            "mean_abs_A_TD": np.nan,
            "median_abs_A_TD": np.nan,
            "p95_abs_A_TD": np.nan,
            "mean_cgm": np.nan,
            "median_cgm": np.nan,
            "min_cgm": np.nan,
            "max_cgm": np.nan,
            "hypo_fraction": np.nan,
            "hyper_fraction": np.nan,
            "mean_teacher_action": np.nan,
            "mean_executed_action": np.nan,
            "mean_teacher_dose": np.nan,
            "mean_executed_dose": np.nan,
            "mean_reward": np.nan,
            "sum_reward": np.nan,
        }

    def arr(key):
        return np.asarray([r.get(key, np.nan) for r in rows], dtype=float)

    def finite_any(x):
        return np.isfinite(x).any()

    def safe_mean(x):
        return float(np.nanmean(x)) if finite_any(x) else np.nan

    def safe_median(x):
        return float(np.nanmedian(x)) if finite_any(x) else np.nan

    def safe_min(x):
        return float(np.nanmin(x)) if finite_any(x) else np.nan

    def safe_max(x):
        return float(np.nanmax(x)) if finite_any(x) else np.nan

    def safe_quantile(x, q):
        return float(np.nanquantile(x, q)) if finite_any(x) else np.nan

    v = arr("V_teacher_s")
    q_td = arr("Q_TD")
    a_td = arr("A_TD")
    abs_a_td = arr("abs_A_TD")
    cgm = arr("cgm")
    reward = arr("reward")
    teacher_action = arr("teacher_action")
    executed_action = arr("executed_action")
    teacher_dose = arr("teacher_dose")
    executed_dose = arr("executed_dose")

    attempted_episodes = []
    for r in rows:
        ep = r.get("attempted_episode", np.nan)
        try:
            if np.isfinite(float(ep)):
                attempted_episodes.append(int(ep))
        except Exception:
            pass

    return {
        "iteration": int(iteration),
        "group": str(group),
        "n_rows": int(len(rows)),
        "n_episodes": int(len(set(attempted_episodes))),

        "mean_V_teacher_s": safe_mean(v),
        "median_V_teacher_s": safe_median(v),
        "p05_V_teacher_s": safe_quantile(v, 0.05),
        "p95_V_teacher_s": safe_quantile(v, 0.95),

        "mean_Q_TD": safe_mean(q_td),
        "median_Q_TD": safe_median(q_td),

        "mean_A_TD": safe_mean(a_td),
        "median_A_TD": safe_median(a_td),

        "mean_abs_A_TD": safe_mean(abs_a_td),
        "median_abs_A_TD": safe_median(abs_a_td),
        "p95_abs_A_TD": safe_quantile(abs_a_td, 0.95),

        "mean_cgm": safe_mean(cgm),
        "median_cgm": safe_median(cgm),
        "min_cgm": safe_min(cgm),
        "max_cgm": safe_max(cgm),
        "hypo_fraction": float(np.nanmean(cgm < 70.0)) if finite_any(cgm) else np.nan,
        "hyper_fraction": float(np.nanmean(cgm > 180.0)) if finite_any(cgm) else np.nan,

        "mean_teacher_action": safe_mean(teacher_action),
        "mean_executed_action": safe_mean(executed_action),
        "mean_teacher_dose": safe_mean(teacher_dose),
        "mean_executed_dose": safe_mean(executed_dose),

        "mean_reward": safe_mean(reward),
        "sum_reward": float(np.nansum(reward)) if finite_any(reward) else np.nan,
    }


def _default_pysr_config():
    return {
        "binary_operators": ["+", "*", "-", "/", "<", ">"],
        "unary_operators": ["square", "exp", "log", "sqrt"],
        "maxsize": 18,
    }


def _build_pysr_policy(env, pysr_config, loss_str):
    """
    Build PySRPolicy from a config dictionary.

    Supported direct keys include the usual PySRPolicy/PySRRegressor arguments,
    for example:
        binary_operators, unary_operators, maxsize, populations,
        niterations, nested_constraints, constraints, parsimony, etc.

    You can also place arbitrary additional kwargs under:
        extra_kwargs={...}
    """
    cfg = dict(pysr_config or {})
    extra_kwargs = dict(cfg.pop("extra_kwargs", {}) or {})

    binary_operators = cfg.pop(
        "binary_operators",
        ["+", "*", "-", "/", "<", ">"],
    )
    unary_operators = cfg.pop(
        "unary_operators",
        ["square", "exp", "log", "sqrt"],
    )
    maxsize = cfg.pop("maxsize", 18)

    verbosity = cfg.pop("verbosity", 0)
    temp_equation_file = cfg.pop("temp_equation_file", False)
    delete_tempfiles = cfg.pop("delete_tempfiles", True)
    output_jax_format = cfg.pop("output_jax_format", False)
    output_torch_format = cfg.pop("output_torch_format", False)
    elementwise_loss = cfg.pop("elementwise_loss", loss_str)
    progress = cfg.pop("progress", True)
    input_stream = cfg.pop("input_stream", "devnull")

    cfg.update(extra_kwargs)

    return PySRPolicy(
        env,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        maxsize=maxsize,
        verbosity=verbosity,
        temp_equation_file=temp_equation_file,
        delete_tempfiles=delete_tempfiles,
        output_jax_format=output_jax_format,
        output_torch_format=output_torch_format,
        elementwise_loss=elementwise_loss,
        progress=progress,
        input_stream=input_stream,
        parsimony = 0.003,
        **cfg,
    )


def train_spid(
        teacher_path,
        teacher_model,
        save_folder_path,
        environment,
        n_iter,
        total_timesteps,
        save_results=False,
        verbose=1,
        n_eval_episodes=100,
        final_n_val_episodes=100,
        hf_repo_id=None,
        hf_filename=None,
        hf_algo=None,
        vecnormalize_path=None,
        skip_initial_steps=10,
        sample_episodes=5,
        drop_terminal_transitions=True,
        discard_early_terminal_episodes=True,
        max_episode_steps=None,
        max_sampling_episodes=200,
        save_sampling_summary=True,
        diagnostics_gamma=None,
        pysr_config=None,
        pysr_config_name="default",
        sampling_policy_mode="mixed",
        max_insulin_action=DEFAULT_MAX_INSULIN_ACTION,
    ):
    """
    Train SPID using gmDAGGER-style dataset aggregation.

    This version supports:
        - skipping first N steps after reset
        - discarding early-terminal episodes
        - saving Option A diagnostics in one CSV
        - passing PySR search configs from the train script
        - using teacher-specific gamma from train_config.json via diagnostics_gamma
        - training symbolic regression on teacher insulin dose labels

    Output diagnostics:
        sampling_iteration_summary.csv

    It stores two rows per iteration:
        group = used
        group = terminal
    """

    loss_str = "loss(pred, target, w) = w .* (pred .- target).^2"

    dataset = []
    policy = None
    policies = []
    rewards = []

    run_dir = create_run_folder(save_folder_path)

    if pysr_config is None:
        raise ValueError(
            "pysr_config was None. The distill script did not pass the selected "
            "PySR configuration into train_spid(...)."
        )
    if save_results:
        with (run_dir / "pysr_config.json").open("w") as f:
            json.dump(
                {
                    "pysr_config_name": str(pysr_config_name),
                    "pysr_config": pysr_config,
                    "sampling_policy_mode": str(sampling_policy_mode),
                    "student_target": "teacher_insulin_dose",
                    "max_insulin_action": float(max_insulin_action),
                },
                f,
                indent=2,
            )

    for i in tqdm(range(n_iter), disable=verbose > 0):
        beta = 1 if i == 0 else 0.5

        (
            new_data,
            used_summary,
            terminal_summary,
        ) = sample_trajectory(
            teacher_path,
            teacher_model,
            environment,
            total_timesteps,
            n_iter,
            policy,
            beta,
            hf_repo_id=hf_repo_id,
            hf_filename=hf_filename,
            hf_algo=hf_algo,
            vecnormalize_path=vecnormalize_path,
            skip_initial_steps=skip_initial_steps,
            sample_episodes=sample_episodes,
            drop_terminal_transitions=drop_terminal_transitions,
            discard_early_terminal_episodes=discard_early_terminal_episodes,
            max_episode_steps=max_episode_steps,
            max_sampling_episodes=max_sampling_episodes,
            iteration=i,
            diagnostics_gamma=diagnostics_gamma,
            sampling_policy_mode=sampling_policy_mode,
            max_insulin_action=max_insulin_action,
        )

        if save_results and save_sampling_summary:
            append_rows_to_csv(
                run_dir / "sampling_iteration_summary.csv",
                [used_summary, terminal_summary],
            )

        if not dataset:
            dataset = new_data.copy()
        else:
            dataset = [
                np.concatenate((x_old, x_new), axis=0)
                for x_old, x_new in zip(dataset, new_data)
            ]

        x = dataset[0]
        y = dataset[1]
        advs = dataset[2]

        weights = np.abs(advs)
        weights = weights / np.max(weights) if np.max(weights) > 0 else weights

        env = create_env(environment, hf_repo_id, vecnormalize_path)
        srr_test = _build_pysr_policy(
            env=env,
            pysr_config=pysr_config,
            loss_str=loss_str,
        )

        print("training")
        srr_test.fit(x, y, weights=weights)

        policies.append(srr_test)
        policy = srr_test

        print("Evaluating trained model")
        eval_env = create_env(environment, hf_repo_id, vecnormalize_path)
        mean_reward, std_reward = evaluate_policy(
            srr_test,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=False,
        )

        eval_env.close()
        env.close()

        rewards.append(float(mean_reward))
        if verbose == 2:
            print(f"Iteration {i}: student reward = {mean_reward:.4f} +/- {std_reward:.4f}")

    best_idx = int(np.argmax(rewards))
    best_policy = policies[best_idx]
    best_wrapper = best_policy

    best_policy_path = run_dir / "best_student_policy.joblib"
    if save_results:
        best_wrapper.save(best_policy_path)

        save_rewards_csv(rewards, run_dir / "student_rewards.csv")
        save_iteration_summary_csv(rewards, run_dir / "summary.csv")

    teacher_eval_env, teacher = load_teacher_env(
        teacher_path,
        teacher_model,
        environment,
        hf_repo_id=hf_repo_id,
        hf_filename=hf_filename,
        hf_algo=hf_algo,
        vecnormalize_path=vecnormalize_path,
    )

    teacher_mean_reward, teacher_std_reward = evaluate_policy(
        teacher,
        teacher_eval_env,
        n_eval_episodes=final_n_val_episodes,
        deterministic=False,
    )
    teacher_eval_env.close()

    student_eval_env = create_env(environment, hf_repo_id, vecnormalize_path)
    student_mean_reward, student_std_reward = evaluate_policy(
        best_wrapper,
        student_eval_env,
        n_eval_episodes=final_n_val_episodes,
        deterministic=False,
    )
    student_eval_env.close()

    dataset_size = int(dataset[0].shape[0]) if dataset else 0

    if save_results:
        save_final_results_json(
            run_dir / "final_results.json",
            teacher_metrics=(teacher_mean_reward, teacher_std_reward),
            student_metrics=(student_mean_reward, student_std_reward),
            best_iteration=best_idx,
            dataset_size=dataset_size,
            pysr_config_name=pysr_config_name,
        )

        with (run_dir / "teacher_student_comparison.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["policy", "mean_reward", "std_reward"])
            writer.writerow(["teacher", float(teacher_mean_reward), float(teacher_std_reward)])
            writer.writerow(["student_best", float(student_mean_reward), float(student_std_reward)])

    print(f"SPID iteration complete. Dataset size: {dataset_size}")
    print(f"Best policy iteration: {best_idx}")
    print(f"Best student reward during search: {np.max(rewards):.4f}")
    print(f"Teacher eval: {teacher_mean_reward:.4f} +/- {teacher_std_reward:.4f}")
    print(f"Student eval: {student_mean_reward:.4f} +/- {student_std_reward:.4f}")
    print(f"Saved results to: {run_dir}")

    best_wrapper.print_info()
    return rewards, best_policy, best_wrapper, run_dir


def load_teacher_env(
    teacher_path,
    teacher_model,
    environment,
    hf_repo_id=None,
    hf_filename=None,
    hf_algo=None,
    vecnormalize_path=None,
):
    env = create_env(environment, hf_repo_id=hf_repo_id, vecnormalize_path=vecnormalize_path)

    if teacher_path is not None:
        try:
            teacher = teacher_model.load(teacher_path)
            print(f"Loaded local teacher from: {teacher_path}")
            return env, teacher
        except Exception as e:
            print(f"Local load failed: {e}")

    if hf_repo_id is not None and hf_filename is not None:
        try:
            algo_class = get_algo_class(hf_algo) if hf_algo is not None else teacher_model

            checkpoint_path = load_from_hub(
                repo_id=hf_repo_id,
                filename=hf_filename,
            )

            teacher = algo_class.load(checkpoint_path)
            print(f"Loaded Hugging Face teacher from: {hf_repo_id}/{hf_filename}")
            return env, teacher
        except Exception as e:
            raise RuntimeError(
                f"Failed to load teacher from Hugging Face repo '{hf_repo_id}' "
                f"with filename '{hf_filename}': {e}"
            )

    raise FileNotFoundError(
        "Could not load teacher model. Provide either a valid local teacher_path "
        "or Hugging Face repo_id + filename."
    )


def _vec_done_to_bool(done) -> bool:
    return bool(np.asarray(done).reshape(-1)[0])


def _first_info(infos) -> dict:
    if isinstance(infos, (list, tuple)):
        return infos[0] if len(infos) > 0 and isinstance(infos[0], dict) else {}
    return infos if isinstance(infos, dict) else {}


def _flatten_vec_value(value):
    arr = np.asarray(value)
    return arr.reshape(-1) if arr.ndim > 1 else arr


def _is_natural_episode_end(
    done_bool: bool,
    info: dict,
    episode_steps: int,
    max_episode_steps=None,
) -> bool:
    """
    Treat time-limit endings as natural and early terminations as terminal failures.
    """
    if not done_bool:
        return False

    if bool(info.get("TimeLimit.truncated", False)):
        return True

    if max_episode_steps is not None and episode_steps >= int(max_episode_steps):
        return True

    return False


def _safe_scalar_from_info(info, key, default=np.nan):
    try:
        value = info.get(key, default)
        return float(value)
    except Exception:
        return default


def sample_trajectory(
        teacher_path,
        teacher_model,
        environment,
        total_timesteps,
        n_iter,
        policy,
        beta,
        hf_repo_id=None,
        hf_filename=None,
        hf_algo=None,
        vecnormalize_path=None,
        skip_initial_steps=10,
        sample_episodes=5,
        drop_terminal_transitions=True,
        discard_early_terminal_episodes=True,
        max_episode_steps=None,
        max_sampling_episodes=200,
        iteration=0,
        diagnostics_gamma=None,
        sampling_policy_mode="mixed",
        max_insulin_action=DEFAULT_MAX_INSULIN_ACTION,
    ):
    """
    Collect gmDAGGER data for SPID.

    The symbolic regression target is the teacher's proposed insulin dose, not
    the raw PPO action. The executed action used for environment rollout remains
    whatever the active policy produces.

    This assumes your PySRPolicy wrapper converts predicted insulin dose back to
    raw env action before stepping the environment. If not, update PySRWrapper.
    """
    if skip_initial_steps < 0:
        raise ValueError("skip_initial_steps must be >= 0.")

    if max_sampling_episodes <= 0:
        raise ValueError("max_sampling_episodes must be > 0.")

    target_steps = int(total_timesteps // n_iter)
    target_episodes = (
        int(sample_episodes)
        if sample_episodes is not None and int(sample_episodes) > 0
        else None
    )

    env, teacher = load_teacher_env(
        teacher_path,
        teacher_model,
        environment,
        hf_repo_id=hf_repo_id,
        hf_filename=hf_filename,
        hf_algo=hf_algo,
        vecnormalize_path=vecnormalize_path,
    )
    policy = policy or teacher

    gamma_used = (
        float(diagnostics_gamma)
        if diagnostics_gamma is not None
        else float(getattr(teacher, "gamma", 0.99))
    )

    print(f"Using gamma for diagnostics and advantage weights: {gamma_used}")

    states = []
    training_actions = []
    teacher_actions = []
    rewards = []
    next_states = []

    pysr_actions = 0
    oracle_actions = 0
    accepted_episodes = 0
    discarded_episodes = 0
    attempted_episodes = 0

    used_diagnostic_rows = []
    terminal_diagnostic_rows = []

    def enough_data() -> bool:
        if target_episodes is not None:
            return accepted_episodes >= target_episodes
        return len(states) >= target_steps

    def compute_value_diagnostics(obs_flat, next_obs_flat, reward_scalar):
        """
        PPO learns V(s), not Q(s,a).

        For PPO/on-policy critics:
            Q_TD = r + gamma * V(s_next)
            A_TD = Q_TD - V(s)

        These are diagnostics and weighting estimates, not exact Q-values.
        """
        try:
            device = torch.device("cpu")
            teacher.policy = teacher.policy.to(device)

            with torch.no_grad():
                s_t = torch.as_tensor(
                    obs_flat,
                    dtype=torch.float32,
                    device=device,
                ).view(1, -1)
                sp_t = torch.as_tensor(
                    next_obs_flat,
                    dtype=torch.float32,
                    device=device,
                ).view(1, -1)

                if hasattr(teacher.policy, "predict_values"):
                    v_s = float(
                        teacher.policy.predict_values(s_t)
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)[0]
                    )
                    v_sp = float(
                        teacher.policy.predict_values(sp_t)
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)[0]
                    )

                    q_td = float(reward_scalar + gamma_used * v_sp)
                    a_td = float(q_td - v_s)
                    return v_s, v_sp, q_td, a_td, abs(a_td)

                return np.nan, np.nan, np.nan, np.nan, np.nan

        except Exception:
            return np.nan, np.nan, np.nan, np.nan, np.nan

    def make_diag_row(
        obs_flat,
        next_obs_flat,
        reward,
        action_flat,
        oracle_action_flat,
        info0,
        done_bool,
        store_transition,
        episode_step_idx,
        sample_type,
        discard_reason,
    ):
        reward_scalar = float(np.asarray(reward).reshape(-1)[0])

        v_s, v_sp, q_td, a_td, abs_a_td = compute_value_diagnostics(
            obs_flat=obs_flat,
            next_obs_flat=next_obs_flat,
            reward_scalar=reward_scalar,
        )

        teacher_dose = raw_action_to_proposed_insulin(
            oracle_action_flat,
            max_insulin_action=max_insulin_action,
        )
        executed_dose = raw_action_to_proposed_insulin(
            action_flat,
            max_insulin_action=max_insulin_action,
        )

        return {
            "iteration": int(iteration),
            "attempted_episode": int(attempted_episodes),
            "episode_step": int(episode_step_idx),
            "used_for_training": bool(store_transition),
            "done": bool(done_bool),
            "sample_type": sample_type,
            "discard_reason": discard_reason,

            "patient_name": info0.get("patient_name", ""),
            "multipatient_episode_idx": info0.get("multipatient_episode_idx", np.nan),

            "cgm": _safe_scalar_from_info(info0, "plot_cgm_raw", np.nan),
            "meal": _safe_scalar_from_info(info0, "plot_meal", np.nan),
            "insulin": _safe_scalar_from_info(info0, "plot_insulin_action", np.nan),

            "obs_0": float(obs_flat[0]) if len(obs_flat) > 0 else np.nan,
            "obs_1": float(obs_flat[1]) if len(obs_flat) > 1 else np.nan,

            "teacher_action": float(np.asarray(oracle_action_flat).reshape(-1)[0]),
            "executed_action": float(np.asarray(action_flat).reshape(-1)[0]),
            "teacher_dose": float(np.asarray(teacher_dose).reshape(-1)[0]),
            "executed_dose": float(np.asarray(executed_dose).reshape(-1)[0]),

            "reward": reward_scalar,

            "V_teacher_s": v_s,
            "V_teacher_s_next": v_sp,
            "Q_TD": q_td,
            "A_TD": a_td,
            "abs_A_TD": abs_a_td,
        }

    try:
        while not enough_data():
            if attempted_episodes >= max_sampling_episodes:
                used_summary = summarize_diagnostic_rows(
                    used_diagnostic_rows,
                    iteration=iteration,
                    group="used",
                )
                terminal_summary = summarize_diagnostic_rows(
                    terminal_diagnostic_rows,
                    iteration=iteration,
                    group="terminal",
                )

                used_summary["gamma"] = float(gamma_used)
                terminal_summary["gamma"] = float(gamma_used)
                used_summary["sampling_policy_mode"] = str(sampling_policy_mode)
                terminal_summary["sampling_policy_mode"] = str(sampling_policy_mode)
                used_summary["max_insulin_action"] = float(max_insulin_action)
                terminal_summary["max_insulin_action"] = float(max_insulin_action)

                print("Used summary before failure:")
                print(used_summary)
                print("Terminal summary before failure:")
                print(terminal_summary)

                raise RuntimeError(
                    "Could not collect enough non-terminal SPID samples. "
                    f"accepted_episodes={accepted_episodes}, "
                    f"discarded_episodes={discarded_episodes}, "
                    f"target_episodes={target_episodes}, "
                    f"target_steps={target_steps}. "
                    "Either increase max_sampling_episodes or inspect why the "
                    "current policy terminates early."
                )

            attempted_episodes += 1
            obs = env.reset()
            episode_step_idx = 0

            ep_states = []
            ep_training_actions = []
            ep_teacher_actions = []
            ep_rewards = []
            ep_next_states = []
            ep_diagnostic_rows = []

            while True:
                if sampling_policy_mode == "mixed":
                    active_policy = [policy, teacher][np.random.binomial(1, beta)]
                elif sampling_policy_mode == "teacher":
                    active_policy = teacher
                elif sampling_policy_mode == "student":
                    active_policy = policy
                else:
                    raise ValueError(
                        f"Unknown sampling_policy_mode={sampling_policy_mode!r}. "
                        "Expected one of: 'mixed', 'teacher', 'student'."
                    )

                if isinstance(active_policy, PySRPolicy):
                    action, _states = active_policy.predict(obs)
                    pysr_actions += 1
                else:
                    action, _states = active_policy.predict(obs, deterministic=False)
                    oracle_actions += 1

                oracle_action, _states = teacher.predict(obs, deterministic=False)

                next_obs, reward, done, infos = env.step(action)
                done_bool = _vec_done_to_bool(done)
                info0 = _first_info(infos)

                obs_flat = _flatten_vec_value(obs)
                reward_flat = _flatten_vec_value(reward)
                action_flat = _flatten_vec_value(action)
                oracle_action_flat = _flatten_vec_value(oracle_action)
                next_obs_flat = _flatten_vec_value(next_obs)

                oracle_dose_flat = raw_action_to_proposed_insulin(
                    oracle_action_flat,
                    max_insulin_action=max_insulin_action,
                )

                store_transition = episode_step_idx >= skip_initial_steps
                if drop_terminal_transitions and done_bool:
                    store_transition = False

                if store_transition:
                    ep_states.append(obs_flat)
                    ep_training_actions.append(action_flat)
                    ep_teacher_actions.append(oracle_dose_flat)
                    ep_rewards.append(reward_flat)
                    ep_next_states.append(next_obs_flat)

                    diag_row = make_diag_row(
                        obs_flat=obs_flat,
                        next_obs_flat=next_obs_flat,
                        reward=reward,
                        action_flat=action_flat,
                        oracle_action_flat=oracle_action_flat,
                        info0=info0,
                        done_bool=done_bool,
                        store_transition=True,
                        episode_step_idx=episode_step_idx,
                        sample_type="used",
                        discard_reason="",
                    )

                else:
                    if done_bool:
                        reason = "terminal_transition"
                    elif episode_step_idx < skip_initial_steps:
                        reason = "initial_skip"
                    else:
                        reason = "not_used"

                    diag_row = make_diag_row(
                        obs_flat=obs_flat,
                        next_obs_flat=next_obs_flat,
                        reward=reward,
                        action_flat=action_flat,
                        oracle_action_flat=oracle_action_flat,
                        info0=info0,
                        done_bool=done_bool,
                        store_transition=False,
                        episode_step_idx=episode_step_idx,
                        sample_type="not_used",
                        discard_reason=reason,
                    )

                ep_diagnostic_rows.append(diag_row)

                episode_step_idx += 1
                obs = next_obs

                if (
                    target_episodes is None
                    and len(states) + len(ep_states) >= target_steps
                    and not done_bool
                ):
                    remaining = target_steps - len(states)

                    states.extend(ep_states[:remaining])
                    training_actions.extend(ep_training_actions[:remaining])
                    teacher_actions.extend(ep_teacher_actions[:remaining])
                    rewards.extend(ep_rewards[:remaining])
                    next_states.extend(ep_next_states[:remaining])
                    accepted_episodes += 1

                    kept_used_rows = [
                        r for r in ep_diagnostic_rows
                        if r["sample_type"] == "used"
                    ][:remaining]

                    for row in kept_used_rows:
                        row["accepted_episode"] = int(accepted_episodes)
                        row["terminal_episode"] = False
                        used_diagnostic_rows.append(row)

                    break

                if done_bool:
                    natural_end = _is_natural_episode_end(
                        done_bool=done_bool,
                        info=info0,
                        episode_steps=episode_step_idx,
                        max_episode_steps=max_episode_steps,
                    )

                    if discard_early_terminal_episodes and not natural_end:
                        discarded_episodes += 1

                        for row in ep_diagnostic_rows:
                            row["accepted_episode"] = np.nan
                            row["terminal_episode"] = True
                            row["used_for_training"] = False
                            row["sample_type"] = "terminal"
                            row["discard_reason"] = "early_terminal_episode"
                            terminal_diagnostic_rows.append(row)

                    else:
                        states.extend(ep_states)
                        training_actions.extend(ep_training_actions)
                        teacher_actions.extend(ep_teacher_actions)
                        rewards.extend(ep_rewards)
                        next_states.extend(ep_next_states)
                        accepted_episodes += 1

                        for row in ep_diagnostic_rows:
                            if row["sample_type"] == "used":
                                row["accepted_episode"] = int(accepted_episodes)
                                row["terminal_episode"] = False
                                used_diagnostic_rows.append(row)

                    break

        if len(states) == 0:
            raise RuntimeError(
                "No SPID samples were collected. Check skip_initial_steps, "
                "sample_episodes, max_episode_steps, and terminal/discard settings."
            )

        print(
            "finished collecting trajectories | "
            f"samples={len(states)}, accepted_episodes={accepted_episodes}, "
            f"discarded_early_terminal_episodes={discarded_episodes}, "
            f"pysr_actions={pysr_actions}, teacher_actions={oracle_actions}"
        )

        weights = get_advantage_weights(
            states,
            training_actions,
            rewards,
            next_states,
            teacher,
            gamma=gamma_used,
        )

        trajectory = [
            np.asarray(states),
            np.asarray(teacher_actions),
            weights,
        ]

        used_summary = summarize_diagnostic_rows(
            used_diagnostic_rows,
            iteration=iteration,
            group="used",
        )

        terminal_summary = summarize_diagnostic_rows(
            terminal_diagnostic_rows,
            iteration=iteration,
            group="terminal",
        )

        used_summary["gamma"] = float(gamma_used)
        terminal_summary["gamma"] = float(gamma_used)
        used_summary["sampling_policy_mode"] = str(sampling_policy_mode)
        terminal_summary["sampling_policy_mode"] = str(sampling_policy_mode)
        used_summary["max_insulin_action"] = float(max_insulin_action)
        terminal_summary["max_insulin_action"] = float(max_insulin_action)

        used_summary["accepted_episodes"] = int(accepted_episodes)
        used_summary["discarded_episodes"] = int(discarded_episodes)
        used_summary["attempted_episodes"] = int(attempted_episodes)
        used_summary["acceptance_rate"] = (
            float(accepted_episodes / attempted_episodes)
            if attempted_episodes > 0
            else np.nan
        )

        terminal_summary["accepted_episodes"] = int(accepted_episodes)
        terminal_summary["discarded_episodes"] = int(discarded_episodes)
        terminal_summary["attempted_episodes"] = int(attempted_episodes)
        terminal_summary["acceptance_rate"] = (
            float(accepted_episodes / attempted_episodes)
            if attempted_episodes > 0
            else np.nan
        )

        return trajectory, used_summary, terminal_summary

    finally:
        env.close()


def get_advantage_weights(
    states,
    actions,
    rewards,
    next_states,
    expert,
    gamma=0.99,
    device=None,
    force_cpu=True,
):
    """
    Compute sample-wise advantage weights.

    For PPO/A2C/TRPO:
        V(s) is used and Q_TD is approximated as r + gamma V(s').

    For off-policy algorithms:
        critic Q(s,a) is used if available.
    """
    print("computing advantages")

    if force_cpu:
        device = torch.device("cpu")
    elif device is None:
        try:
            device = next(expert.policy.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    expert.policy = expert.policy.to(device)

    def to_tensor(x, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, dtype=dtype, device=device)

    def ensure_2d_action(x):
        if x.ndim == 1:
            return x.unsqueeze(-1)
        return x

    with torch.no_grad():
        states_t = to_tensor(np.stack(states), dtype=torch.float32)
        actions_t = ensure_2d_action(to_tensor(np.stack(actions), dtype=torch.float32))
        rewards_t = to_tensor(np.asarray(rewards), dtype=torch.float32).view(-1)
        next_states_t = to_tensor(np.stack(next_states), dtype=torch.float32)

        if hasattr(expert.policy, "predict_values"):
            v_s = expert.policy.predict_values(states_t).squeeze(-1)
            v_sp = expert.policy.predict_values(next_states_t).squeeze(-1)
            q_sa = rewards_t + gamma * v_sp
            adv_t = q_sa - v_s

        else:
            try:
                if not hasattr(expert.policy, "critic"):
                    raise AttributeError("No critic found on policy")

                algorithm_name = expert.__class__.__name__.lower()

                if hasattr(expert.policy.critic, "q1_forward"):
                    q_s = expert.policy.critic.q1_forward(
                        states_t,
                        actions_t,
                    ).squeeze(-1)

                    if "sac" in algorithm_name:
                        try:
                            actor_output = expert.policy.actor(next_states_t)
                            next_actions = (
                                actor_output[0]
                                if isinstance(actor_output, tuple)
                                else actor_output
                            )
                        except Exception as e1:
                            print(f"SAC actor call method 1 failed: {e1}")
                            try:
                                latent_pi = expert.policy.actor.latent_pi(next_states_t)
                                next_actions = expert.policy.actor.mu(latent_pi)
                                next_actions = torch.tanh(next_actions)
                            except Exception as e2:
                                print(f"SAC actor call method 2 failed: {e2}")
                                next_states_np = next_states_t.detach().cpu().numpy()
                                next_actions_np, _ = expert.predict(
                                    next_states_np,
                                    deterministic=False,
                                )
                                next_actions = to_tensor(
                                    next_actions_np,
                                    dtype=torch.float32,
                                )
                    else:
                        actor_output = expert.policy.actor(next_states_t)
                        next_actions = (
                            actor_output[0]
                            if isinstance(actor_output, tuple)
                            else actor_output
                        )

                    next_actions = ensure_2d_action(next_actions)
                    next_actions = next_actions.to(device=device, dtype=torch.float32)

                    q_sp = expert.policy.critic.q1_forward(
                        next_states_t,
                        next_actions,
                    ).squeeze(-1)

                elif hasattr(expert.policy.critic, "forward"):
                    q_s = expert.policy.critic(states_t, actions_t).squeeze(-1)

                    next_actions = expert.policy.actor(next_states_t)
                    if isinstance(next_actions, tuple):
                        next_actions = next_actions[0]

                    next_actions = ensure_2d_action(next_actions)
                    next_actions = next_actions.to(device=device, dtype=torch.float32)

                    q_sp = expert.policy.critic(
                        next_states_t,
                        next_actions,
                    ).squeeze(-1)

                else:
                    raise AttributeError("Critic method not found")

                target_q = rewards_t + gamma * q_sp
                adv_t = q_s - target_q

                print(f"Successfully computed Q-based advantages for {algorithm_name} on {device}")

            except Exception as e:
                print(
                    "Warning: Q-network computation failed "
                    f"({e}), using simplified advantage computation"
                )
                rewards_np = rewards_t.detach().cpu().numpy()
                adv_t = torch.as_tensor(
                    rewards_np - rewards_np.mean(),
                    dtype=torch.float32,
                    device=device,
                )

    adv = adv_t.detach().cpu().numpy()
    adv = np.squeeze(adv)
    return adv