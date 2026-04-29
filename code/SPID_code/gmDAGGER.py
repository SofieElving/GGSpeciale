from pathlib import Path
import csv
import json
from datetime import datetime

import gymnasium as gym
import numpy as np

from tqdm import tqdm
from pysr import PySRRegressor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG
from sb3_contrib import TRPO, TQC, ARS, CrossQ

from huggingface_hub import hf_hub_download
from huggingface_sb3 import load_from_hub

from PySRWrapper import PySRPolicy

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch
import warnings

warnings.filterwarnings(
    "ignore",
    message="Could not deserialize object learning_rate"
)

warnings.filterwarnings(
    "ignore",
    message="You loaded a model that was trained using OpenAI Gym"
)

warnings.filterwarnings(
    "ignore",
    message="Could not deserialize object clip_range",
    module="stable_baselines3.common.save_util"
)

warnings.filterwarnings(
    "ignore",
    message="Could not deserialize object lr_schedule"
)

warnings.filterwarnings(
    "ignore",
    message="You are trying to run PPO on the GPU"
)

warnings.filterwarnings(
    "ignore",
    message="Note: it looks like you are running in Jupyter"
)


def make_env(environment, env_kwargs=None):
    env_kwargs = env_kwargs or {}

    def _init():
        return Monitor(gym.make(environment, **env_kwargs))

    return _init


def create_env(environment, hf_repo_id=None, vecnormalize_path=None, env_kwargs=None):
    env_kwargs = env_kwargs or {}

    try:
        # Load normalized environment from Hugging Face if requested
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
            # Plain non-normalized vectorized env
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
    """
    Ensure that a directory exists at the given path.
    If it does not exist, create it.

    Returns
    -------
    Path
        Path object to the created/existing directory.
    """
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

def save_final_results_json(path, teacher_metrics, student_metrics, best_iteration, dataset_size):
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

    with Path(path).open("w") as f:
        json.dump(payload, f, indent=2)


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
        vecnormalize_path=None
    ):
    
    # print(f"Training SPID on {env_name}")

    loss_str = "loss(pred, target, w) = w .* (pred .- target).^2"
    # loss_str = "loss(pred, target) = (pred .- target).^2"

    dataset = []
    policy = None
    policies = []
    rewards = []

    run_dir = create_run_folder(save_folder_path)

    for i in tqdm(range(n_iter), disable=verbose > 0):
        beta = 1 if i == 0 else 0.5

        new_data = sample_trajectory(
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
            )
        
        if not dataset:
            dataset = new_data.copy()
        else: 
            dataset = [np.concatenate((x, y), axis=0) for x, y in zip(dataset, new_data)]
 
        x = dataset[0]
        y = dataset[1]
        advs = dataset[2]

        weights = np.abs(advs)
        weights = weights / np.max(weights) if np.max(weights) > 0 else weights

        env = create_env(environment, hf_repo_id, vecnormalize_path)
        srr_test = PySRPolicy(env, 
                              binary_operators=["+", "*", "-", "/"],
                              #populations=64, 
                              maxsize=18, 
                              #niterations=100, 
                              verbosity=0, 
                              temp_equation_file=False,
                              delete_tempfiles=True,
                              output_jax_format=False,
                              output_torch_format=False,
                              elementwise_loss=loss_str, 
                              progress=True,
                              input_stream='devnull')
        
        print("training")
        srr_test.fit(x, y, weights=weights)

        # print("training")
        # srr_test.fit(x, y)


        policies.append(srr_test)
        policy = srr_test

        print(f"Evaluating trained model")
        eval_env = create_env(environment, hf_repo_id, vecnormalize_path)
        mean_reward, std_reward = evaluate_policy(
            srr_test,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
        )

        eval_env.close()

        rewards.append(float(mean_reward))
        if verbose == 2:
            print(f"Iteration {i}: student reward = {mean_reward:.4f} +/- {std_reward:.4f}")

    best_idx = int(np.argmax(rewards))
    best_policy = policies[best_idx]
    best_wrapper = best_policy

    # Save best symbolic policy
    best_policy_path = run_dir / "best_student_policy.joblib"
    if save_results:
        best_wrapper.save(best_policy_path)

        # Save reward history
        save_rewards_csv(rewards, run_dir / "student_rewards.csv")
        save_iteration_summary_csv(rewards, run_dir / "summary.csv")

    # Evaluate teacher
    teacher_eval_env, teacher = load_teacher_env(
        teacher_path,
        teacher_model,
        environment,
        hf_repo_id=hf_repo_id,
        hf_filename=hf_filename,
        hf_algo=hf_algo,
        vecnormalize_path=vecnormalize_path
    )

    teacher_mean_reward, teacher_std_reward = evaluate_policy(
        teacher,
        teacher_eval_env,
        n_eval_episodes=final_n_val_episodes,
        deterministic=True,
    )
    teacher_eval_env.close()

    # Evaluate best student
    student_eval_env = create_env(environment, hf_repo_id, vecnormalize_path)
    student_mean_reward, student_std_reward = evaluate_policy(
        best_wrapper,
        student_eval_env,
        n_eval_episodes=final_n_val_episodes,
        deterministic=True,
    )
    student_eval_env.close()

    # Save final comparison
    if save_results:
        save_final_results_json(
            run_dir / "final_results.json",
            teacher_metrics=(teacher_mean_reward, teacher_std_reward),
            student_metrics=(student_mean_reward, student_std_reward),
            best_iteration=best_idx,
            dataset_size=len(dataset),
        )

        # Optional: save a simple CSV comparison too
        with (run_dir / "teacher_student_comparison.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["policy", "mean_reward", "std_reward"])
            writer.writerow(["teacher", float(teacher_mean_reward), float(teacher_std_reward)])
            writer.writerow(["student_best", float(student_mean_reward), float(student_std_reward)])

    print(f"SPID iteration complete. Dataset size: {len(dataset)}")
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
    vecnormalize_path=None
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
    ):
    # We create a new environment for each step since
    # vectorized stable baseline environments can only be reset once
    env, teacher = load_teacher_env(
        teacher_path,
        teacher_model,
        environment,
        hf_repo_id=hf_repo_id,
        hf_filename=hf_filename,
        hf_algo=hf_algo,
        vecnormalize_path=vecnormalize_path
    )
    policy = policy or teacher

    trajectory = []

    obs = env.reset()
    n_steps = total_timesteps // n_iter
    i = 1

    states = []
    training_actions = []
    teacher_actions = []
    rewards = []
    next_states = []

    # status counter 
    pysr_actions = 0
    oracle_actions = 0

    while len(states) < n_steps:
        
        active_policy = [policy, teacher][np.random.binomial(1, beta)]

        if isinstance(active_policy, PySRPolicy):
            action, _states = active_policy.predict(obs)
            pysr_actions += 1
        else:
            action, _states = active_policy.predict(obs, deterministic=True)
            oracle_actions += 1

        oracle_action, _states = teacher.predict(obs, deterministic=True)

        # Normalize policy action for Gym Pendulum / VecEnv
        # action = np.asarray(action, dtype=np.float32)
        # if action.ndim == 0:
        #     action = action.reshape(1, 1)
        # elif action.ndim == 1:
        #     action = action.reshape(1, -1)

        next_obs, reward, done, _ = env.step(action)
        
        # flatten if needed
        obs = obs.reshape(-1) if obs.ndim > 1 else obs
        reward = reward.reshape(-1) if reward.ndim > 1 else reward
        action = action.reshape(-1) if action.ndim > 1 else action
        oracle_action = oracle_action.reshape(-1) if oracle_action.ndim > 1 else oracle_action

        states.append(obs)
        training_actions.append(action)
        teacher_actions.append(oracle_action)
        rewards.append(reward)
        next_states.append(next_obs)

        obs = next_obs
        i += 1

        if done:
            obs = env.reset()
    
    print(f"finished collecting trajectories")

    # print("\ntraining actions: ")
    # print(np.array(training_actions).shape)
    # print("teacher actions")
    # print(np.array(teacher_actions).shape)
    
    # Note: advantage is training actions, and L2 loss is teacher actions (?) 
    weights = get_advantage_weights(states, training_actions, rewards, next_states, teacher)
    trajectory = [np.array(states), np.array(teacher_actions), weights]

    return trajectory



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

    Parameters
    ----------
    states, actions, rewards, next_states : sequence-like
        Transition data.
    expert : SB3 model
        Trained expert policy.
    gamma : float
        Discount factor.
    device : str | torch.device | None
        Target device. If None, infer from policy unless force_cpu=True.
    force_cpu : bool
        If True, move everything to CPU and run there.

    Returns
    -------
    adv : np.ndarray
        1D array of advantage weights.
    """
    print("computing advantages")

    # Resolve device
    if force_cpu:
        device = torch.device("cpu")
    elif device is None:
        try:
            device = next(expert.policy.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    # Move policy to target device so model/tensors always match
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
        # Build tensors directly on the chosen device
        states_t = to_tensor(np.stack(states), dtype=torch.float32)
        actions_t = ensure_2d_action(to_tensor(np.stack(actions), dtype=torch.float32))
        rewards_t = to_tensor(np.asarray(rewards), dtype=torch.float32).view(-1)
        next_states_t = to_tensor(np.stack(next_states), dtype=torch.float32)

        # On-policy algorithms: PPO, A2C, TRPO...
        if hasattr(expert.policy, "predict_values"):
            v_s = expert.policy.predict_values(states_t).squeeze(-1)
            v_sp = expert.policy.predict_values(next_states_t).squeeze(-1)
            q_sa = rewards_t + gamma * v_sp
            adv_t = q_sa - v_s

        # Off-policy algorithms: SAC, TD3, DDPG, TQC...
        else:
            try:
                if not hasattr(expert.policy, "critic"):
                    raise AttributeError("No critic found on policy")

                algorithm_name = expert.__class__.__name__.lower()

                # ---- SAC / TQC-style critic with q1_forward ----
                if hasattr(expert.policy.critic, "q1_forward"):
                    q_s = expert.policy.critic.q1_forward(states_t, actions_t).squeeze(-1)

                    if "sac" in algorithm_name:
                        try:
                            actor_output = expert.policy.actor(next_states_t)
                            next_actions = actor_output[0] if isinstance(actor_output, tuple) else actor_output
                        except Exception as e1:
                            print(f"SAC actor call method 1 failed: {e1}")
                            try:
                                latent_pi = expert.policy.actor.latent_pi(next_states_t)
                                next_actions = expert.policy.actor.mu(latent_pi)
                                next_actions = torch.tanh(next_actions)
                            except Exception as e2:
                                print(f"SAC actor call method 2 failed: {e2}")
                                # Fallback through expert.predict; returns numpy on CPU
                                next_states_np = next_states_t.detach().cpu().numpy()
                                next_actions_np, _ = expert.predict(next_states_np, deterministic=True)
                                next_actions = to_tensor(next_actions_np, dtype=torch.float32)
                    else:
                        actor_output = expert.policy.actor(next_states_t)
                        next_actions = actor_output[0] if isinstance(actor_output, tuple) else actor_output

                    next_actions = ensure_2d_action(next_actions)
                    next_actions = next_actions.to(device=device, dtype=torch.float32)

                    q_sp = expert.policy.critic.q1_forward(next_states_t, next_actions).squeeze(-1)

                # ---- TD3 / DDPG-style critic forward ----
                elif hasattr(expert.policy.critic, "forward"):
                    q_s = expert.policy.critic(states_t, actions_t).squeeze(-1)

                    next_actions = expert.policy.actor(next_states_t)
                    if isinstance(next_actions, tuple):
                        next_actions = next_actions[0]

                    next_actions = ensure_2d_action(next_actions)
                    next_actions = next_actions.to(device=device, dtype=torch.float32)

                    q_sp = expert.policy.critic(next_states_t, next_actions).squeeze(-1)

                else:
                    raise AttributeError("Critic method not found")

                # Advantage from TD error:
                # A(s,a) ≈ Q(s,a) - [r + gamma * Q(s', pi(s'))]
                target_q = rewards_t + gamma * q_sp
                adv_t = q_s - target_q

                print(f"Successfully computed Q-based advantages for {algorithm_name} on {device}")

            except Exception as e:
                print(f"Warning: Q-network computation failed ({e}), using simplified advantage computation")
                rewards_np = rewards_t.detach().cpu().numpy()
                adv_t = torch.as_tensor(
                    rewards_np - rewards_np.mean(),
                    dtype=torch.float32,
                    device=device,
                )

    adv = adv_t.detach().cpu().numpy()
    adv = np.squeeze(adv)
    return adv
