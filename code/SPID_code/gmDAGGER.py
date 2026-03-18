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
from stable_baselines3 import PPO, DDPG, SAC, TD3
from sb3_contrib import TRPO

from PySRWrapper import PySRWrapper

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from baseline_code.baseline_enviroments.cartpole_env import ContinuousCartPoleEnv


import torch


def make_eval_env():
    return make_vec_env(
        lambda: gym.wrappers.TimeLimit(ContinuousCartPoleEnv(), max_episode_steps=500),
        n_envs=1,
    )


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


def train_spid(teacher_path, 
               teacher_model,
               save_folder_path, 
               environment, 
               n_iter, 
               total_timesteps, 
               verbose=1,
                n_eval_episodes=100):
    
    # print(f"Training SPID on {env_name}")

    dataset = []
    policy = None
    policies = []
    rewards = []

    run_dir = create_run_folder(save_folder_path)

    for i in tqdm(range(n_iter), disable=verbose > 0):
        beta = 1 if i == 0 else 0.5

        dataset += sample_trajectory(teacher_path, 
                                     teacher_model, 
                                     environment, 
                                     total_timesteps, 
                                     n_iter, 
                                     policy, 
                                     beta)
 
        srr = PySRRegressor(binary_operators=["+", "*", "-"], verbosity=0, maxsize=12, run_id=f"")
        x = np.array([traj[0] for traj in dataset])
        y = np.array([traj[1] for traj in dataset])
        # weights = np.array([np.sqrt(score[2]) for score in dataset])

        # srr.fit(x, y, weights=weights)
        srr.fit(x, y)

        policies.append(srr)
        policy = srr

        eval_env = make_eval_env()
        mean_reward, std_reward = evaluate_policy(
            PySRWrapper(policy),
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
    best_wrapper = PySRWrapper(best_policy)

    # Save best symbolic policy
    best_policy_path = run_dir / "best_student_policy.joblib"
    best_wrapper.save(best_policy_path)

    # Save reward history
    save_rewards_csv(rewards, run_dir / "student_rewards.csv")
    save_iteration_summary_csv(rewards, run_dir / "summary.csv")

    # Evaluate teacher
    teacher = teacher_model.load(teacher_path)
    teacher_eval_env = make_eval_env()
    teacher_mean_reward, teacher_std_reward = evaluate_policy(
        teacher,
        teacher_eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
    teacher_eval_env.close()

    # Evaluate best student
    student_eval_env = make_eval_env()
    student_mean_reward, student_std_reward = evaluate_policy(
        best_wrapper,
        student_eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
    student_eval_env.close()

    # Save final comparison
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

    # # TODO: Save best policy
    # print(f"SPID iteration complete. Dataset size: {len(dataset)}")
    # best_policy = policies[np.argmax(rewards)]
    # print(f"Best policy:\t{np.argmax(rewards)}")
    # print(f"Mean reward:\t{np.max(rewards):0.4f}")
    # wrapper = PySRWrapper(best_policy)
    # wrapper.print_info()
    # return rewards, best_policy, wrapper


def load_teacher_env(teacher_path, teacher_model, environment):
    if isinstance(teacher_model, PPO):
        # env = make_vec_env(make_continuous_cartpole, n_envs=1)
        env = make_vec_env(lambda: gym.wrappers.TimeLimit(ContinuousCartPoleEnv(), max_episode_steps=500))
    else: 
        #env = make_vec_env(environment)
        env = make_vec_env(lambda: gym.wrappers.TimeLimit(ContinuousCartPoleEnv(), max_episode_steps=500))
    
    #env = make_vec_env(make_continuous_cartpole, n_envs=1)
    env = make_vec_env(lambda: gym.wrappers.TimeLimit(ContinuousCartPoleEnv(), max_episode_steps=500))
    teacher = teacher_model.load(teacher_path)

    return env, teacher



def sample_trajectory(teacher_path, teacher_model, environment, total_timesteps, n_iter, policy, beta):
    # We create a new environment for each viper step since
    # vectorized stable baseline environments can only be reset once
    env, teacher = load_teacher_env(teacher_path, 
                                   teacher_model, 
                                   environment)
    policy = policy or teacher

    trajectory = []

    obs = env.reset()
    n_steps = total_timesteps // n_iter
    i = 1
    # print(" ===== sampling trajectories =====")
    while len(trajectory) < n_steps:
        # print(f"\niteration {i}")
        
        active_policy = [policy, teacher][np.random.binomial(1, beta)]

        if isinstance(active_policy, PySRRegressor):
            # print("SR policy chosen")
            action = active_policy.predict(obs)
        else:
            # print("Teacher chosen")
            action, _states = active_policy.predict(obs, deterministic=True)
        
        if not isinstance(active_policy, PySRRegressor):
            oracle_action = action
        else:
            oracle_action = teacher.predict(obs, deterministic=True)[0]

        # print(f"Chose action: {action}. Oracle action: {oracle_action}")

        next_obs, reward, done, info = env.step(action)

        # if args.render:
        #     env.render()

        # state_loss = get_loss(env, teacher, obs)
        trajectory += list(zip(obs, oracle_action))

        obs = next_obs
        i += 1

    return trajectory


def get_loss(env, model, obs):
    """
    This is the ~l loss from the paper that tries to capture
    how "critical" a state is, i.e. how much of a difference
    it makes to choose the best vs the worst action

    Instead of training the decision tree with this loss directly (which is not possible because it is not convex)
    we use it as a weight for the samples in the dataset which in expectation leads to the same result
    """
 
    if isinstance(model, DQN) or isinstance(model, DDPG): # For RL algorithms with Q-values 

        # For q-learners it is the difference between the best and worst q value
        q_values = model.q_net(torch.from_numpy(obs)).detach().numpy()
        # q_values n_env x n_actions
        return q_values.max(axis=1) - q_values.min(axis=1)
    
    if isinstance(model, PPO): # For RL algorithms without Q-values 

        # For policy gradient methods we use the max entropy formulation
        # to get Q(s, a) \approx log pi(a|s)
        # See Ziebart et al. 2008
        # assert isinstance(env.action_space,
        #                   gym.spaces.Discrete), "Only discrete action spaces supported for loss function"
        # possible_actions = np.arange(env.action_space.n)

        possible_actions = np.arange(2)

        obs = torch.from_numpy(obs).to("cuda")
        log_probs = []
        for action in possible_actions:
            action = torch.from_numpy(np.array([action])).repeat(obs.shape[0]).to("cuda")
            _, log_prob, _ = model.policy.evaluate_actions(obs, action)
            log_probs.append(log_prob.cpu().detach().numpy().flatten())

        log_probs = np.array(log_probs).T
        return log_probs.max(axis=1) - log_probs.min(axis=1)

    raise NotImplementedError(f"Model type {type(model)} not supported")