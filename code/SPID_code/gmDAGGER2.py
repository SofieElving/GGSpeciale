import warnings
from pysr import PySRRegressor
import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import DQN, PPO, DDPG
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from SPID_code.PySRWrapper import PySRWrapper
from baseline_code.baseline_enviroments.cartpole_env import ContinuousCartPoleEnv
from baseline_code.baseline_enviroments.MountainCar import make_continuous_mountaincar

from tqdm import tqdm


def make_env_fn(environment: str):
    env_key = environment.lower()

    if env_key in {"cartpole", "cartpole-v1"}:
        return lambda: gym.wrappers.TimeLimit(
            ContinuousCartPoleEnv(),
            max_episode_steps=500,
        )

    if env_key in {"mountaincarcontinuous-v0", "mountaincarcontinuous", "mountaincar"}:
        return lambda: make_continuous_mountaincar()

    raise ValueError(f"Unsupported environment: {environment}")

def format_action_for_env(action, env):
    action = np.asarray(action, dtype=np.float32)

    # VecEnv wants shape (n_envs, action_dim)
    if action.ndim == 0:
        action = action.reshape(1, 1)
    elif action.ndim == 1:
        action = action.reshape(1, -1)

    # clip to valid bounds
    low = env.action_space.low.reshape(1, -1)
    high = env.action_space.high.reshape(1, -1)
    action = np.clip(action, low, high)

    return action

def train_spid(
    teacher_path,
    teacher_model,
    save_path,
    environment,
    n_iter,
    total_timesteps,
    verbose=1,
):
    dataset = []
    policy = None
    policies = []
    rewards = []

    for i in tqdm(range(n_iter), disable=verbose > 0):
        beta = 1 if i == 0 else 0.5

        dataset += sample_trajectory(
            teacher_path,
            teacher_model,
            environment,
            total_timesteps,
            n_iter,
            policy,
            beta,
        )

        srr = PySRRegressor(
            binary_operators=["+", "*", "-"],
            verbosity=0,
            maxsize=12,
        )

        x = np.array([traj[0] for traj in dataset])
        y = np.array([traj[1] for traj in dataset])
        srr.fit(x, y)

        policies.append(srr)
        policy = srr

        env = make_vec_env(make_env_fn(environment), n_envs=1)

        mean_reward, std_reward = evaluate_policy(
            PySRWrapper(policy),
            env,
            n_eval_episodes=100,
        )

        if verbose == 2:
            print(f"Policy score: {mean_reward:0.4f} +/- {std_reward:0.4f}")

        rewards.append(mean_reward)

    print(f"SPID iteration complete. Dataset size: {len(dataset)}")
    best_policy = policies[np.argmax(rewards)]
    print(f"Best policy:\t{np.argmax(rewards)}")
    print(f"Mean reward:\t{np.max(rewards):0.4f}")

    wrapper = PySRWrapper(best_policy)
    wrapper.print_info()
    return rewards, best_policy, wrapper


def load_teacher_env(teacher_path, teacher_model, environment):
    env = make_vec_env(make_env_fn(environment), n_envs=1)
    teacher = teacher_model.load(teacher_path, env=env, device="cpu")
    return env, teacher


def sample_trajectory(teacher_path, teacher_model, environment, total_timesteps, n_iter, policy, beta):
    env, teacher = load_teacher_env(teacher_path, teacher_model, environment)
    policy = policy or teacher

    trajectory = []
    obs = env.reset()
    n_steps = total_timesteps // n_iter
    i = 1

    print(" ===== sampling trajectories =====")
    while len(trajectory) < n_steps:
        print(f"\niteration {i}")

        active_policy = [policy, teacher][np.random.binomial(1, beta)]

        if isinstance(active_policy, PySRRegressor):
            print("SR policy chosen")
            action = active_policy.predict(obs)
            action = format_action_for_env(action, env)
        else:
            print("Teacher chosen")
            action, _states = active_policy.predict(obs, deterministic=True)

        if not isinstance(active_policy, PySRRegressor):
            oracle_action = action
        else:
            oracle_action = teacher.predict(obs, deterministic=True)[0]

        print(f"Chose action: {action}. Oracle action: {oracle_action}")

        next_obs, reward, done, info = env.step(action)
        trajectory += list(zip(obs, oracle_action))

        obs = next_obs
        i += 1

    return trajectory