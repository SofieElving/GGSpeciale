import pandas as pd
from stable_baselines3 import PPO, DDPG, SAC, TD3
from sb3_contrib import TRPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize
import sys
import os
from pathlib import Path
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise

sys.path.append(os.path.abspath(".."))

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from baseline_enviroments.acrobot_env import make_continuous_acrobot

model_saves_folder = PROJECT_ROOT / "baseline_models" / "acrobot"
model_saves_folder.mkdir(parents=True, exist_ok=True)


def make_action_noise(n_actions: int, sigma: float = 0.2):
    return NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=sigma * np.ones(n_actions),
    )


methods = {
    "PPO": {
        "model": PPO,
        "n_envs": 16,
        "args": {
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "clip_range": 0.2,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
    },
    "DDPG": {
        "model": DDPG,
        "n_envs": 16,
        "args": {
            "learning_rate": 1e-3,
            "buffer_size": 200_000,
            "learning_starts": 10_000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": (1, "step"),
            "gradient_steps": 1,
        },
        "needs_action_noise": True,
        "noise_sigma": 0.2,
    },
    "TRPO": {
        "model": TRPO,
        "n_envs": 16,
        "args": {
            "learning_rate": 1e-3,
            "n_steps": 1024,
            "batch_size": 128,
            "gamma": 0.99,
            "gae_lambda": 0.98,
            "cg_max_steps": 15,
            "cg_damping": 0.1,
            "line_search_shrinking_factor": 0.8,
            "n_critic_updates": 10,
        },
    },
    "SAC": {
        "model": SAC,
        "n_envs": 16,
        "args": {
            "learning_rate": 3e-4,
            "buffer_size": 200_000,
            "learning_starts": 10_000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": (1, "step"),
            "gradient_steps": 1,
            "ent_coef": "auto",
        },
    },
    "TD3": {
        "model": TD3,
        "n_envs": 16,
        "args": {
            "learning_rate": 1e-3,
            "buffer_size": 200_000,
            "learning_starts": 10_000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": (1, "step"),
            "gradient_steps": 1,
            "policy_delay": 2,
        },
        "needs_action_noise": True,
        "noise_sigma": 0.2,
    },
}

ENV_FACTORY = make_continuous_acrobot
ENV_NAME = "ContinuousAcrobot-v1"
TOTAL_TIMESTEPS = 1_000_000

results = []

for method_name, spec in methods.items():
    print(f"\nRunning {method_name} on {ENV_NAME}")

    n_envs = spec.get("n_envs", 1)

    # Base vectorized env
    train_env = make_vec_env(ENV_FACTORY, n_envs=n_envs, seed=0)

    # Normalize observations and rewards during training
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=spec["args"].get("gamma", 0.99),
    )

    algo_kwargs = dict(spec["args"])

    if spec.get("needs_action_noise", False):
        temp_env = ENV_FACTORY()
        n_actions = temp_env.action_space.shape[-1]
        temp_env.close()
        algo_kwargs["action_noise"] = make_action_noise(
            n_actions,
            sigma=spec.get("noise_sigma", 0.2),
        )

    Algo = spec["model"]

    model = Algo(
        "MlpPolicy",
        train_env,
        verbose=1,
        device="cuda",
        **algo_kwargs,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # Save model and normalization statistics separately
    model_path = model_saves_folder / f"{method_name}_acrobot"
    vecnorm_path = model_saves_folder / f"{method_name}_acrobot_vecnormalize.pkl"

    model.save(str(model_path))
    train_env.save(str(vecnorm_path))

    # Fresh eval env
    eval_env = make_vec_env(ENV_FACTORY, n_envs=1, seed=0)

    # Load normalization stats from training
    eval_env = VecNormalize.load(str(vecnorm_path), eval_env)

    # Freeze normalization during evaluation
    eval_env.training = False
    eval_env.norm_reward = False

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=50,
        deterministic=True,
    )

    print(
        "==============\n"
        f"Average reward: {mean_reward}\n"
        f"Std reward: {std_reward}\n"
        "=============="
    )

    results.append(
        {
            "algorithm": method_name,
            "env": ENV_NAME,
            "mean reward": mean_reward,
            "std reward": std_reward,
        }
    )

    train_env.close()
    eval_env.close()

results = pd.DataFrame(results)
print(results)