import pandas as pd
from stable_baselines3 import PPO, DDPG, SAC, TD3
from sb3_contrib import TRPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import sys
import os

sys.path.append(os.path.abspath(".."))

from baseline_enviroments.MountainCar import make_continuous_mountaincar


methods = {
    "PPO": {
        "model": PPO,
        "args": { # args inspired by https://arxiv.org/pdf/1707.06347
            "learning_rate": 3e-4,      # Gradient step size (smaller = more stable learning)
            "n_steps": 2048,            # Rollout length before each update (long horizon credit)
            "batch_size": 64,           # Minibatch size for optimization (adds useful gradient noise)
            "gamma": 0.999,             # Discount factor (keeps distant +100 reward relevant)
            "gae_lambda": 0.95,         # Bias-variance tradeoff in advantage estimation
            "clip_range": 0.2,          # PPO trust-region style update constraint
            "ent_coef": 0.01,           # Encourages exploration (prevents zero-action collapse)
            "vf_coef": 0.5,             # Weight of value function loss
            "max_grad_norm": 0.5,       # Gradient clipping for stability
            },
    },
    "DDPG": {
        "model": DDPG,
        "args": {
            "learning_rate": 1e-3,
            "buffer_size": 200_000,
            "learning_starts": 5_000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": (1, "step"),
            "gradient_steps": 1,
        },
    },
    "TRPO": {
        "model": TRPO,
        "args": {"gamma": 0.99},
    },
    "SAC": {
        "model": SAC,
        "args": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "buffer_size": 200_000,
            "learning_starts": 5_000,
            "batch_size": 256,
            "tau": 0.005,
        },
    },
    "TD3": {
        "model": TD3,
        "args": {
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "buffer_size": 200_000,
            "learning_starts": 5_000,
            "batch_size": 256,
            "tau": 0.005,
            "train_freq": (1, "step"),
            "gradient_steps": 1,
            # TD3 has its own target policy smoothing internally; still benefits from exploration noise
            # action_noise injected below
        },
    }
}

ENV_FACTORY = make_continuous_mountaincar
ENV_NAME = "MountainCar"  

TOTAL_TIMESTEPS = 25_000
results = []

for method_name, spec in methods.items():
    print(f"\nRunning {method_name} on {ENV_NAME}")

    train_env = make_vec_env(ENV_FACTORY, n_envs=1)

    Algo = spec["model"]
    model = Algo("MlpPolicy", train_env, verbose=0, **spec["args"])
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    eval_env = make_vec_env(ENV_FACTORY, n_envs=1)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

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
model.save(f"baseline_models/{method_name}_{ENV_NAME}")