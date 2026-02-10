# Proximal Policy Optimization (PPO)
# https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html


import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=1)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=10_000)
model.save("ppo_cartpole")

obs = vec_env.reset()
env = gym.make("CartPole-v1")

mean_reward, std_reward = evaluate_policy(model, env)

print(f"==============\n\
Average reward: {mean_reward}\n\
Std reward: {mean_reward}\n\
==============")


while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    #print(rewards)
    vec_env.render("human")
