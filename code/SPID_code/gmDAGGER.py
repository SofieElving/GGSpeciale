import warnings
from pysr import PySRRegressor 
import gymnasium as gym
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier
from stable_baselines3 import DQN, PPO, DDPG
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from SPID_code.PySRWrapper import PySRWrapper

from baseline_code.baseline_enviroments.cartpole_env import ContinuousCartPoleEnv

from tqdm import tqdm

#from gym_env import make_env
#from model.paths import get_oracle_path, get_viper_path
#from model.tree_wrapper import TreeWrapper
#from test.evaluate import evaluate_policy
#from train.oracle import get_model_cls
from stable_baselines3.common.evaluation import evaluate_policy


def train_spid(teacher_path, 
               teacher_model,
               save_path, 
               environment, 
               n_iter, 
               total_timesteps, 
               verbose=1):
    
    # print(f"Training SPID on {env_name}")

    dataset = []
    policy = None
    policies = []
    rewards = []

    for i in tqdm(range(n_iter), disable=verbose > 0):
        beta = 1 if i == 0 else 0.5

        dataset += sample_trajectory(teacher_path, 
                                     teacher_model, 
                                     environment, 
                                     total_timesteps, 
                                     n_iter, 
                                     policy, 
                                     beta)
 
        srr = PySRRegressor(binary_operators=["+", "*", "-"], verbosity=0, maxsize=12)
        x = np.array([traj[0] for traj in dataset])
        y = np.array([traj[1] for traj in dataset])
        # weights = np.array([np.sqrt(score[2]) for score in dataset])

        # srr.fit(x, y, weights=weights)
        srr.fit(x, y)

        policies.append(srr)
        policy = srr

        if isinstance(teacher_model, PPO):
            # env = make_vec_env(make_continuous_cartpole, n_envs=1)
            env = make_vec_env(lambda: gym.wrappers.TimeLimit(ContinuousCartPoleEnv(), max_episode_steps=500))
        else: 
            #env = make_vec_env(environment)
            env = make_vec_env(lambda: gym.wrappers.TimeLimit(ContinuousCartPoleEnv(), max_episode_steps=500))
        
        #env = make_vec_env(make_continuous_cartpole, n_envs=1)

        # mean_reward, std_reward = evaluate_policy(PySRWrapper(policy), env, n_eval_episodes=100)
        mean_reward, std_reward = evaluate_policy(
            PySRWrapper(policy),
            env,
            n_eval_episodes=100
        )
        if verbose == 2:
            print(f"Policy score: {mean_reward:0.4f} +/- {std_reward:0.4f}")
        rewards.append(mean_reward)

    # TODO: Save best policy
    print(f"SPID iteration complete. Dataset size: {len(dataset)}")
    best_policy = policies[np.argmax(rewards)]
    print(f"Best policy:\t{np.argmax(rewards)}")
    print(f"Mean reward:\t{np.max(rewards):0.4f}")
    wrapper = PySRWrapper(best_policy)
    wrapper.print_info()
    return rewards, best_policy, wrapper


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
    print(" ===== sampling trajectories =====")
    while len(trajectory) < n_steps:
        print(f"\niteration {i}")
        
        active_policy = [policy, teacher][np.random.binomial(1, beta)]

        if isinstance(active_policy, PySRRegressor):
            print("SR policy chosen")
            action = active_policy.predict(obs)
        else:
            print("Teacher chosen")
            action, _states = active_policy.predict(obs, deterministic=True)
        
        if not isinstance(active_policy, PySRRegressor):
            oracle_action = action
        else:
            oracle_action = teacher.predict(obs, deterministic=True)[0]

        print(f"Chose action: {action}. Oracle action: {oracle_action}")

        next_obs, reward, done, info = env.step(action)

        # if args.render:
        #     env.render()

        # state_loss = get_loss(env, teacher, obs)
        trajectory += list(zip(obs, oracle_action))

        obs = next_obs
        i += 1

    return trajectory


def get_loss(env, model: BaseAlgorithm, obs):
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