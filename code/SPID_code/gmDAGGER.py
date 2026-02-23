import warnings
import gymnasium as gym
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from pysr import PySRRegressor 
from PySRWrapper import PySRWrapper

from tqdm import tqdm

#from gym_env import make_env
#from model.paths import get_oracle_path, get_viper_path
#from model.tree_wrapper import TreeWrapper
#from test.evaluate import evaluate_policy
#from train.oracle import get_model_cls
from stable_baselines3.common.evaluation import evaluate_policy


def train_spid(teacher_path, 
               teacher_model, 
               environment, 
               n_iter, 
               total_timesteps, 
               verbose=False):
    
    #print(f"Training SPID on {env_name}")

    dataset = []
    policy = None
    policies = []
    rewards = []

    for i in tqdm(range(n_iter), disable=verbose > 0):
        beta = 1 if i == 0 else 0

        # TODO: adjust sample_trajectory function
        dataset += sample_trajectory(teacher_path, 
                                     teacher_model, 
                                     environment, 
                                     total_timesteps, 
                                     n_iter, 
                                     policy, 
                                     beta)

        # TODO: implement PySR here 
        srr = PySRRegressor()
        x = np.array([traj[0] for traj in dataset])
        y = np.array([traj[1] for traj in dataset])
        weight = np.array([traj[2] for traj in dataset])

        # TODO: define loss function 
        srr.fit(x, y, weights=weight)

        policies.append(srr)
        policy = srr


        # TODO: implement policy evaluation - for this, make a wrapper 
        env = make_vec_env(environment)
        mean_reward, std_reward = evaluate_policy(PySRWrapper(policy), env, n_eval_episodes=100)
        if args.verbose == 2:
            print(f"Policy score: {mean_reward:0.4f} +/- {std_reward:0.4f}")
        rewards.append(mean_reward)

    # TODO: Save best policy
    print(f"Viper iteration complete. Dataset size: {len(dataset)}")
    best_policy = policies[np.argmax(rewards)]
    print(f"Best policy:\t{np.argmax(rewards)}")
    print(f"Mean reward:\t{np.max(rewards):0.4f}")
    wrapper = PySRWrapper(best_policy)
    wrapper.print_info()


def load_teacher_env(teacher_path, teacher_model, environment):
    # TODO: load teacher environment 

    # Load correct environment for model 
    # Load model settings - with arguments. 

    env = make_vec_env(environment)
    teacher = teacher_model.load(teacher_path, env=env)

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
    while len(trajectory) < n_steps:
        active_policy = [policy, teacher][np.random.binomial(1, beta)]

        # TODO: replace with PySR classifier 
        if isinstance(active_policy, DecisionTreeClassifier):
            action = active_policy.predict(obs)
        else:
            action, _states = active_policy.predict(obs, deterministic=True)
        
        # TODO: replace with PySR classifier 
        if not isinstance(active_policy, DecisionTreeClassifier):
            oracle_action = action
        else:
            oracle_action = teacher.predict(obs, deterministic=True)[0]

        next_obs, reward, done, info = env.step(action)

        # if args.render:
        #     env.render()

        state_loss = get_loss(env, teacher, obs)
        trajectory += list(zip(obs, oracle_action, state_loss))

        obs = next_obs

    return trajectory


def get_loss(env, model: BaseAlgorithm, obs):
    """
    This is the ~l loss from the paper that tries to capture
    how "critical" a state is, i.e. how much of a difference
    it makes to choose the best vs the worst action

    Instead of training the decision tree with this loss directly (which is not possible because it is not convex)
    we use it as a weight for the samples in the dataset which in expectation leads to the same result
    """
    # TODO: implement GM loss


    if isinstance(model, DQN): # For RL algorithms with Q-values 

        # For q-learners it is the difference between the best and worst q value
        q_values = model.q_net(torch.from_numpy(obs)).detach().numpy()
        # q_values n_env x n_actions
        return q_values.max(axis=1) - q_values.min(axis=1)
    
    if isinstance(model, PPO): # For RL algorithms without Q-values 

        # For policy gradient methods we use the max entropy formulation
        # to get Q(s, a) \approx log pi(a|s)
        # See Ziebart et al. 2008
        assert isinstance(env.action_space,
                          gym.spaces.Discrete), "Only discrete action spaces supported for loss function"
        possible_actions = np.arange(env.action_space.n)

        obs = torch.from_numpy(obs)
        log_probs = []
        for action in possible_actions:
            action = torch.from_numpy(np.array([action])).repeat(obs.shape[0])
            _, log_prob, _ = model.policy.evaluate_actions(obs, action)
            log_probs.append(log_prob.detach().numpy().flatten())

        log_probs = np.array(log_probs).T
        return log_probs.max(axis=1) - log_probs.min(axis=1)

    raise NotImplementedError(f"Model type {type(model)} not supported")