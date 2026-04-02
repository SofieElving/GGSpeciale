# expert_path = f"/home/peilang/github_repo/distillation/trained_model/{args.env_name}/{args.expert_algo.lower()}/{args.expert_algo.lower()}_{env.__class__.__name__}_model"
import argparse
import json
from pysr import PySRRegressor
import torch, numpy as np, gym
from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG
from sb3_contrib import TRPO, TQC, ARS, CrossQ
from stable_baselines3.common.evaluation import evaluate_policy

# Import environments
from env.cartpole2 import ContinuousCartPoleEnv
from env.mountain_car import Continuous_MountainCarEnv
from env.pendulum import PendulumEnv
from env.hopper import HopperEnv
from env.walker2d import Walker2dEnv
from env.swimmer import SwimmerEnv
from env.reacher import ReacherEnv
from env.inverted_double_pendulum import InvertedDoublePendulumEnv
from env.lunar_lander import LunarLanderContinuous
from env.bipedalwalker import BipedalWalker
from env.acrobot import AcrobotEnv


def load_expert_model(model_path, env, algorithm="ppo"):
    """Load the pre-trained expert model"""
    if algorithm.lower() == "ppo":
        return PPO.load(model_path, env=env, device="cpu")
    elif algorithm.lower() == "sac":
        return SAC.load(model_path, env=env, device="cpu")
    elif algorithm.lower() == "td3":
        return TD3.load(model_path, env=env, device="cpu")
    elif algorithm.lower() == "a2c":
        return A2C.load(model_path, env=env, device="cpu")
    elif algorithm.lower() == "ddpg":
        return DDPG.load(model_path, env=env, device="cpu")
    elif algorithm.lower() == "trpo":
        return TRPO.load(model_path, env=env, device="cpu")
    elif algorithm.lower() == "tqc":
        return TQC.load(model_path, env=env, device="cpu")
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def compute_advantage(states, actions, rewards, next_states, expert, gamma=0.99):
    """
    Compute advantages for both on-policy and off-policy algorithms
    states, actions, rewards, next_states: numpy arrays of same length (batch, ...)
    returns: numpy array of advantages A(s,a)
    """
    with torch.no_grad():
        states_t = torch.tensor(states, dtype=torch.float32)
        next_t = torch.tensor(next_states, dtype=torch.float32)
        # Ensure actions have proper shape - should be (batch_size, action_dim)
        if len(actions.shape) == 1:
            actions_t = torch.tensor(actions, dtype=torch.float32).unsqueeze(-1)
        else:
            actions_t = torch.tensor(actions, dtype=torch.float32)

        # Check if the policy has predict_values (on-policy algorithms)
        if hasattr(expert.policy, 'predict_values'):
            # On-policy algorithms (PPO, A2C, TRPO)
            v_s = expert.policy.predict_values(states_t).squeeze(-1).numpy()
            v_sp = expert.policy.predict_values(next_t).squeeze(-1).numpy()
            q_sa = rewards + gamma * v_sp
            adv = q_sa - v_s
        else:
            # Off-policy algorithms (SAC, TD3, DDPG, TQC)
            try:
                # Use Q-values to approximate advantage
                if hasattr(expert.policy, 'critic'):
                    # Determine algorithm type and handle actor calls appropriately
                    algorithm_name = expert.__class__.__name__.lower()
                    
                    # Get current Q-values
                    if hasattr(expert.policy.critic, 'q1_forward'):
                        # SAC has q1_forward method
                        q_s = expert.policy.critic.q1_forward(states_t, actions_t).squeeze(-1).numpy()
                        
                        # Get next actions from policy - handle SAC's variable output
                        if 'sac' in algorithm_name:
                            # SAC's actor can return different outputs depending on context
                            # Try different approaches to get the action
                            try:
                                # First try: actor might return (action, log_prob)
                                actor_output = expert.policy.actor(next_t)
                                if isinstance(actor_output, tuple):
                                    next_actions, _ = actor_output
                                else:
                                    next_actions = actor_output
                            except Exception as e:
                                print(f"SAC actor call method 1 failed: {e}")
                                try:
                                    # Second try: use the action_net directly
                                    latent_pi = expert.policy.actor.latent_pi(next_t)
                                    next_actions = expert.policy.actor.mu(latent_pi)
                                    # Apply tanh squashing like SAC does
                                    next_actions = torch.tanh(next_actions)
                                except Exception as e2:
                                    print(f"SAC actor call method 2 failed: {e2}")
                                    # Fallback: use expert.predict to get actions
                                    next_actions_np, _ = expert.predict(next_t.numpy(), deterministic=True)
                                    next_actions = torch.tensor(next_actions_np, dtype=torch.float32)
                        else:
                            # Other algorithms with q1_forward (like TQC)
                            actor_output = expert.policy.actor(next_t)
                            if isinstance(actor_output, tuple):
                                next_actions, _ = actor_output
                            else:
                                next_actions = actor_output
                        
                        # Ensure proper shape
                        if len(next_actions.shape) == 1:
                            next_actions = next_actions.unsqueeze(-1)
                        q_sp = expert.policy.critic.q1_forward(next_t, next_actions).squeeze(-1).numpy()
                        
                    elif hasattr(expert.policy.critic, 'forward'):
                        # TD3, DDPG use forward method
                        q_s = expert.policy.critic.forward(states_t, actions_t).squeeze(-1).numpy()
                        # Get next actions from policy (TD3/DDPG returns only action)
                        next_actions = expert.policy.actor(next_t)
                        if len(next_actions.shape) == 1:
                            next_actions = next_actions.unsqueeze(-1)
                        q_sp = expert.policy.critic.forward(next_t, next_actions).squeeze(-1).numpy()
                    else:
                        raise AttributeError("Critic method not found")
                    
                    # Compute advantage as TD error: Q(s,a) - (r + γ * Q(s',π(s')))
                    target_q = rewards + gamma * q_sp
                    adv = q_s - target_q
                    print(f"Successfully computed Q-based advantages for {algorithm_name}")
                else:
                    raise AttributeError("No critic found")
                    
            except Exception as e:
                # More robust fallback for off-policy algorithms
                print(f"Warning: Q-network computation failed ({str(e)}), using simplified advantage computation")
                # Use a simple advantage approximation based on rewards
                # This is a reasonable fallback since we're doing imitation learning
                adv = rewards - np.mean(rewards)  # Center rewards around 0
                
        print(adv)
                
    return adv

def collect_rollout(env, expert, n_steps=3000, policy=None, mix_alpha=1.0, gamma=0.99):
    """
    policy: callable(state) -> action
    mix_alpha=1 → pure expert, 0 → pure policy
    returns obs, acts, advs arrays
    
    If actions are out of bounds, they are clipped to the environment's action space
    but the original (unclipped) actions are stored in the training buffer.
    """
    obs_buf, act_buf, rew_buf, next_buf = [], [], [], []
    obs, _ = env.reset()
    
    # Get action space bounds for clipping
    if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
        action_low = env.action_space.low
        action_high = env.action_space.high
        has_bounds = True
        print(f"Action space bounds: [{action_low}, {action_high}]")
    else:
        has_bounds = False
        print("Action space has no explicit bounds")
    
    clipped_actions_count = 0
    
    for step in range(n_steps):
        if policy is None or np.random.rand() < mix_alpha:
            # Use expert policy
            act, _ = expert.predict(obs, deterministic=True)
            original_act = act.copy()
        else:
            # Use symbolic policy
            act = policy(obs)
            act = act.astype(np.float32)  # ensure correct type (match expert's action type)
            original_act = act.copy()
        
        # Store the original (potentially out-of-bounds) action for training
        # This is important because we want the symbolic model to learn the correct actions
        training_act = original_act.copy()
        
        # Clip action for environment execution if bounds exist
        if has_bounds:
            act_clipped = np.clip(act, action_low, action_high)
            if not np.array_equal(act, act_clipped):
                clipped_actions_count += 1
                if clipped_actions_count <= 5:  # Only print first few occurrences
                    print(f"Step {step}: Action {act} clipped to {act_clipped}")
                elif clipped_actions_count == 6:
                    print("... (further clipping messages suppressed)")
            act = act_clipped
        
        # Execute the (potentially clipped) action in the environment
        try:
            next_obs, rew, done, _, _ = env.step(act)
        except AssertionError as e:
            # If action is still invalid, print error and use expert action as fallback
            print(f"Action still invalid after clipping: {act}, using expert action as fallback")
            expert_act, _ = expert.predict(obs, deterministic=True)
            next_obs, rew, done, _, _ = env.step(expert_act)
            # Keep the original symbolic action for training, but note the issue
            print(f"Original symbolic action: {training_act}, Expert fallback: {expert_act}")

        # Store data for training (use original action, not clipped)
        obs_buf.append(obs)
        act_buf.append(training_act)  # Store original action for training
        rew_buf.append(rew)
        next_buf.append(next_obs)

        obs = next_obs
        if done:
            obs, _ = env.reset()
    
    if clipped_actions_count > 0:
        print(f"Total actions clipped: {clipped_actions_count}/{n_steps} ({100*clipped_actions_count/n_steps:.1f}%)")

    obs = np.array(obs_buf)
    acts = np.array(act_buf)  # Keep original shape - don't reshape here!
    advs = compute_advantage(obs, acts, np.array(rew_buf), np.array(next_buf), expert, gamma)
    return obs, acts, advs


def train_symbolic(obs, acts, advs, 
                  # PySR core parameters
                  populations=64, maxsize=15, niterations=100,
                  # Operators
                  binary_operators=None, unary_operators=None,
                  # Constraints
                  nested_constraints=None, complexity_of_operators=None,
                  # Loss and optimization
                  loss="L2DistLoss()", weights=None, optimize_hof=True,
                  # Parsimony and complexity
                  parsimony=0.0032, alpha=0.1, 
                  # Tournament and mutation
                  tournament_selection_n=10, tournament_selection_p=0.86,
                  topn=12, fraction_replaced=0.000364, fraction_replaced_hof=0.035,
                  # Mutation probabilities
                  weight_add_node=0.79, weight_insert_node=5.1, weight_delete_node=1.7,
                  weight_do_nothing=0.21, weight_mutate_constant=0.048, weight_mutate_operator=0.47,
                  weight_randomize=0.00023, weight_simplify=0.0020,
                  # Advanced options
                  crossover_probability=0.066, annealing=False, batching=False,
                  batch_size=50, fast_cycle=False, turbo=False, precision=32,
                  # Early stopping and timeouts
                  early_stop_condition=None, timeout_in_seconds=None,
                  # Selection and evaluation
                  ncycles_per_iteration=550, fraction_replaced_hof_params=None,
                  # Additional parameters
                  should_optimize_constants=True, warmup_maxsize_by=0.0,
                  **kwargs):
    """
    Train symbolic regression model with comprehensive hyperparameters
    """
    
    # Set default operators if not provided
    if binary_operators is None:
        binary_operators = ["+", "*", "-", "/"]
    if unary_operators is None:
        unary_operators = ["cos", "sin", "exp", "square", "sqrt", "log"]
    
    # Set default nested constraints if not provided
    if nested_constraints is None:
        nested_constraints = {
            "square": {"square": 1, "exp": 0},
            "exp": {"square": 1, "exp": 0, "log": 0},
            "log": {"exp": 0, "log": 0},
            "sqrt": {"sqrt": 1, "square": 0}
        }
    
    # Set default complexity constraints if not provided
    if complexity_of_operators is None:
        complexity_of_operators = {
            "+": 1, "-": 1, "*": 1, "/": 2,
            "cos": 2, "sin": 2, "exp": 3, "log": 3,
            "square": 1, "sqrt": 2
        }
    
    # Handle multi-dimensional actions - train separate models for each dimension
    action_dim = acts.shape[1] if len(acts.shape) > 1 else 1
    models = []
    
    for dim in range(action_dim):
        print(f"Training symbolic model for action dimension {dim+1}/{action_dim}")
        
        # Extract actions for this dimension
        if action_dim == 1:
            acts_dim = acts.flatten() if len(acts.shape) > 1 else acts
        else:
            acts_dim = acts[:, dim]
        
        # Prepare loss function and weights
        if loss == "weighted_L2" or "weighted" in loss.lower():
            loss_str = "loss(pred, target, w) = w .* (pred .- target).^2"
            # Prepare weights using advantage values (matching original implementation)
            if weights is None:
                weights_dim = np.abs(advs)
                weights_dim = weights_dim / np.max(weights_dim) if np.max(weights_dim) > 0 else weights_dim
            else:
                weights_dim = weights
        else:
            loss_str = "loss(prediction, target) = (prediction - target)^2"
            # For non-weighted losses, don't use advantage-based weights
            weights_dim = None
        
        model = PySRRegressor(
            populations=populations,
            maxsize=maxsize,
            niterations=niterations,
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            nested_constraints=nested_constraints,
            complexity_of_operators=complexity_of_operators,
            elementwise_loss=loss_str,
            parsimony=parsimony,
            alpha=alpha,
            tournament_selection_n=tournament_selection_n,
            tournament_selection_p=tournament_selection_p,
            topn=topn,
            fraction_replaced=fraction_replaced,
            fraction_replaced_hof=fraction_replaced_hof,
            weight_add_node=weight_add_node,
            weight_insert_node=weight_insert_node,
            weight_delete_node=weight_delete_node,
            weight_do_nothing=weight_do_nothing,
            weight_mutate_constant=weight_mutate_constant,
            weight_mutate_operator=weight_mutate_operator,
            weight_randomize=weight_randomize,
            weight_simplify=weight_simplify,
            crossover_probability=crossover_probability,
            annealing=annealing,
            batching=batching,
            batch_size=batch_size,
            fast_cycle=fast_cycle,
            turbo=turbo,
            precision=precision,
            early_stop_condition=early_stop_condition,
            timeout_in_seconds=timeout_in_seconds,
            ncycles_per_iteration=ncycles_per_iteration,
            should_optimize_constants=should_optimize_constants,
            warmup_maxsize_by=warmup_maxsize_by,
            verbosity=0,
            **kwargs
        )
        
        model.fit(obs, acts_dim, weights=weights_dim)
        models.append(model)
    
    # If single dimension, return the model directly; otherwise return list
    return models[0] if action_dim == 1 else models


def create_symbolic_policy(symbolic_model):
    """Create a callable wrapper around PySR expression(s)"""
    def symbolic_policy(state):
        if isinstance(symbolic_model, list):
            # Multi-dimensional action space
            actions = []
            for model in symbolic_model:
                action_dim = model.predict(state.reshape(1, -1)).item()
                actions.append(action_dim)
            return np.array(actions)
        else:
            # Single-dimensional action space
            action = symbolic_model.predict(state.reshape(1, -1))
            if np.isscalar(action):
                return np.array([action])
            elif hasattr(action, 'item'):
                return np.array([action.item()])
            else:
                return np.array([action[0]]) if len(action) > 0 else np.array([0.0])
    return symbolic_policy


def evaluate_symbolic_policy(env, symbolic_policy, n_eval=10):
    """Evaluate the symbolic policy"""
    returns = []
    
    # Get action space bounds for clipping (same as in collect_rollout)
    if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
        action_low = env.action_space.low
        action_high = env.action_space.high
        has_bounds = True
        print(f"Action space bounds for evaluation: [{action_low}, {action_high}]")
    else:
        has_bounds = False
        print("Action space has no explicit bounds for evaluation")
    
    clipped_actions_count = 0
    total_actions = 0
    
    for _ in range(n_eval):
        obs, _ = env.reset()
        tot = 0
        episode_steps = 0
        while True:
            obs = obs.reshape(1, -1)
            act = symbolic_policy(obs)
            
            # Ensure action is in the correct format
            if isinstance(act, (list, tuple)):
                act = np.array(act).astype(np.float32)
            elif np.isscalar(act):
                act = np.array([act]).astype(np.float32)
            else:
                act = np.array(act).astype(np.float32)
                
            # Ensure act is at least 1D
            if act.ndim == 0:
                act = np.array([act.item()]).astype(np.float32)
            
            # Store original action for debugging
            original_act = act.copy()
            
            # Clip action to environment bounds if they exist
            if has_bounds:
                act_clipped = np.clip(act, action_low, action_high)
                if not np.array_equal(act, act_clipped):
                    clipped_actions_count += 1
                    if clipped_actions_count <= 3:  # Only print first few occurrences
                        print(f"Evaluation: Action {act} clipped to {act_clipped}")
                act = act_clipped
            
            total_actions += 1
            
            try:
                obs, rew, terminated, truncated, _ = env.step(act)
            except AssertionError as e:
                print(f"Action still invalid after clipping: {act} (original: {original_act})")
                print(f"Error: {e}")
                # Use a safe fallback action (center of action space)
                if has_bounds:
                    safe_action = (action_low + action_high) / 2
                else:
                    safe_action = np.array([0.0])
                print(f"Using safe fallback action: {safe_action}")
                obs, rew, terminated, truncated, _ = env.step(safe_action)
            
            tot += rew
            episode_steps += 1
            
            if terminated or truncated:
                returns.append(tot)
                break
                
            # Safety check to prevent infinite episodes
            if episode_steps > 1000:
                print(f"Episode exceeded 1000 steps, terminating")
                returns.append(tot)
                break
    
    if clipped_actions_count > 0:
        print(f"Total actions clipped during evaluation: {clipped_actions_count}/{total_actions} ({100*clipped_actions_count/total_actions:.1f}%)")
    
    return np.mean(returns), np.std(returns)

def save_symbolic_model_and_performance(symbolic, round_num, env_name, expert_algo, 
                                      mean_return, std_return, data_size, equation_str, args):
    """Save symbolic model and its performance metrics"""
    import pickle
    import os
    import json
    from datetime import datetime
    
    # Create directories if they don't exist
    os.makedirs("model", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Generate filenames
    if round_num == 0:
        round_name = "initial"
    else:
        round_name = f"round_{round_num}"
    
    base_name = f"symbolic_{env_name}_{expert_algo}_{round_name}"
    
    # Save the model
    model_path = f"model/{base_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(symbolic, f)
    
    # Save the equation as text
    equation_path = f"model/{base_name}_equation.txt"
    with open(equation_path, 'w') as f:
        f.write(equation_str)
    
    # Save performance metrics
    performance_data = {
        'round': round_num,
        'round_name': round_name,
        'environment': env_name,
        'expert_algorithm': expert_algo,
        'mean_return': float(mean_return),
        'std_return': float(std_return),
        'data_size': int(data_size),
        'equation': equation_str,
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': {
            'populations': args.populations,
            'maxsize': args.maxsize,
            'niterations': args.niterations,
            'parsimony': args.parsimony,
            'alpha': args.alpha,
            'advantage_threshold': args.advantage_threshold,
            'mix_alpha': args.mix_alpha,
            'gamma': args.gamma
        }
    }
    
    performance_path = f"results/{base_name}_performance.json"
    with open(performance_path, 'w') as f:
        json.dump(performance_data, f, indent=2)
    
    print(f"Saved model: {model_path}")
    print(f"Saved equation: {equation_path}")
    print(f"Saved performance: {performance_path}")
    
    return performance_data


def train_symbolic_policy(env, expert, args):
    """
    Main training function for symbolic policy with DAgger
    """
    import pickle
    import os
    import json
    from datetime import datetime
    
    print(f"Training symbolic policy for {env.__class__.__name__}...")
    print(f"Expert algorithm: {args.expert_algo}")
    
    # Create results tracking
    os.makedirs("results", exist_ok=True)
    training_log = []
    
    # Initial data collection from pure expert
    print("Collecting initial expert data...")
    obs0, act0, adv0 = collect_rollout(
        env=env, 
        expert=expert, 
        n_steps=args.initial_steps,
        mix_alpha=1.0,
        gamma=args.gamma
    )
    
    # Train initial symbolic policy
    print("Training initial symbolic policy...")
    symbolic = train_symbolic(
        obs0, act0, adv0,
        populations=args.populations,
        maxsize=args.maxsize,
        niterations=args.niterations,
        binary_operators=args.binary_operators,
        unary_operators=args.unary_operators,
        nested_constraints=args.nested_constraints,
        complexity_of_operators=args.complexity_of_operators,
        loss=args.loss,
        parsimony=args.parsimony,
        alpha=args.alpha,
        tournament_selection_n=args.tournament_selection_n,
        tournament_selection_p=args.tournament_selection_p,
        topn=args.topn,
        fraction_replaced=args.fraction_replaced,
        fraction_replaced_hof=args.fraction_replaced_hof,
        weight_add_node=args.weight_add_node,
        weight_insert_node=args.weight_insert_node,
        weight_delete_node=args.weight_delete_node,
        weight_do_nothing=args.weight_do_nothing,
        weight_mutate_constant=args.weight_mutate_constant,
        weight_mutate_operator=args.weight_mutate_operator,
        weight_randomize=args.weight_randomize,
        weight_simplify=args.weight_simplify,
        crossover_probability=args.crossover_probability,
        annealing=args.annealing,
        batching=args.batching,
        batch_size=args.batch_size,
        fast_cycle=args.fast_cycle,
        turbo=args.turbo,
        precision=args.precision,
        early_stop_condition=args.early_stop_condition,
        timeout_in_seconds=args.timeout_in_seconds,
        ncycles_per_iteration=args.ncycles_per_iteration,
        should_optimize_constants=args.should_optimize_constants,
        warmup_maxsize_by=args.warmup_maxsize_by
    )
    
    symbolic_policy = create_symbolic_policy(symbolic)
    
    print("Initial symbolic policy:")
    print(symbolic)
    mean_r, std_r = evaluate_symbolic_policy(env, symbolic_policy, args.n_eval)
    print(f"Initial symbolic policy return: {mean_r:.1f} ± {std_r:.1f}")
    
    # Save initial model and performance
    initial_performance = save_symbolic_model_and_performance(
        symbolic, 0, args.env_name, args.expert_algo, 
        mean_r, std_r, len(obs0), str(symbolic), args
    )
    training_log.append(initial_performance)
    
    # DAgger iterations
    for dagger_round in range(args.dagger_rounds):
        print(f"=== DAgger round {dagger_round+1} ===")
        
        # Collect new data with current policy
        obs_new, act_new, adv_new = collect_rollout(
            env=env,
            expert=expert,
            n_steps=args.dagger_steps,
            policy=symbolic_policy,
            mix_alpha=args.mix_alpha,
            gamma=args.gamma
        )
        
        # Aggregate datasets
        obs0 = np.vstack([obs0, obs_new])
        act0 = np.vstack([act0, act_new])
        adv0 = np.concatenate([adv0, adv_new])
        
        # Advantage filtering: remove data with low advantage
        if args.advantage_threshold > 0:
            mask = adv0 > args.advantage_threshold
            obs0 = obs0[mask]
            act0 = act0[mask]
            adv0 = adv0[mask]
            print(f"Data after filtering: {len(obs0)} samples")
        
        # Retrain symbolic policy
        symbolic = train_symbolic(
            obs0, act0, adv0,
            populations=args.populations,
            maxsize=args.maxsize,
            niterations=args.niterations,
            binary_operators=args.binary_operators,
            unary_operators=args.unary_operators,
            nested_constraints=args.nested_constraints,
            complexity_of_operators=args.complexity_of_operators,
            loss=args.loss,
            parsimony=args.parsimony,
            alpha=args.alpha,
            tournament_selection_n=args.tournament_selection_n,
            tournament_selection_p=args.tournament_selection_p,
            topn=args.topn,
            fraction_replaced=args.fraction_replaced,
            fraction_replaced_hof=args.fraction_replaced_hof,
            weight_add_node=args.weight_add_node,
            weight_insert_node=args.weight_insert_node,
            weight_delete_node=args.weight_delete_node,
            weight_do_nothing=args.weight_do_nothing,
            weight_mutate_constant=args.weight_mutate_constant,
            weight_mutate_operator=args.weight_mutate_operator,
            weight_randomize=args.weight_randomize,
            weight_simplify=args.weight_simplify,
            crossover_probability=args.crossover_probability,
            annealing=args.annealing,
            batching=args.batching,
            batch_size=args.batch_size,
            fast_cycle=args.fast_cycle,
            turbo=args.turbo,
            precision=args.precision,
            early_stop_condition=args.early_stop_condition,
            timeout_in_seconds=args.timeout_in_seconds,
            ncycles_per_iteration=args.ncycles_per_iteration,
            should_optimize_constants=args.should_optimize_constants,
            warmup_maxsize_by=args.warmup_maxsize_by
        )
        
        symbolic_policy = create_symbolic_policy(symbolic)
        
        print(f"Symbolic policy after round {dagger_round+1}:")
        print(symbolic)
        mean_r, std_r = evaluate_symbolic_policy(env, symbolic_policy, args.n_eval)
        print(f"Symbolic policy return: {mean_r:.1f} ± {std_r:.1f}")
        
        # Save model and performance for this round
        round_performance = save_symbolic_model_and_performance(
            symbolic, dagger_round+1, args.env_name, args.expert_algo,
            mean_r, std_r, len(obs0), str(symbolic), args
        )
        training_log.append(round_performance)
    
    # Save final training summary
    summary_path = f"results/training_summary_{args.env_name}_{args.expert_algo}.json"
    training_summary = {
        'experiment_info': {
            'environment': args.env_name,
            'expert_algorithm': args.expert_algo,
            'total_rounds': args.dagger_rounds + 1,  # +1 for initial
            'timestamp': datetime.now().isoformat()
        },
        'hyperparameters': {
            'populations': args.populations,
            'maxsize': args.maxsize,
            'niterations': args.niterations,
            'parsimony': args.parsimony,
            'alpha': args.alpha,
            'advantage_threshold': args.advantage_threshold,
            'mix_alpha': args.mix_alpha,
            'gamma': args.gamma,
            'initial_steps': args.initial_steps,
            'dagger_steps': args.dagger_steps,
            'n_eval': args.n_eval
        },
        'training_progress': training_log,
        'best_performance': max(training_log, key=lambda x: x['mean_return']),
        'final_performance': training_log[-1]
    }
    
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    print(f"\nTraining Summary:")
    print(f"Best performance: {training_summary['best_performance']['mean_return']:.1f} ± {training_summary['best_performance']['std_return']:.1f} (Round {training_summary['best_performance']['round']})")
    print(f"Final performance: {training_summary['final_performance']['mean_return']:.1f} ± {training_summary['final_performance']['std_return']:.1f}")
    print(f"Saved training summary: {summary_path}")
    
    return symbolic

# Utility function to convert string to float or None
def float_or_none(value):
    if value.lower() == 'none':
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float or None: {value}")

# Utility function to properly parse boolean values
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    # Initialize environment
    if args.env_name == 'pendulum':
        env = PendulumEnv(record_csv=True)
    elif args.env_name == 'cartpole':
        env = ContinuousCartPoleEnv(record_csv=True)
    elif args.env_name == 'mountaincar':
        env = Continuous_MountainCarEnv(record_csv=True)
    elif args.env_name == 'hopper':
        env = HopperEnv(record_csv=True)
    elif args.env_name == 'walker2d':
        env = Walker2dEnv(record_csv=True)
    elif args.env_name == 'swimmer':
        env = SwimmerEnv(record_csv=True)
    elif args.env_name == 'reacher':
        env = ReacherEnv(record_csv=True)
    elif args.env_name == 'inverteddoublependulum':
        env = InvertedDoublePendulumEnv(record_csv=True)
    elif args.env_name == 'lunarlander':
        env = LunarLanderContinuous(record_csv=True)
    elif args.env_name == 'bipedalwalker':
        env = BipedalWalker(record_csv=True)
    elif args.env_name == 'acrobot':
        env = AcrobotEnv(record_csv=True)
    else:
        raise ValueError('Invalid Env, please choose from: pendulum, cartpole, mountaincar, hopper, walker2d, swimmer, reacher, inverteddoublependulum, lunarlander, bipedalwalker, acrobot')

    # Load expert model
    if args.expert_model_path:
        expert_path = args.expert_model_path
    else:
        expert_path = f"/home/peilang/github_repo/distillation/trained_model/{args.env_name}/{args.expert_algo.lower()}/{args.expert_algo.lower()}_{env.__class__.__name__}_model"
    
    expert = load_expert_model(expert_path, env, args.expert_algo)
    
    # Train symbolic policy
    symbolic_model = train_symbolic_policy(env, expert, args)
    
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                   description="Symbolic Policy Training with Comprehensive Hyperparameters")
    
    # Environment and expert model
    parser.add_argument('-env', dest='env_name', type=str, default='cartpole', required=False,
                       help='Environment to use')
    parser.add_argument('-expertalgo', dest='expert_algo', type=str, default='ppo', required=False,
                       help='Expert algorithm used (ppo, sac, td3, etc.)')
    parser.add_argument('-expert_model', dest='expert_model_path', type=str, default=None, required=False,
                       help='Path to expert model (if not provided, uses default naming)')
    
    # DAgger parameters
    parser.add_argument('-gamma', dest='gamma', type=float, default=0.99, required=False,
                       help='Discount factor for advantage computation')
    parser.add_argument('-initial_steps', dest='initial_steps', type=int, default=3000, required=False,
                       help='Initial expert data collection steps')
    parser.add_argument('-dagger_rounds', dest='dagger_rounds', type=int, default=5, required=False,
                       help='Number of DAgger rounds')
    parser.add_argument('-dagger_steps', dest='dagger_steps', type=int, default=1000, required=False,
                       help='Steps per DAgger round')
    parser.add_argument('-mix_alpha', dest='mix_alpha', type=float, default=0.5, required=False,
                       help='Mixing ratio for expert vs policy data (1.0 = pure expert)')
    parser.add_argument('-advantage_threshold', dest='advantage_threshold', type=float, default=0.0001, required=False,
                       help='Threshold for advantage filtering')
    parser.add_argument('-n_eval', dest='n_eval', type=int, default=10, required=False,
                       help='Number of evaluation episodes')
    
    # PySR Core Parameters
    parser.add_argument('-populations', dest='populations', type=int, default=64, required=False,
                       help='Number of populations for symbolic regression')
    parser.add_argument('-maxsize', dest='maxsize', type=int, default=15, required=False,
                       help='Maximum complexity of expressions')
    parser.add_argument('-niterations', dest='niterations', type=int, default=100, required=False,
                       help='Number of iterations for symbolic regression')
    
    # Operators
    parser.add_argument('-binary_operators', dest='binary_operators', type=str, nargs='+', 
                       default=["+", "*", "-", "/"], required=False,
                       help='Binary operators for symbolic regression')
    parser.add_argument('-unary_operators', dest='unary_operators', type=str, nargs='+',
                       default=["cos", "sin", "exp", "square", "sqrt", "log"], required=False, ############### peilang
                       help='Unary operators for symbolic regression')
    
    # Constraints (JSON strings)
    parser.add_argument('-nested_constraints', dest='nested_constraints', type=str, default=None, required=False,
                       help='Nested constraints as JSON string')
    parser.add_argument('-complexity_of_operators', dest='complexity_of_operators', type=str, default=None, required=False,
                       help='Complexity of operators as JSON string')
    
    # Loss and optimization
    parser.add_argument('-loss', dest='loss', type=str, default="weighted_L2", required=False,
                       help='Loss function for symbolic regression (weighted_L2, L2DistLoss, etc.)')
    parser.add_argument('-parsimony', dest='parsimony', type=float, default=0.0032, required=False,
                       help='Parsimony coefficient')
    parser.add_argument('-alpha', dest='alpha', type=float, default=0.1, required=False,
                       help='Alpha parameter for loss weighting')
    
    # Tournament and selection
    parser.add_argument('-tournament_selection_n', dest='tournament_selection_n', type=int, default=10, required=False,
                       help='Tournament selection parameter n')
    parser.add_argument('-tournament_selection_p', dest='tournament_selection_p', type=float, default=0.86, required=False,
                       help='Tournament selection parameter p')
    parser.add_argument('-topn', dest='topn', type=int, default=12, required=False,
                       help='Number of top expressions to keep')
    
    # Replacement fractions
    parser.add_argument('-fraction_replaced', dest='fraction_replaced', type=float, default=0.000364, required=False,
                       help='Fraction of population replaced each iteration')
    parser.add_argument('-fraction_replaced_hof', dest='fraction_replaced_hof', type=float, default=0.035, required=False,
                       help='Fraction of hall of fame replaced')
    
    # Mutation weights
    parser.add_argument('-weight_add_node', dest='weight_add_node', type=float, default=0.79, required=False,
                       help='Weight for adding nodes')
    parser.add_argument('-weight_insert_node', dest='weight_insert_node', type=float, default=5.1, required=False,
                       help='Weight for inserting nodes')
    parser.add_argument('-weight_delete_node', dest='weight_delete_node', type=float, default=1.7, required=False,
                       help='Weight for deleting nodes')
    parser.add_argument('-weight_do_nothing', dest='weight_do_nothing', type=float, default=0.21, required=False,
                       help='Weight for doing nothing')
    parser.add_argument('-weight_mutate_constant', dest='weight_mutate_constant', type=float, default=0.048, required=False,
                       help='Weight for mutating constants')
    parser.add_argument('-weight_mutate_operator', dest='weight_mutate_operator', type=float, default=0.47, required=False,
                       help='Weight for mutating operators')
    parser.add_argument('-weight_randomize', dest='weight_randomize', type=float, default=0.00023, required=False,
                       help='Weight for randomizing expressions')
    parser.add_argument('-weight_simplify', dest='weight_simplify', type=float, default=0.0020, required=False,
                       help='Weight for simplifying expressions')
    
    # Advanced options - SUPPORTS BOTH OLD AND NEW BOOLEAN SYNTAX
    parser.add_argument('-crossover_probability', dest='crossover_probability', type=float, default=0.066, required=False,
                       help='Crossover probability')
    
    # Boolean arguments - support both -flag=value and --flag/--no-flag syntax
    parser.add_argument('-annealing', dest='annealing', type=str2bool, nargs='?', const=True, default=False,
                       help='Use annealing (can use -annealing=true/false or --annealing)')
    parser.add_argument('--annealing', dest='annealing', action='store_true',
                       help='Use annealing (flag version)')
    parser.add_argument('--no-annealing', dest='annealing', action='store_false',
                       help='Do not use annealing')
    
    parser.add_argument('-batching', dest='batching', type=str2bool, nargs='?', const=True, default=False,
                       help='Use batching (can use -batching=true/false or --batching)')
    parser.add_argument('--batching', dest='batching', action='store_true',
                       help='Use batching (flag version)')
    parser.add_argument('--no-batching', dest='batching', action='store_false',
                       help='Do not use batching')
    
    parser.add_argument('-batch_size', dest='batch_size', type=int, default=50, required=False,
                       help='Batch size for batching')
    
    parser.add_argument('-fast_cycle', dest='fast_cycle', type=str2bool, nargs='?', const=True, default=False,
                       help='Use fast cycle (can use -fast_cycle=true/false or --fast_cycle)')
    parser.add_argument('--fast_cycle', dest='fast_cycle', action='store_true',
                       help='Use fast cycle (flag version)')
    parser.add_argument('--no-fast_cycle', dest='fast_cycle', action='store_false',
                       help='Do not use fast cycle')
    
    parser.add_argument('-turbo', dest='turbo', type=str2bool, nargs='?', const=True, default=False,
                       help='Use turbo mode (can use -turbo=true/false or --turbo)')
    parser.add_argument('--turbo', dest='turbo', action='store_true',
                       help='Use turbo mode (flag version)')
    parser.add_argument('--no-turbo', dest='turbo', action='store_false',
                       help='Do not use turbo mode')
    
    parser.add_argument('-precision', dest='precision', type=int, default=32, required=False,
                       help='Precision for calculations')
    
    # Early stopping and timeouts
    parser.add_argument('-early_stop_condition', dest='early_stop_condition', type=str, default=None, required=False,
                       help='Early stopping condition')
    parser.add_argument('-timeout_in_seconds', dest='timeout_in_seconds', type=float_or_none, default=None, required=False,
                       help='Timeout in seconds')
    
    # Additional parameters
    parser.add_argument('-ncycles_per_iteration', dest='ncycles_per_iteration', type=int, default=550, required=False,
                       help='Number of cycles per iteration')
    
    parser.add_argument('-should_optimize_constants', dest='should_optimize_constants', type=str2bool, nargs='?', const=True, default=True,
                       help='Should optimize constants (can use -should_optimize_constants=true/false or --should_optimize_constants)')
    parser.add_argument('--should_optimize_constants', dest='should_optimize_constants', action='store_true',
                       help='Should optimize constants (flag version)')
    parser.add_argument('--no-should_optimize_constants', dest='should_optimize_constants', action='store_false',
                       help='Should not optimize constants')
    
    parser.add_argument('-warmup_maxsize_by', dest='warmup_maxsize_by', type=float, default=0.0, required=False,
                       help='Warmup max size parameter')
    
    args = parser.parse_args()
    
    # Parse JSON constraints if provided
    if args.nested_constraints:
        try:
            args.nested_constraints = json.loads(args.nested_constraints)
        except json.JSONDecodeError:
            print("Warning: Invalid JSON for nested_constraints, using defaults")
            args.nested_constraints = None
    
    if args.complexity_of_operators:
        try:
            args.complexity_of_operators = json.loads(args.complexity_of_operators)
        except json.JSONDecodeError:
            print("Warning: Invalid JSON for complexity_of_operators, using defaults")
            args.complexity_of_operators = None
    
    main(args)