import sys
import os
import time
import yaml
import numpy as np
import random

import copy
import math
from scipy.stats import norm
from scipy.special import logsumexp
from scipy.special import gammaln, psi
from scipy.spatial.distance import cdist

# Get current and parent directory to handle import paths
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from agent.q_learning_agent_ import ValueIteration, PolicyEvaluation


from concurrent.futures import ProcessPoolExecutor
import time


from concurrent.futures import ProcessPoolExecutor
import time


def _vi_worker(args):
    """Worker receives a single picklable tuple."""
    env, epsilon = args
    v = ValueIteration(env)
    v.run_value_iteration(epsilon=epsilon)
    return v.get_q_values()


def parallel_value_iteration(
    envs,
    *,
    epsilon=1e-10,
    n_jobs=None,
    log=print
):
    n_envs = len(envs)
    log("[3/12] Running Value Iteration on all MDPs... (parallel)")
    t0 = time.time()

    worker_args = [(env, epsilon) for env in envs]

    Q_list = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:

        # No lambda, no closure — fully picklable
        for i, Q in enumerate(executor.map(_vi_worker, worker_args)):
            Q_list.append(Q)

            # progress
            if (i + 1) % max(1, n_envs // 5) == 0:
                log(f"       VI progress: {i+1}/{n_envs} MDPs solved...")

    log(f"       ✔ VI completed in {time.time() - t0:.2f}s\n")
    return Q_list

import numpy as np

def sa_pairs_to_action_list(env, sa_pairs, default_action=0, include_terminals=False):
    S = env.get_num_states()
    terminals = set(getattr(env, "terminal_states", []) or [])
    policy = [default_action] * S

    for s, a in sa_pairs:
        s = int(s); a = int(a)
        if (not include_terminals) and (s in terminals):
            continue
        policy[s] = a

    return policy



def calculate_percentage_optimal_actions(policy, env, epsilon=0.0001):
    """
    Calculate the percentage of actions in the given policy that are optimal under the environment's Q-values.

    Args:
        policy (list): List of actions for each state.
        env: The environment object.
        epsilon (float): Tolerance for determining optimal actions.

    Returns:
        float: Percentage of optimal actions in the policy.
    """
    # Compute Q-values using value iteration
    q_values = ValueIteration(env).get_q_values()
    
    # Count how many actions in the policy are optimal under the environment
    optimal_actions_count = sum(
        1 for state, action in enumerate(policy) if action in _arg_max_set(q_values[state], epsilon)
    )
    
    return optimal_actions_count / env.num_states

def _arg_max_set(values, epsilon=0.0001):
    """
    Returns the indices corresponding to the maximum element(s) in a set of values, within a tolerance.

    Args:
        values (list or np.array): List of values to evaluate.
        epsilon (float): Tolerance for determining equality of maximum values.

    Returns:
        list: Indices of the maximum value(s).
    """
    max_val = max(values)
    return [i for i, v in enumerate(values) if abs(max_val - v) < epsilon]

def calculate_expected_value_difference(eval_policy, env, epsilon=0.0001, normalize_with_random_policy=False):
    """
    Calculates the difference in expected returns between an optimal policy for an MDP and the eval_policy.

    Args:
        eval_policy (list): The policy to evaluate.
        env: The environment object.
        storage (dict): A storage dictionary (not used in this version, but passed for consistency).
        epsilon (float): Convergence threshold for value iteration and policy evaluation.
        normalize_with_random_policy (bool): Whether to normalize using a random policy.

    Returns:
        float: The difference in expected returns between the optimal policy and eval_policy.
    """
    
    # Run value iteration to get the optimal state values
    V_opt = ValueIteration(env).run_value_iteration(epsilon=epsilon)
    
    print("Inside the expected value difference")
    print("Env weight: ", env.get_feature_weights())

    eval_policy = sa_pairs_to_action_list(env, eval_policy)
    # Perform policy evaluation for the provided eval_policy
    V_eval = PolicyEvaluation(env, policy=eval_policy).run_policy_evaluation(epsilon=epsilon)
    
    # Optional: Normalize using a random policy if the flag is set
    if normalize_with_random_policy:
        V_rand = PolicyEvaluation(env, uniform_random=True).run_policy_evaluation(epsilon=epsilon)
        #if np.mean(V_opt) - np.mean(V_eval) == 0:
        #    return 0.0

        return (np.mean(V_opt) - np.mean(V_eval)) / (np.mean(V_opt) - np.mean(V_rand))
        #return (np.mean(V_opt) - np.mean(V_eval)) / (np.mean(V_opt))

    # Return the unnormalized difference in expected returns between optimal and eval_policy
    return np.mean(V_opt) - np.mean(V_eval)

def calculate_policy_accuracy(opt_pi, eval_pi):
    assert len(opt_pi) == len(eval_pi)
    matches = 0
    for i in range(len(opt_pi)):
        matches += opt_pi[i] == eval_pi[i]
    return matches / len(opt_pi)


###################### Multi process

from joblib import Parallel, delayed
import copy
import numpy as np

def compute_policy_loss(sample, env, map_policy, random_normalization):
    learned_env = copy.deepcopy(env)
    learned_env.set_feature_weights(sample)
    return calculate_expected_value_difference(
        map_policy, learned_env, normalize_with_random_policy=random_normalization
    )

def compute_policy_loss_avar_bounds(mcmc_samples, env, map_policy, random_normalization, alphas, delta):
    # Parallelize policy loss computation
    policy_losses = Parallel(n_jobs=-1)(
        delayed(compute_policy_loss)(sample, env, map_policy, random_normalization)
        for sample in mcmc_samples
    )

    # Sort the policy losses
    policy_losses = sorted(policy_losses)

    # Compute a-VaR bounds
    N_burned = len(mcmc_samples)
    avar_bounds = {}
    for alpha in alphas:
        k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned * alpha * (1 - alpha)) - 0.5)
        k = min(k, N_burned - 1)
        avar_bounds[alpha] = policy_losses[k]

    return avar_bounds


def compute_reward_for_trajectory(env, trajectory, discount_factor=None):
    """
    Computes the cumulative reward for a given trajectory in the environment. If a discount factor 
    is provided, the function calculates the discounted cumulative reward, where rewards received 
    later in the trajectory are given less weight.

    :param env: The environment (MDP) which provides the reward function. This should have a 
                `compute_reward(state)` method.
    :param trajectory: List of tuples (state, action), where `state` is the current state and 
                       `action` is the action taken in that state. The action is ignored in reward 
                       computation but kept for compatibility with the trajectory format.
    :param discount_factor: (Optional) A float representing the discount factor (gamma) for 
                            future rewards. It should be between 0 and 1. If None, no discounting 
                            is applied, and rewards are summed without any decay.
    :return: The cumulative reward for the trajectory, either discounted or non-discounted.
             If discount_factor is provided, it applies a discount based on the time step of 
             the trajectory.
    """
    cumulative_reward = 0
    discount = 1 if discount_factor is None else discount_factor
    
    for t, (state, action) in enumerate(trajectory):
        if state is None:  # Terminal state reached
            break
        
        # Compute the reward for the current state
        reward = env.compute_reward(state)
        
        # If a discount factor is provided, apply it to the reward
        if discount_factor:
            cumulative_reward += reward * (discount_factor ** t)
        else:
            cumulative_reward += reward

    return cumulative_reward

def logsumexp(x):
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from scipy.special import gammaln

from joblib import Parallel, delayed
import numpy as np
from scipy.special import logsumexp
import copy

def log_prob_demo(env, demos, theta, beta):
    """
    Computes the log probability of a set of demonstrations given a reward function.

    Args:
        env: The GridWorld environment.
        demos: A list of (state, action) pairs representing demonstrations.
        theta: The reward function parameters.
        beta: The rationality parameter for the likelihood model.

    Returns:
        float: The log-likelihood of the demonstrations given the reward function.
    """
    env = copy.deepcopy(env)  # Create a deep copy of the environment
    env.set_feature_weights(theta)
    q_values = ValueIteration(env).get_q_values()

    log_sum = 0.0
    for s, a in demos:
        if s not in env.terminal_states:
            log_sum += beta * q_values[s][a] - logsumexp(beta * q_values[s])

    return log_sum

def compute_entropy_importance_sampling_parallel(env, demos, samples, beta):
    num_mcmc_samples = len(samples)
    
    # Parallelize log P(H_{1:i} | theta) computation for each sample
    log_probs = np.array(Parallel(n_jobs=-1)(
        delayed(log_prob_demo)(env, demos, theta, beta) for theta in samples
    ))
    
    # First term: - (1 / m_i) * sum(log P(H_{1:i} | theta^{(k)}))
    first_term = -np.mean(log_probs)
    
    # Harmonic mean term: -log( (1 / m_i) * sum( 1 / P(H_{1:i} | theta^{(k)}) ) )
    inverse_likelihoods = 1.0 / np.exp(log_probs)  # 1 / P(H | theta) = exp(-log P(H | theta))
    harmonic_mean_term = -np.log(np.mean(inverse_likelihoods))
    
    # Prior term: -log A_d, where A_d is the surface area of the unit sphere
    d = len(samples[0])  # Dimensionality of theta
    log_A_d = compute_log_surface_area(d)
    prior_term = -log_A_d
    
    # Entropy estimate
    entropy = first_term + prior_term + harmonic_mean_term
    return entropy


def log_prob_estop(env, demos, theta, beta):
    
    env = copy.deepcopy(env)  # Create a deep copy of the environment

    env.set_feature_weights(theta)

    # Initialize the log prior as 0, assuming an uninformative prior
    log_prior = 0.0
    log_sum = log_prior  # Start the log sum with the log prior value

    for estop in demos:
        # Unpack the trajectory and stopping point
        trajectory, t = estop
        traj_len = len(trajectory)

        # Compute the cumulative reward up to the stopping point t
        reward_up_to_t = sum(env.compute_reward(s) for s, _ in trajectory[:t+1])

        # Add repeated rewards for the last step at time t until the trajectory horizon
        #reward_up_to_t += (traj_len - t - 1) * env.compute_reward(trajectory[t][0])

        # Numerator: P(off | r, C) -> exp(beta * reward_up_to_t)
        stop_prob_numerator = beta * reward_up_to_t

        # reward of the whole trajectory
        traj_reward = sum(env.compute_reward(s) for s, _ in trajectory[:])
        
        #denominator = np.exp(self.beta * traj_reward) + np.exp(stop_prob_numerator)
    
        log_denominator = logsumexp([beta * traj_reward, stop_prob_numerator])
        
        # Use the Log-Sum-Exp trick for the denominator
        #max_reward = max(self.beta * traj_reward, stop_prob_numerator)
        #log_denominator = max_reward + np.log(
        #    np.exp(self.beta * traj_reward - max_reward) +
        #   np.exp(stop_prob_numerator - max_reward)
        #)

        # Add the log probability to the log sum
        log_sum += stop_prob_numerator - log_denominator
    
    return log_sum


def log_prob_comparison(env, demos, theta, beta):
    """
    Computes the log probability of preference demonstrations using a Boltzmann distribution.

    Args:
        env: The GridWorld environment.
        demos: A list of (trajectory1, trajectory2) pairs representing preference demonstrations.
        theta: The reward function parameters.
        beta: The rationality parameter for the likelihood model.

    Returns:
        float: The log-likelihood of the preferences given the reward function.
    """
    env = copy.deepcopy(env)  # Create a deep copy of the environment
    env.set_feature_weights(theta)

    log_sum = 0.0
    for traj1, traj2 in demos:
        reward1 = compute_reward_for_trajectory(env, traj1)
        reward2 = compute_reward_for_trajectory(env, traj2)
        log_sum += beta * reward1 - logsumexp([beta * reward1, beta * reward2])

    return log_sum

# removing dublicates in a given trajectory
def dedup_traj(traj):
    seen = set()
    out = []
    for sa in traj:
        if sa not in seen:
            seen.add(sa)
            out.append(sa)
    return out

def bucket_and_dedup(chosen, num_envs, dedup_across_trajs=False):
    demos_by_env = [[] for _ in range(num_envs)]
    if dedup_across_trajs:
        seen_env = [set() for _ in range(num_envs)]

    for mdp_idx, traj in chosen:
        t = dedup_traj(traj)  # de-dup within this trajectory
        if dedup_across_trajs:
            # add only pairs not yet seen in this env
            for sa in t:
                if sa not in seen_env[mdp_idx]:
                    seen_env[mdp_idx].add(sa)
                    demos_by_env[mdp_idx].append(sa)
        else:
            # keep all (s,a) from this de-duped trajectory
            demos_by_env[mdp_idx].extend(t)

    return demos_by_env