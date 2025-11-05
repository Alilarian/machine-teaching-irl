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
from agent.q_learning_agent import ValueIteration, PolicyEvaluation

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
'''
def compute_policy_loss_avar_bound(mcmc_samples, env, map_policy, random_normalization, alpha, delta):

    policy_losses = []

    # Step 1: Calculate policy loss for each MCMC sample
    for sample in mcmc_samples:
        learned_env = copy.deepcopy(env)  # Create a copy of the environment
        learned_env.set_feature_weights(sample)   # Set the reward function to the current sample
        
        # Calculate the policy loss (Expected Value Difference)
        policy_loss = calculate_expected_value_difference(
            map_policy, learned_env, normalize_with_random_policy=random_normalization
        )
        policy_losses.append(policy_loss)

    # Step 2: Sort the policy losses
    policy_losses.sort()

    # Step 3: Compute the VaR (Value at Risk) bound
    N_burned = len(mcmc_samples)
    k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned * alpha * (1 - alpha)) - 0.5)
    k = min(k, N_burned - 1)  # Ensure k doesn't exceed the number of samples

    # Return the computed a-VaR bound
    return policy_losses[k]
'''


# def compute_policy_loss_avar_bounds(mcmc_samples, env, map_policy, random_normalization, alphas, delta):
#     """
#     Computes the counterfactual policy losses and calculates the a-VaR (Value at Risk) bound for multiple alpha values.

#     Args:
#         mcmc_samples (list): List of MCMC sampled rewards from the BIRL process.
#         env: The environment object.
#         map_policy: The MAP (Maximum a Posteriori) policy from BIRL.
#         random_normalization (bool): Whether to normalize using a random policy.
#         alphas (list of float): List of confidence level parameters.
#         delta (float): Risk level parameter.

#     Returns:
#         dict: A dictionary mapping each alpha to its computed a-VaR bound.
#     """
#     policy_losses = []

#     # Step 1: Calculate policy loss for each MCMC sample
#     for sample in mcmc_samples:
#         learned_env = copy.deepcopy(env)  # Create a copy of the environment
#         learned_env.set_feature_weights(sample)   # Set the reward function to the current sample
        
#         # Calculate the policy loss (Expected Value Difference)
#         policy_loss = calculate_expected_value_difference(
#             map_policy, learned_env, normalize_with_random_policy=random_normalization
#         )
#         policy_losses.append(policy_loss)

#     # Step 2: Sort the policy losses
#     policy_losses.sort()

#     # Step 3: Compute the a-VaR bound for each alpha
#     N_burned = len(mcmc_samples)
#     avar_bounds = {}

#     for alpha in alphas:
#         k = math.ceil(N_burned * alpha + norm.ppf(1 - delta) * np.sqrt(N_burned * alpha * (1 - alpha)) - 0.5)
#         k = min(k, N_burned - 1)  # Ensure k doesn't exceed the number of samples
#         avar_bounds[alpha] = policy_losses[k]

#     return avar_bounds

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

def knn_entropy(samples, k=3, epsilon=1e-10, dimension=3):
    """
    Estimate differential entropy using the k-Nearest Neighbors estimator.
    
    Args:
        samples: NumPy array of shape (m, d), where m is number of samples.
        k: Number of neighbors to consider (default: 3).
        epsilon: Small value to avoid log(0) issues (default: 1e-10).
        dimension: Dimensionality for entropy calculation (default: 3 for S^3).
    
    Returns:
        Estimated differential entropy (float).
    """
    m, d = samples.shape
    if m <= k:
        raise ValueError(f"Need at least {k+1} samples for k-NN entropy estimation.")
    
    # Check for duplicates
    unique_samples = np.unique(samples, axis=0)

    
    # Verify samples are on the unit sphere
    norms = np.linalg.norm(samples, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6):
      
        samples = samples / norms[:, np.newaxis]
    
    # Compute pairwise Euclidean distances
    distances = cdist(samples, samples, metric='euclidean')
    np.fill_diagonal(distances, np.inf)  # Exclude self-distances
    # Get k-th nearest neighbor distances
    rho_k = np.partition(distances, k-1, axis=1)[:, k-1]
    
    # Handle zero distances
    if np.any(rho_k == 0):
        
        rho_k = np.maximum(rho_k, epsilon)
    

    # Constants
    log_ball_volume = gammaln(dimension / 2 + 1) - (dimension / 2) * np.log(np.pi)
    
    # k-NN entropy estimate
    entropy = psi(k) - psi(m) + (dimension / m) * np.sum(np.log(rho_k)) + log_ball_volume
    return entropy
def kozachenko_leonenko_entropy(samples, epsilon=1e-10):
    """
    Compute entropy using the Kozachenko-Leonenko estimator.
    
    Parameters:
    - samples: numpy array of shape (m, d), where m is the number of samples and d is the dimension
    - epsilon: small value to avoid log(0) issues (default: 1e-10)
    
    Returns:
    - entropy: estimated differential entropy
    """
    m, d = samples.shape
    if m < 2:
        raise ValueError("Need at least 2 samples for entropy estimation.")

    # Compute pairwise Euclidean distances
    distances = cdist(samples, samples, metric='euclidean')
    # Set diagonal to infinity to exclude self-distances
    np.fill_diagonal(distances, np.inf)
    # Get nearest neighbor distances
    rho = np.min(distances, axis=1)

    # Replace zero distances with epsilon to avoid log(0)
    if np.any(rho == 0):
        print("Warning: Zero distances found; replacing with epsilon.")
        rho = np.maximum(rho, epsilon)

    # Constants for the KL estimator
    euler_mascheroni = 0.5772156649  # Euler-Mascheroni constant
    log_ball_volume = gammaln(d / 2 + 1) - (d / 2) * np.log(np.pi)  # Log of volume of d-dimensional unit ball

    # Compute entropy
    entropy = (d / m) * np.sum(np.log(rho)) + log_ball_volume + np.log(m - 1) + euler_mascheroni
    return entropy


def compute_log_surface_area(d):
    """
    Compute the log surface area of a (d-1)-dimensional unit sphere in d dimensions.
    A_d = 2 * pi^(d/2) / Gamma(d/2)
    
    Args:
        d (int): Dimensionality of the space.
    
    Returns:
        float: Log of the surface area.
    """
    log_A_d = np.log(2) + (d/2) * np.log(np.pi) - gammaln(d/2)
    return log_A_d


# # Harmonic mean
def compute_entropy_importance_sampling(env, demos, samples, beta, log_prob_func):
    num_mcmc_samples = len(samples)
    
    env = copy.deepcopy(env)
    # Compute log P(H_{1:i} | theta) for each sample
    log_probs = np.array([log_prob_func(env, demos, theta, beta) for theta in samples])
    
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