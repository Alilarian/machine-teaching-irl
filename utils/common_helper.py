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


def logsumexp(x):
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

