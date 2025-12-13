# test/test_birl_reward_recovery_5d.py

import numpy as np
import os
import sys

# ----------------------------
# Project imports
# ----------------------------
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from mdp.gridworld_env_layout import GridWorldMDPFromLayoutEnv  # <-- adjust import path to your env file
from agent.q_learning_agent_ import ValueIteration
from reward_learning.birl import BIRL  # <-- adjust import path to your BIRL file


# -----------------------------
# Robust helpers (works with slightly different VI APIs)
# -----------------------------
def _run_vi_and_get_q(env, epsilon=1e-4):
    """
    Try to run value iteration and return Q-values with minimal assumptions.
    """
    vi = ValueIteration(env)

    # Try common method names
    if hasattr(vi, "run_value_iteration"):
        vi.run_value_iteration(epsilon=epsilon)
    elif hasattr(vi, "run"):
        vi.run(epsilon=epsilon)
    # Some implementations compute lazily inside get_q_values()

    if hasattr(vi, "get_q_values"):
        Q = vi.get_q_values()
    elif hasattr(vi, "Q"):
        Q = vi.Q
    elif hasattr(vi, "q_values"):
        Q = vi.q_values
    else:
        raise AttributeError("ValueIteration: couldn't find Q-values accessor (get_q_values/Q/q_values).")

    return np.asarray(Q, dtype=float)


def optimal_sa_pairs_all_states(env, epsilon=1e-4, tie_tol=1e-12):
    """
    Returns list of (s,a) for ALL non-terminal states.
    If multiple optimal actions exist (ties), includes them all.
    """
    Q = _run_vi_and_get_q(env, epsilon=epsilon)
    demos = []
    terminals = set(env.terminal_states or [])

    for s in range(env.get_num_states()):
        if s in terminals:
            continue
        row = Q[s]
        vmax = np.max(row)
        # all ties within tolerance
        opt_actions = np.where(row >= vmax - tie_tol)[0].tolist()
        for a in opt_actions:
            demos.append((s, int(a)))

    return demos, Q


def compute_greedy_policy(Q, env):
    """
    Greedy (argmax) policy; ties broken by first max.
    """
    terminals = set(env.terminal_states or [])
    policy = {}
    for s in range(env.get_num_states()):
        if s in terminals:
            continue
        policy[s] = int(np.argmax(Q[s]))
    return policy


def reward_vector_over_states(env, w):
    """
    r(s) = phi(s)·w for all states.
    """
    env = env  # no copy needed if you restore weights afterward
    old = env.get_feature_weights().copy()
    env.set_feature_weights(w)
    r = np.array([env.compute_reward(s) for s in range(env.get_num_states())], dtype=float)
    env.set_feature_weights(old)
    return r


# -----------------------------
# 5D test environment builder
# -----------------------------
def make_5d_env(seed=0):
    rng = np.random.default_rng(seed)

    # Simple layout (you can change size/colors freely)
    layout = [
        ["W", "W", "W", "W", "G"],
        ["W", "R", "R", "W", "G"],
        ["W", "R", "B", "W", "G"],
        ["W", "W", "W", "W", "G"],
        ["Y", "Y", "Y", "Y", "G"],
    ]

    # 5D features per color (example: one-hot-ish but not identical)
    color_to_feature_map = {
        "W": [1, 0, 0, 0, 0],
        "R": [0, 1, 0, 0, 0],
        "B": [0, 0, 1, 0, 0],
        "G": [0, 0, 0, 1, 0],
        "Y": [0, 0, 0, 0, 1],
    }

    # Pick a terminal (bottom-right)
    rows, cols = len(layout), len(layout[0])
    terminal_state = (rows - 1) * cols + (cols - 1)

    # True reward weights (unit norm, 5D)
    true_w = rng.normal(size=5)
    true_w = true_w / (np.linalg.norm(true_w) + 1e-12)

    env_true = GridWorldMDPFromLayoutEnv(
        gamma=0.95,
        layout=layout,
        color_to_feature_map=color_to_feature_map,
        noise_prob=0.10,
        terminal_states=[terminal_state],
        custom_feature_weights=true_w,
        render_mode=None,
    )

    # Base env passed to BIRL (weights don't matter; BIRL overwrites them anyway)
    env_base = GridWorldMDPFromLayoutEnv(
        gamma=0.95,
        layout=layout,
        color_to_feature_map=color_to_feature_map,
        noise_prob=0.10,
        terminal_states=[terminal_state],
        custom_feature_weights=rng.normal(size=5),
        render_mode=None,
    )

    env_base.set_random_seed(seed)
    env_true.set_random_seed(seed)

    return env_true, env_base, true_w


# -----------------------------
# Main test
# -----------------------------
def test_birl_reward_recovery_5d(
    seed=0,
    beta=10.0,
    vi_epsilon=1e-4,
    mcmc_samples=8000,
    mcmc_stepsize=0.08,
    adaptive=True,
):
    np.random.seed(seed)

    env_true, env_base, true_w = make_5d_env(seed=seed)

    # 1) Generate ALL optimal (s,a) pairs (including ties) under true reward
    demos, Q_true = optimal_sa_pairs_all_states(env_true, epsilon=vi_epsilon, tie_tol=1e-12)
    true_policy = compute_greedy_policy(Q_true, env_true)

    print(f"[INFO] num_states={env_true.get_num_states()}, terminals={env_true.terminal_states}")
    print(f"[INFO] demos generated (all optimal actions for all non-terminal states): {len(demos)}")
    print(f"[INFO] true_w = {true_w}")

    # 2) Run BIRL
    birl = BIRL(env=env_base, demos=demos, beta=beta, epsilon=vi_epsilon)
    birl.run_mcmc(samples=mcmc_samples, stepsize=mcmc_stepsize, normalize=True, adaptive=adaptive)
    w_map = birl.get_map_solution()

    # 3) Compare MAP vs True (directional + policy)
    # Cosine similarity
    cos = float(np.dot(w_map, true_w) / ((np.linalg.norm(w_map) * np.linalg.norm(true_w)) + 1e-12))
    # (optional) if you want to show best sign alignment:
    cos_flip = float(np.dot(-w_map, true_w) / ((np.linalg.norm(w_map) * np.linalg.norm(true_w)) + 1e-12))
    cos_best = max(cos, cos_flip)

    # Reward correlation across states
    r_true = reward_vector_over_states(env_true, true_w)
    r_map = reward_vector_over_states(env_true, w_map)
    r_corr = float(np.corrcoef(r_true, r_map)[0, 1])

    # Policy agreement
    # Compute greedy policy under MAP weights
    env_tmp = make_5d_env(seed=seed)[0]  # env_true-like dynamics; we only need same layout/dynamics
    env_tmp.set_feature_weights(w_map)
    Q_map = _run_vi_and_get_q(env_tmp, epsilon=vi_epsilon)
    map_policy = compute_greedy_policy(Q_map, env_tmp)

    agree = 0
    total = 0
    for s, a_true in true_policy.items():
        a_map = map_policy.get(s, None)
        if a_map is None:
            continue
        total += 1
        agree += int(a_map == a_true)
    policy_agreement = (agree / total) if total > 0 else 0.0

    print("\n==== RESULTS ====")
    print(f"accept_rate = {getattr(birl, 'accept_rate', None)}")
    print(f"MAP w       = {w_map}")
    print(f"cos(w_map, true_w)        = {cos:.4f}")
    print(f"cos(-w_map, true_w)       = {cos_flip:.4f}")
    print(f"best cosine (allow flip)  = {cos_best:.4f}")
    print(f"reward corr over states   = {r_corr:.4f}")
    print(f"policy agreement (greedy) = {policy_agreement:.4f}")

    # Lightweight “pass/fail” checks (tune thresholds if needed)
    # Note: exact recovery is not guaranteed in all MDPs; policy agreement is often the most meaningful.
    assert policy_agreement >= 0.85, f"Policy agreement too low: {policy_agreement:.3f}"
    assert r_corr >= 0.80, f"Reward correlation too low: {r_corr:.3f}"

    return {
        "true_w": true_w,
        "w_map": w_map,
        "cos": cos,
        "cos_flip": cos_flip,
        "cos_best": cos_best,
        "reward_corr": r_corr,
        "policy_agreement": policy_agreement,
        "num_demos": len(demos),
    }


if __name__ == "__main__":
    out = test_birl_reward_recovery_5d(
        seed=0,
        beta=10.0,
        vi_epsilon=1e-4,
        mcmc_samples=4000,
        mcmc_stepsize=0.8,
        adaptive=True,
    )
    print("\n[OK] Test finished.")
