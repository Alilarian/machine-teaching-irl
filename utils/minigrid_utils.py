from multiprocessing import Pool, cpu_count

import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
from utils import remove_redundant_constraints

from __future__ import annotations
import numpy as np



ACT_LEFT = 0
ACT_RIGHT = 1
ACT_FORWARD = 2
ACTIONS = [ACT_LEFT, ACT_RIGHT, ACT_FORWARD]




def policy_evaluation_next_state(
    T: np.ndarray,
    r_next: np.ndarray,
    policy: np.ndarray,
    terminal_mask: np.ndarray,
    gamma: float,
    theta: float = 1e-8,
    max_iters: int = 200000,
) -> np.ndarray:
    """
    Evaluate a fixed policy with NEXT-state reward:
      V(s) = Σ_{s'} T[s,a,s'] * ( r_next[s'] + gamma * 1[~terminal(s')] * V(s') )
    Terminal states are kept at V=0 (consistent with your VI done-cutoff).
    """
    S, A, S2 = T.shape
    assert S == S2
    V = np.zeros(S, dtype=float)

    cont = (~terminal_mask).astype(float)  # 1 if nonterminal, 0 if terminal

    for _ in range(max_iters):
        delta = 0.0
        for s in range(S):
            if terminal_mask[s]:
                continue
            a = int(policy[s])
            v_new = float(np.sum(T[s, a] * (r_next + gamma * (cont * V))))
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break
    return V

def value_iteration_next_state(
    T: np.ndarray,
    r_next: np.ndarray,
    terminal_mask: np.ndarray,
    gamma: float,
    theta: float = 1e-8,
    max_iters: int = 200000,
):
    """
    NEXT-state reward value iteration:
      Q(s,a) = Σ_{s'} T[s,a,s'] * ( r_next[s'] + gamma * 1[~terminal(s')] * V(s') )
      V(s) = max_a Q(s,a)
    Terminal states fixed at V=0.
    Returns: V, Q, pi
    """
    S, A, S2 = T.shape
    assert S == S2
    V = np.zeros(S, dtype=float)
    Q = np.zeros((S, A), dtype=float)

    cont = (~terminal_mask).astype(float)

    for _ in range(max_iters):
        delta = 0.0
        for s in range(S):
            if terminal_mask[s]:
                continue

            # compute Q(s,a) for all a
            for a in range(A):
                Q[s, a] = float(np.sum(T[s, a] * (r_next + gamma * (cont * V))))

            v_new = float(np.max(Q[s]))
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new

        if delta < theta:
            break

    # greedy policy
    pi = np.zeros(S, dtype=int)
    for s in range(S):
        if terminal_mask[s]:
            pi[s] = ACT_FORWARD
        else:
            pi[s] = int(np.argmax(Q[s]))

    return V, Q, pi

def compute_successor_features_from_q_next_state(
    T: np.ndarray,
    Phi: np.ndarray,
    Q: np.ndarray,
    terminal_mask: np.ndarray,
    gamma: float,
    tol: float = 1e-10,
    max_iters: int = 100000,
):
    """
    Successor Features with NEXT-STATE (entering) convention, consistent with your code.

    Definitions:
      π(s)      = argmax_a Q(s,a)
      ψ(s)      = E_π [ sum_t γ^t φ(s_{t+1}) | s0 = s ]
      ψ(s,a)    = E [ φ(s1) + γ ψ(s1) | s0=s, a0=a ]

    Bellman equation:
      ψ(s) = Σ_{s'} P_π(s,s') [ φ(s') + γ * 1[~terminal(s')] * ψ(s') ]

    Inputs:
      T             : (S,A,S) transition matrix
      Phi           : (S,D) state feature matrix (φ(s))
      Q             : (S,A) Q-values (used to extract greedy policy)
      terminal_mask : (S,) boolean
      gamma         : discount factor

    Returns:
      Psi_sa : (S,A,D) successor features for state-action
      Psi_s  : (S,D)   successor features for state
    """
    S, A, S2 = T.shape
    assert S == S2
    D = Phi.shape[1]

    # -----------------------------
    # Greedy policy from Q
    # -----------------------------
    Pi = np.zeros((S, A), dtype=float)
    for s in range(S):
        if terminal_mask[s]:
            continue
        Pi[s, np.argmax(Q[s])] = 1.0

    # -----------------------------
    # Policy transition matrix
    # P_pi[s,s'] = Σ_a π(a|s) T[s,a,s']
    # -----------------------------
    P_pi = np.zeros((S, S), dtype=float)
    for s in range(S):
        for a in range(A):
            if Pi[s, a] > 0:
                P_pi[s] += Pi[s, a] * T[s, a]

        # absorbing fallback (safety)
        if P_pi[s].sum() == 0:
            P_pi[s, s] = 1.0

    cont = (~terminal_mask).astype(float)

    # -----------------------------
    # Iterative policy SFs ψ(s)
    # -----------------------------
    Psi_s = np.zeros((S, D), dtype=float)

    for _ in range(max_iters):
        Psi_old = Psi_s.copy()

        for s in range(S):
            if terminal_mask[s]:
                continue

            exp_phi_next = P_pi[s] @ Phi
            exp_psi_next = P_pi[s] @ Psi_old

            Psi_s[s] = exp_phi_next + gamma * cont[s] * exp_psi_next

        if np.max(np.abs(Psi_s - Psi_old)) < tol:
            break

    # -----------------------------
    # State–action successor features ψ(s,a)
    # -----------------------------
    Psi_sa = np.zeros((S, A, D), dtype=float)
    for s in range(S):
        for a in range(A):
            p_next = T[s, a]
            exp_phi_next = p_next @ Phi
            exp_psi_next = p_next @ Psi_s
            Psi_sa[s, a] = exp_phi_next + gamma * cont[s] * exp_psi_next

    return Psi_sa, Psi_s

def _policy_eval_worker(args):
    T, r_next, policy, terminal_mask, gamma, theta, max_iters = args
    return policy_evaluation_next_state(
        T=T,
        r_next=r_next,
        policy=policy,
        terminal_mask=terminal_mask,
        gamma=gamma,
        theta=theta,
        max_iters=max_iters,
    )

def policy_evaluation_next_state_multi(
    mdps,
    r_next_list,
    policy_list,
    gamma,
    theta=1e-8,
    max_iters=200000,
    n_jobs=None,
):
    """
    Parallel policy evaluation over multiple envs.

    mdps        : list of mdp dicts
    r_next_list : list of r_next vectors (one per env)
    policy_list : list of policies (one per env)
    """
    if n_jobs is None:
        n_jobs = cpu_count()

    args = [
        (
            mdp["T"],
            r_next,
            policy,
            mdp["terminal"],
            gamma,
            theta,
            max_iters,
        )
        for mdp, r_next, policy in zip(mdps, r_next_list, policy_list)
    ]

    with Pool(n_jobs) as pool:
        Vs = pool.map(_policy_eval_worker, args)

    return Vs

def _vi_worker(args):
    T, r_next, terminal_mask, gamma, theta, max_iters = args
    return value_iteration_next_state(
        T=T,
        r_next=r_next,
        terminal_mask=terminal_mask,
        gamma=gamma,
        theta=theta,
        max_iters=max_iters,
    )

def value_iteration_next_state_multi(
    mdps,
    r_next_list,
    gamma,
    theta=1e-8,
    max_iters=200000,
    n_jobs=None,
):
    """
    Parallel value iteration over multiple envs.

    Returns:
        V_list, Q_list, pi_list
    """
    if n_jobs is None:
        n_jobs = cpu_count()

    args = [
        (
            mdp["T"],
            r_next,
            mdp["terminal"],
            gamma,
            theta,
            max_iters,
        )
        for mdp, r_next in zip(mdps, r_next_list)
    ]

    with Pool(n_jobs) as pool:
        results = pool.map(_vi_worker, args)

    V_list, Q_list, pi_list = zip(*results)
    return list(V_list), list(Q_list), list(pi_list)

def _sf_worker(args):
    T, Phi, Q, terminal_mask, gamma, tol, max_iters = args
    return compute_successor_features_from_q_next_state(
        T=T,
        Phi=Phi,
        Q=Q,
        terminal_mask=terminal_mask,
        gamma=gamma,
        tol=tol,
        max_iters=max_iters,
    )

def compute_successor_features_multi(
    mdps,
    Q_list,
    gamma,
    tol=1e-10,
    max_iters=100000,
    n_jobs=None,
):
    """
    Parallel successor feature computation.
    """
    if n_jobs is None:
        n_jobs = cpu_count()

    args = [
        (
            mdp["T"],
            mdp["Phi"],
            Q,
            mdp["terminal"],
            gamma,
            tol,
            max_iters,
        )
        for mdp, Q in zip(mdps, Q_list)
    ]

    with Pool(n_jobs) as pool:
        results = pool.map(_sf_worker, args)

    Psi_sa_list, Psi_s_list = zip(*results)
    return list(Psi_sa_list), list(Psi_s_list)