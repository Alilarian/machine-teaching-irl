from multiprocessing import Pool, cpu_count

import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
#from utils import remove_redundant_constraints

#from __future__ import annotations
import numpy as np

from .minigrid_lava_generator import rollout_random_trajectory, enumerate_states

from scipy.special import logsumexp
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from typing import List
import numpy as np

ACT_LEFT = 0
ACT_RIGHT = 1
ACT_FORWARD = 2
ACTIONS = [ACT_LEFT, ACT_RIGHT, ACT_FORWARD]

def l2_normalize(w, eps=1e-8):
    n = np.linalg.norm(w)
    return w if n < eps else w / n

def policy_evaluation_next_state(
    mdp,
    theta,
    policy,
    gamma,
    tol=1e-8,
    max_iters=200000,
):
    T = mdp["T"]
    Phi = mdp["Phi"]
    terminal_mask = mdp["terminal"]

    r_next = Phi @ theta

    S, A, S2 = T.shape
    assert S == S2

    V = np.zeros(S, dtype=float)
    cont = (~terminal_mask).astype(float)

    for _ in range(max_iters):
        delta = 0.0
        for s in range(S):
            if terminal_mask[s]:
                continue

            a = int(policy[s])
            v_new = np.sum(T[s, a] * (r_next + gamma * cont * V))
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new

        if delta < tol:
            break

    return V

def value_iteration_next_state(
    mdp,
    theta,
    gamma,
    tol=1e-8,
    max_iters=200000,
):
    T = mdp["T"]
    Phi = mdp["Phi"]
    terminal_mask = mdp["terminal"]

    r_next = Phi @ theta

    S, A, S2 = T.shape
    assert S == S2

    V = np.zeros(S)
    Q = np.zeros((S, A))
    cont = (~terminal_mask).astype(float)

    for _ in range(max_iters):
        delta = 0.0
        for s in range(S):
            if terminal_mask[s]:
                continue

            for a in range(A):
                Q[s, a] = np.sum(T[s, a] * (r_next + gamma * cont * V))

            v_new = np.max(Q[s])
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new

        if delta < tol:
            break

    pi = np.zeros(S, dtype=int)
    for s in range(S):
        if not terminal_mask[s]:
            pi[s] = np.argmax(Q[s])

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
    mdp, theta, policy, gamma, tol, max_iters = args
    return policy_evaluation_next_state(
        mdp=mdp,
        theta=theta,
        policy=policy,
        gamma=gamma,
        tol=tol,
        max_iters=max_iters,
    )

def policy_evaluation_next_state_multi(
    mdps,
    theta,
    policy_list,
    gamma,
    tol=1e-8,
    max_iters=200000,
    n_jobs=None,
):
    if n_jobs is None:
        n_jobs = cpu_count()

    args = [
        (mdp, theta, policy, gamma, tol, max_iters)
        for mdp, policy in zip(mdps, policy_list)
    ]

    with Pool(n_jobs) as pool:
        return pool.map(_policy_eval_worker, args)

def _vi_worker(args):
    mdp, theta, gamma, tol, max_iters = args
    return value_iteration_next_state(
        mdp=mdp,
        theta=theta,
        gamma=gamma,
        tol=tol,
        max_iters=max_iters,
    )

def value_iteration_next_state_multi(
    mdps,
    theta,
    gamma,
    tol=1e-8,
    max_iters=200000,
    n_jobs=None,
):
    if n_jobs is None:
        n_jobs = cpu_count()

    args = [
        (mdp, theta, gamma, tol, max_iters)
        for mdp in mdps
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

def generate_state_action_demos(states, pi, terminal_mask, idx_of):
    """
    states : iterable of (y,x,d)
    pi     : policy over MDP state indices
    """
    demos = []

    for s in states:
        i = idx_of[s]

        if terminal_mask[i]:
            continue

        demos.append((i, int(pi[i])))

    return demos

def _generate_demos_only_worker(args):
    mdp, pi = args

    states = enumerate_states(mdp["size"], mdp["wall_mask"])

    demos = generate_state_action_demos(
        states=states,
        pi=pi,
        terminal_mask=mdp["terminal"],
        idx_of=mdp["idx_of"],
    )

    return demos

def generate_demos_from_policies_multi(
    mdps,
    pi_list,
    n_jobs=None,
):
    """
    Generate state–action demos for all envs in parallel,
    given precomputed policies.

    Parameters
    ----------
    mdps : list of mdp dicts
    pi_list : list of policies (output of value_iteration_next_state_multi)
    n_jobs : number of processes

    Returns
    -------
    demos_list : list[list[(s,a)]]
        One demo list per env
    """
    assert len(mdps) == len(pi_list)

    if n_jobs is None:
        n_jobs = cpu_count()

    args = [
        (mdp, pi)
        for mdp, pi in zip(mdps, pi_list)
    ]

    with Pool(n_jobs) as pool:
        demos_list = pool.map(_generate_demos_only_worker, args)

    return demos_list

def constraints_from_demos_next_state(
    demos,
    Psi_sa,
    terminal_mask=None,
    normalize=True,
    tol=1e-12,
):
    """
    Builds linear reward constraints from demos using successor features.

    Each constraint is:
        (ψ(s,a*) - ψ(s,a)) · θ >= 0     for all a != a*

    Inputs:
      demos         : list of (s, a_star) pairs (state index, optimal action)
      Psi_sa        : (S, A, D) successor features (NEXT-state convention)
      terminal_mask : optional (S,) boolean mask
      normalize     : L2-normalize constraint vectors
      tol           : skip near-zero constraints

    Returns:
      constraints : list of constraint vectors v ∈ R^D
    """
    Psi_sa = np.asarray(Psi_sa)
    S, A, D = Psi_sa.shape
    constraints = []

    if demos is None:
        return constraints

    for s, a_star in demos:
        if s is None or a_star is None:
            continue

        s = int(s)
        a_star = int(a_star)

        if not (0 <= s < S) or not (0 <= a_star < A):
            continue

        if terminal_mask is not None and terminal_mask[s]:
            continue

        psi_star = Psi_sa[s, a_star]

        for a in range(A):
            if a == a_star:
                continue

            diff = psi_star - Psi_sa[s, a]
            norm = np.linalg.norm(diff)

            if norm <= tol:
                continue

            constraints.append(diff / norm if normalize else diff)

    return constraints

def _constraints_from_demos_worker(args):
    """
    Worker: extract constraints for ONE env.
    """
    demos, Psi_sa, terminal_mask, normalize, tol = args

    return constraints_from_demos_next_state(
        demos=demos,
        Psi_sa=Psi_sa,
        terminal_mask=terminal_mask,
        normalize=normalize,
        tol=tol,
    )

def constraints_from_demos_next_state_multi(
    demos_list,
    Psi_sa_list,
    terminal_mask_list=None,
    normalize=True,
    tol=1e-12,
    n_jobs=None,
):
    """
    Parallel constraint extraction across envs.

    Parameters
    ----------
    demos_list : list[list[(s,a)]]
        One demo list per env
    Psi_sa_list : list[np.ndarray]
        One (S,A,D) successor-feature tensor per env
    terminal_mask_list : list[np.ndarray] or None
        One terminal mask per env (optional)
    normalize : bool
    tol : float
    n_jobs : int

    Returns
    -------
    constraints_per_env : list[list[np.ndarray]]
        constraints_per_env[i] = constraints from env i
    """
    assert len(demos_list) == len(Psi_sa_list)

    if terminal_mask_list is None:
        terminal_mask_list = [None] * len(demos_list)
    else:
        assert len(terminal_mask_list) == len(demos_list)

    if n_jobs is None:
        n_jobs = cpu_count()

    args = [
        (demos, Psi_sa, terminal_mask, normalize, tol)
        for demos, Psi_sa, terminal_mask in zip(
            demos_list, Psi_sa_list, terminal_mask_list
        )
    ]

    with Pool(n_jobs) as pool:
        constraints_per_env = pool.map(_constraints_from_demos_worker, args)

    return constraints_per_env

# ---------------------------
# Extract constraints from feedbacks
# ---------------------------
def constraints_from_demo_atom_next_state(
    demo_payload,
    Psi_sa,
    idx_of,
    terminal_mask=None,
    normalize=True,
    tol=1e-12,
):
    """
    Demo payload is either:
      - (s, a_star) where s is a tuple state, needs idx_of
      - (s_idx, a_star) where s_idx is already int

    Returns: list[np.ndarray] constraint vectors
    """
    if demo_payload is None:
        return []

    s, a_star = demo_payload

    # Map state to index if needed
    if isinstance(s, (tuple, list, np.ndarray)):
        s_idx = idx_of[tuple(s)]
    else:
        s_idx = int(s)

    a_star = int(a_star)

    Psi_sa = np.asarray(Psi_sa)
    S, A, D = Psi_sa.shape

    if not (0 <= s_idx < S) or not (0 <= a_star < A):
        return []

    if terminal_mask is not None and terminal_mask[s_idx]:
        return []

    psi_star = Psi_sa[s_idx, a_star]
    out = []

    for a in range(A):
        if a == a_star:
            continue

        diff = psi_star - Psi_sa[s_idx, a]
        n = np.linalg.norm(diff)
        if n <= tol:
            continue

        out.append(diff / n if normalize else diff)

    return out



def trajectory_expected_features(
    traj,
    mdp,
):
    """
    Compute discounted feature count:

        Φ(τ) = Σ γ^t φ(s_{t+1})

    Uses mdp["Phi"] and mdp["idx_of"] directly.
    """

    Phi = mdp["Phi"]          # (S, D)
    idx_of = mdp["idx_of"]
    gamma = mdp["gamma"]

    D = Phi.shape[1]
    phi_sum = np.zeros(D, dtype=float)

    g = 1.0

    for (_, _, sp) in traj:
        s_idx = idx_of[sp]
        phi_sum += g * Phi[s_idx]
        g *= gamma

    return phi_sum

def constraints_from_pairwise_preferences(
    pairwise_prefs,
    mdp,
    normalize=True,
    tol=1e-12,
):
    constraints = []

    for tau_pos, tau_neg in pairwise_prefs:
        psi_pos = trajectory_expected_features(tau_pos, mdp)
        psi_neg = trajectory_expected_features(tau_neg, mdp)

        diff = psi_pos - psi_neg
        norm = np.linalg.norm(diff)

        if norm <= tol:
            continue

        constraints.append(diff / norm if normalize else diff)

    return constraints

def constraints_from_correction_feedback(
    corrections,
    mdp,
    normalize=True,
    tol=1e-12,
):
    constraints = []

    for tau_new, tau_old in corrections:
        psi_new = trajectory_expected_features(tau_new, mdp)
        psi_old = trajectory_expected_features(tau_old, mdp)

        diff = psi_new - psi_old
        norm = np.linalg.norm(diff)

        if norm <= tol:
            continue

        constraints.append(diff / norm if normalize else diff)

    return constraints
def constraints_from_estop_feedback(
    estop_feedback,
    mdp,
    normalize=True,
    tol=1e-12,
):
    constraints = []

    for traj, t_stop in estop_feedback:
        prefix = traj[: t_stop + 1]

        psi_prefix = trajectory_expected_features(prefix, mdp)
        psi_full   = trajectory_expected_features(traj, mdp)

        diff = psi_prefix - psi_full
        norm = np.linalg.norm(diff)

        if norm <= tol:
            continue

        constraints.append(diff / norm if normalize else diff)

    return constraints

def constraints_from_single_atom(
    atom,
    *,
    mdp,
    Psi_sa=None,
    terminal_mask=None,
    normalize=True,
    tol=1e-12,
):
    t = atom.atom_type

    if t == "demo":
        if Psi_sa is None:
            raise ValueError("Demo atom encountered but Psi_sa was not provided.")
        return constraints_from_demo_atom_next_state(
            atom.payload,
            Psi_sa=Psi_sa,
            idx_of=mdp["idx_of"],
            terminal_mask=terminal_mask,
            normalize=normalize,
            tol=tol,
        )

    if t == "pairwise":
        return constraints_from_pairwise_preferences(
            [atom.payload], mdp,
            normalize=normalize, tol=tol
        )

    if t == "improvement":
        return constraints_from_correction_feedback(
            [atom.payload], mdp,
            normalize=normalize, tol=tol
        )

    if t == "estop":
        return constraints_from_estop_feedback(
            [atom.payload], mdp,
            normalize=normalize, tol=tol
        )

    raise ValueError(f"Unknown atom_type: {t}")

from multiprocessing import Pool, cpu_count


def _constraints_from_atoms_env_per_atom_worker(args):
    """
    Worker: extract constraints PER ATOM for ONE environment.
    """
    (
        atoms,
        mdp,
        Psi_sa,
        terminal_mask,
        normalize,
        tol,
    ) = args

    env_constraints = []

    for atom in atoms:
        atom_constraints = constraints_from_single_atom(
            atom,
            mdp=mdp,
            Psi_sa=Psi_sa,
            terminal_mask=terminal_mask,
            normalize=normalize,
            tol=tol,
        )

        # Always normalize to list[np.ndarray]
        if atom_constraints is None:
            atom_constraints = []
        elif isinstance(atom_constraints, np.ndarray):
            if atom_constraints.ndim == 1:
                atom_constraints = [atom_constraints]
            elif atom_constraints.ndim == 2:
                atom_constraints = [
                    atom_constraints[i, :]
                    for i in range(atom_constraints.shape[0])
                ]
            else:
                raise ValueError(
                    f"Unexpected constraint array shape {atom_constraints.shape}"
                )
        else:
            atom_constraints = list(atom_constraints)

        env_constraints.append(atom_constraints)

    return env_constraints

def constraints_from_atoms_multi_env(
    atoms_per_env,
    mdps,
    *,
    Psi_sa_list=None,
    terminal_mask_list=None,
    normalize=True,
    tol=1e-12,
    n_jobs=None,
):
    """
    Convert atoms into constraints for ALL environments, preserving atom identity.

    Returns:
        constraints_per_env[e][i] = list of constraint vectors
                                      from atom i in env e
    """

    assert len(atoms_per_env) == len(mdps), \
        "Length mismatch among atoms_per_env / mdps"

    E = len(atoms_per_env)

    # Optional lists
    if Psi_sa_list is None:
        Psi_sa_list = [None] * E
    else:
        assert len(Psi_sa_list) == E, \
            "Psi_sa_list length mismatch"

    if terminal_mask_list is None:
        terminal_mask_list = [None] * E
    else:
        assert len(terminal_mask_list) == E, \
            "terminal_mask_list length mismatch"

    # Early check for demo atoms
    for e, atoms in enumerate(atoms_per_env):
        if any(getattr(a, "atom_type", None) == "demo" for a in atoms):
            if Psi_sa_list[e] is None:
                raise ValueError(
                    f"Env {e} has demo atoms but Psi_sa_list[{e}] is None."
                )

    if n_jobs is None:
        n_jobs = cpu_count()

    args = [
        (
            atoms,
            mdp,
            Psi_sa,
            terminal_mask,
            normalize,
            tol,
        )
        for atoms, mdp, Psi_sa, terminal_mask in zip(
            atoms_per_env,
            mdps,
            Psi_sa_list,
            terminal_mask_list,
        )
    ]

    with Pool(processes=n_jobs) as pool:
        constraints_per_env = pool.map(
            _constraints_from_atoms_env_per_atom_worker,
            args,
        )

    # HARD invariant check
    for e in range(E):
        if len(constraints_per_env[e]) != len(atoms_per_env[e]):
            raise RuntimeError(
                f"Env {e}: atom/constraint mismatch "
                f"({len(atoms_per_env[e])} atoms vs "
                f"{len(constraints_per_env[e])} constraint lists)"
            )

    return constraints_per_env

################################# regret computation

import numpy as np

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _as_action_list(policy, S, idx_of=None, default_action=0):
    """
    Convert policy into an action list of length S.

    Supported inputs:
      1) action-list: array/list shape (S,)
      2) list of (s, a) where s is int state-index
      3) list of (s, a) where s is tuple state and idx_of is provided

    Anything missing gets default_action.
    """
    # Case 1: already action list
    if isinstance(policy, (list, np.ndarray)) and len(policy) == S and \
       not (len(policy) > 0 and isinstance(policy[0], (tuple, list)) and len(policy[0]) == 2):
        return np.asarray(policy, dtype=int)

    # Case 2/3: list of (s,a)
    pi = np.full(S, int(default_action), dtype=int)
    if policy is None:
        return pi

    for s, a in policy:
        if isinstance(s, (tuple, list, np.ndarray)):
            if idx_of is None:
                raise ValueError("Policy uses tuple-states but idx_of was not provided.")
            s = idx_of[tuple(s)]
        s = int(s)
        if 0 <= s < S:
            pi[s] = int(a)
    return pi


def policy_evaluation_next_state_stochastic(
    mdp,
    theta,
    Pi,               # (S,A) rows sum to 1 for non-terminal states
    gamma=None,
    tol=1e-8,
    max_iters=200000,
):
    """
    Policy evaluation for STOCHASTIC policies under NEXT-STATE rewards:
        V(s) = Σ_a π(a|s) Σ_{s'} T[s,a,s'] [ r(s') + γ * 1[~terminal(s')] * V(s') ]

    Uses r(s') = Phi[s'] @ theta.

    Returns: V (S,)
    """
    T = mdp["T"]
    Phi = mdp["Phi"]
    terminal = mdp["terminal"]
    if gamma is None:
        gamma = mdp.get("gamma", 0.99)

    S, A, S2 = T.shape
    assert S == S2
    Pi = np.asarray(Pi, dtype=float)
    assert Pi.shape == (S, A)

    r_next = Phi @ theta
    cont = (~terminal).astype(float)

    V = np.zeros(S, dtype=float)

    for _ in range(max_iters):
        delta = 0.0
        V_old = V.copy()

        for s in range(S):
            if terminal[s]:
                continue

            # Expected next-state value under π
            v_new = 0.0
            for a in range(A):
                if Pi[s, a] == 0:
                    continue
                v_new += Pi[s, a] * np.sum(T[s, a] * (r_next + gamma * cont * V_old))

            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new

        if delta < tol:
            break

    return V


def policy_evaluation_next_state_uniform_random(
    mdp,
    theta,
    gamma=None,
    tol=1e-8,
    max_iters=200000,
):
    """
    Exact evaluation for uniform random policy (stochastic).
    """
    S = mdp["T"].shape[0]
    A = mdp["T"].shape[1]
    Pi = np.full((S, A), 1.0 / A, dtype=float)
    # For terminal states, it doesn't matter, but make it well-formed anyway
    terminal = mdp["terminal"]
    Pi[terminal, :] = 0.0
    return policy_evaluation_next_state_stochastic(
        mdp=mdp, theta=theta, Pi=Pi, gamma=gamma, tol=tol, max_iters=max_iters
    )


def mean_state_value(V, terminal_mask=None, include_terminal=False):
    """
    Consistent "np.mean(V)" but with an option to ignore terminals,
    which is usually what you actually want.
    """
    V = np.asarray(V, dtype=float)
    if terminal_mask is None or include_terminal:
        return float(np.mean(V))
    keep = ~np.asarray(terminal_mask, dtype=bool)
    if not np.any(keep):
        return 0.0
    return float(np.mean(V[keep]))


# ------------------------------------------------------------
# Expected Value Difference (single env)
# ------------------------------------------------------------
def expected_value_difference_next_state(
    eval_policy,
    mdp,
    theta,
    *,
    epsilon=1e-8,   # kept for API similarity; mapped to tol
    gamma=None,
    normalize_with_random_policy=False,
    include_terminal_in_mean=False,
    max_iters=200000,
):
    """
    EVD compatible with your mdp dict + VI/PE utilities.

    Computes:
        EVD = mean(V_opt) - mean(V_eval)

    Optional normalization (uniform random policy):
        NEVD = (mean(V_opt) - mean(V_eval)) / (mean(V_opt) - mean(V_rand))

    Notes:
      - Uses NEXT-STATE reward: r(s') = Phi[s']·theta (matches your VI/PE)
      - eval_policy can be:
          * action list length S
          * list of (s_idx, a)
          * list of (state_tuple, a) if mdp["idx_of"] exists
    """
    if gamma is None:
        gamma = mdp.get("gamma", 0.99)

    # 1) Optimal values via your VI (next-state reward)
    V_opt, Q_opt, pi_opt = value_iteration_next_state(
        mdp=mdp, theta=theta, gamma=gamma, tol=epsilon, max_iters=max_iters
    )

    # 2) Eval policy deterministic evaluation
    S = mdp["T"].shape[0]
    pi_eval = _as_action_list(eval_policy, S, idx_of=mdp.get("idx_of", None))
    V_eval = policy_evaluation_next_state(
        mdp=mdp, theta=theta, policy=pi_eval, gamma=gamma, tol=epsilon, max_iters=max_iters
    )

    m_opt  = mean_state_value(V_opt,  mdp["terminal"], include_terminal=include_terminal_in_mean)
    m_eval = mean_state_value(V_eval, mdp["terminal"], include_terminal=include_terminal_in_mean)

    diff = m_opt - m_eval

    if not normalize_with_random_policy:
        return diff

    # 3) Proper uniform random stochastic evaluation (not "random action per state")
    V_rand = policy_evaluation_next_state_uniform_random(
        mdp=mdp, theta=theta, gamma=gamma, tol=epsilon, max_iters=max_iters
    )

    m_rand = mean_state_value(V_rand, mdp["terminal"], include_terminal=include_terminal_in_mean)

    denom = (m_opt - m_rand)
    if abs(denom) < 1e-12:
        # Degenerate case: random is as good as optimal under this averaging
        return 0.0

    return diff / denom


# ------------------------------------------------------------
# Multi-env wrapper (no extra multiprocessing needed)
# ------------------------------------------------------------
def expected_value_difference_next_state_multi(
    eval_policies,
    mdps,
    theta,
    *,
    epsilon=1e-8,
    gamma=None,
    normalize_with_random_policy=False,
    include_terminal_in_mean=False,
    max_iters=200000,
):
    """
    Convenience wrapper: returns list of EVDs, one per env.
    """
    assert len(eval_policies) == len(mdps)
    out = []
    for pol, mdp in zip(eval_policies, mdps):
        out.append(
            expected_value_difference_next_state(
                eval_policy=pol,
                mdp=mdp,
                theta=theta,
                epsilon=epsilon,
                gamma=gamma,
                normalize_with_random_policy=normalize_with_random_policy,
                include_terminal_in_mean=include_terminal_in_mean,
                max_iters=max_iters,
            )
        )
    return out




############################################################################## this block is update with correct traj expected feature instead of using Psi

# def trajectory_successor_features(
#     traj,
#     Psi_s,
#     idx_of,
#     gamma=0.99,
# ):
#     """
#     Compute discounted feature sum of a trajectory:

#         Φ(τ) = Σ γ^t φ(s_{t+1})

#     where φ is raw state feature (NOT successor feature).
#     """

#     D = Psi_s.shape[1]
#     phi_sum = np.zeros(D, dtype=float)

#     g = 1.0

#     for (_, _, sp) in traj:
#         s_idx = idx_of[sp]
#         phi_sum += g * Psi_s[s_idx]
#         g *= gamma

#     return phi_sum



# def constraints_from_pairwise_preferences(
#     pairwise_prefs,
#     Psi_s,
#     idx_of,
#     gamma=0.99,
#     normalize=True,
#     tol=1e-12,
# ):
#     """
#     Extract linear constraints from pairwise trajectory preferences.
#     """
#     constraints = []

#     for tau_pos, tau_neg in pairwise_prefs:
#         psi_pos = trajectory_successor_features(tau_pos, Psi_s, idx_of, gamma)
#         psi_neg = trajectory_successor_features(tau_neg, Psi_s, idx_of, gamma)

#         diff = psi_pos - psi_neg
#         norm = np.linalg.norm(diff)

#         if norm <= tol:
#             continue

#         constraints.append(diff / norm if normalize else diff)

#     return constraints

# def constraints_from_correction_feedback(
#     corrections,
#     Psi_s,
#     idx_of,
#     gamma=0.99,
#     normalize=True,
#     tol=1e-12,
# ):
#     """
#     Extract constraints from correction (improvement) feedback.
#     """
#     constraints = []

#     for tau_new, tau_old in corrections:
#         psi_new = trajectory_successor_features(tau_new, Psi_s, idx_of, gamma)
#         psi_old = trajectory_successor_features(tau_old, Psi_s, idx_of, gamma)

#         diff = psi_new - psi_old
#         norm = np.linalg.norm(diff)

#         if norm <= tol:
#             continue

#         constraints.append(diff / norm if normalize else diff)

#     return constraints

# def constraints_from_estop_feedback(
#     estop_feedback,
#     Psi_s,
#     idx_of,
#     gamma=0.99,
#     normalize=True,
#     tol=1e-12,
# ):
#     """
#     Extract constraints from E-stop (trajectory truncation) feedback.
#     """
#     constraints = []

#     for traj, t_stop in estop_feedback:
#         # prefix trajectory
#         prefix = traj[: t_stop + 1]

#         psi_prefix = trajectory_successor_features(prefix, Psi_s, idx_of, gamma)
#         psi_full   = trajectory_successor_features(traj,    Psi_s, idx_of, gamma)

#         diff = psi_prefix - psi_full
#         norm = np.linalg.norm(diff)

#         if norm <= tol:
#             continue

#         constraints.append(diff / norm if normalize else diff)

#     return constraints

# def constraints_from_single_atom(
#     atom,
#     *,
#     Psi_s,
#     Psi_sa,
#     idx_of,
#     terminal_mask=None,
#     gamma=0.99,
#     normalize=True,
#     tol=1e-12,
# ):
#     """
#     Returns list[np.ndarray] constraints generated by THIS atom only.
#     """
#     t = atom.atom_type

#     if t == "demo":
#         if Psi_sa is None:
#             raise ValueError("Demo atom encountered but Psi_sa was not provided.")
#         return constraints_from_demo_atom_next_state(
#             atom.payload,
#             Psi_sa=Psi_sa,
#             idx_of=idx_of,
#             terminal_mask=terminal_mask,
#             normalize=normalize,
#             tol=tol,
#         )

#     if t == "pairwise":
#         return constraints_from_pairwise_preferences(
#             [atom.payload], Psi_s, idx_of,
#             gamma=gamma, normalize=normalize, tol=tol
#         )

#     if t == "improvement":
#         return constraints_from_correction_feedback(
#             [atom.payload], Psi_s, idx_of,
#             gamma=gamma, normalize=normalize, tol=tol
#         )

#     if t == "estop":
#         return constraints_from_estop_feedback(
#             [atom.payload], Psi_s, idx_of,
#             gamma=gamma, normalize=normalize, tol=tol
#         )

#     raise ValueError(f"Unknown atom_type: {t}")

# def _constraints_from_atoms_env_per_atom_worker(args):
#     """
#     Worker: extract constraints PER ATOM for ONE environment.
#     """
#     (
#         atoms,
#         Psi_s,
#         Psi_sa,
#         idx_of,
#         terminal_mask,
#         gamma,
#         normalize,
#         tol,
#     ) = args

#     env_constraints = []

#     for atom in atoms:
#         atom_constraints = constraints_from_single_atom(
#             atom,
#             Psi_s=Psi_s,
#             Psi_sa=Psi_sa,
#             idx_of=idx_of,
#             terminal_mask=terminal_mask,
#             gamma=gamma,
#             normalize=normalize,
#             tol=tol,
#         )

#         # Always normalize to list[np.ndarray]
#         if atom_constraints is None:
#             atom_constraints = []
#         elif isinstance(atom_constraints, np.ndarray):
#             if atom_constraints.ndim == 1:
#                 atom_constraints = [atom_constraints]
#             elif atom_constraints.ndim == 2:
#                 atom_constraints = [atom_constraints[i, :] for i in range(atom_constraints.shape[0])]
#             else:
#                 raise ValueError(f"Unexpected constraint array shape {atom_constraints.shape}")
#         else:
#             atom_constraints = list(atom_constraints)

#         env_constraints.append(atom_constraints)

#     return env_constraints


# def constraints_from_atoms_multi_env(
#     atoms_per_env,
#     Psi_s_list,
#     idx_of_list,
#     *,
#     Psi_sa_list=None,
#     terminal_mask_list=None,
#     gamma=0.99,
#     normalize=True,
#     tol=1e-12,
#     n_jobs=None,
# ):
#     """
#     Convert atoms into constraints for ALL environments, preserving atom identity.

#     Returns:
#       constraints_per_env[e][i] = list of constraint vectors from atom i in env e
#     """
#     assert len(atoms_per_env) == len(Psi_s_list) == len(idx_of_list), \
#         "Length mismatch among atoms_per_env / Psi_s_list / idx_of_list"

#     E = len(atoms_per_env)

#     # Default optionals
#     if Psi_sa_list is None:
#         Psi_sa_list = [None] * E
#     else:
#         assert len(Psi_sa_list) == E, "Psi_sa_list length mismatch"

#     if terminal_mask_list is None:
#         terminal_mask_list = [None] * E
#     else:
#         assert len(terminal_mask_list) == E, "terminal_mask_list length mismatch"

#     # Early check: if ANY demo atoms exist, Psi_sa must be provided for those envs
#     for e, atoms in enumerate(atoms_per_env):
#         if any(getattr(a, "atom_type", None) == "demo" for a in atoms):
#             if Psi_sa_list[e] is None:
#                 raise ValueError(
#                     f"Env {e} has demo atoms but Psi_sa_list[{e}] is None. "
#                     f"Pass Psi_sa_list to constraints_from_atoms_multi_env."
#                 )

#     if n_jobs is None:
#         n_jobs = cpu_count()

#     args = [
#         (
#             atoms,
#             Psi_s,
#             Psi_sa,
#             idx_of,
#             terminal_mask,
#             gamma,
#             normalize,
#             tol,
#         )
#         for atoms, Psi_s, Psi_sa, idx_of, terminal_mask in zip(
#             atoms_per_env, Psi_s_list, Psi_sa_list, idx_of_list, terminal_mask_list
#         )
#     ]

#     with Pool(processes=n_jobs) as pool:
#         constraints_per_env = pool.map(_constraints_from_atoms_env_per_atom_worker, args)

#     # HARD invariant check (keep it)
#     for e in range(E):
#         if len(constraints_per_env[e]) != len(atoms_per_env[e]):
#             raise RuntimeError(
#                 f"Env {e}: atom/constraint mismatch "
#                 f"({len(atoms_per_env[e])} atoms vs {len(constraints_per_env[e])} constraint lists)"
#             )

#     return constraints_per_env




###############################################################################################

def trajectory_expected_features(
    traj,
    mdp,
):
    """
    Compute discounted feature count:

        Φ(τ) = Σ γ^t φ(s_{t+1})

    Uses mdp["Phi"] and mdp["idx_of"] directly.
    """

    Phi = mdp["Phi"]          # (S, D)
    idx_of = mdp["idx_of"]
    gamma = mdp["gamma"]

    D = Phi.shape[1]
    phi_sum = np.zeros(D, dtype=float)

    g = 1.0

    for (_, _, sp) in traj:
        s_idx = idx_of[sp]
        phi_sum += g * Phi[s_idx]
        g *= gamma

    return phi_sum

def constraints_from_pairwise_preferences(
    pairwise_prefs,
    mdp,
    normalize=True,
    tol=1e-12,
):
    constraints = []

    for tau_pos, tau_neg in pairwise_prefs:
        psi_pos = trajectory_expected_features(tau_pos, mdp)
        psi_neg = trajectory_expected_features(tau_neg, mdp)

        diff = psi_pos - psi_neg
        norm = np.linalg.norm(diff)

        if norm <= tol:
            continue

        constraints.append(diff / norm if normalize else diff)

    return constraints

def constraints_from_correction_feedback(
    corrections,
    mdp,
    normalize=True,
    tol=1e-12,
):
    constraints = []

    for tau_new, tau_old in corrections:
        psi_new = trajectory_expected_features(tau_new, mdp)
        psi_old = trajectory_expected_features(tau_old, mdp)

        diff = psi_new - psi_old
        norm = np.linalg.norm(diff)

        if norm <= tol:
            continue

        constraints.append(diff / norm if normalize else diff)

    return constraints
def constraints_from_estop_feedback(
    estop_feedback,
    mdp,
    normalize=True,
    tol=1e-12,
):
    constraints = []

    for traj, t_stop in estop_feedback:
        prefix = traj[: t_stop + 1]

        psi_prefix = trajectory_expected_features(prefix, mdp)
        psi_full   = trajectory_expected_features(traj, mdp)

        diff = psi_prefix - psi_full
        norm = np.linalg.norm(diff)

        if norm <= tol:
            continue

        constraints.append(diff / norm if normalize else diff)

    return constraints

def constraints_from_single_atom(
    atom,
    *,
    mdp,
    Psi_sa=None,
    terminal_mask=None,
    normalize=True,
    tol=1e-12,
):
    t = atom.atom_type

    if t == "demo":
        if Psi_sa is None:
            raise ValueError("Demo atom encountered but Psi_sa was not provided.")
        return constraints_from_demo_atom_next_state(
            atom.payload,
            Psi_sa=Psi_sa,
            idx_of=mdp["idx_of"],
            terminal_mask=terminal_mask,
            normalize=normalize,
            tol=tol,
        )

    if t == "pairwise":
        return constraints_from_pairwise_preferences(
            [atom.payload], mdp,
            normalize=normalize, tol=tol
        )

    if t == "improvement":
        return constraints_from_correction_feedback(
            [atom.payload], mdp,
            normalize=normalize, tol=tol
        )

    if t == "estop":
        return constraints_from_estop_feedback(
            [atom.payload], mdp,
            normalize=normalize, tol=tol
        )

    raise ValueError(f"Unknown atom_type: {t}")

from multiprocessing import Pool, cpu_count


def _constraints_from_atoms_env_per_atom_worker(args):
    """
    Worker: extract constraints PER ATOM for ONE environment.
    """
    (
        atoms,
        mdp,
        Psi_sa,
        terminal_mask,
        normalize,
        tol,
    ) = args

    env_constraints = []

    for atom in atoms:
        atom_constraints = constraints_from_single_atom(
            atom,
            mdp=mdp,
            Psi_sa=Psi_sa,
            terminal_mask=terminal_mask,
            normalize=normalize,
            tol=tol,
        )

        # Always normalize to list[np.ndarray]
        if atom_constraints is None:
            atom_constraints = []
        elif isinstance(atom_constraints, np.ndarray):
            if atom_constraints.ndim == 1:
                atom_constraints = [atom_constraints]
            elif atom_constraints.ndim == 2:
                atom_constraints = [
                    atom_constraints[i, :]
                    for i in range(atom_constraints.shape[0])
                ]
            else:
                raise ValueError(
                    f"Unexpected constraint array shape {atom_constraints.shape}"
                )
        else:
            atom_constraints = list(atom_constraints)

        env_constraints.append(atom_constraints)

    return env_constraints

def constraints_from_atoms_multi_env(
    atoms_per_env,
    mdps,
    *,
    Psi_sa_list=None,
    terminal_mask_list=None,
    normalize=True,
    tol=1e-12,
    n_jobs=None,
):
    """
    Convert atoms into constraints for ALL environments, preserving atom identity.

    Returns:
        constraints_per_env[e][i] = list of constraint vectors
                                      from atom i in env e
    """

    assert len(atoms_per_env) == len(mdps), \
        "Length mismatch among atoms_per_env / mdps"

    E = len(atoms_per_env)

    # Optional lists
    if Psi_sa_list is None:
        Psi_sa_list = [None] * E
    else:
        assert len(Psi_sa_list) == E, \
            "Psi_sa_list length mismatch"

    if terminal_mask_list is None:
        terminal_mask_list = [None] * E
    else:
        assert len(terminal_mask_list) == E, \
            "terminal_mask_list length mismatch"

    # Early check for demo atoms
    for e, atoms in enumerate(atoms_per_env):
        if any(getattr(a, "atom_type", None) == "demo" for a in atoms):
            if Psi_sa_list[e] is None:
                raise ValueError(
                    f"Env {e} has demo atoms but Psi_sa_list[{e}] is None."
                )

    if n_jobs is None:
        n_jobs = cpu_count()

    args = [
        (
            atoms,
            mdp,
            Psi_sa,
            terminal_mask,
            normalize,
            tol,
        )
        for atoms, mdp, Psi_sa, terminal_mask in zip(
            atoms_per_env,
            mdps,
            Psi_sa_list,
            terminal_mask_list,
        )
    ]

    with Pool(processes=n_jobs) as pool:
        constraints_per_env = pool.map(
            _constraints_from_atoms_env_per_atom_worker,
            args,
        )

    # HARD invariant check
    for e in range(E):
        if len(constraints_per_env[e]) != len(atoms_per_env[e]):
            raise RuntimeError(
                f"Env {e}: atom/constraint mismatch "
                f"({len(atoms_per_env[e])} atoms vs "
                f"{len(constraints_per_env[e])} constraint lists)"
            )

    return constraints_per_env
