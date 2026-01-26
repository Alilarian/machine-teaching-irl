from multiprocessing import Pool, cpu_count

import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
#from utils import remove_redundant_constraints

#from __future__ import annotations
import numpy as np

from .minigrid_lava_generator import rollout_random_trajectory

from scipy.special import logsumexp

ACT_LEFT = 0
ACT_RIGHT = 1
ACT_FORWARD = 2
ACTIONS = [ACT_LEFT, ACT_RIGHT, ACT_FORWARD]

def l2_normalize(w, eps=1e-8):
    n = np.linalg.norm(w)
    return w if n < eps else w / n

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

def generate_state_action_demos(states, pi, terminal_mask):
    demos = []
    for i, _s in enumerate(states):
        if terminal_mask[i]:
            continue
        demos.append((i, int(pi[i])))
    return demos

def _generate_demos_only_worker(args):
    """
    Worker that ONLY generates state–action demos
    from a given policy.
    """
    mdp, pi = args

    states = np.arange(mdp["T"].shape[0])
    demos = generate_state_action_demos(
        states=states,
        pi=pi,
        terminal_mask=mdp["terminal"],
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
# Feedback generation functions
# ---------------------------

def generate_random_trajectories_from_state(
    start_state,
    n_trajs,
    wall_mask,
    goal_yx,
    lava_mask,
    max_horizon=30,
    seed=0,
):
    rng = np.random.default_rng(seed)
    return [
        rollout_random_trajectory(
            start_state,
            wall_mask,
            goal_yx,
            lava_mask,
            max_horizon=max_horizon,
            rng=rng,
        )
        for _ in range(n_trajs)
    ]

def generate_trajectory_pool(
    states,
    terminal_mask,
    wall_mask,
    goal_yx,
    lava_mask,
    n_trajs_per_state=5,
    max_horizon=30,
):
    
    
    pool = []
    for i, s in enumerate(states):
        if terminal_mask[i]:
            continue

        trajs = generate_random_trajectories_from_state(
            start_state=s,
            n_trajs=n_trajs_per_state,
            wall_mask=wall_mask,
            goal_yx=goal_yx,
            lava_mask=lava_mask,
            max_horizon=max_horizon,
            seed=i,
        )
        pool.extend(trajs)

    return pool

def _trajectory_pool_worker(args):
    (
        mdp,
        n_trajs_per_state,
        max_horizon,
    ) = args

    states = np.arange(mdp["T"].shape[0])

    pool = generate_trajectory_pool(
        states=states,
        terminal_mask=mdp["terminal"],
        wall_mask=mdp["wall_mask"],
        goal_yx=mdp["goal_yx"],
        lava_mask=mdp["lava_mask"],
        n_trajs_per_state=n_trajs_per_state,
        max_horizon=max_horizon,
    )

    return pool

def generate_trajectory_pools_multi(
    mdps,
    n_trajs_per_state=5,
    max_horizon=30,
    n_jobs=None,
):
    if n_jobs is None:
        n_jobs = cpu_count()

    args = [
        (mdp, n_trajs_per_state, max_horizon)
        for mdp in mdps
    ]

    with Pool(n_jobs) as pool:
        traj_pools = pool.map(_trajectory_pool_worker, args)

    return traj_pools

def simulate_human_estop_one_mdp(
    traj,
    mdp,
    theta_true,
    beta=2.0,
):
    """
    Compatible E-stop simulation.

    traj : list of (s, a, s_next)
    mdp  : mdp dict containing Phi
    """
    Phi = mdp["Phi"]
    idx_of = mdp["idx_of"] if "idx_of" in mdp else None

    def reward(sp):
        if idx_of is not None:
            sp_idx = idx_of[sp]
        else:
            sp_idx = sp
        return Phi[sp_idx] @ theta_true

    traj_len = len(traj)

    # full trajectory return
    full_reward = sum(reward(sp) for (_, _, sp) in traj)

    log_probs = []
    cumulative = 0.0

    for t in range(traj_len):
        _, _, sp = traj[t]
        cumulative += reward(sp)

        num = beta * cumulative
        den = logsumexp([beta * full_reward, num])
        log_probs.append(num - den)

    t_stop = int(np.argmax(log_probs))
    return (traj, t_stop)

def trajectory_return(
    traj,
    Phi,
    theta,
    gamma=0.99,
):
    """
    traj: list of (s, a, s_next)
    """
    theta = l2_normalize(theta)
    ret = 0.0
    g = 1.0

    for (_s, _a, sp) in traj:
        sp_idx = Phi["idx_of"][sp]
        r = Phi["Phi"][sp_idx] @ theta
        ret += g * r
        g *= gamma

    return ret

def generate_pairwise_preferences(
    trajectories,
    mdp,
    theta_true,
    gamma=0.99,
    n_pairs=1000,
    seed=0,
):
    rng = np.random.default_rng(seed)
    prefs = []

    returns = [
        trajectory_return(traj, mdp, theta_true, gamma)
        for traj in trajectories
    ]

    N = len(trajectories)

    for _ in range(n_pairs):
        i, j = rng.choice(N, size=2, replace=False)

        if returns[i] == returns[j]:
            continue

        if returns[i] > returns[j]:
            prefs.append((trajectories[i], trajectories[j]))
        else:
            prefs.append((trajectories[j], trajectories[i]))

    return prefs

def simulate_correction_one(
    traj,
    mdp,
    theta_true,
    num_random_trajs=10,
    max_horizon=None,
):
    """
    Given an existing trajectory τ, attempt to find a better trajectory
    starting from the SAME start state.

    Returns:
        (tau_improved, tau_original)
        or None if no improvement found
    """
    start_state = traj[0][0]

    # use original length unless overridden
    horizon = len(traj) if max_horizon is None else max_horizon

    original_return = trajectory_return(traj, mdp, theta_true)
    best_traj = traj
    best_return = original_return

    rng = np.random.default_rng()

    for _ in range(num_random_trajs):
        new_traj = rollout_random_trajectory(
            start_state=start_state,
            wall_mask=mdp["wall_mask"],
            goal_yx=mdp["goal_yx"],
            lava_mask=mdp["lava_mask"],
            max_horizon=horizon,
            rng=rng,
        )

        if len(new_traj) == 0:
            continue

        new_return = trajectory_return(new_traj, mdp, theta_true)

        if new_return > best_return:
            best_return = new_return
            best_traj = new_traj

    if best_traj is traj:
        return None  # no improvement found

    return (best_traj, traj)

def generate_correction_feedback(
    trajectories,
    mdp,
    theta_true,
    num_random_trajs=10,
):
    """
    Generate correction (improvement) feedback:
    (tau_improved ≻ tau_original), same start state.
    """
    corrections = []

    for traj in trajectories:
        if len(traj) == 0:
            continue

        corr = simulate_correction_one(
            traj=traj,
            mdp=mdp,
            theta_true=theta_true,
            num_random_trajs=num_random_trajs,
        )

        if corr is not None:
            corrections.append(corr)

    return corrections

def _feedback_worker(args):
    (
        trajectories,
        mdp,
        theta_true,
        gamma,
        n_pairs,
        seed,
        num_random_trajs,
        estop_beta,
    ) = args

    # ---------------------------
    # Pairwise preferences
    # ---------------------------
    pairwise = generate_pairwise_preferences(
        trajectories=trajectories,
        mdp=mdp,
        theta_true=theta_true,
        gamma=gamma,
        n_pairs=n_pairs,
        seed=seed,
    )

    # ---------------------------
    # Correction feedback
    # ---------------------------
    corrections = generate_correction_feedback(
        trajectories=trajectories,
        mdp=mdp,
        theta_true=theta_true,
        num_random_trajs=num_random_trajs,
    )

    # ---------------------------
    # E-stop feedback
    # ---------------------------
    estops = []
    for traj in trajectories:
        if len(traj) == 0:
            continue
        estops.append(
            simulate_human_estop_one_mdp(
                traj=traj,
                mdp=mdp,
                theta_true=theta_true,
                beta=estop_beta,
            )
        )

    return {
        "pairwise": pairwise,
        "corrections": corrections,
        "estop": estops,
    }

def generate_feedback_multi(
    traj_pools,
    mdps,
    theta_true_list,
    gamma=0.99,
    n_pairs=1000,
    num_random_trajs=10,
    estop_beta=10.0,
    n_jobs=None,
):
    if n_jobs is None:
        n_jobs = cpu_count()

    args = [
        (
            trajs,
            mdp,
            theta_true,
            gamma,
            n_pairs,
            i,              # seed
            num_random_trajs,
            estop_beta,
        )
        for i, (trajs, mdp, theta_true) in enumerate(
            zip(traj_pools, mdps, theta_true_list)
        )
    ]

    with Pool(n_jobs) as pool:
        results = pool.map(_feedback_worker, args)

    pairwise_list   = [r["pairwise"]    for r in results]
    correction_list = [r["corrections"] for r in results]
    estop_list      = [r["estop"]       for r in results]

    return pairwise_list, correction_list, estop_list

def generate_random_feedback_pipeline_multi(
    mdps,
    theta_true_list,
    n_trajs_per_state=5,
    max_horizon=30,
    gamma=0.99,
    n_pairs=1000,
    num_random_trajs=10,
    estop_beta=10.0,
    n_jobs=None,
):
    # --------------------------------------------------
    # 1) Trajectory pools
    # --------------------------------------------------
    traj_pools = generate_trajectory_pools_multi(
        mdps=mdps,
        n_trajs_per_state=n_trajs_per_state,
        max_horizon=max_horizon,
        n_jobs=n_jobs,
    )

    # --------------------------------------------------
    # 2) Feedback: pairwise + corrections + e-stop
    # --------------------------------------------------
    pairwise_list, correction_list, estop_list = generate_feedback_multi(
        traj_pools=traj_pools,
        mdps=mdps,
        theta_true_list=theta_true_list,
        gamma=gamma,
        n_pairs=n_pairs,
        num_random_trajs=num_random_trajs,
        estop_beta=estop_beta,
        n_jobs=n_jobs,
    )

    return traj_pools, pairwise_list, correction_list, estop_list