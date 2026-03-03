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

ACT_LEFT = 0
ACT_RIGHT = 1
ACT_FORWARD = 2
ACTIONS = [ACT_LEFT, ACT_RIGHT, ACT_FORWARD]


def l2_normalize(w, eps=1e-8):
    n = np.linalg.norm(w)
    return w if n < eps else w / n

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
# Feedback generation functions
# ---------------------------

def generate_random_trajectories_from_state(
    start_state,
    n_trajs,
    wall_mask,
    goal_yx,
    lava_mask,
    max_horizon=150,
    seed=0,
):
    rng = np.random.default_rng(seed)
    #print(start_state)
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
    n_trajs_per_state=200,
    max_horizon=150,
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

    #states = np.arange(mdp["T"].shape[0])
    #enumerate_states()

    pool = generate_trajectory_pool(
        states=mdp["states"],
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
    n_trajs_per_state=200,
    max_horizon=150,
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
            mdp["true_w"],   # ← pull ground-truth reward from the MDP
            gamma,
            n_pairs,
            i,               # seed
            num_random_trajs,
            estop_beta,
        )
        for i, (trajs, mdp) in enumerate(zip(traj_pools, mdps))
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
    n_trajs_per_state=200,
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

##############################################
# Budgetting
##############################################

from dataclasses import dataclass
from typing import Any, Literal, Optional, Dict

AtomType = Literal["demo", "pairwise", "estop", "improvement"]

@dataclass(frozen=True)
class Atom:
    atom_type: AtomType
    env_id: int
    payload: Any


AllocMethod = Literal["uniform", "dirichlet", "sparse_poisson"]

@dataclass
class DemoSpec_minigrid:
    enabled: bool
    env_fraction: float
    state_fraction: Optional[float] = None
    total_state_budget: Optional[int] = None
    num_rollouts_per_state: int = 1
    max_steps: int = 1

@dataclass
class FeedbackSpec_minigrid:
    enabled: bool
    total_budget: int
    alloc_method: AllocMethod
    alloc_params: Dict

@dataclass
class GenerationSpec_minigrid:
    seed: int
    demo: Optional[DemoSpec_minigrid] = None
    pairwise: Optional[FeedbackSpec_minigrid] = None
    estop: Optional[FeedbackSpec_minigrid] = None
    improvement: Optional[FeedbackSpec_minigrid] = None

def allocate_budget(num_envs, total_budget, method, params, rng):
    if total_budget <= 0:
        return np.zeros(num_envs, dtype=int)

    if method == "uniform":
        base = total_budget // num_envs
        rem = total_budget % num_envs
        alloc = np.full(num_envs, base)
        alloc[:rem] += 1
        return alloc

    if method == "dirichlet":
        alpha = params.get("alpha", 1.0)
        w = rng.dirichlet([alpha] * num_envs)
        alloc = np.floor(w * total_budget).astype(int)
        alloc[0] += total_budget - alloc.sum()
        return alloc

    if method == "sparse_poisson":
        p = params["p_active"]
        mean = params["mean"]
        alloc = np.zeros(num_envs, dtype=int)
        active = rng.random(num_envs) < p
        alloc[active] = rng.poisson(mean, size=active.sum())
        scale = total_budget / max(alloc.sum(), 1)
        alloc = np.floor(alloc * scale).astype(int)
        return alloc

    raise ValueError(f"Unknown alloc_method {method}")


## Can I multiprocess that
def generate_demo_atoms(mdps, pi_list, spec, rng, enumerate_states):
    num_envs = len(mdps)
    atoms_per_env = [[] for _ in range(num_envs)]

    env_mask = rng.random(num_envs) < spec.env_fraction

    for env_id, (mdp, pi) in enumerate(zip(mdps, pi_list)):
        if not env_mask[env_id]:
            continue

        states = enumerate_states(mdp["size"], mdp["wall_mask"])
        valid = [
            s for s in states
            if not mdp["terminal"][mdp["idx_of"][s]]
        ]

        if not valid:
            continue

        if spec.total_state_budget is not None:
            k = min(spec.total_state_budget, len(valid))
        else:
            k = int(len(valid) * spec.state_fraction)

        if k <= 0:
            continue

        # ✅ SAMPLE INDICES, NOT STATES
        idx = rng.choice(len(valid), size=k, replace=False)
        states_sel = [valid[i] for i in idx]

        for s in states_sel:
            a = int(pi[mdp["idx_of"][s]])
            atoms_per_env[env_id].append(
                Atom("demo", env_id, (s, a))
            )

    return atoms_per_env

def subsample_atoms(atoms_per_env, spec, rng):
    num_envs = len(atoms_per_env)
    alloc = allocate_budget(
        num_envs,
        spec.total_budget,
        spec.alloc_method,
        spec.alloc_params,
        rng,
    )

    selected = [[] for _ in range(num_envs)]
    for env_id in range(num_envs):
        atoms = atoms_per_env[env_id]
        if not atoms or alloc[env_id] == 0:
            continue
        k = min(len(atoms), alloc[env_id])
        idx = rng.choice(len(atoms), size=k, replace=False)
        selected[env_id] = [atoms[i] for i in idx]

    return selected

## multi process that across env?
def generate_feedback_candidate_atoms(
    mdps,
    traj_pools,
    spec,
    rng,
    atom_type,
    pairwise_fn,
    estop_fn,
    improvement_fn,
):
    atoms_per_env = [[] for _ in range(len(mdps))]

    for env_id, (mdp, trajs) in enumerate(zip(mdps, traj_pools)):
        if atom_type == "pairwise":
            pairs = pairwise_fn(
                trajs, mdp, mdp["true_w"],
                n_pairs=spec.total_budget,
                seed=env_id,
            )
            for p in pairs:
                atoms_per_env[env_id].append(
                    Atom("pairwise", env_id, p)
                )

        elif atom_type == "estop":
            for traj in trajs:
                atoms_per_env[env_id].append(
                    Atom("estop", env_id, estop_fn(traj, mdp, mdp["true_w"]))
                )

        elif atom_type == "improvement":
            corrs = improvement_fn(trajs, mdp, mdp["true_w"])
            for c in corrs:
                atoms_per_env[env_id].append(
                    Atom("improvement", env_id, c)
                )

    return subsample_atoms(atoms_per_env, spec, rng)


def generate_candidate_atoms_for_scot_minigrid(
    mdps,
    pi_list,
    spec,
    enumerate_states,
    n_trajs_per_state=200,
    max_horizon=100,
):
    """
    Self-contained atom generation for SCOT in MiniGrid LavaWorlds.

    This function internally:
      - generates trajectory pools
      - generates demo atoms
      - generates pairwise / estop / improvement atoms
      - applies budgeting

    No external function handles required.
    """

    rng = np.random.default_rng(spec.seed)
    num_envs = len(mdps)
    atoms_per_env = [[] for _ in range(num_envs)]

    # --------------------------------------------------
    # 1) Trajectory pools (internal)
    # --------------------------------------------------
    traj_pools = generate_trajectory_pools_multi(
        mdps=mdps,
        n_trajs_per_state=n_trajs_per_state,
        max_horizon=max_horizon,
    )

    # --------------------------------------------------
    # 2) Demo atoms
    # --------------------------------------------------
    if spec.demo and spec.demo.enabled:
        demo_atoms = generate_demo_atoms(
            mdps=mdps,
            pi_list=pi_list,
            spec=spec.demo,
            rng=rng,
            enumerate_states=enumerate_states,
        )
        for i in range(num_envs):
            atoms_per_env[i].extend(demo_atoms[i])

    # --------------------------------------------------
    # 3) Pairwise atoms
    # --------------------------------------------------
    if spec.pairwise and spec.pairwise.enabled:
        sel = generate_feedback_candidate_atoms(
            mdps=mdps,
            traj_pools=traj_pools,
            spec=spec.pairwise,
            rng=rng,
            atom_type="pairwise",
            pairwise_fn=generate_pairwise_preferences,
            estop_fn=None,
            improvement_fn=None,
        )
        for i in range(num_envs):
            atoms_per_env[i].extend(sel[i])

    # --------------------------------------------------
    # 4) E-stop atoms
    # --------------------------------------------------
    if spec.estop and spec.estop.enabled:
        sel = generate_feedback_candidate_atoms(
            mdps=mdps,
            traj_pools=traj_pools,
            spec=spec.estop,
            rng=rng,
            atom_type="estop",
            pairwise_fn=None,
            estop_fn=simulate_human_estop_one_mdp,
            improvement_fn=None,
        )
        for i in range(num_envs):
            atoms_per_env[i].extend(sel[i])

    # --------------------------------------------------
    # 5) Improvement atoms
    # --------------------------------------------------
    if spec.improvement and spec.improvement.enabled:
        sel = generate_feedback_candidate_atoms(
            mdps=mdps,
            traj_pools=traj_pools,
            spec=spec.improvement,
            rng=rng,
            atom_type="improvement",
            pairwise_fn=None,
            estop_fn=None,
            improvement_fn=generate_correction_feedback,
        )
        for i in range(num_envs):
            atoms_per_env[i].extend(sel[i])

    return atoms_per_env
