# ============================================================
# generate_feedback.py  (FULL FIXED VERSION)
# ============================================================

import numpy as np
from scipy.special import logsumexp
import random
from .successor_features import build_Pi_from_q
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# 0. Atom abstraction
# ============================================================

class Atom:
    def __init__(self, env_idx, feedback_type, data, metadata=None):
        """
        env_idx: index of environment/MDP
        feedback_type: 'demo', 'random_traj', 'pairwise', 'estop', 'improvement', 'optimal_sa'
        data: payload (trajectory, pairwise tuple, (traj, t_stop), etc.)
        """
        self.env_idx = env_idx
        self.feedback_type = feedback_type
        self.data = data
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Atom(env={self.env_idx}, type={self.feedback_type})"


# ============================================================
# 1. Trajectory utilities
# ============================================================

def evaluate_trajectory(env, traj):
    """Compute total reward of a trajectory."""
    return sum(env.compute_reward(s) for s, _ in traj)


def generate_random_trajectory(env, max_horizon=25):
    """
    Generate a random trajectory using uniformly random actions.
    """
    traj = []
    obs = env.reset()
    terminal_states = obs["terminal states"]

    try:
        state = obs["agent"][0] * env.columns + obs["agent"][1]
    except Exception:
        state = obs["agent"][0] * env.size + obs["agent"][1]

    for _ in range(max_horizon):
        if state in terminal_states:
            traj.append((state, None))
            break

        action = np.random.choice(env.num_actions)
        next_state = np.random.choice(env.num_states, p=env.transitions[state][action])

        traj.append((state, action))
        state = next_state

    return traj


def generate_random_trajectory_from_state(env, start_state, length):
    traj = []
    state = start_state
    terminals = env.terminal_states

    for _ in range(length):
        if state in terminals:
            traj.append((state, None))
            break

        action = np.random.choice(env.num_actions)
        next_state = np.random.choice(env.num_states, p=env.transitions[state][action])

        traj.append((state, action))
        state = next_state

    return traj


# def generate_valid_trajectories(env, n, min_length=3, max_horizon=25):
#     trajs = []
#     while len(trajs) < n:
#         t = generate_random_trajectory(env, max_horizon=max_horizon)
#         if len(t) >= min_length:
#             trajs.append(t)
#     return trajs

def _rollout_one(env, min_length, max_horizon):
    t = generate_random_trajectory(env, max_horizon)
    return t if len(t) >= min_length else None


def generate_valid_trajectories(
    env,
    n,
    min_length=3,
    max_horizon=25,
    max_workers=8,
    oversample_factor=2,
):
    """
    Thread-parallel trajectory generation inside ONE env.
    """
    trajs = []
    needed = n

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        while len(trajs) < n:
            batch = oversample_factor * needed
            futures = [
                ex.submit(_rollout_one, env, min_length, max_horizon)
                for _ in range(batch)
            ]

            for f in futures:
                t = f.result()
                if t is not None:
                    trajs.append(t)
                    if len(trajs) >= n:
                        break

            needed = n - len(trajs)

    return trajs[:n]


# ============================================================
# 2. Q-based (optimal) trajectories
# ============================================================

def generate_q_optimal_trajectories(
    env,
    q_values,
    num_rollouts_per_state=1,
    max_steps=1,
    tie_eps=1e-10,
):
    S = env.get_num_states()
    A = env.get_num_actions()
    terminals = set(env.terminal_states or [])
    T = env.transitions

    opt_actions = [[] for _ in range(S)]
    for s in range(S):
        if s in terminals:
            continue
        row = q_values[s]
        max_q = np.max(row)
        opt_actions[s] = [a for a in range(A) if abs(row[a] - max_q) < tie_eps]

    trajectories = []
    for start_s in range(S):
        if start_s in terminals or not opt_actions[start_s]:
            continue

        for _ in range(num_rollouts_per_state):
            tau, s, steps = [], int(start_s), 0
            while steps < max_steps and s not in terminals:
                acts = opt_actions[s]
                if not acts:
                    break
                a = int(np.random.choice(acts))
                tau.append((s, a))
                s = int(np.random.choice(S, p=T[s, a]))
                steps += 1
            trajectories.append(tau)

    return trajectories


# ============================================================
# 3. Corrections
# ============================================================

# def simulate_corrections(env, trajs, num_random_trajs=25):
#     paired = []

#     for traj in trajs:
#         start_state = traj[0][0]
#         length = len(traj)

#         original_return = evaluate_trajectory(env, traj)
#         best_traj = traj
#         best_return = original_return

#         for _ in range(num_random_trajs):
#             new_traj = generate_random_trajectory_from_state(env, start_state, length)
#             new_return = evaluate_trajectory(env, new_traj)
#             if new_return > best_return:
#                 best_return = new_return
#                 best_traj = new_traj

#         paired.append((best_traj, traj))
#     return paired

def _simulate_improvement_one(env, traj, num_random_trajs):
    start_state = traj[0][0]
    length = len(traj)

    original_return = evaluate_trajectory(env, traj)
    best_traj = traj
    best_return = original_return

    for _ in range(num_random_trajs):
        new_traj = generate_random_trajectory_from_state(env, start_state, length)
        new_return = evaluate_trajectory(env, new_traj)
        if new_return > best_return:
            best_return = new_return
            best_traj = new_traj

    return (best_traj, traj)

def simulate_corrections(
    env,
    trajectories,
    *,
    num_random_trajs=25,
    max_workers=8,
):
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(
            ex.map(
                lambda t: _simulate_improvement_one(env, t, num_random_trajs),
                trajectories,
            )
        )


def compute_rewards(env, trajectories):
    return np.array([evaluate_trajectory(env, t) for t in trajectories])

def generate_pairwise_comparisons(
    env,
    trajectories,
    num_comparisons=10,
    max_trials=50,
):
    """
    O(K) expected time, no quadratic blowup.
    """
    rewards = compute_rewards(env, trajectories)
    n = len(trajectories)

    pairs = []
    seen = set()
    trials = 0

    while len(pairs) < num_comparisons and trials < max_trials * num_comparisons:
        i, j = np.random.choice(n, size=2, replace=False)
        if rewards[i] == rewards[j]:
            trials += 1
            continue

        key = (min(i, j), max(i, j))
        if key in seen:
            trials += 1
            continue

        seen.add(key)

        if rewards[i] > rewards[j]:
            pairs.append((trajectories[i], trajectories[j]))
        else:
            pairs.append((trajectories[j], trajectories[i]))

        trials += 1

    return pairs



def simulate_human_estop_one(env, full_trajectory, beta=2.0):
    traj_len = len(full_trajectory)
    full_reward = sum(env.compute_reward(s) for s, _ in full_trajectory)

    log_probs = []
    for t in range(traj_len):
        reward_to_t = sum(env.compute_reward(s) for s, _ in full_trajectory[:t+1])
        num = beta * reward_to_t
        den = logsumexp([beta * full_reward, num])
        log_probs.append(num - den)

    t_stop = int(np.argmax(log_probs))
    return (full_trajectory, t_stop)

def _simulate_estop_one(env, traj, beta):
    return simulate_human_estop_one(env, traj, beta)

from concurrent.futures import ThreadPoolExecutor

def simulate_human_estops(
    env,
    trajectories,
    *,
    beta=10.0,
    max_workers=8,
):
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(
            ex.map(lambda t: _simulate_estop_one(env, t, beta), trajectories)
        )


# ============================================================
# 5. Atom constructors
# ============================================================

def trajs_to_atoms(env_idx, trajs, feedback_type):
    return [Atom(env_idx, feedback_type, t) for t in trajs]

def pairwise_to_atoms(env_idx, pairs):
    return [Atom(env_idx, "pairwise", p) for p in pairs]

def estops_to_atoms(env_idx, estops):
    return [Atom(env_idx, "estop", e) for e in estops]

def corrections_to_atoms(env_idx, imps):
    return [Atom(env_idx, "improvement", imp) for imp in imps]


# ============================================================
# 6. Unified feedback → atoms
# ============================================================


def _generate_candidates_for_one_env(args):
    """
    Worker-safe, picklable function.
    """
    (
        env_idx,
        env,
        qv,
        use_q_demos,
        num_q_rollouts_per_state,
        q_demo_max_steps,
        tie_eps,
        use_pairwise,
        n_pairwise,
        use_estop,
        n_estops,
        use_improvement,
        n_improvements,
        n_random_for_improvement,
        base_min_length,
        base_max_horizon,
    ) = args

    C = []

    # ---------------- Q demos ----------------
    if use_q_demos:
        q_trajs = generate_q_optimal_trajectories(
            env,
            qv,
            num_rollouts_per_state=num_q_rollouts_per_state,
            max_steps=q_demo_max_steps,
            tie_eps=tie_eps,
        )
        C.extend(trajs_to_atoms(env_idx, q_trajs, "demo"))

    # ---------------- base trajectories ----------------
    needs_base = use_pairwise or use_estop or use_improvement
    if needs_base:
        base_count = max(n_pairwise, n_estops, n_improvements)
        base_trajs = generate_valid_trajectories(
            env,
            n=base_count,
            min_length=base_min_length,
            max_horizon=base_max_horizon,
        )

    # ---------------- pairwise ----------------
    if use_pairwise:
        pw = generate_pairwise_comparisons(
            env, base_trajs, num_comparisons=n_pairwise
        )
        C.extend(pairwise_to_atoms(env_idx, pw))

    # ---------------- estop ----------------
    if use_estop:
        estop_trajs = random.sample(base_trajs, min(n_estops, len(base_trajs)))
        estops  = simulate_human_estops(env, estop_trajs)
        #estops = [simulate_human_estop(env, t) for t in estop_trajs]
        C.extend(estops_to_atoms(env_idx, estops))

    # ---------------- improvement ----------------
    if use_improvement:
        imp_trajs = random.sample(base_trajs, min(n_improvements, len(base_trajs)))
        imps = simulate_corrections(
            env,
            imp_trajs,
            num_random_trajs=n_random_for_improvement,
        )
        C.extend(corrections_to_atoms(env_idx, imps))

    return env_idx, C

def generate_candidate_atoms_for_scot(
    envs,
    Q_list,
    *,
    max_workers=None,
    **kwargs,
):
    """
    Parallel version: one process per environment.
    """

    if max_workers is None:
        max_workers = min(len(envs), mp.cpu_count())

    tasks = []
    for env_idx, (env, qv) in enumerate(zip(envs, Q_list)):
        tasks.append((
            env_idx,
            env,
            qv,
            kwargs.get("use_q_demos", True),
            kwargs.get("num_q_rollouts_per_state", 10),
            kwargs.get("q_demo_max_steps", 1),
            kwargs.get("tie_eps", 1e-10),
            kwargs.get("use_pairwise", False),
            kwargs.get("n_pairwise", 10),
            kwargs.get("use_estop", False),
            kwargs.get("n_estops", 10),
            kwargs.get("use_improvement", False),
            kwargs.get("n_improvements", 10),
            kwargs.get("n_random_for_improvement", 300),
            kwargs.get("base_min_length", 3),
            kwargs.get("base_max_horizon", 100),
        ))

    results = [None] * len(envs)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_generate_candidates_for_one_env, t) for t in tasks]

        for f in as_completed(futures):
            env_idx, atoms = f.result()
            results[env_idx] = atoms

    return results

## Need to think about this part. how to generate mpre fairly than random
def sample_random_atoms_like_scot(candidates_per_env, chosen_scot, seed=None):
    """
    Random baseline for SCOT that returns:
        [(env_idx, Atom), ...]
    exactly matching SCOT format.

    Inputs:
        candidates_per_env : list[list[Atom]]
        chosen_scot        : list[(env_idx, Atom)] or list[(env_idx, atom.data)]
                             (we only use env_idx counts)

    Returns:
        random_chosen : list[(env_idx, Atom)]
    """

    if seed is not None:
        np.random.seed(seed)

    out = []

    # --- 1. Count how many atoms SCOT selected per environment ---
    scot_counts = {}
    for env_idx, atom_or_data in chosen_scot:
        scot_counts.setdefault(env_idx, 0)
        scot_counts[env_idx] += 1

    # --- 2. Randomly sample the same number of Atoms per env ---
    for env_idx, count in scot_counts.items():
        pool = candidates_per_env[env_idx]
        if len(pool) == 0:
            continue

        # sample indices
        if len(pool) >= count:
            idxs = np.random.choice(len(pool), size=count, replace=False)
        else:
            idxs = np.random.choice(len(pool), size=count, replace=True)

        # full Atom objects — NOT atom.data
        for idx in idxs:
            atom = pool[idx]
            out.append((env_idx, atom))

    return out

def sample_random_atoms_global_pool(
    candidates_per_env,
    chosen_scot,
    seed=None,
):
    """
    Global random baseline:
    - Ignores environments completely
    - Pools all atoms from all envs
    - Randomly selects the same TOTAL number of atoms as SCOT

    Returns:
        random_chosen : list[(env_idx, Atom)]
        (same format as SCOT output)
    """

    if seed is not None:
        np.random.seed(seed)

    n_scot = len(chosen_scot)

    global_pool = []
    for env_idx, atoms in enumerate(candidates_per_env):
        for atom in atoms:
            global_pool.append((env_idx, atom))

    if len(global_pool) == 0 or n_scot == 0:
        return []

    replace = len(global_pool) < n_scot

    idxs = np.random.choice(
        len(global_pool),
        size=n_scot,
        replace=replace,
    )

    random_chosen = [global_pool[i] for i in idxs]

    return random_chosen
