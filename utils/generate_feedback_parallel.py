# ============================================================
# generate_feedback_parallel.py
# ============================================================

import numpy as np
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.special import logsumexp
import multiprocessing as mp


# ============================================================
# 0. Atom abstraction
# ============================================================

class Atom:
    def __init__(self, env_idx, feedback_type, data, metadata=None):
        self.env_idx = env_idx
        self.feedback_type = feedback_type
        self.data = data          # index-based payload
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Atom(env={self.env_idx}, type={self.feedback_type})"


# ============================================================
# 1. Trajectory utilities
# ============================================================

def evaluate_trajectory(env, traj):
    r = 0.0
    for s, _ in traj:
        r += env.compute_reward(s)
    return r


def generate_random_trajectory(env, max_horizon=25):
    traj = []
    obs = env.reset()
    terminals = obs["terminal states"]

    try:
        state = obs["agent"][0] * env.columns + obs["agent"][1]
    except Exception:
        state = obs["agent"][0] * env.size + obs["agent"][1]

    for _ in range(max_horizon):
        if state in terminals:
            traj.append((state, None))
            break

        a = np.random.randint(env.num_actions)
        next_state = np.random.choice(env.num_states, p=env.transitions[state][a])
        traj.append((state, a))
        state = next_state

    return traj


def generate_random_trajectory_from_state(env, start_state, length):
    traj = []
    s = start_state
    terminals = env.terminal_states

    for _ in range(length):
        if s in terminals:
            traj.append((s, None))
            break

        a = np.random.randint(env.num_actions)
        s = np.random.choice(env.num_states, p=env.transitions[s][a])
        traj.append((s, a))

    return traj


def generate_valid_trajectories(env, n, min_length=3, max_horizon=25):
    trajs = []
    while len(trajs) < n:
        t = generate_random_trajectory(env, max_horizon)
        if len(t) >= min_length:
            trajs.append(t)
    return trajs


# ============================================================
# 2. Q-optimal demonstrations
# ============================================================

def generate_q_optimal_trajectories(
    env, Q, num_rollouts_per_state=1, max_steps=1, tie_eps=1e-10
):
    S, A = Q.shape
    terminals = set(env.terminal_states or [])
    T = env.transitions

    opt_actions = [[] for _ in range(S)]
    for s in range(S):
        if s in terminals:
            continue
        row = Q[s]
        m = np.max(row)
        opt_actions[s] = [a for a in range(A) if abs(row[a] - m) < tie_eps]

    trajs = []
    for s0 in range(S):
        if s0 in terminals or not opt_actions[s0]:
            continue
        for _ in range(num_rollouts_per_state):
            s = s0
            tau = []
            for _ in range(max_steps):
                if s in terminals:
                    break
                a = random.choice(opt_actions[s])
                tau.append((s, a))
                s = np.random.choice(S, p=T[s][a])
            trajs.append(tau)
    return trajs


# ============================================================
# 3. Fast pairwise (O(N))
# ============================================================

def generate_pairwise_fast(trajs, returns, n_pairs, seed=None):
    rng = np.random.default_rng(seed)
    M = len(trajs)
    pairs = []

    i_idx = rng.integers(0, M, size=n_pairs)
    j_idx = rng.integers(0, M, size=n_pairs)

    for i, j in zip(i_idx, j_idx):
        if i == j:
            continue
        ri, rj = returns[i], returns[j]
        if ri == rj:
            continue
        pairs.append((i, j) if ri > rj else (j, i))

    return pairs


# ============================================================
# 4. E-stop (prefix-based, fast)
# ============================================================

def compute_prefix_returns(env, trajs):
    returns = np.zeros(len(trajs))
    prefix = []

    for i, t in enumerate(trajs):
        pr = np.zeros(len(t))
        acc = 0.0
        for k, (s, _) in enumerate(t):
            acc += env.compute_reward(s)
            pr[k] = acc
        returns[i] = acc
        prefix.append(pr)

    return returns, prefix


def generate_estops_fast(returns, prefix_returns, n_estops, beta=2.0, seed=None):
    rng = np.random.default_rng(seed)
    M = len(returns)
    estops = []

    idxs = rng.integers(0, M, size=n_estops)
    for i in idxs:
        pr = prefix_returns[i]
        logits = beta * pr - logsumexp([beta * returns[i], beta * pr.max()])
        t_stop = int(np.argmax(logits))
        estops.append((i, t_stop))

    return estops


# ============================================================
# 5. Improvement (SAME START STATE, RANDOM ROLLOUTS)
# ============================================================

def group_by_start_state(trajs):
    g = defaultdict(list)
    for i, t in enumerate(trajs):
        g[t[0][0]].append(i)
    return g


def generate_improvements_same_start(
    env,
    base_trajs,
    base_returns,
    n_imps,
    n_random_rollouts=50,
    seed=None,
):
    rng = np.random.default_rng(seed)
    groups = group_by_start_state(base_trajs)
    start_states = list(groups.keys())

    improvements = []

    for _ in range(n_imps):
        s0 = rng.choice(start_states)
        base_idx = rng.choice(groups[s0])
        base_traj = base_trajs[base_idx]
        L = len(base_traj)

        best_traj = base_traj
        best_r = base_returns[base_idx]

        for _ in range(n_random_rollouts):
            cand = generate_random_trajectory_from_state(env, s0, L)
            r = evaluate_trajectory(env, cand)
            if r > best_r:
                best_r = r
                best_traj = cand

        if best_r > base_returns[base_idx]:
            improvements.append((best_traj, base_traj))

    return improvements


# ============================================================
# 6. Worker per environment (PARALLEL)
# ============================================================

def _worker_env(
    env_idx,
    env_builder,
    Q,
    cfg,
    seed,
):
    np.random.seed(seed)
    random.seed(seed)

    env = env_builder(env_idx)
    atoms = []

    # --- Q demos ---
    if cfg["use_q_demos"]:
        q_trajs = generate_q_optimal_trajectories(
            env,
            Q,
            cfg["num_q_rollouts_per_state"],
            cfg["q_demo_max_steps"],
            cfg["tie_eps"],
        )
        atoms += [Atom(env_idx, "demo", t) for t in q_trajs]

    # --- Base pool ---
    if cfg["needs_base"]:
        base_trajs = generate_valid_trajectories(
            env,
            cfg["base_pool_size"],
            cfg["base_min_length"],
            cfg["base_max_horizon"],
        )

        returns, prefix = compute_prefix_returns(env, base_trajs)

    # --- Pairwise ---
    if cfg["use_pairwise"]:
        pw = generate_pairwise_fast(base_trajs, returns, cfg["n_pairwise"], seed)
        atoms += [Atom(env_idx, "pairwise", p) for p in pw]

    # --- Estop ---
    if cfg["use_estop"]:
        estops = generate_estops_fast(returns, prefix, cfg["n_estops"], seed=seed + 1)
        atoms += [Atom(env_idx, "estop", e) for e in estops]

    # --- Improvement ---
    if cfg["use_improvement"]:
        imps = generate_improvements_same_start(
            env,
            base_trajs,
            returns,
            cfg["n_improvements"],
            cfg["n_random_for_improvement"],
            seed + 2,
        )
        atoms += [Atom(env_idx, "improvement", imp) for imp in imps]

    return env_idx, atoms


# ============================================================
# 7. Public API — PARALLEL SCOT CANDIDATES
# ============================================================

def generate_candidate_atoms_for_scot_parallel(
    env_builder,
    Q_list,
    *,
    max_workers=None,
    seed=0,
    **kwargs,
):
    cfg = dict(
        use_q_demos=True,
        num_q_rollouts_per_state=5,
        q_demo_max_steps=1,
        tie_eps=1e-10,

        use_pairwise=False,
        n_pairwise=1000,

        use_estop=False,
        n_estops=1000,

        use_improvement=False,
        n_improvements=1000,
        n_random_for_improvement=50,

        base_min_length=3,
        base_max_horizon=50,
        base_pool_size=200,
    )
    cfg.update(kwargs)
    cfg["needs_base"] = (
        cfg["use_pairwise"] or cfg["use_estop"] or cfg["use_improvement"]
    )

    out = [None] * len(Q_list)
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = []
        for i, Q in enumerate(Q_list):
            futures.append(
                ex.submit(
                    _worker_env,
                    i,
                    env_builder,
                    Q,
                    cfg,
                    seed + 1000 * i,
                )
            )

        for f in as_completed(futures):
            idx, atoms = f.result()
            out[idx] = atoms

    return out
