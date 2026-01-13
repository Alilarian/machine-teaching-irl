# ============================================================
# generate_feedback.py  (BUDGETED + REPRODUCIBLE + SPARSE VERSION)
# ============================================================

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

import numpy as np
from scipy.special import logsumexp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# from .successor_features import build_Pi_from_q   # keep if you need it


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
# 1. RNG helpers (IMPORTANT for reproducibility with parallelism)
# ============================================================

def _make_rng(seed: Optional[int] = None) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


def _split_seed(rng: np.random.Generator) -> int:
    # returns a fresh uint32-ish seed for child RNGs
    return int(rng.integers(0, 2**32 - 1))


# ============================================================
# 2. Budget samplers (env-level diversity / sparsity)
# ============================================================

def _fix_budget_sum(budgets: np.ndarray, total: int, rng: np.random.Generator) -> np.ndarray:
    """
    Ensure budgets.sum() == total by distributing the remainder (after flooring).
    """
    budgets = budgets.astype(int, copy=True)
    diff = int(total - budgets.sum())
    if diff == 0:
        return budgets

    n = len(budgets)
    if diff > 0:
        # add +1 to diff envs
        idx = rng.choice(n, size=min(diff, n), replace=False)
        budgets[idx] += 1
        diff2 = int(total - budgets.sum())
        # if still diff (diff > n), loop
        while diff2 > 0:
            idx = rng.choice(n, size=min(diff2, n), replace=False)
            budgets[idx] += 1
            diff2 = int(total - budgets.sum())
    else:
        # remove -1 from envs with budget > 0
        diff = -diff
        while diff > 0:
            pos = np.where(budgets > 0)[0]
            if len(pos) == 0:
                break
            k = min(diff, len(pos))
            idx = rng.choice(pos, size=k, replace=False)
            budgets[idx] -= 1
            diff = int(budgets.sum() - total)

    return budgets

def dirichlet_env_budgets(
    total: int,
    n_envs: int,
    *,
    alpha: float = 0.3,
    rng: np.random.Generator,
    allow_zeros: bool = True,
) -> np.ndarray:
    """
    Split `total` across `n_envs` with Dirichlet weights.
    alpha < 1 => sparse/skewed, alpha ~ 1 => ~uniform, alpha > 1 => smooth.
    """
    if total <= 0:
        return np.zeros(n_envs, dtype=int)

    a = float(alpha)
    a = max(a, 1e-6)
    weights = rng.dirichlet(a * np.ones(n_envs))
    raw = weights * total
    budgets = np.floor(raw).astype(int)

    budgets = _fix_budget_sum(budgets, total, rng)

    if not allow_zeros:
        # ensure every env gets at least 1 if total >= n_envs
        if total >= n_envs:
            zeros = np.where(budgets == 0)[0]
            if len(zeros) > 0:
                donors = np.where(budgets > 1)[0]
                rng.shuffle(donors)
                for z in zeros:
                    if len(donors) == 0:
                        break
                    d = donors[0]
                    budgets[d] -= 1
                    budgets[z] += 1
                    donors = np.where(budgets > 1)[0]
        # else impossible; keep zeros
    return budgets

def sparse_poisson_env_budgets(
    total: int,
    n_envs: int,
    *,
    p_active: float = 0.4,
    mean: float = 2000.0,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Very sparse allocation:
      - each env becomes active w.p. p_active
      - active env gets ~Poisson(mean)
      - then scaled to hit `total`
    """
    if total <= 0:
        return np.zeros(n_envs, dtype=int)

    p = float(p_active)
    p = min(max(p, 0.0), 1.0)
    mean = max(float(mean), 1e-6)

    raw = np.zeros(n_envs, dtype=float)
    for i in range(n_envs):
        if rng.random() < p:
            raw[i] = rng.poisson(mean)

    if raw.sum() <= 0:
        # fallback: pick one env and give it all
        j = int(rng.integers(0, n_envs))
        out = np.zeros(n_envs, dtype=int)
        out[j] = int(total)
        return out

    scaled = raw * (total / raw.sum())
    budgets = np.floor(scaled).astype(int)
    budgets = _fix_budget_sum(budgets, total, rng)
    return budgets

def allocate_budgets(
    total: int,
    n_envs: int,
    *,
    rng: np.random.Generator,
    method: str = "dirichlet",
    params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    params = params or {}
    m = method.lower()
    if total <= 0:
        return np.zeros(n_envs, dtype=int)

    if m == "dirichlet":
        return dirichlet_env_budgets(total, n_envs, rng=rng, **params)
    if m in ("sparse_poisson", "poisson_sparse", "bernoulli_poisson"):
        return sparse_poisson_env_budgets(total, n_envs, rng=rng, **params)
    if m == "uniform":
        base = total // n_envs
        budgets = np.full(n_envs, base, dtype=int)
        budgets = _fix_budget_sum(budgets, total, rng)
        return budgets

    raise ValueError(f"Unknown budget allocation method: {method}")

# ============================================================
# 3. Trajectory utilities (ALL randomness uses rng)
# ============================================================

def evaluate_trajectory(env, traj):
    """Compute total reward of a trajectory."""
    return sum(env.compute_reward(s) for s, _ in traj)


def generate_random_trajectory(env, *, max_horizon=25, rng: np.random.Generator):
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

        action = int(rng.integers(0, env.num_actions))
        next_state = int(rng.choice(env.num_states, p=env.transitions[state][action]))

        traj.append((state, action))
        state = next_state

    return traj


def generate_random_trajectory_from_state(env, start_state, length, *, rng: np.random.Generator):
    traj = []
    state = int(start_state)
    terminals = set(env.terminal_states or [])

    for _ in range(length):
        if state in terminals:
            traj.append((state, None))
            break

        action = int(rng.integers(0, env.num_actions))
        next_state = int(rng.choice(env.num_states, p=env.transitions[state][action]))

        traj.append((state, action))
        state = next_state

    return traj


def _rollout_one(env, min_length, max_horizon, seed: int):
    # thread worker: uses its own RNG
    rng = _make_rng(seed)
    t = generate_random_trajectory(env, max_horizon=max_horizon, rng=rng)
    return t if len(t) >= min_length else None


def generate_valid_trajectories(
    env,
    n,
    *,
    min_length=3,
    max_horizon=25,
    max_workers=8,
    oversample_factor=2,
    rng: np.random.Generator,
):
    """
    Thread-parallel trajectory generation inside ONE env.
    Reproducible: we generate per-rollout seeds from rng.
    """
    trajs = []
    needed = int(n)
    if needed <= 0:
        return []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        while len(trajs) < n:
            batch = int(oversample_factor * needed)
            seeds = [int(rng.integers(0, 2**32 - 1)) for _ in range(batch)]
            futures = [
                ex.submit(_rollout_one, env, min_length, max_horizon, s)
                for s in seeds
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
# 4. Q-based (optimal) demos with env+state budgeting
# ============================================================

def generate_q_optimal_trajectories(
    env,
    q_values,
    *,
    # you can control demos by fraction OR by explicit state_budget:
    state_fraction: Optional[float] = None,   # e.g. 0.4 means 40% states
    state_budget: Optional[int] = None,       # e.g. 10 states (overrides fraction)
    num_rollouts_per_state=1,
    max_steps=1,
    tie_eps=1e-10,
    rng: np.random.Generator,
):
    """
    Returns demo trajectories (typically short, max_steps=1).
    Budget controls WHICH start states are used.
    """
    S = env.get_num_states()
    A = env.get_num_actions()
    terminals = set(env.terminal_states or [])
    T = env.transitions

    # eligible start states
    eligible = [s for s in range(S) if s not in terminals]
    if not eligible:
        return []

    # choose how many states to demo
    if state_budget is not None:
        k = int(max(0, min(len(eligible), state_budget)))
    else:
        frac = 1.0 if state_fraction is None else float(state_fraction)
        frac = min(max(frac, 0.0), 1.0)
        k = int(math.floor(frac * len(eligible)))

    if k <= 0:
        return []

    chosen_states = rng.choice(np.array(eligible, dtype=int), size=k, replace=False)

    # precompute optimal actions with tie handling
    opt_actions = [[] for _ in range(S)]
    for s in range(S):
        if s in terminals:
            continue
        row = q_values[s]
        max_q = np.max(row)
        opt_actions[s] = [a for a in range(A) if abs(row[a] - max_q) < tie_eps]

    trajectories = []
    for start_s in chosen_states:
        start_s = int(start_s)
        if start_s in terminals or not opt_actions[start_s]:
            continue

        for _ in range(int(num_rollouts_per_state)):
            tau, s, steps = [], start_s, 0
            while steps < max_steps and s not in terminals:
                acts = opt_actions[s]
                if not acts:
                    break
                a = int(rng.choice(np.array(acts, dtype=int)))
                tau.append((s, a))
                s = int(rng.choice(S, p=T[s, a]))
                steps += 1
            trajectories.append(tau)

    return trajectories

# ============================================================
# 5. Improvements / corrections (reproducible)
# ============================================================

def _simulate_improvement_one(env, traj, num_random_trajs, seed: int):
    rng = _make_rng(seed)
    start_state = traj[0][0]
    length = len(traj)

    original_return = evaluate_trajectory(env, traj)
    best_traj = traj
    best_return = original_return

    for _ in range(num_random_trajs):
        new_traj = generate_random_trajectory_from_state(env, start_state, length, rng=rng)
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
    rng: np.random.Generator,
):
    seeds = [int(rng.integers(0, 2**32 - 1)) for _ in range(len(trajectories))]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(lambda args: _simulate_improvement_one(env, args[0], num_random_trajs, args[1]),
                           zip(trajectories, seeds)))


# ============================================================
# 6. Pairwise (reproducible, no quadratic blowup)
# ============================================================

def compute_rewards(env, trajectories):
    return np.array([evaluate_trajectory(env, t) for t in trajectories])


def generate_pairwise_comparisons(
    env,
    trajectories,
    *,
    num_comparisons=10,
    max_trials=50,
    rng: np.random.Generator,
):
    """
    O(K) expected time, avoids O(n^2).
    """
    n = len(trajectories)
    if n <= 1 or num_comparisons <= 0:
        return []

    rewards = compute_rewards(env, trajectories)

    pairs = []
    seen = set()
    trials = 0

    # we cap trials to avoid infinite loops when many ties exist
    cap = int(max_trials * num_comparisons)

    while len(pairs) < num_comparisons and trials < cap:
        i, j = rng.choice(n, size=2, replace=False)
        i, j = int(i), int(j)

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


# ============================================================
# 7. E-Stop (no randomness internally, but sampling is rng-driven)
# ============================================================

def simulate_human_estop_one(env, full_trajectory, beta=2.0):
    traj_len = len(full_trajectory)
    full_reward = sum(env.compute_reward(s) for s, _ in full_trajectory)

    log_probs = []
    for t in range(traj_len):
        reward_to_t = sum(env.compute_reward(s) for s, _ in full_trajectory[:t + 1])
        num = beta * reward_to_t
        den = logsumexp([beta * full_reward, num])
        log_probs.append(num - den)

    t_stop = int(np.argmax(log_probs))
    return (full_trajectory, t_stop)


def simulate_human_estops(
    env,
    trajectories,
    *,
    beta=10.0,
    max_workers=8,
):
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(lambda t: simulate_human_estop_one(env, t, beta=beta), trajectories))


# ============================================================
# 8. Atom constructors
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
# 9. Configs (clean knobs)
# ============================================================

@dataclass
class DemoSpec:
    enabled: bool = True
    # env-level: fraction of envs that even get demos
    env_fraction: float = 1.0  # e.g. 0.6 means 60% envs get demos
    # state-level: either use fraction OR budgets (if total_state_budget given)
    state_fraction: Optional[float] = 1.0  # e.g. 0.4 means 40% states in each selected env
    # OR: global total budget of demo-states across envs (distributed by allocator)
    total_state_budget: Optional[int] = None
    alloc_method: str = "dirichlet"
    alloc_params: Optional[Dict[str, Any]] = None

    num_rollouts_per_state: int = 1
    max_steps: int = 1
    tie_eps: float = 1e-10


@dataclass
class FeedbackSpec:
    enabled: bool = False
    total_budget: int = 0
    alloc_method: str = "dirichlet"
    alloc_params: Optional[Dict[str, Any]] = None


@dataclass
class GenerationSpec:
    seed: int = 0
    max_workers: Optional[int] = None

    demo: DemoSpec = DemoSpec()

    pairwise: FeedbackSpec = FeedbackSpec(enabled=False, total_budget=0)
    estop: FeedbackSpec = FeedbackSpec(enabled=False, total_budget=0)
    improvement: FeedbackSpec = FeedbackSpec(enabled=False, total_budget=0)

    # base trajectory pool (used by pairwise/estop/improvement)
    base_min_length: int = 2
    base_max_horizon: int = 100
    base_threads: int = 8

    # improvement internals
    n_random_for_improvement: int = 300

    # estop
    estop_beta: float = 10.0


# ============================================================
# 10. Worker (per-env generation) — picklable
# ============================================================

def _generate_candidates_for_one_env(args):
    (
        env_idx,
        env,
        qv,
        env_seed,
        demo_state_budget,        # number of start-states to demo in this env (or None if using fraction)
        demo_state_fraction,      # fraction of eligible states in this env (if budget is None)
        do_demo,
        do_pairwise,
        do_estop,
        do_improvement,
        pw_budget,
        estop_budget,
        imp_budget,
        spec_dict,
    ) = args

    spec = GenerationSpec(**spec_dict)  # reconstruct (dataclasses are picklable too, but dict is safest)
    rng = _make_rng(env_seed)

    C: List[Atom] = []

    # ---------------- demos (Q-optimal) ----------------
    if do_demo and spec.demo.enabled:
        q_trajs = generate_q_optimal_trajectories(
            env,
            qv,
            state_fraction=demo_state_fraction,
            state_budget=demo_state_budget,
            num_rollouts_per_state=spec.demo.num_rollouts_per_state,
            max_steps=spec.demo.max_steps,
            tie_eps=spec.demo.tie_eps,
            rng=rng,
        )
        C.extend(trajs_to_atoms(env_idx, q_trajs, "demo"))

    # ---------------- base trajectories ----------------
    needs_base = do_pairwise or do_estop or do_improvement
    base_trajs = []
    if needs_base:
        base_count = max(int(pw_budget), int(estop_budget), int(imp_budget))
        if base_count > 0:
            base_trajs = generate_valid_trajectories(
                env,
                n=base_count,
                min_length=spec.base_min_length,
                max_horizon=spec.base_max_horizon,
                max_workers=spec.base_threads,
                rng=rng,
            )

    # ---------------- pairwise ----------------
    if do_pairwise and spec.pairwise.enabled and pw_budget > 0 and base_trajs:
        pw = generate_pairwise_comparisons(
            env,
            base_trajs,
            num_comparisons=int(pw_budget),
            rng=rng,
        )
        C.extend(pairwise_to_atoms(env_idx, pw))

    # ---------------- estop ----------------
    if do_estop and spec.estop.enabled and estop_budget > 0 and base_trajs:
        k = min(int(estop_budget), len(base_trajs))
        idx = rng.choice(len(base_trajs), size=k, replace=False)
        estop_trajs = [base_trajs[int(i)] for i in idx]
        estops = simulate_human_estops(
            env,
            estop_trajs,
            beta=spec.estop_beta,
            max_workers=spec.base_threads,
        )
        C.extend(estops_to_atoms(env_idx, estops))

    # ---------------- improvement ----------------
    if do_improvement and spec.improvement.enabled and imp_budget > 0 and base_trajs:
        k = min(int(imp_budget), len(base_trajs))
        idx = rng.choice(len(base_trajs), size=k, replace=False)
        imp_trajs = [base_trajs[int(i)] for i in idx]
        imps = simulate_corrections(
            env,
            imp_trajs,
            num_random_trajs=spec.n_random_for_improvement,
            max_workers=spec.base_threads,
            rng=rng,
        )
        C.extend(corrections_to_atoms(env_idx, imps))

    return env_idx, C


# ============================================================
# 11. Public API: generate_candidate_atoms_for_scot (budgeted)
# ============================================================

def generate_candidate_atoms_for_scot(
    envs: Sequence[Any],
    Q_list: Sequence[np.ndarray],
    *,
    spec: Optional[GenerationSpec] = None,
    max_workers: Optional[int] = None,
) -> List[List[Atom]]:
    """
    Budgeted, sparse, reproducible generator.

    Key behaviors:
      - each feedback type has a GLOBAL total budget (e.g., 10k pairwise)
      - budgets are distributed across envs (Dirichlet / sparse_poisson / uniform)
      - demos are controlled at two levels:
          (i) env_fraction: which envs receive demos
          (ii) state_fraction or total_state_budget: which states receive demos
      - all randomness is controlled by spec.seed and per-env child seeds
    """
    if spec is None:
        spec = GenerationSpec()

    n_envs = len(envs)
    if n_envs != len(Q_list):
        raise ValueError("envs and Q_list must have the same length")

    if max_workers is None:
        max_workers = spec.max_workers
    if max_workers is None:
        max_workers = min(n_envs, mp.cpu_count())

    master_rng = _make_rng(spec.seed)

    # per-env seeds (parallel-safe reproducibility)
    env_seeds = master_rng.integers(0, 2**32 - 1, size=n_envs, dtype=np.uint32).astype(int)

    # ---------------- allocate budgets for pairwise/estop/improvement ----------------
    pw_budgets = allocate_budgets(
        spec.pairwise.total_budget,
        n_envs,
        rng=master_rng,
        method=spec.pairwise.alloc_method,
        params=spec.pairwise.alloc_params,
    ) if spec.pairwise.enabled else np.zeros(n_envs, dtype=int)

    estop_budgets = allocate_budgets(
        spec.estop.total_budget,
        n_envs,
        rng=master_rng,
        method=spec.estop.alloc_method,
        params=spec.estop.alloc_params,
    ) if spec.estop.enabled else np.zeros(n_envs, dtype=int)

    imp_budgets = allocate_budgets(
        spec.improvement.total_budget,
        n_envs,
        rng=master_rng,
        method=spec.improvement.alloc_method,
        params=spec.improvement.alloc_params,
    ) if spec.improvement.enabled else np.zeros(n_envs, dtype=int)

    # ---------------- demos: env mask + state budgets/fractions ----------------
    do_demo_env = np.zeros(n_envs, dtype=bool)
    if spec.demo.enabled and spec.demo.env_fraction > 0:
        p = min(max(float(spec.demo.env_fraction), 0.0), 1.0)
        do_demo_env = master_rng.random(n_envs) < p

    # If you want a GLOBAL total demo-state budget across envs, allocate it.
    # Otherwise use per-env state_fraction (same fraction in each demo-enabled env).
    demo_state_budgets = np.full(n_envs, None, dtype=object)  # Optional[int] per env
    demo_state_fraction = np.full(n_envs, None, dtype=object) # Optional[float] per env

    if spec.demo.enabled and spec.demo.total_state_budget is not None:
        # allocate only across the envs that are demo-enabled
        active = np.where(do_demo_env)[0]
        if len(active) == 0:
            pass
        else:
            alloc = allocate_budgets(
                int(spec.demo.total_state_budget),
                len(active),
                rng=master_rng,
                method=spec.demo.alloc_method,
                params=spec.demo.alloc_params,
            )
            for j, env_idx in enumerate(active):
                demo_state_budgets[env_idx] = int(alloc[j])
            # fraction unused in this mode
    else:
        # fraction mode
        frac = spec.demo.state_fraction
        frac = 1.0 if frac is None else float(frac)
        frac = min(max(frac, 0.0), 1.0)
        for i in range(n_envs):
            demo_state_fraction[i] = frac

    # ---------------- build tasks ----------------
    spec_dict = {
        "seed": spec.seed,
        "max_workers": spec.max_workers,
        "demo": spec.demo,
        "pairwise": spec.pairwise,
        "estop": spec.estop,
        "improvement": spec.improvement,
        "base_min_length": spec.base_min_length,
        "base_max_horizon": spec.base_max_horizon,
        "base_threads": spec.base_threads,
        "n_random_for_improvement": spec.n_random_for_improvement,
        "estop_beta": spec.estop_beta,
    }

    # dataclasses nested inside dict aren’t JSON-serializable but ARE picklable.
    # ProcessPool uses pickle, so this is OK. If you prefer, you can manually convert
    # nested dataclasses to dicts too — but it’s not required here.

    tasks = []
    for env_idx, (env, qv) in enumerate(zip(envs, Q_list)):
        tasks.append((
            env_idx,
            env,
            qv,
            int(env_seeds[env_idx]),
            demo_state_budgets[env_idx],        # Optional[int]
            demo_state_fraction[env_idx],       # Optional[float]
            bool(do_demo_env[env_idx]),
            bool(spec.pairwise.enabled),
            bool(spec.estop.enabled),
            bool(spec.improvement.enabled),
            int(pw_budgets[env_idx]),
            int(estop_budgets[env_idx]),
            int(imp_budgets[env_idx]),
            spec_dict,
        ))

    results: List[Optional[List[Atom]]] = [None] * n_envs

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_generate_candidates_for_one_env, t) for t in tasks]
        for f in as_completed(futures):
            env_idx, atoms = f.result()
            results[env_idx] = atoms

    return results  # type: ignore


# ============================================================
# 12. Example usage (copy-paste)
# ============================================================
#
# spec = GenerationSpec(
#     seed=123,
#     demo=DemoSpec(
#         enabled=True,
#         env_fraction=0.6,          # only 60% envs get demos
#         state_fraction=0.4,        # 40% states per demo-env
#         # OR: total_state_budget=50,  # instead of state_fraction
#         num_rollouts_per_state=1,
#         max_steps=1,
#     ),
#     pairwise=FeedbackSpec(
#         enabled=True,
#         total_budget=10_000,
#         alloc_method="dirichlet",
#         alloc_params={"alpha": 0.3},
#     ),
#     estop=FeedbackSpec(
#         enabled=True,
#         total_budget=2_000,
#         alloc_method="sparse_poisson",
#         alloc_params={"p_active": 0.4, "mean": 400},
#     ),
#     improvement=FeedbackSpec(
#         enabled=True,
#         total_budget=3_000,
#         alloc_method="dirichlet",
#         alloc_params={"alpha": 0.5},
#     ),
# )
#
# atoms_per_env = generate_candidate_atoms_for_scot(envs, Q_list, spec=spec)
#
