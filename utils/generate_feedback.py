# ============================================================
# generate_feedback.py  (FULL FIXED VERSION)
# ============================================================

import numpy as np
from scipy.special import logsumexp
import random
from .successor_features import build_Pi_from_q


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


def generate_valid_trajectories(env, n, min_length=3, max_horizon=25):
    trajs = []
    while len(trajs) < n:
        t = generate_random_trajectory(env, max_horizon=max_horizon)
        if len(t) >= min_length:
            trajs.append(t)
    return trajs


# ============================================================
# 2. Q-based (optimal) trajectories
# ============================================================

def generate_q_optimal_trajectories(
    env,
    q_values,
    num_rollouts_per_state=10,
    max_steps=15,
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

def simulate_corrections(env, trajs, num_random_trajs=25):
    paired = []

    for traj in trajs:
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

        paired.append((best_traj, traj))
    return paired


# ============================================================
# 4. Pairwise & E-Stop
# ============================================================

def generate_pairwise_comparisons(env, trajectories, num_comparisons=10):
    rewarded = []
    for t in trajectories:
        r = evaluate_trajectory(env, t)
        rewarded.append((t, r))

    all_pairs = []
    for i in range(len(rewarded)):
        for j in range(i + 1, len(rewarded)):
            t1, r1 = rewarded[i]
            t2, r2 = rewarded[j]
            if r1 == r2:
                continue
            if r1 > r2:
                all_pairs.append((t1, t2))
            else:
                all_pairs.append((t2, t1))

    if not all_pairs:
        return []

    num = min(num_comparisons, len(all_pairs))
    return random.sample(all_pairs, num)


def simulate_human_estop(env, full_trajectory, beta=2.0):
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

def simulate_all_feedback(
    envs,
    Q_list,
    *,
    n_base_trajs=20,
    base_min_length=3,
    base_max_horizon=25,
    n_pairwise=5,
    n_estops=5,
    n_improvements=5,
    n_random_demos=5,
    q_demo_rollouts=10,
    q_demo_max_steps=15,
):
    atoms_per_env = []

    for i, (env, qv) in enumerate(zip(envs, Q_list)):
        A = []

        # (1) Optimal Q demonstrations
        q_trajs = generate_q_optimal_trajectories(env, qv,
            num_rollouts_per_state=q_demo_rollouts,
            max_steps=q_demo_max_steps
        )
        A.extend(trajs_to_atoms(i, q_trajs, "demo"))

        # (2) Base trajectories for pairwise/estop/improvement
        base_trajs = generate_valid_trajectories(
            env,
            n=n_base_trajs,
            min_length=base_min_length,
            max_horizon=base_max_horizon
        )

        # (3) Pairwise
        pw = generate_pairwise_comparisons(env, base_trajs, num_comparisons=n_pairwise)
        A.extend(pairwise_to_atoms(i, pw))

        # (4) E-Stop
        estop_trajs = random.sample(base_trajs, min(n_estops, len(base_trajs)))
        estops = [simulate_human_estop(env, t) for t in estop_trajs]
        A.extend(estops_to_atoms(i, estops))

        # (5) Improvement
        imp_trajs = random.sample(base_trajs, min(n_improvements, len(base_trajs)))
        imps = simulate_corrections(env, imp_trajs)
        A.extend(corrections_to_atoms(i, imps))

        atoms_per_env.append(A)

    return atoms_per_env

# ============================================================
# 7. Unified SCOT candidate generator
# ============================================================

# ============================================================
# 7. Unified SCOT candidate generator  (NO RANDOM DEMOS)
# ============================================================

def generate_candidate_atoms_for_scot(
    envs,
    Q_list,
    *,
    # ---- Q-demos ----
    use_q_demos=True,
    num_q_rollouts_per_state=10,
    q_demo_max_steps=1,
    tie_eps=1e-10,

    # ---- Pairwise ----
    use_pairwise=False,
    n_pairwise=10,

    # ---- E-stop ----
    use_estop=False,
    n_estops=10,

    # ---- Improvement ----
    use_improvement=False,
    n_improvements=10,
    n_random_for_improvement=300,

    # ---- Base traj parameters (for pairwise/estop/improvement) ----
    base_min_length=3,
    base_max_horizon=100,
):
    """
    Generate candidate ATOMS for SCOT.
    Random demonstrations are REMOVED as requested.

    Returns:
        candidates_per_env: List[List[Atom]]
    """
    candidates_per_env = []

    for env_idx, (env, qv) in enumerate(zip(envs, Q_list)):
        C = []

        # --------------------------------------------------------
        # 1) Q-optimal demonstration atoms  (EXACT original logic)
        # --------------------------------------------------------
        if use_q_demos:
            q_trajs = generate_q_optimal_trajectories(
                env,
                qv,
                num_rollouts_per_state=num_q_rollouts_per_state,
                max_steps=q_demo_max_steps,
                tie_eps=tie_eps
            )
            C.extend(trajs_to_atoms(env_idx, q_trajs, "demo"))

        # --------------------------------------------------------
        # Prepare base trajectories if needed
        # --------------------------------------------------------
        needs_base = use_pairwise or use_estop or use_improvement

        if needs_base:
            base_count = max(n_pairwise, n_estops, n_improvements)
            base_trajs = generate_valid_trajectories(
                env,
                n=base_count,
                min_length=base_min_length,
                max_horizon=base_max_horizon
            )

        # --------------------------------------------------------
        # 2) Pairwise atoms
        # --------------------------------------------------------
        if use_pairwise:
            pw = generate_pairwise_comparisons(env, base_trajs, num_comparisons=n_pairwise)
            C.extend(pairwise_to_atoms(env_idx, pw))

        # --------------------------------------------------------
        # 3) E-stop atoms
        # --------------------------------------------------------
        if use_estop:
            estop_trajs = random.sample(base_trajs, min(n_estops, len(base_trajs)))
            estops = [simulate_human_estop(env, t) for t in estop_trajs]
            C.extend(estops_to_atoms(env_idx, estops))

        # --------------------------------------------------------
        # 4) Improvement atoms
        # --------------------------------------------------------
        if use_improvement:
            imp_trajs = random.sample(base_trajs, min(n_improvements, len(base_trajs)))
            imps = simulate_corrections(env, imp_trajs, num_random_trajs=n_random_for_improvement)
            C.extend(corrections_to_atoms(env_idx, imps))

        candidates_per_env.append(C)

    return candidates_per_env

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



# # ============================================================
# # generate_feedback.py — FULL DETERMINISTIC VERSION
# # ============================================================



# import numpy as np
# from scipy.special import logsumexp
# import random
# from .successor_features import build_Pi_from_q


# # ============================================================
# # 0. Global RNG (deterministic)
# # ============================================================

# _FEEDBACK_RNG = np.random.default_rng()
# _PY_RNG = random.Random()

# def set_feedback_seed(seed):
#     """
#     Set deterministic seeds for *all* feedback generation functions.
#     Ensures identical trajectories, Q-demos, pairwise, estops, and improvements.
#     """
#     global _FEEDBACK_RNG, _PY_RNG
#     _FEEDBACK_RNG = np.random.default_rng(seed)
#     _PY_RNG = random.Random(seed)


# # ============================================================
# # 1. Atom abstraction
# # ============================================================

# class Atom:
#     def __init__(self, env_idx, feedback_type, data, metadata=None):
#         self.env_idx = env_idx
#         self.feedback_type = feedback_type
#         self.data = data
#         self.metadata = metadata or {}

#     def __repr__(self):
#         return f"Atom(env={self.env_idx}, type={self.feedback_type})"


# # ============================================================
# # 2. Trajectory utilities (ALL RNG CONTROLLED)
# # ============================================================

# def evaluate_trajectory(env, traj):
#     return sum(env.compute_reward(s) for s, _ in traj)


# def generate_random_trajectory(env, max_horizon=25):
#     traj = []
#     obs = env.reset()
#     terminal_states = obs["terminal states"]

#     try:
#         state = obs["agent"][0] * env.columns + obs["agent"][1]
#     except Exception:
#         state = obs["agent"][0] * env.size + obs["agent"][1]

#     for _ in range(max_horizon):
#         if state in terminal_states:
#             traj.append((state, None))
#             break

#         action = _FEEDBACK_RNG.integers(env.num_actions)
#         next_state = _FEEDBACK_RNG.choice(env.num_states, p=env.transitions[state][action])

#         traj.append((state, action))
#         state = next_state

#     return traj


# def generate_random_trajectory_from_state(env, start_state, length):
#     traj = []
#     state = start_state
#     terminals = env.terminal_states

#     for _ in range(length):
#         if state in terminals:
#             traj.append((state, None))
#             break

#         action = _FEEDBACK_RNG.integers(env.num_actions)
#         next_state = _FEEDBACK_RNG.choice(env.num_states, p=env.transitions[state][action])

#         traj.append((state, action))
#         state = next_state

#     return traj


# def generate_valid_trajectories(env, n, min_length=3, max_horizon=25):
#     trajs = []
#     while len(trajs) < n:
#         t = generate_random_trajectory(env, max_horizon=max_horizon)
#         if len(t) >= min_length:
#             trajs.append(t)
#     return trajs


# # ============================================================
# # 3. Q-based (optimal) trajectories (deterministic)
# # ============================================================

# def generate_q_optimal_trajectories(
#     env,
#     q_values,
#     num_rollouts_per_state=10,
#     max_steps=15,
#     tie_eps=1e-10,
# ):
#     S = env.get_num_states()
#     A = env.get_num_actions()
#     terminals = set(env.terminal_states or [])
#     T = env.transitions

#     opt_actions = [[] for _ in range(S)]
#     for s in range(S):
#         if s in terminals:
#             continue
#         row = q_values[s]
#         max_q = np.max(row)
#         opt_actions[s] = [a for a in range(A) if abs(row[a] - max_q) < tie_eps]

#     trajectories = []
#     for start_s in range(S):
#         if start_s in terminals or not opt_actions[start_s]:
#             continue

#         for _ in range(num_rollouts_per_state):
#             tau, s, steps = [], int(start_s), 0
#             while steps < max_steps and s not in terminals:
#                 acts = opt_actions[s]
#                 if not acts:
#                     break
#                 a = int(_FEEDBACK_RNG.choice(acts))
#                 tau.append((s, a))
#                 s = int(_FEEDBACK_RNG.choice(S, p=T[s, a]))
#                 steps += 1
#             trajectories.append(tau)

#     return trajectories


# # ============================================================
# # 4. Corrections (deterministic)
# # ============================================================

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


# # ============================================================
# # 5. Pairwise & E-Stop (deterministic)
# # ============================================================

# def generate_pairwise_comparisons(env, trajectories, num_comparisons=10):
#     rewarded = []
#     for t in trajectories:
#         r = evaluate_trajectory(env, t)
#         rewarded.append((t, r))

#     all_pairs = []
#     for i in range(len(rewarded)):
#         for j in range(i + 1, len(rewarded)):
#             t1, r1 = rewarded[i]
#             t2, r2 = rewarded[j]
#             if r1 == r2:
#                 continue
#             if r1 > r2:
#                 all_pairs.append((t1, t2))
#             else:
#                 all_pairs.append((t2, t1))

#     if not all_pairs:
#         return []

#     num = min(num_comparisons, len(all_pairs))
#     return _PY_RNG.sample(all_pairs, num)


# def simulate_human_estop(env, full_trajectory, beta=2.0):
#     traj_len = len(full_trajectory)
#     full_reward = sum(env.compute_reward(s) for s, _ in full_trajectory)

#     log_probs = []
#     for t in range(traj_len):
#         reward_to_t = sum(env.compute_reward(s) for s, _ in full_trajectory[:t+1])
#         num = beta * reward_to_t
#         den = logsumexp([beta * full_reward, num])
#         log_probs.append(num - den)

#     t_stop = int(np.argmax(log_probs))
#     return (full_trajectory, t_stop)


# # ============================================================
# # 6. Atom constructors
# # ============================================================

# def trajs_to_atoms(env_idx, trajs, feedback_type):
#     return [Atom(env_idx, feedback_type, t) for t in trajs]

# def pairwise_to_atoms(env_idx, pairs):
#     return [Atom(env_idx, "pairwise", p) for p in pairs]

# def estops_to_atoms(env_idx, estops):
#     return [Atom(env_idx, "estop", e) for e in estops]

# def corrections_to_atoms(env_idx, imps):
#     return [Atom(env_idx, "improvement", imp) for imp in imps]


# # ============================================================
# # 7. Unified feedback → atoms  (DETERMINISTIC)
# # ============================================================

# def simulate_all_feedback(
#     envs,
#     Q_list,
#     *,
#     n_base_trajs=20,
#     base_min_length=3,
#     base_max_horizon=25,
#     n_pairwise=5,
#     n_estops=5,
#     n_improvements=5,
#     q_demo_rollouts=10,
#     q_demo_max_steps=15,
# ):
#     atoms_per_env = []

#     for i, (env, qv) in enumerate(zip(envs, Q_list)):
#         A = []

#         # (1) Q-optimal demonstrations
#         q_trajs = generate_q_optimal_trajectories(
#             env,
#             qv,
#             num_rollouts_per_state=q_demo_rollouts,
#             max_steps=q_demo_max_steps
#         )
#         A.extend(trajs_to_atoms(i, q_trajs, "demo"))

#         # (2) Base trajectories
#         base_trajs = generate_valid_trajectories(
#             env,
#             n=n_base_trajs,
#             min_length=base_min_length,
#             max_horizon=base_max_horizon
#         )

#         # (3) Pairwise
#         pw = generate_pairwise_comparisons(env, base_trajs, num_comparisons=n_pairwise)
#         A.extend(pairwise_to_atoms(i, pw))

#         # (4) E-Stop
#         estop_trajs = _PY_RNG.sample(base_trajs, min(n_estops, len(base_trajs)))
#         estops = [simulate_human_estop(env, t) for t in estop_trajs]
#         A.extend(estops_to_atoms(i, estops))

#         # (5) Improvement
#         imp_trajs = _PY_RNG.sample(base_trajs, min(n_improvements, len(base_trajs)))
#         imps = simulate_corrections(env, imp_trajs)
#         A.extend(corrections_to_atoms(i, imps))

#         atoms_per_env.append(A)

#     return atoms_per_env


# # ============================================================
# # 8. Unified SCOT candidate generator (DETERMINISTIC)
# # ============================================================

# def generate_candidate_atoms_for_scot(
#     envs,
#     Q_list,
#     *,
#     use_q_demos=True,
#     num_q_rollouts_per_state=10,
#     q_demo_max_steps=15,
#     tie_eps=1e-10,
#     use_pairwise=False,
#     n_pairwise=10,
#     use_estop=False,
#     n_estops=10,
#     use_improvement=False,
#     n_improvements=10,
#     n_random_for_improvement=25,
#     base_min_length=3,
#     base_max_horizon=25,
# ):
#     candidates_per_env = []

#     for env_idx, (env, qv) in enumerate(zip(envs, Q_list)):
#         C = []

#         # Q-demos
#         if use_q_demos:
#             q_trajs = generate_q_optimal_trajectories(
#                 env,
#                 qv,
#                 num_rollouts_per_state=num_q_rollouts_per_state,
#                 max_steps=q_demo_max_steps,
#                 tie_eps=tie_eps
#             )
#             C.extend(trajs_to_atoms(env_idx, q_trajs, "demo"))

#         needs_base = use_pairwise or use_estop or use_improvement

#         if needs_base:
#             base_count = max(n_pairwise, n_estops, n_improvements)
#             base_trajs = generate_valid_trajectories(
#                 env,
#                 n=base_count,
#                 min_length=base_min_length,
#                 max_horizon=base_max_horizon
#             )

#         # Pairwise
#         if use_pairwise:
#             pw = generate_pairwise_comparisons(env, base_trajs, num_comparisons=n_pairwise)
#             C.extend(pairwise_to_atoms(env_idx, pw))

#         # Estop
#         if use_estop:
#             estop_trajs = _PY_RNG.sample(base_trajs, min(n_estops, len(base_trajs)))
#             estops = [simulate_human_estop(env, t) for t in estop_trajs]
#             C.extend(estops_to_atoms(env_idx, estops))

#         # Improvement
#         if use_improvement:
#             imp_trajs = _PY_RNG.sample(base_trajs, min(n_improvements, len(base_trajs)))
#             imps = simulate_corrections(env, imp_trajs, num_random_trajs=n_random_for_improvement)
#             C.extend(corrections_to_atoms(env_idx, imps))

#         candidates_per_env.append(C)

#     return candidates_per_env

# def sample_random_atoms_like_scot(envs, candidates_per_env, chosen_scot, seed=None):
#     """
#     Random baseline for SCOT:
#     Sample random atoms from *candidate atoms*, matching SCOT's 
#     per-environment selection count.

#     Returns:
#         [(env_idx, atom.data), ...]
#     """
#     rng = np.random.default_rng(seed)
#     out = []

#     # Count how many atoms SCOT picked from each environment
#     scot_counts = {}
#     for env_idx, atom_data in chosen_scot:
#         scot_counts.setdefault(env_idx, 0)
#         scot_counts[env_idx] += 1

#     # Sample random atoms accordingly
#     for env_idx, count in scot_counts.items():
#         pool = candidates_per_env[env_idx]
#         if len(pool) == 0:
#             continue

#         # deterministic sampling
#         if len(pool) >= count:
#             chosen = rng.choice(pool, size=count, replace=False)
#         else:
#             chosen = rng.choice(pool, size=count, replace=True)

#         for atom in chosen:
#             out.append((env_idx, atom.data))

#     return out
