import numpy as np
from scipy.special import logsumexp
import random

# ============================================================
# 0. Atom abstraction
# ============================================================

class Atom:
    def __init__(self, env_idx, feedback_type, data, metadata=None):
        """
        env_idx: index of environment/MDP
        feedback_type: 'demo', 'random_traj', 'pairwise', 'estop', 'improvement', ...
        data: payload (trajectory, pairwise tuple, (traj, t_stop), etc.)
        metadata: optional dict
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
    State is stored as integer index.
    """
    traj = []
    obs = env.reset()
    terminal_states = obs["terminal states"]

    # state index from agent position
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
    """
    Generate a random trajectory from a fixed start state with given length
    (or shorter if terminal reached).
    """
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
    """
    Generate n random trajectories with length >= min_length.
    """
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
    q_values,                      # shape (S, A)
    num_rollouts_per_state=10,
    max_steps=15,
    tie_eps=1e-10,
):
    """
    Generate trajectories by following a greedy policy derived from q_values,
    sampling next states from env.transitions.
    """
    S = env.get_num_states()
    A = env.get_num_actions()
    terminals = set(env.terminal_states or [])
    T = env.transitions

    # Precompute greedy action sets (ties allowed within tie_eps)
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
# 3. Correction feedback
# ============================================================


def simulate_corrections(env, trajs, num_random_trajs=25):
    """
    For each trajectory, generate random trajectories from same start state and
    choose the best (improvement) vs original.

    Returns: list of (better_traj, original_traj)
    """
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
# 4. Pairwise comparisons & E-stop
# ============================================================

def generate_pairwise_comparisons(env, trajectories, num_comparisons=10):
    """
    Generate pairwise preferences from a pool of trajectories.
    Returns: list of (better_traj, worse_traj)
    """
    # evaluate each trajectory
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

    # sample for diversity
    num = min(num_comparisons, len(all_pairs))
    return random.sample(all_pairs, num)


def simulate_human_estop(env, full_trajectory, beta=2.0):
    """
    Simulate an E-stop using a Boltzmann-like preference over partial cumulative reward.
    Returns (trajectory, t_stop).
    """
    traj_len = len(full_trajectory)
    full_reward = sum(env.compute_reward(s) for s, _ in full_trajectory)

    log_probs = []
    for t in range(traj_len):
        reward_up_to_t = sum(env.compute_reward(s) for s, _ in full_trajectory[:t+1])
        num = beta * reward_up_to_t
        den = logsumexp([beta * full_reward, num])
        log_stop_prob = num - den
        log_probs.append(log_stop_prob)

    log_probs = np.asarray(log_probs)
    t_stop = int(np.argmax(log_probs))
    return (full_trajectory, t_stop)


# ============================================================
# 5. Converting to atoms
# ============================================================

def trajs_to_atoms(env_idx, trajs, feedback_type):
    return [Atom(env_idx, feedback_type, t) for t in trajs]


def pairwise_to_atoms(env_idx, pairs):
    return [Atom(env_idx, "pairwise", p) for p in pairs]


def estops_to_atoms(env_idx, estops):
    return [Atom(env_idx, "estop", e) for e in estops]


def corrections_to_atoms(env_idx, improvements):
    return [Atom(env_idx, "improvement", imp) for imp in improvements]


# ============================================================
# 6. Unified feedback simulation → atoms
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
    """
    For each env:
      - Q-optimal demos
      - random demos
      - base pool for pairwise/estop/improvements
    Returns:
      atoms_per_env: List[List[Atom]]
    """
    atoms_per_env = []

    for i, (env, qv) in enumerate(zip(envs, Q_list)):
        A = []

        # 1) Q-optimal demonstrations
        q_trajs = generate_q_optimal_trajectories(
            env, qv,
            num_rollouts_per_state=q_demo_rollouts,
            max_steps=q_demo_max_steps,
        )
        A.extend(trajs_to_atoms(i, q_trajs, "demo"))

        # # 2) Random demonstrations (treated distinctly)
        # rand_trajs = generate_valid_trajectories(
        #     env, n=n_random_demos,
        #     min_length=base_min_length,
        #     max_horizon=base_max_horizon,
        # )
        # A.extend(trajs_to_atoms(i, rand_trajs, "random_traj"))

        # 3) Base pool for pairwise / estop / improvements
        base_trajs = generate_valid_trajectories(
            env,
            n=n_base_trajs,
            min_length=base_min_length,
            max_horizon=base_max_horizon,
        )

        # 4) Pairwise
        pw = generate_pairwise_comparisons(env, base_trajs, num_comparisons=n_pairwise)
        A.extend(pairwise_to_atoms(i, pw))

        # 5) E-stop
        estop_trajs = random.sample(base_trajs, min(n_estops, len(base_trajs)))
        estops = [simulate_human_estop(env, t) for t in estop_trajs]
        A.extend(estops_to_atoms(i, estops))

        # 6) Improvement (correction)
        imp_trajs = random.sample(base_trajs, min(n_improvements, len(base_trajs)))
        imps = simulate_corrections(env, imp_trajs)
        A.extend(corrections_to_atoms(i, imps))

        atoms_per_env.append(A)

    return atoms_per_env