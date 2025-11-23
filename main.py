"""Command-line tool to sample feedback signals and save quick plots per environment.

Inputs:
  - seed: base seed for reproducibility. Each env gets seed+idx.
  - n_envs: how many environments to generate.
  - rows/cols: grid dimensions for the layout-based GridWorld.
  - true_reward: two numbers (comma/space separated) for the reward vector.
  - num_samples: how many trajectories/feedback samples to draw.
  - feedback: which feedback type to run (estop, pairwise, correction, demonstration).
  - output_dir: where to place the generated graphs (default: outputs/<feedback>/...).

The script builds simple layout-based GridWorlds, samples the requested feedback,
creates a small summary plot per environment, and saves it to disk.
"""
from __future__ import annotations

import argparse
import pathlib
import random
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from data_generation.generate_data import (
    generate_pairwise_comparisons,
    generate_random_trajectory,
    generate_suboptimal_demonstrations,
    simulate_human_estop_v2,
    simulate_improvement_feedback_v4,
)
from agent.q_learning_agent import ValueIteration
from mdp.gridworld_env_layout import GridWorldMDPFromLayoutEnv
from utils.machine_teaching_utils import remove_redundant_constraints
from utils.mdp_generator import generate_random_gridworld_envs


# ----------------------------- parsing helpers -----------------------------

def parse_reward_vector(raw: str) -> np.ndarray:
    """Parse comma/space separated floats into a 2-D reward vector."""
    pieces = raw.replace(",", " ").split()
    if not pieces:
        raise argparse.ArgumentTypeError("true_reward must contain at least one number")
    try:
        values = np.array([float(p) for p in pieces], dtype=float)
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError("true_reward must be numeric") from exc
    if len(values) != 2:
        raise argparse.ArgumentTypeError("true_reward must have exactly two numbers, e.g., '-0.97 0.24'")
    if np.allclose(values, 0):
        raise argparse.ArgumentTypeError("true_reward cannot be all zeros")
    return values

# ----------------------------- feedback sampling -----------------------------

def sample_trajectories(env, num_samples: int, max_horizon: int):
    return [
        generate_random_trajectory(env, max_horizon=max_horizon, fixed_start=True)
        for _ in range(num_samples)
    ]


def plot_estop(env, trajectories: Sequence, out_path: pathlib.Path, title: str):
    """Generate estop constraints (notebook style)."""
    # Track unique normalized vectors (notebook style)
    normalized_unique_vectors = []
    seen = set()

    for traj in trajectories:
        estop_traj, stop_time = simulate_human_estop_v2(env, traj, beta=100, gamma=env.gamma)
        if stop_time is None:
            continue

        # Compute features up to t (notebook style: explicit computation)
        features_up_to_t = [env.get_state_feature(s) for s, _ in estop_traj[:int(stop_time) + 1]]
        sum_feat_up_to_t = np.sum(features_up_to_t, axis=0)

        # Compute features for full trajectory
        full_traj_features = [env.get_state_feature(s) for s, _ in estop_traj]
        traj_feat = np.sum(full_traj_features, axis=0)

        # Difference
        diff = sum_feat_up_to_t - traj_feat

        # Normalize (L2 norm)
        norm = np.linalg.norm(diff)
        if norm == 0:
            normalized = np.zeros_like(diff)
        else:
            normalized = diff / norm

        # Convert to tuple for hashing and check uniqueness (notebook style)
        key = tuple(np.round(normalized, decimals=6))  # rounding to avoid float precision issues
        if key not in seen:
            seen.add(key)
            normalized_unique_vectors.append(normalized)

    # Remove redundant constraints (notebook style: epsilon=0.0001, no orientation toward true_reward)
    constraints = remove_redundant_constraints(normalized_unique_vectors, epsilon=0.0001)
    plot_constraints(constraints, env.feature_weights, out_path, title, highlight=None)


def plot_pairwise(env, trajectories: Sequence, num_samples: int, out_path: pathlib.Path, title: str):
    """Generate pairwise comparison constraints (notebook style)."""
    max_pairs = max(0, len(trajectories) * (len(trajectories) - 1))
    num_to_draw = min(num_samples, max_pairs)
    comparisons = (
        generate_pairwise_comparisons(env, trajectories, num_comparisons=num_to_draw)
        if num_to_draw
        else []
    )

    # Track unique normalized vectors (notebook style)
    seen = set()
    unique_vectors = []

    for preferred, other in comparisons:
        # Compute features explicitly (notebook style)
        preferred_feats = [env.get_state_feature(s) for s, _ in preferred]
        other_feats = [env.get_state_feature(s) for s, _ in other]

        preferred_sum = np.sum(preferred_feats, axis=0)
        other_sum = np.sum(other_feats, axis=0)

        # Difference
        diff = preferred_sum - other_sum
        norm = np.linalg.norm(diff)

        # Normalize
        normalized = diff / norm if norm != 0 else np.zeros_like(diff)
        key = tuple(np.round(normalized, decimals=6))  # use rounding for stable hashing

        if key not in seen:
            seen.add(key)
            unique_vectors.append(normalized)

    # Remove redundant constraints (notebook style: epsilon=0.0001, no orientation toward true_reward)
    constraints = remove_redundant_constraints(unique_vectors, epsilon=0.0001)
    plot_constraints(constraints, env.feature_weights, out_path, title, highlight=None)


def plot_correction(env, trajectories: Sequence, num_samples: int, out_path: pathlib.Path, title: str):
    """Generate correction constraints using improvement feedback (notebook style)."""
    pairs = simulate_improvement_feedback_v4(env, trajectories, num_random_trajs=max(2, num_samples))

    # Track unique normalized vectors (notebook style)
    normalized_unique_vectors = []
    seen = set()

    for improved, other in pairs:
        # Compute features for improved trajectory
        improved_traj = [env.get_state_feature(s) for s, _ in improved]
        sum_feat_improved = np.sum(improved_traj, axis=0)

        # Compute features for original trajectory
        other_traj = [env.get_state_feature(s) for s, _ in other]
        sum_feat_other = np.sum(other_traj, axis=0)

        # Difference
        diff = sum_feat_improved - sum_feat_other

        # Normalize (L2 norm)
        norm = np.linalg.norm(diff)
        if norm == 0:
            normalized = np.zeros_like(diff)
        else:
            normalized = diff / norm

        # Convert to tuple for hashing and check uniqueness (notebook style)
        key = tuple(np.round(normalized, decimals=6))  # rounding to avoid float precision issues
        if key not in seen:
            seen.add(key)
            normalized_unique_vectors.append(normalized)

    # Remove redundant constraints (notebook style: epsilon=0.0001, no orientation toward true_reward)
    constraints = remove_redundant_constraints(normalized_unique_vectors, epsilon=0.0001)
    plot_constraints(constraints, env.feature_weights, out_path, title, highlight=None)


def plot_demonstration(env, trajectories: Sequence, num_samples: int, out_path: pathlib.Path, title: str):
    """Generate demonstration constraints using optimal vs random trajectory feature differences (notebook style)."""
    # Get Q-values for optimal trajectory generation
    q_values = ValueIteration(env).get_q_values()

    def get_intended_next_state(state, action):
        """Get next state by sampling from transition probabilities."""
        probs = env.transitions[state][action]
        return np.random.choice(list(range(env.num_states)), p=probs)

    def generate_optimal_trajectory(start_state):
        """Generate an optimal trajectory starting from start_state."""
        if start_state in env.terminal_states:
            return [start_state], []

        current_state = start_state
        states = [current_state]
        actions = []
        terminal_states = set(env.terminal_states)
        max_steps = 100

        for _ in range(max_steps):
            if current_state in terminal_states:
                break
            max_q = np.max(q_values[current_state])
            epsilon = 1e-10
            optimal_actions = [
                a for a in range(env.get_num_actions())
                if abs(q_values[current_state][a] - max_q) < epsilon
            ]
            if not optimal_actions:
                break
            action = np.random.choice(optimal_actions)
            actions.append(action)
            current_state = get_intended_next_state(current_state, action)
            states.append(current_state)

        return states, actions

    def generate_random_trajectory_for_demo(start_state):
        """Generate a random trajectory starting from start_state."""
        current_state = start_state
        states = [current_state]
        actions = []
        terminal_states = set(env.terminal_states)
        max_steps = 100

        for _ in range(max_steps):
            if current_state in terminal_states:
                break
            action = random.randrange(env.get_num_actions())
            actions.append(action)
            current_state = get_intended_next_state(current_state, action)
            states.append(current_state)

        return states, actions

    def compute_feature_sum(states):
        """Compute sum of features for a trajectory."""
        feature_sum = np.zeros(env.num_features)
        for state in states:
            feature_sum += env.get_state_feature(state)
        return feature_sum

    # Get non-terminal states
    num_states = env.get_num_states()
    terminal_states = env.terminal_states
    non_terminal_states = [s for s in range(num_states) if s not in terminal_states]

    # Generate optimal and random trajectories for each non-terminal state
    num_optimal_trajectories = 5
    num_random_trajectories = 5

    optimal_trajectories = {}
    random_trajectories = {}

    for start_state in non_terminal_states:
        optimal_trajectories[start_state] = [
            generate_optimal_trajectory(start_state) for _ in range(num_optimal_trajectories)
        ]
        random_trajectories[start_state] = [
            generate_random_trajectory_for_demo(start_state) for _ in range(num_random_trajectories)
        ]

    # Compute difference vectors: optimal_feature_sum - random_feature_sum
    # Only keep differences where dot product with true reward > 0
    diff_vectors = []
    for start_state in non_terminal_states:
        for opt_states, _ in optimal_trajectories[start_state]:
            opt_feature_sum = compute_feature_sum(opt_states)
            for rand_states, _ in random_trajectories[start_state]:
                rand_feature_sum = compute_feature_sum(rand_states)
                diff_vector = opt_feature_sum - rand_feature_sum
                # Only keep if dot product with true reward > 0 (notebook style)
                if np.dot(env.feature_weights, diff_vector) > 0:
                    diff_vectors.append(diff_vector)

    # Convert to numpy array for easier manipulation (notebook style)
    all_difference_vectors = np.array(diff_vectors)

    # Normalize the difference vectors using L2 norm (notebook style: vectorized)
    norms = np.linalg.norm(all_difference_vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized_difference_vectors = all_difference_vectors / norms

    # Convert back to a list if needed
    normalized_difference_vectors_list = normalized_difference_vectors.tolist()

    # Convert normalized vectors to a list of tuples for use with set() (notebook style)
    normalized_vectors_tuples = [tuple(vec) for vec in normalized_difference_vectors_list]

    # Get unique normalized vectors using set() (notebook style)
    unique_normalized_vectors_tuples = set(normalized_vectors_tuples)

    # Convert back to a list of lists (or arrays) if needed
    unique_normalized_vectors = [list(vec) for vec in unique_normalized_vectors_tuples]

    # Remove redundant constraints (notebook style: epsilon=0.0001)
    constraints = remove_redundant_constraints(unique_normalized_vectors, epsilon=0.0001)
    plot_constraints(constraints, env.feature_weights, out_path, title, highlight=None)


FEEDBACK_HANDLERS = {
    "estop": plot_estop,
    "pairwise": plot_pairwise,
    "correction": plot_correction,
    "demonstration": plot_demonstration,
}


# ----------------------------- constraint utilities -----------------------------

def traj_feature(env, traj):
    """Sum state-feature vectors along a trajectory."""
    feats = np.zeros(2, dtype=float)
    for state, _ in traj:
        if state is None:
            continue
        feats += env.get_state_feature(state)
    return feats


def process_constraints(
    constraints: Sequence[np.ndarray],
    normalize: bool = True,
    true_reward: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Normalize, orient toward true_reward (if provided), drop zeros, and remove redundant constraints."""
    cleaned = []
    for h in constraints:
        h = np.asarray(h, dtype=float)
        if h.shape[0] != 2:
            continue
        if normalize:
            nrm = np.linalg.norm(h)
            if nrm == 0:
                continue
            h = h / nrm
        if true_reward is not None and np.dot(h, true_reward) < 0:
            h = -h
        cleaned.append(h)
    return remove_redundant_constraints(cleaned, epsilon=1e-4)


def plot_constraints(
    constraints: Sequence[np.ndarray],
    true_reward: np.ndarray,
    out_path: pathlib.Path,
    title: str,
    highlight: int | None = None,
):
    """Plot halfspaces h^T w >= 0 in 2D, shade feasible region, mark true reward (notebook style)."""
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ["#d81159", "#218380"]

    # High resolution grid for feasibility shading (notebook style: 8000x8000)
    w1_grid = np.linspace(-1, 1, 8000)
    w2_grid = np.linspace(-1, 1, 8000)
    W1, W2 = np.meshgrid(w1_grid, w2_grid)
    feasible = np.ones_like(W1, dtype=bool)

    # High resolution for line plotting (notebook style: 4000 points)
    w1 = np.linspace(-1, 1, 4000)

    # deterministic order by angle for repeatability (notebook-esque)
    angles = []
    for h in constraints:
        h = np.asarray(h, dtype=float)
        if h.shape[0] != 2:
            angles.append(0.0)
        else:
            angles.append(np.arctan2(h[1], h[0]))
    plot_order: list[int] = sorted(range(len(constraints)), key=lambda i: angles[i])
    # Show all constraints (notebook style: binding_indices includes all)
    if highlight is not None:
        plot_order = plot_order[:highlight]

    # plot all constraints
    for idx in plot_order:
        h = np.asarray(constraints[idx], dtype=float)
        if h.shape[0] != 2:
            continue
        c = colors[idx % len(colors)]
        if abs(h[1]) < 1e-12:
            ax.axvline(0.0, color=c, linewidth=5, label=f"Constraint {idx+1}")
        else:
            y_line = -(h[0] / h[1]) * w1
            ax.plot(w1, y_line, color=c, linewidth=5, label=f"Constraint {idx+1}")

    # feasibility uses all constraints
    for h in constraints:
        h = np.asarray(h, dtype=float)
        if h.shape[0] != 2:
            continue
        feasible &= (W1 * h[0] + W2 * h[1]) >= -1e-12

    # shade feasible region
    if constraints:
        ax.contourf(W1, W2, feasible, levels=[0.5, 1], colors="orange", alpha=1, hatches=["//"])

    # true reward point
    ax.scatter(true_reward[0], true_reward[1], marker="*", color="black", s=140, edgecolors="black", zorder=5, label="True Reward")

    ax.axhline(0, color="black", linewidth=3)
    ax.axvline(0, color="black", linewidth=3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("w1", fontsize=32)
    ax.set_ylabel("w2", fontsize=32)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_title(title)
    if constraints:
        ax.legend(loc="upper right", fontsize=22)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- entry point -----------------------------

def run(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_root = pathlib.Path(args.output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    handler = FEEDBACK_HANDLERS[args.feedback]
    feedback_dir = out_root / args.feedback
    feedback_dir.mkdir(parents=True, exist_ok=True)

    envs, _meta = generate_random_gridworld_envs(
        n_envs=args.n_envs,
        rows=args.rows, cols=args.cols,
        color_to_feature_map={"red":[1.0,0.0], "blue":[0.0,1.0]},
        palette=("red","blue"),
        p_color_range={"red":(0.2,0.6), "blue":(0.4,0.8)},
        terminal_policy=dict(kind="random_k", k_min=0, k_max=1, p_no_terminal=0.1),
        gamma_range=(0.98, 0.995),
        noise_prob_range=(0.0, 0.0),
        w_mode="fixed",
        W_fixed=args.true_reward,
        seed=args.seed,
        GridEnvClass=GridWorldMDPFromLayoutEnv,
        ensure_unique_layouts=True,
        ensure_unique_layout_terminals=True,
    )

    for env_idx, env in enumerate(envs):
        env.set_random_seed(args.seed + env_idx)
        horizon = args.max_horizon or (env.num_states * 2)
        trajectories = sample_trajectories(env, args.num_samples, max_horizon=horizon)
        print(envs[env_idx].layout)
        title = f"{args.feedback} env{env_idx + 1}"
        out_path = feedback_dir / f"{args.feedback}_env{env_idx + 1}.png"
        if args.feedback in {"pairwise", "correction", "demonstration"}:
            handler(env, trajectories, args.num_samples, out_path, title)
        else:
            handler(env, trajectories, out_path, title)
        print(f"Saved {title} -> {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate feedback plots for synthetic GridWorlds.")
    parser.add_argument("--seed", type=int, required=True, help="Base seed. Each env uses seed + idx.")
    parser.add_argument("--n_envs", type=int, required=True, help="How many environments to generate.")
    parser.add_argument("--rows", type=int, required=True, help="Grid rows for the layout env.")
    parser.add_argument("--cols", type=int, required=True, help="Grid cols for the layout env.")
    parser.add_argument(
        "--true_reward",
        type=parse_reward_vector,
        required=True,
        help="Two comma/space separated reward weights (e.g., '-0.97 0.24').",
    )
    parser.add_argument("--num_samples", type=int, required=True, help="Number of trajectories/samples per env.")
    parser.add_argument(
        "--feedback",
        choices=sorted(FEEDBACK_HANDLERS.keys()),
        required=True,
        help="Which feedback type to simulate and plot.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory where graphs are stored (graphs go under output_dir/<feedback>/).",
    )
    parser.add_argument(
        "--max_horizon",
        type=int,
        default=None,
        help="Optional trajectory horizon (defaults to rows * cols).",
    )
    return parser


if __name__ == "__main__":
    cli = build_arg_parser()
    run(cli.parse_args())
