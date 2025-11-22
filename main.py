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
    constraints = []
    for traj in trajectories:
        estop_traj, stop_time = simulate_human_estop_v2(env, traj, beta=2.0, gamma=env.gamma)
        if stop_time is None:
            continue
        prefix = estop_traj[: int(stop_time) + 1]
        if not prefix:
            continue

        # diff = features up to stop - features of full trajectory
        feat_prefix = traj_feature(env, prefix)
        feat_full = traj_feature(env, estop_traj)
        diff = feat_prefix - feat_full
        norm = np.linalg.norm(diff)
        if norm == 0:
            continue
        constraints.append(diff / norm)

    constraints = process_constraints(constraints, normalize=False, true_reward=env.feature_weights)
    plot_constraints(constraints, env.feature_weights, out_path, title, highlight=2)


def plot_pairwise(env, trajectories: Sequence, num_samples: int, out_path: pathlib.Path, title: str):
    max_pairs = max(0, len(trajectories) * (len(trajectories) - 1))
    num_to_draw = min(num_samples, max_pairs)
    comparisons = (
        generate_pairwise_comparisons(env, trajectories, num_comparisons=num_to_draw)
        if num_to_draw
        else []
    )
    constraints = []
    for t1, t2 in comparisons:
        phi1 = traj_feature(env, t1)
        phi2 = traj_feature(env, t2)
        diff = phi1 - phi2
        nrm = np.linalg.norm(diff)
        if nrm == 0:
            continue
        constraints.append(diff / nrm)
    constraints = process_constraints(constraints, true_reward=env.feature_weights)
    plot_constraints(constraints, env.feature_weights, out_path, title, highlight=2)


def plot_correction(env, trajectories: Sequence, num_samples: int, out_path: pathlib.Path, title: str):
    pairs = simulate_improvement_feedback_v4(env, trajectories, num_random_trajs=max(2, num_samples))
    constraints = []
    for improved, original in pairs:
        phi_improved = traj_feature(env, improved)
        phi_original = traj_feature(env, original)
        diff = phi_improved - phi_original
        nrm = np.linalg.norm(diff)
        if nrm == 0:
            continue
        constraints.append(diff / nrm)
    constraints = process_constraints(constraints, true_reward=env.feature_weights)
    plot_constraints(constraints, env.feature_weights, out_path, title, highlight=2)


def plot_demonstration(env, trajectories: Sequence, num_samples: int, out_path: pathlib.Path, title: str):
    # Use only demonstration trajectories as constraints (notebook-style: positives only).
    demos = list(trajectories[:num_samples])
    constraints = [traj_feature(env, demo) for demo in demos if demo]
    constraints = process_constraints(constraints, normalize=True, true_reward=env.feature_weights)
    plot_constraints(constraints, env.feature_weights, out_path, title, highlight=2)


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
    colors = ["#d81159", "#218380", "#ff7f0e", "#1f77b4", "#9467bd"]

    # dense grid for feasibility shading (use moderate resolution to avoid huge memory)
    w1 = np.linspace(-1, 1, 1000)
    w2 = np.linspace(-1, 1, 1000)
    W1, W2 = np.meshgrid(w1, w2)
    feasible = np.ones_like(W1, dtype=bool)

    # deterministic order by angle for repeatability (notebook-esque)
    angles = []
    for h in constraints:
        h = np.asarray(h, dtype=float)
        if h.shape[0] != 2:
            angles.append(0.0)
        else:
            angles.append(np.arctan2(h[1], h[0]))
    plot_order: list[int] = sorted(range(len(constraints)), key=lambda i: angles[i])
    if highlight is not None:
        plot_order = plot_order[:highlight]

    # plot highlighted constraints
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
