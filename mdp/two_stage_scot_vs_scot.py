# ============================================================
# test_compare_scot_algorithms.py
# ============================================================

import argparse
import json
import os
import sys
from datetime import datetime
import numpy as np

# ============================================================
# Project path
# ============================================================

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

# ============================================================
# Imports from repo
# ============================================================

from mdp.gridworld_env_layout import GridWorldMDPFromLayoutEnv
from utils import (
    generate_random_gridworld_envs,
    parallel_value_iteration,
    compute_successor_features_family,
    derive_constraints_from_q_family,
    derive_constraints_from_atoms,
    generate_candidate_atoms_for_scot,
    remove_redundant_constraints,
)
from teaching.two_stage_scot import two_stage_scot_no_cost
from teaching import scot_greedy_family_atoms_tracked


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    # Environment / MDP parameters
    parser.add_argument("--n_envs", type=int, default=30)
    parser.add_argument("--mdp_size", type=int, default=10)
    parser.add_argument("--feature_dim", type=int, default=2)
    parser.add_argument(
        "--w_true_mode",
        type=str,
        default="random_signed",
        choices=["random_signed", "one_hot", "biased"],
    )

    # Feedback settings
    parser.add_argument(
        "--feedback",
        nargs="+",
        default=["demo", "pairwise", "estop", "improvement"],
    )
    parser.add_argument("--feedback_count", type=int, default=50)

    # General
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result_dir", type=str, default="results_universal")

    return parser.parse_args()


# ============================================================
# Utility
# ============================================================

def log(msg):
    print(msg, flush=True)


# ============================================================
# Main experiment
# ============================================================

def run_experiment(args):
    log("\n==============================================")
    log(" SCOT vs TWO-STAGE SCOT — FULL COMPARISON RUN")
    log("==============================================\n")

    rng = np.random.default_rng(args.seed)

    # --------------------------------------------------------
    # Ground-truth reward
    # --------------------------------------------------------
    if args.w_true_mode == "random_signed":
        W_TRUE = rng.normal(size=args.feature_dim)
        W_TRUE /= np.linalg.norm(W_TRUE)
    elif args.w_true_mode == "one_hot":
        W_TRUE = np.zeros(args.feature_dim)
        W_TRUE[rng.integers(args.feature_dim)] = 1.0
    else:  # biased
        W_TRUE = np.ones(args.feature_dim)
        W_TRUE /= np.linalg.norm(W_TRUE)

    # --------------------------------------------------------
    # Environments
    # --------------------------------------------------------
    color_to_feature_map = {
        f"f{i}": [1 if j == i else 0 for j in range(args.feature_dim)]
        for i in range(args.feature_dim)
    }

    envs, _ = generate_random_gridworld_envs(
        n_envs=args.n_envs,
        rows=args.mdp_size,
        cols=args.mdp_size,
        color_to_feature_map=color_to_feature_map,
        palette=list(color_to_feature_map.keys()),
        p_color_range={c: (0.3, 0.7) for c in color_to_feature_map},
        terminal_policy=dict(kind="random_k", k_min=1, k_max=1, p_no_terminal=0.0),
        gamma_range=(0.99, 0.99),
        noise_prob_range=(0.0, 0.0),
        w_mode="fixed",
        W_fixed=W_TRUE,
        seed=args.seed,
        GridEnvClass=GridWorldMDPFromLayoutEnv,
    )

    # --------------------------------------------------------
    # Value iteration
    # --------------------------------------------------------
    Q_list = parallel_value_iteration(envs, epsilon=1e-10)

    # --------------------------------------------------------
    # Successor features
    # --------------------------------------------------------
    SFs = compute_successor_features_family(
        envs,
        Q_list,
        convention="entering",
        zero_terminal_features=True,
        tol=1e-10,
        max_iters=10000,
    )

    # --------------------------------------------------------
    # Q-based constraints
    # --------------------------------------------------------
    U_per_env_q, U_q_global = derive_constraints_from_q_family(
        SFs,
        Q_list,
        envs,
        skip_terminals=True,
        normalize=True,
    )

    # --------------------------------------------------------
    # Candidate atoms
    # --------------------------------------------------------
    candidates_per_env = generate_candidate_atoms_for_scot(
        envs,
        Q_list,
        use_q_demos="demo" in args.feedback,
        num_q_rollouts_per_state=1,
        q_demo_max_steps=1,
        use_pairwise="pairwise" in args.feedback,
        use_estop="estop" in args.feedback,
        use_improvement="improvement" in args.feedback,
        n_pairwise=args.feedback_count,
        n_estops=args.feedback_count,
        n_improvements=args.feedback_count,
    )

    # --------------------------------------------------------
    # Atom constraints
    # --------------------------------------------------------
    U_per_env_atoms, U_atoms_global = derive_constraints_from_atoms(
        candidates_per_env,
        SFs,
        envs,
    )

    # --------------------------------------------------------
    # Universal constraint set
    # --------------------------------------------------------
    U_universal = remove_redundant_constraints(
        np.vstack([U_q_global, U_atoms_global]),
        epsilon=1e-4,
    )

    # --------------------------------------------------------
    # Per-env stats
    # --------------------------------------------------------
    per_env_stats = {}
    for i in range(args.n_envs):
        per_env_stats[i] = {
            "q_constraints": len(U_per_env_q[i]),
            "atom_constraints": len(U_per_env_atoms[i]),
            "total_constraints": len(U_per_env_q[i]) + len(U_per_env_atoms[i]),
        }

    # --------------------------------------------------------
    # TWO-STAGE SCOT
    # --------------------------------------------------------
    two_stage = two_stage_scot_no_cost(
        U_universal=U_universal,
        U_per_env_atoms=U_per_env_atoms,
        U_per_env_q=U_per_env_q,
        candidates_per_env=candidates_per_env,
        SFs=SFs,
        envs=envs,
    )

    two_stage_envs = sorted(two_stage["activated_envs"])

    # --------------------------------------------------------
    # FLAT SCOT
    # --------------------------------------------------------
    chosen_flat, flat_stats, _ = scot_greedy_family_atoms_tracked(
        U_universal,
        candidates_per_env,
        SFs,
        envs,
    )

    flat_envs = sorted({i for i, _ in chosen_flat})

    # --------------------------------------------------------
    # Comparison metrics
    # --------------------------------------------------------
    comparison = {
        "same_activated_envs": set(two_stage_envs) == set(flat_envs),
        "env_activation_ratio": {
            "two_stage": len(two_stage_envs) / args.n_envs,
            "flat": len(flat_envs) / args.n_envs,
        },
        "atoms_per_activated_env": {
            "two_stage": len(two_stage["chosen_atoms"]) / max(1, len(two_stage_envs)),
            "flat": len(chosen_flat) / max(1, len(flat_envs)),
        },
        "constraints_per_atom": {
            "two_stage": len(U_universal) / max(1, len(two_stage["chosen_atoms"])),
            "flat": len(U_universal) / max(1, len(chosen_flat)),
        },
    }

    # --------------------------------------------------------
    # Final JSON
    # --------------------------------------------------------
    result = {
        "config": vars(args),
        "universal_constraints": {
            "total": len(U_universal),
            "q_constraints": len(U_q_global),
            "atom_constraints": len(U_atoms_global),
        },
        "per_env_stats": per_env_stats,
        "two_stage_scot": {
            "selected_mdps_stage1": two_stage["selected_mdps"],
            "activated_envs": two_stage_envs,
            "num_atoms": len(two_stage["chosen_atoms"]),
        },
        "flat_scot": {
            "activated_envs": flat_envs,
            "num_atoms": len(chosen_flat),
        },
        "comparison": comparison,
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    os.makedirs(args.result_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    feedback_tag = "_".join(sorted(args.feedback))
    # fallback if empty (just in case)
    if not feedback_tag:
        feedback_tag = "no_feedback"

    out_path = os.path.join(
        args.result_dir,
        f"scot_compare_{feedback_tag}_{run_id}.json"
    )

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    log(f"\n[SAVED] Results written to {out_path}")
    log("==============================================\n")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
