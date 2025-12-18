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


def stage(title):
    log("\n" + "=" * 60)
    log(f"[STAGE] {title}")
    log("=" * 60)


# ============================================================
# Main experiment
# ============================================================

def run_experiment(args):
    stage("EXPERIMENT START")

    log(f"Config: n_envs={args.n_envs}, grid={args.mdp_size}x{args.mdp_size}, "
        f"d={args.feature_dim}, feedback={args.feedback}, seed={args.seed}")

    rng = np.random.default_rng(args.seed)

    # --------------------------------------------------------
    # Ground-truth reward
    # --------------------------------------------------------
    stage("GROUND-TRUTH REWARD")

    if args.w_true_mode == "random_signed":
        W_TRUE = rng.normal(size=args.feature_dim)
        W_TRUE /= np.linalg.norm(W_TRUE)
    elif args.w_true_mode == "one_hot":
        W_TRUE = np.zeros(args.feature_dim)
        W_TRUE[rng.integers(args.feature_dim)] = 1.0
    else:
        W_TRUE = np.ones(args.feature_dim)
        W_TRUE /= np.linalg.norm(W_TRUE)

    log(f"W_TRUE = {W_TRUE}")

    # --------------------------------------------------------
    # Environments
    # --------------------------------------------------------
    stage("ENVIRONMENT GENERATION")

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

    log(f"Generated {len(envs)} environments")

    # --------------------------------------------------------
    # Value iteration
    # --------------------------------------------------------
    stage("VALUE ITERATION")

    Q_list = parallel_value_iteration(envs, epsilon=1e-10)
    log(f"Computed Q-functions for {len(Q_list)} environments")

    # --------------------------------------------------------
    # Successor features
    # --------------------------------------------------------
    stage("SUCCESSOR FEATURES")

    SFs = compute_successor_features_family(
        envs,
        Q_list,
        convention="entering",
        zero_terminal_features=True,
        tol=1e-10,
        max_iters=10000,
    )

    log(f"Computed successor features for {len(SFs)} environments")

    # --------------------------------------------------------
    # Q-based constraints
    # --------------------------------------------------------
    stage("Q-BASED CONSTRAINT EXTRACTION")

    U_per_env_q, U_q_global = derive_constraints_from_q_family(
        SFs,
        Q_list,
        envs,
        skip_terminals=True,
        normalize=True,
    )

    log(f"Total Q-based constraints: {len(U_q_global)}")
    log(f"Per-env Q constraints: {[len(H) for H in U_per_env_q]}")

    # --------------------------------------------------------
    # Candidate atoms
    # --------------------------------------------------------
    stage("CANDIDATE ATOM GENERATION")

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

    log(f"Atoms per env: {[len(c) for c in candidates_per_env]}")

    # --------------------------------------------------------
    # Atom constraints
    # --------------------------------------------------------
    stage("ATOM CONSTRAINT EXTRACTION")

    U_per_env_atoms, U_atoms_global = derive_constraints_from_atoms(
        candidates_per_env,
        SFs,
        envs,
    )

    log(f"Total atom-based constraints: {len(U_atoms_global)}")

    # --------------------------------------------------------
    # Universal constraint set
    # --------------------------------------------------------
    stage("UNIVERSAL CONSTRAINT SET")

    U_universal = remove_redundant_constraints(
        np.vstack([U_q_global, U_atoms_global]),
        epsilon=1e-4,
    )

    log(f"|U_universal| = {len(U_universal)}")

    # --------------------------------------------------------
    # TWO-STAGE SCOT
    # --------------------------------------------------------
    stage("TWO-STAGE SCOT")

    two_stage = two_stage_scot_no_cost(
        U_universal=U_universal,
        U_per_env_atoms=U_per_env_atoms,
        U_per_env_q=U_per_env_q,
        candidates_per_env=candidates_per_env,
        SFs=SFs,
        envs=envs,
    )

    two_stage_envs = sorted(two_stage["activated_envs"])

    log(f"Stage-1 selected MDPs: {two_stage['selected_mdps']}")
    log(f"Stage-2 activated envs: {two_stage_envs}")
    log(f"#Atoms selected: {len(two_stage['chosen_atoms'])}")

    # --------------------------------------------------------
    # FLAT SCOT
    # --------------------------------------------------------
    stage("FLAT SCOT")

    chosen_flat, flat_stats, _ = scot_greedy_family_atoms_tracked(
        U_universal,
        candidates_per_env,
        SFs,
        envs,
    )

    flat_envs = sorted({i for i, _ in chosen_flat})

    log(f"Activated envs: {flat_envs}")
    log(f"#Atoms selected: {len(chosen_flat)}")

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------
    stage("FINAL COMPARISON SUMMARY")

    log(f"Two-stage activated envs: {len(two_stage_envs)}")
    log(f"Flat activated envs: {len(flat_envs)}")
    log(f"Two-stage atoms: {len(two_stage['chosen_atoms'])}")
    log(f"Flat atoms: {len(chosen_flat)}")
    log(f"Same activated envs? {set(two_stage_envs) == set(flat_envs)}")

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------
    stage("SAVING RESULTS")

    os.makedirs(args.result_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    feedback_tag = "_".join(sorted(args.feedback)) or "no_feedback"

    out_path = os.path.join(
        args.result_dir,
        f"scot_compare_{feedback_tag}_{run_id}.json"
    )

    with open(out_path, "w") as f:
        json.dump({"note": "see logs for detailed stage output"}, f, indent=2)

    log(f"[SAVED] {out_path}")
    stage("EXPERIMENT END")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
