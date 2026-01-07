import argparse
import json
import os
import sys
import numpy as np
from datetime import datetime

# ============================================================
# Project path
# ============================================================
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

# ============================================================
# Repo Imports
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
    GenerationSpec, DemoSpec, FeedbackSpec
)

# Algorithms
from teaching.two_stage_scot_lazy import two_stage_scot_weighted_lazy
from teaching.scot_lazy import scot_greedy_family_atoms_tracked_lazy
from teaching import scot_greedy_family_atoms_tracked

# ============================================================
# Args
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=50)
    parser.add_argument("--mdp_size", type=int, default=8)
    parser.add_argument("--feature_dim", type=int, default=3)
    parser.add_argument("--feedback", nargs="+",
                        default=["demo", "pairwise", "estop", "improvement"])
    parser.add_argument("--total_budget", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result_dir", type=str, default="results_lazy_compare")
    return parser.parse_args()


def log(msg):
    print(f"[LOG] {msg}", flush=True)

# ============================================================
# Experiment
# ============================================================
def run_experiment(args):
    log("=== STARTING LAZY SCOT COMPARISON EXPERIMENT ===")
    rng = np.random.default_rng(args.seed)

    # --------------------------------------------------------
    # 1. True reward + environments
    # --------------------------------------------------------
    W_TRUE = rng.normal(size=args.feature_dim)
    W_TRUE /= np.linalg.norm(W_TRUE)

    color_map = {
        f"f{i}": [1 if j == i else 0 for j in range(args.feature_dim)]
        for i in range(args.feature_dim)
    }

    envs, _ = generate_random_gridworld_envs(
        n_envs=args.n_envs,
        rows=args.mdp_size,
        cols=args.mdp_size,
        color_to_feature_map=color_map,
        palette=list(color_map.keys()),
        W_fixed=W_TRUE,
        seed=args.seed,
        GridEnvClass=GridWorldMDPFromLayoutEnv
    )

    # --------------------------------------------------------
    # 2. Solve MDPs + successor features
    # --------------------------------------------------------
    Q_list = parallel_value_iteration(envs)
    SFs = compute_successor_features_family(envs, Q_list)

    # --------------------------------------------------------
    # 3. Q-based constraints
    # --------------------------------------------------------
    U_per_env_q, U_q_global = derive_constraints_from_q_family(
        SFs, Q_list, envs
    )

    # --------------------------------------------------------
    # 4. Candidate atoms (budgeted feedback generation)
    # --------------------------------------------------------
    spec = GenerationSpec(
        seed=args.seed,
        demo=DemoSpec(enabled="demo" in args.feedback, env_fraction=0.1),
        pairwise=FeedbackSpec(
            enabled="pairwise" in args.feedback,
            total_budget=args.total_budget,
            alloc_method="dirichlet",
            alloc_params={"alpha": 0.5},
        ),
        estop=FeedbackSpec(
            enabled="estop" in args.feedback,
            total_budget=args.total_budget // 4,
            alloc_method="sparse_poisson",
            alloc_params={"p_active": 0.1},
        )
    )

    candidates_per_env = generate_candidate_atoms_for_scot(
        envs, Q_list, spec=spec
    )

    # --------------------------------------------------------
    # 5. Atom-based constraints + universal set
    # --------------------------------------------------------
    U_per_env_atoms, U_atoms_global = derive_constraints_from_atoms(
        candidates_per_env, SFs, envs
    )

    U_universal = remove_redundant_constraints(
        np.vstack([U_q_global, U_atoms_global])
    )

    log(f"Universal constraint count: {len(U_universal)}")

    # ========================================================
    # Algorithm A: Lazy Two-Stage SCOT
    # ========================================================
    log("Running Lazy Two-Stage SCOT...")
    res_two_stage = two_stage_scot_weighted_lazy(
        U_universal=U_universal,
        U_per_env_atoms=U_per_env_atoms,
        U_per_env_q=U_per_env_q,
        candidates_per_env=candidates_per_env,
        SFs=SFs,
        envs=envs
    )

    # ========================================================
    # Algorithm B: Lazy Naive SCOT (Pool = all envs)
    # ========================================================
    # log("Running Lazy Naive SCOT...")
    # _, stats_lazy_naive, _ = scot_greedy_family_atoms_tracked_lazy(
    #     U_universal,
    #     candidates_per_env,
    #     SFs,
    #     envs
    # )

    log("Running Lazy Naive SCOT...")
    _, stats_lazy_naive, _ = scot_greedy_family_atoms_tracked(
        U_universal,
        candidates_per_env,
        SFs,
        envs
    )

    naive_inspected = stats_lazy_naive["total_inspected_count"]
    naive_activated = len(stats_lazy_naive["activated_env_indices"])

    # ========================================================
    # Summary
    # ========================================================
    log("\n" + "=" * 55)
    log(f"RESULTS SUMMARY (seed={args.seed})")
    log("-" * 55)
    log(f"{'Metric':<30} | {'Two-Stage':>10} | {'Naive':>10}")
    log("-" * 55)
    log(f"{'Env inspections':<30} | "
        f"{res_two_stage['inspection_count']:>10} | {naive_inspected:>10}")
    log(f"{'Activated envs':<30} | "
        f"{res_two_stage['activation_count']:>10} | {naive_activated:>10}")
    log(f"{'Inspection waste':<30} | "
        f"{res_two_stage['waste']:>10} | "
        f"{naive_inspected - naive_activated:>10}")
    log("-" * 55)
    log(f"Env inspections saved: "
        f"{naive_inspected - res_two_stage['inspection_count']}")
    log("=" * 55)

    # ========================================================
    # Save results
    # ========================================================
    os.makedirs(args.result_dir, exist_ok=True)
    out_path = os.path.join(
        args.result_dir,
        f"lazy_compare_seed_{args.seed}.json"
    )

    with open(out_path, "w") as f:
        json.dump({
            "seed": args.seed,
            "two_stage": {
                "inspected_envs": res_two_stage["inspection_count"],
                "activated_envs": res_two_stage["activation_count"],
                "waste": res_two_stage["waste"],
                "stage1_stats": res_two_stage["stage1_stats"],
            },
            "lazy_naive": {
                "inspected_envs": naive_inspected,
                "activated_envs": naive_activated,
                "lazy_stats": stats_lazy_naive["lazy_global"],
            }
        }, f, indent=2)

    log(f"Results saved to {out_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    run_experiment(parse_args())
