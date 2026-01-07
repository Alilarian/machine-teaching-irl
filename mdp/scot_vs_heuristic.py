import argparse
import json
import os
import sys
import time
from datetime import datetime
import numpy as np

# ============================================================
# Project path
# ============================================================
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

# ============================================================
# Imports
# ============================================================
from mdp.gridworld_env_layout import GridWorldMDPFromLayoutEnv
from utils import (
    generate_random_gridworld_envs,
    parallel_value_iteration,
    compute_successor_features_family,
    derive_constraints_from_q_family,
    derive_constraints_from_atoms,
    
    remove_redundant_constraints,
    GenerationSpec,
    DemoSpec,
    FeedbackSpec,
)

from utils.feedback_budgeting import generate_candidate_atoms_for_scot

from teaching.scot import scot_greedy_family_atoms_tracked as scot_naive
from teaching.scot_heuristic import (
    scot_with_env_heuristic,
    rank_envs_by_constraint_informativeness,
)

from collections import Counter, defaultdict

def print_feedback_distribution(atoms_per_env):
    """
    atoms_per_env: List[List[Atom]]
    """

    n_envs = len(atoms_per_env)

    # Per-env stats
    per_env_stats = []
    global_counter = Counter()
    env_has_type = defaultdict(int)

    for env_idx, atoms in enumerate(atoms_per_env):
        c = Counter(a.feedback_type for a in atoms)
        total = len(atoms)

        per_env_stats.append((env_idx, total, c))

        global_counter.update(c)
        for k in c:
            env_has_type[k] += 1

    # ----------------------------
    # Pretty print per-env table
    # ----------------------------
    print("\n" + "=" * 80)
    print("[FEEDBACK DISTRIBUTION PER ENV]")
    print("=" * 80)
    print(f"{'Env':>4} | {'Total':>5} | demo | pairwise | estop | improvement")
    print("-" * 80)

    for env_idx, total, c in per_env_stats:
        print(f"{env_idx:4d} | {total:5d} | "
              f"{c.get('demo', 0):4d} | "
              f"{c.get('pairwise', 0):8d} | "
              f"{c.get('estop', 0):5d} | "
              f"{c.get('improvement', 0):11d}")

    # ----------------------------
    # Global summary
    # ----------------------------
    print("\n" + "=" * 80)
    print("[GLOBAL FEEDBACK SUMMARY]")
    print("=" * 80)

    total_atoms = sum(global_counter.values())
    for k, v in global_counter.items():
        print(f"{k:12s}: {v:6d} atoms "
              f"({v / max(total_atoms, 1):.2%}), "
              f"active envs = {env_has_type[k]}/{n_envs}")

    print(f"\nTotal atoms: {total_atoms}")


# ============================================================
# Args
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("Naive SCOT vs Heuristic SCOT (full pipeline)")
    p.add_argument("--n_envs", type=int, default=50)
    p.add_argument("--mdp_size", type=int, default=8)
    p.add_argument("--feature_dim", type=int, default=4)
    p.add_argument("--total_budget", type=int, default=2000)
    p.add_argument("--demo_env_fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)

    # NEW: feedback selection
    p.add_argument(
        "--feedback",
        nargs="+",
        choices=["demo", "pairwise", "estop", "improvement"],
        default=["demo", "pairwise", "estop", "improvement"],
        help="Which feedback types to enable",
    )

    # Heuristic
    p.add_argument("--heuristic_k", type=int, default=10)
    p.add_argument("--top_frac", type=float, default=0.10)

    p.add_argument("--result_dir", type=str, default="results_comparison")
    return p.parse_args()


# ============================================================
# Logging helpers
# ============================================================

def stage(title):
    print("\n" + "=" * 80)
    print(f"[STAGE] {title}")
    print("=" * 80)

def log(msg):
    print(f"[INFO] {msg}")


# ============================================================
# Main experiment
# ============================================================

def main(args):

    rng = np.random.default_rng(args.seed)

    # ========================================================
    # SETUP
    # ========================================================
    stage("EXPERIMENT SETUP")
    log(vars(args))

    W_TRUE = rng.normal(size=args.feature_dim)
    W_TRUE /= np.linalg.norm(W_TRUE)

    # ========================================================
    # ENV GENERATION
    # ========================================================
    stage("GENERATING ENVIRONMENTS")

    color_to_feature_map = {
        f"f{i}": [1 if j == i else 0 for j in range(args.feature_dim)]
        for i in range(args.feature_dim)
    }

    envs, _ = generate_random_gridworld_envs(
        n_envs=args.n_envs,
        rows=args.mdp_size,
        cols=args.mdp_size,
        color_to_feature_map=color_to_feature_map,
        palette=list(color_to_feature_map),
        p_color_range={c: (0.3, 0.7) for c in color_to_feature_map},
        terminal_policy=dict(kind="random_k", k_min=1, k_max=1),
        gamma_range=(0.99, 0.99),
        noise_prob_range=(0.0, 0.0),
        w_mode="fixed",
        W_fixed=W_TRUE,
        seed=args.seed,
        GridEnvClass=GridWorldMDPFromLayoutEnv,
    )

    log(f"Generated {len(envs)} envs")

    # ========================================================
    # VALUE ITERATION + SFS
    # ========================================================
    stage("VALUE ITERATION & SUCCESSOR FEATURES")

    Q_list = parallel_value_iteration(envs)
    SFs = compute_successor_features_family(
        envs, Q_list,
        convention="entering",
        zero_terminal_features=True,
        tol=1e-10,
    )

    # ========================================================
    # CONSTRAINT GENERATION
    # ========================================================
    stage("GENERATING CONSTRAINTS")

    _, U_q = derive_constraints_from_q_family(
        SFs, Q_list, envs,
        skip_terminals=True,
        normalize=True,
    )

    enabled = set(args.feedback)

    spec = GenerationSpec(
        seed=args.seed,

        demo=DemoSpec(
            enabled=("demo" in enabled),
            env_fraction=1,
            state_fraction=args.demo_env_fraction,
        ),

        pairwise=FeedbackSpec(
            enabled=("pairwise" in enabled),
            total_budget=args.total_budget if "pairwise" in enabled else 0,
        ),

        estop=FeedbackSpec(
            enabled=("estop" in enabled),
            total_budget=(args.total_budget) if "estop" in enabled else 0,
        ),

        improvement=FeedbackSpec(
            enabled=("improvement" in enabled),
            total_budget=(args.total_budget) if "improvement" in enabled else 0,
        ),
    )

    atoms_per_env = generate_candidate_atoms_for_scot(envs, Q_list, spec=spec)
    atom_counts = [len(a) for a in atoms_per_env]
    log(f"Atoms per env: mean={np.mean(atom_counts):.1f}, total={sum(atom_counts)}")
        
    print_feedback_distribution(atoms_per_env)

    _, U_atoms = derive_constraints_from_atoms(atoms_per_env, SFs, envs)



    U_global = remove_redundant_constraints(
        np.vstack([U_q, U_atoms]),
    )

    log(f"|U_global| = {len(U_global)}")

    # ========================================================
    # HEURISTIC DIAGNOSTIC (BEFORE SCOT)
    # ========================================================
    stage("ENV HEURISTIC RANKING")

    ranked_envs, heuristic_stats = rank_envs_by_constraint_informativeness(
        atoms_per_env,
        SFs,
        envs,
        K=args.heuristic_k,
    )

    n_keep = max(1, int(args.top_frac * args.n_envs))
    selected_envs = ranked_envs[:n_keep]

    log(f"Selected top {n_keep}/{args.n_envs} envs")

    print("\nTop environments by heuristic:")
    for i in selected_envs:
        print(f"  Env {i:02d} → unique_constraints = "
              f"{heuristic_stats[i]['unique_constraints']}")

    # ========================================================
    # NAIVE SCOT
    # ========================================================
    stage("RUNNING NAIVE SCOT")

    t0 = time.time()
    chosen_naive, stats_naive, _ = scot_naive(
        U_global,
        atoms_per_env,
        SFs,
        envs,
    )
    t_naive = time.time() - t0

    # ========================================================
    # HEURISTIC SCOT
    # ========================================================
    stage("RUNNING HEURISTIC SCOT")

    t0 = time.time()
    chosen_h, stats_h, global_h = scot_with_env_heuristic(
        U_global,
        atoms_per_env,
        SFs,
        envs,
        K=args.heuristic_k,
        top_frac=args.top_frac,
    )
    t_h = time.time() - t0

    # ========================================================
    # SUMMARY
    # ========================================================
    stage("SUMMARY")

    def summarize(name, chosen, stats, runtime, U_global_size):
        coverage_frac = stats["final_coverage"] / max(U_global_size, 1)

        log(f"{name}")
        log(f"  atoms selected      = {len(chosen)}")
        log(f"  envs inspected      = {stats['total_inspected_count']}")
        log(f"  envs activated      = {stats['total_activated_count']}")
        log(f"  final coverage      = {stats['final_coverage']} / {U_global_size}")
        log(f"  coverage fraction   = {coverage_frac:.3f}")
        log(f"  runtime (s)         = {runtime:.2f}")


    summarize("NAIVE SCOT", chosen_naive, stats_naive, t_naive, len(U_global))
    summarize("HEURISTIC SCOT", chosen_h, stats_h, t_h, len(U_global))

    # ========================================================
    # SAVE RESULTS
    # ========================================================
    stage("SAVING RESULTS")

    os.makedirs(args.result_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    save = {
        "args": vars(args),
        "U_global_size": len(U_global),
        "heuristic_ranking": {
            int(i): heuristic_stats[i]["unique_constraints"]
            for i in ranked_envs
        },
        "selected_envs": selected_envs,
        "naive": {
            "atoms": len(chosen_naive),
            "inspected_envs": stats_naive["total_inspected_count"],
            "activated_envs": stats_naive["total_activated_count"],
            "final_coverage": stats_naive["final_coverage"],
            "coverage_fraction": stats_naive["final_coverage"] / max(len(U_global), 1),
            "runtime": t_naive,
        },
        "heuristic": {
            "atoms": len(chosen_h),
            "inspected_envs": stats_h["total_inspected_count"],
            "activated_envs": stats_h["total_activated_count"],
            "final_coverage": stats_h["final_coverage"],
            "coverage_fraction": stats_h["final_coverage"] / max(len(U_global), 1),
            "runtime": t_h,
        },

    }

    path = os.path.join(args.result_dir, f"scot_comparison_{run_id}.json")
    with open(path, "w") as f:
        json.dump(save, f, indent=2)

    log(f"Saved results → {path}")


# ============================================================
if __name__ == "__main__":
    args = parse_args()
    main(args)
