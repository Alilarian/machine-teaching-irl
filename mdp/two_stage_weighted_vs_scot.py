import argparse
import json
import os
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================
# Project path setup
# ============================================================
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

# ============================================================
# Imports from your repo
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
    GenerationSpec,
    DemoSpec,
    FeedbackSpec,
)
# Import both versions (assume scot_naive.py and scot_heuristic.py exist in teaching/)
from teaching.scot import scot_greedy_family_atoms_tracked as scot_naive
from teaching.scot_heuristic import scot_greedy_family_atoms_tracked as scot_heuristic

# ============================================================
# Argument parsing
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Compare Naive vs Heuristic SCOT with Budgeted Generation")
    # Environment params
    parser.add_argument("--n_envs", type=int, default=50, help="Number of MDPs")
    parser.add_argument("--mdp_size", type=int, default=8, help="Grid size (NxN)")
    parser.add_argument("--feature_dim", type=int, default=4)
    # Feedback params
    parser.add_argument("--feedback", nargs="+", default=["demo", "pairwise", "estop", "improvement"])
    parser.add_argument("--total_budget", type=int, default=2000, help="Total budget for feedback atoms")
    parser.add_argument("--demo_env_fraction", type=float, default=0.05, help="Fraction of envs for demos")
    parser.add_argument("--pairwise_alloc_method", type=str, default="dirichlet", help="Allocation for pairwise")
    parser.add_argument("--pairwise_alpha", type=float, default=0.2, help="Dirichlet alpha for pairwise")
    parser.add_argument("--estop_alloc_method", type=str, default="sparse_poisson", help="Allocation for estop")
    parser.add_argument("--estop_p_active", type=float, default=0.05, help="Poisson p_active for estop")
    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--result_dir", type=str, default="results_comparison")
    # Heuristic params
    parser.add_argument("--sample_size", type=int, default=10, help="Heuristic sample size")
    parser.add_argument("--initial_k", type=int, default=10)
    parser.add_argument("--increment", type=int, default=5)
    return parser.parse_args()

# ============================================================
# Utilities
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
def run_comparison(args):
    stage("EXPERIMENT SETUP")
    log(f"Config: {args.n_envs} envs, {args.mdp_size}x{args.mdp_size} grid, "
        f"d={args.feature_dim}, feedback={args.feedback}, total_budget={args.total_budget}, seed={args.seed}")

    rng = np.random.default_rng(args.seed)

    # Ground-truth reward
    W_TRUE = rng.normal(size=args.feature_dim)
    W_TRUE /= np.linalg.norm(W_TRUE)
    log(f"W_TRUE norm = {np.linalg.norm(W_TRUE):.3f}")

    # Generate environments
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
        palette=list(color_to_feature_map.keys()),
        p_color_range={c: (0.3, 0.7) for c in color_to_feature_map},
        terminal_policy=dict(kind="random_k", k_min=1, k_max=1, p_no_terminal=0.0),
        gamma_range=(0.99, 0.99),
        noise_prob_range=(0.2, 0.5),
        w_mode="fixed",
        W_fixed=W_TRUE,
        seed=args.seed,
        GridEnvClass=GridWorldMDPFromLayoutEnv,
    )
    log(f"Generated {len(envs)} gridworlds")

    # Value iteration + successor features
    stage("VALUE ITERATION & SUCCESSOR FEATURES")
    Q_list = parallel_value_iteration(envs, epsilon=1e-10)
    SFs = compute_successor_features_family(
        envs, Q_list, convention="entering", zero_terminal_features=True, tol=1e-10
    )

    # Universal constraints from Q-values
    stage("Q-BASED CONSTRAINTS")
    _, U_q_global = derive_constraints_from_q_family(SFs, Q_list, envs, skip_terminals=True, normalize=True)
    log(f"|U_q_global| = {len(U_q_global)}")

    # Budgeted Sparse Candidate Generation
    stage("CANDIDATE ATOM GENERATION")
    spec = GenerationSpec(
        seed=args.seed,
        demo=DemoSpec(enabled="demo" in args.feedback, env_fraction=args.demo_env_fraction),
        pairwise=FeedbackSpec(
            enabled="pairwise" in args.feedback,
            total_budget=args.total_budget,
            alloc_method=args.pairwise_alloc_method,
            alloc_params={"alpha": args.pairwise_alpha}
        ),
        estop=FeedbackSpec(
            enabled="estop" in args.feedback,
            total_budget=args.total_budget // 4,
            alloc_method=args.estop_alloc_method,
            alloc_params={"p_active": args.estop_p_active}
        ),
        improvement=FeedbackSpec(
            enabled="improvement" in args.feedback,
            total_budget=args.total_budget // 4,
            alloc_method="dirichlet",  # Using same as pairwise for simplicity
            alloc_params={"alpha": 0.2}
        )
    )
    candidates_per_env = generate_candidate_atoms_for_scot(envs, Q_list, spec=spec)
    atom_counts = [len(c) for c in candidates_per_env]
    log(f"Atoms per env: mean={np.mean(atom_counts):.1f}, total={sum(atom_counts)}")

    # Atom-based constraints + universal set
    stage("ATOM CONSTRAINTS & UNIVERSAL SET")
    _, U_atoms_global = derive_constraints_from_atoms(candidates_per_env, SFs, envs)
    U_universal = remove_redundant_constraints(np.vstack([U_q_global, U_atoms_global]), epsilon=1e-4)
    log(f"|U_universal| = {len(U_universal)} (target coverage)")

    # Run both algorithms
    results = {}

    # --- Naive SCOT ---
    stage("RUNNING NAIVE SCOT")
    start = time.time()
    chosen_naive, stats_naive, _ = scot_naive(
        U_universal, candidates_per_env, SFs, envs
    )
    results["naive"] = {
        "chosen": [(i, str(atom)) for i, atom in chosen_naive],  # Serialize atoms if needed
        "stats": stats_naive,
        "runtime": time.time() - start,
        "solution_size": len(chosen_naive),
        "activated_envs": len(stats_naive["activated_env_indices"]),
        "inspected_envs": stats_naive["total_inspected_count"],
        "full_coverage": stats_naive["final_coverage"] == len(U_universal),
    }
    log(f"Naive: {results['naive']['solution_size']} atoms, "
        f"{results['naive']['activated_envs']} envs activated, "
        f"{results['naive']['inspected_envs']} inspected, "
        f"runtime={results['naive']['runtime']:.2f}s")

    # --- Heuristic SCOT ---
    stage("RUNNING HEURISTIC SCOT")
    start = time.time()
    chosen_heuristic, stats_heuristic, _ = scot_heuristic(
        U_universal,
        candidates_per_env,
        SFs,
        envs,
        sample_size=args.sample_size,
        initial_k=args.initial_k,
        increment=args.increment,
    )
    results["heuristic"] = {
        "chosen": [(i, str(atom)) for i, atom in chosen_heuristic],  # Serialize
        "stats": stats_heuristic,
        "runtime": time.time() - start,
        "solution_size": len(chosen_heuristic),
        "activated_envs": len(stats_heuristic["activated_env_indices"]),
        "inspected_envs": stats_heuristic["total_inspected_count"],
        "full_coverage": stats_heuristic["final_coverage"] == len(U_universal),
    }
    log(f"Heuristic: {results['heuristic']['solution_size']} atoms, "
        f"{results['heuristic']['activated_envs']} envs activated, "
        f"{results['heuristic']['inspected_envs']} inspected, "
        f"runtime={results['heuristic']['runtime']:.2f}s")

    # Save results
    stage("SAVING RESULTS")
    os.makedirs(args.result_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    feedback_tag = "_".join(sorted(args.feedback))
    json_path = os.path.join(args.result_dir, f"comparison_{feedback_tag}_{run_id}.json")
    plot_path = os.path.join(args.result_dir, f"comparison_{feedback_tag}_{run_id}.png")

    save_data = {
        "args": vars(args),
        "W_TRUE": W_TRUE.tolist(),
        "n_envs": args.n_envs,
        "U_universal_size": len(U_universal),
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2)
    log(f"Saved JSON: {json_path}")

    # Visualize
    stage("GENERATING PLOTS")
    visualize_comparison(results, plot_path)
    log(f"Saved plot: {plot_path}")

    stage("COMPARISON SUMMARY")
    n = results["naive"]
    h = results["heuristic"]
    print(f"\n{'Metric':<25} {'Naive':<15} {'Heuristic':<15} {'Improvement'}")
    print("-" * 65)
    print(f"{'Runtime (s)':<25} {n['runtime']:<15.2f} {h['runtime']:<15.2f} "
          f"{n['runtime']/h['runtime']:.2f}x faster")
    print(f"{'Inspected Envs':<25} {n['inspected_envs']:<15} {h['inspected_envs']:<15} "
          f"{(1 - h['inspected_envs']/n['inspected_envs'])*100:.1f}% reduction")
    print(f"{'Activated Envs':<25} {n['activated_envs']:<15} {h['activated_envs']:<15} "
          f"{(1 - h['activated_envs']/n['activated_envs'])*100:.1f}% reduction")
    print(f"{'Solution Size':<25} {n['solution_size']:<15} {h['solution_size']:<15} "
          f"{h['solution_size']/n['solution_size']:.3f}x")
    print(f"{'Full Coverage':<25} {n['full_coverage']:<15} {h['full_coverage']:<15}")

    stage("EXPERIMENT COMPLETE")

def visualize_comparison(results, save_path):
    n = results["naive"]["stats"]
    h = results["heuristic"]["stats"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Runtime breakdown
    axs[0,0].bar(['Naive', 'Heuristic'],
                 [n["total_precompute_time"] + n["total_greedy_time"],
                  h["total_precompute_time"] + h["total_greedy_time"]])
    axs[0,0].set_title("Total Runtime (precompute + greedy)")
    axs[0,0].set_ylabel("Time (s)")

    # 2. Inspected & Activated
    labels = ['Inspected', 'Activated']
    naive_vals = [n["total_inspected_count"], n["total_activated_count"]]
    heur_vals = [h["total_inspected_count"], h["total_activated_count"]]
    x = np.arange(len(labels))
    width = 0.35
    axs[0,1].bar(x - width/2, naive_vals, width, label='Naive')
    axs[0,1].bar(x + width/2, heur_vals, width, label='Heuristic')
    axs[0,1].set_xticks(x)
    axs[0,1].set_xticklabels(labels)
    axs[0,1].set_title("Environments Inspected vs Activated")
    axs[0,1].legend()

    # 3. Solution size
    axs[1,0].bar(['Naive', 'Heuristic'],
                 [results["naive"]["solution_size"], results["heuristic"]["solution_size"]])
    axs[1,0].set_title("Number of Selected Feedback Atoms")
    axs[1,0].set_ylabel("Count")

    # 4. Cumulative coverage curve
    naive_cum = np.cumsum(n["coverage_counts"])
    heur_cum = np.cumsum(h["coverage_counts"])
    axs[1,1].plot(naive_cum, label='Naive', marker='o')
    axs[1,1].plot(heur_cum, label='Heuristic', marker='s')
    axs[1,1].set_xlabel("Atoms Selected")
    axs[1,1].set_ylabel("Cumulative Unique Constraints Covered")
    axs[1,1].set_title("Coverage Efficiency")
    axs[1,1].legend()
    axs[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    run_comparison(args)
    
    # 4. Universal Constraints
    U_per_env_atoms, U_atoms_global = derive_constraints_from_atoms(candidates_per_env, SFs, envs)
    U_universal = remove_redundant_constraints(np.vstack([U_q_global, U_atoms_global]))
    log(f"Universal Constraint Set Size: {len(U_universal)}")

    # 5. Algorithm 1: Weighted Two-Stage SCOT
    log("Running Weighted Two-Stage...")
    res_2s = two_stage_scot_weighted(
        U_universal=U_universal, U_per_env_atoms=U_per_env_atoms,
        U_per_env_q=U_per_env_q, candidates_per_env=candidates_per_env,
        SFs=SFs, envs=envs
    )

    # 6. Algorithm 2: Flat SCOT (The Naive Approach)
    log("Running Flat SCOT...")
    # This call now returns tracking stats including iterations and deep evaluations
    stats_flat = scot_greedy_family_atoms_tracked(U_universal, candidates_per_env, SFs, envs)
    
    # 7. Final Metrics & Comparison
    # We calculate total "Deep Effort" = atoms evaluated across all iterations
    # In Two-Stage, Stage 2 only looked at selected_mdps atoms
    # In Flat, it looked at all MDP atoms.
    
    log("\n" + "="*60)
    log(f"RESULTS SUMMARY (Seed: {args.seed})")
    log("-" * 60)
    log(f"{'METRIC':<25} | {'TWO-STAGE':<12} | {'FLAT (NAIVE)':<12}")
    log("-" * 60)
    log(f"{'Inspected Envs (Deep)':<25} | {res_2s['inspected_count']:<12} | {stats_flat['total_inspected_count']:<12}")
    log(f"{'Selected (Final) Envs':<25} | {res_2s['selection_count']:<12} | {len(stats_flat['activated_env_indices']):<12}")
    
    # These are the new "Process" metrics
    log(f"{'Teaching Size (Atoms)':<25} | {len(res_2s['chosen_atoms']):<12} | {len(stats_flat['activated_env_indices']):<12}")
    # Note: s2_deep_checks should be added to your two_stage_scot_weighted return
    # If not yet there, it's len(pool_atoms) * iterations_in_s2
    log(f"{'Inspection Waste':<25} | {res_2s['waste']:<12} | {stats_flat['total_inspected_count'] - len(stats_flat['activated_env_indices']):<12}")
    log("-" * 60)
    
    log(f"Efficiency Gain: {stats_flat['total_inspected_count'] / max(1, res_2s['inspected_count']):.2f}x fewer deep inspections.")
    log("="*60)

    # 8. Save results
    os.makedirs(args.result_dir, exist_ok=True)
    out_path = os.path.join(args.result_dir, f"weighted_results_{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump({
            "two_stage": res_2s,
            "flat": {
                "inspected": stats_flat["total_inspected_count"],
                "activated_envs": stats_flat["activated_env_indices"],
                "total_atoms": len(stats_flat["activated_env_indices"])
            }
        }, f, indent=2, default=str)

if __name__ == "__main__":
    run_experiment(parse_args())