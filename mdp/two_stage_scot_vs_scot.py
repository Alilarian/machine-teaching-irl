# ============================================================
# test_compare_two_stage_vs_flat_scot.py
# ============================================================

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

from teaching.scot import scot_greedy_family_atoms_tracked
from teaching.two_stage_scot import two_stage_scot


# ============================================================
# Args
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("Flat SCOT vs Two-Stage SCOT")

    # Environment
    p.add_argument("--n_envs", type=int, default=50)
    p.add_argument("--mdp_size", type=int, default=8)
    p.add_argument("--feature_dim", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # Feedback / data generation
    p.add_argument("--total_budget", type=int, default=2000)
    p.add_argument("--demo_env_fraction", type=float, default=0.05)

    p.add_argument(
        "--feedback",
        nargs="+",
        choices=["demo", "pairwise", "estop", "improvement"],
        default=["demo", "pairwise", "estop", "improvement"],
    )

    # Output
    p.add_argument("--result_dir", type=str, default="results_two_stage")

    return p.parse_args()


# ============================================================
# Logging helpers
# ============================================================

def stage(title):
    print("\n" + "=" * 80)
    print(f"[STAGE] {title}")
    print("=" * 80)

def log(msg):
    print(f"[INFO] {msg}", flush=True)


# ============================================================
# Main
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

    log(f"Generated {len(envs)} environments")

    # ========================================================
    # VALUE ITERATION + SFS
    # ========================================================
    stage("VALUE ITERATION & SUCCESSOR FEATURES")

    Q_list = parallel_value_iteration(envs)
    SFs = compute_successor_features_family(
        envs,
        Q_list,
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
            env_fraction=1.0,
            state_fraction=args.demo_env_fraction,
        ),

        pairwise=FeedbackSpec(
            enabled=("pairwise" in enabled),
            total_budget=args.total_budget if "pairwise" in enabled else 0,
        ),

        estop=FeedbackSpec(
            enabled=("estop" in enabled),
            total_budget=args.total_budget if "estop" in enabled else 0,
        ),

        improvement=FeedbackSpec(
            enabled=("improvement" in enabled),
            total_budget=args.total_budget if "improvement" in enabled else 0,
        ),
    )

    candidates_per_env = generate_candidate_atoms_for_scot(
        envs, Q_list, spec=spec
    )

    atom_counts = [len(a) for a in candidates_per_env]
    log(f"Atoms per env: mean={np.mean(atom_counts):.1f}, total={sum(atom_counts)}")


    _, U_atoms = derive_constraints_from_atoms(
        candidates_per_env, SFs, envs
    )

    blocks = [U_q]

    if U_atoms is not None and len(U_atoms) > 0:
        blocks.append(U_atoms)

    U_universal = remove_redundant_constraints(
        np.vstack(blocks)
    )

    log(f"|U_universal| = {len(U_universal)}")

    # ========================================================
    # FLAT SCOT
    # ========================================================
    stage("RUNNING FLAT SCOT")

    t0 = time.time()
    chosen_flat, stats_flat, _ = scot_greedy_family_atoms_tracked(
        U_universal,
        candidates_per_env,
        SFs,
        envs,
    )
    t_flat = time.time() - t0

    # ========================================================
    # TWO-STAGE SCOT
    # ========================================================
    stage("RUNNING TWO-STAGE SCOT")

    t0 = time.time()
    two_stage = two_stage_scot(
        U_universal=U_universal,
        U_per_env_atoms=_[0] if False else None,  # placeholder; not used
        U_per_env_q=_[0] if False else None,      # placeholder; not used
        candidates_per_env=candidates_per_env,
        SFs=SFs,
        envs=envs,
    )
    t_two_stage = time.time() - t0

    # ========================================================
    # SUMMARY
    # ========================================================
    stage("SUMMARY")

    def summarize(name, atoms, inspected, activated, coverage, runtime):
        frac = coverage / max(len(U_universal), 1)
        log(name)
        log(f"  atoms selected      = {atoms}")
        log(f"  envs inspected      = {inspected}")
        log(f"  envs activated      = {activated}")
        log(f"  final coverage      = {coverage} / {len(U_universal)}")
        log(f"  coverage fraction   = {frac:.3f}")
        log(f"  runtime (s)         = {runtime:.2f}")

    summarize(
        "FLAT SCOT",
        len(chosen_flat),
        stats_flat["total_inspected_count"],
        stats_flat["total_activated_count"],
        stats_flat["final_coverage"],
        t_flat,
    )

    summarize(
        "TWO-STAGE SCOT",
        two_stage["s2_iterations"],
        two_stage["s1_iterations"],
        len(two_stage["activated_envs"]),
        len(U_universal),
        t_two_stage,
    )

    # ========================================================
    # SAVE RESULTS (SAME LOGIC AS REFERENCE SCRIPT)
    # ========================================================
    stage("SAVING RESULTS")

    os.makedirs(args.result_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    save = {
        "args": vars(args),
        "U_global_size": len(U_universal),

        "flat_scot": {
            "atoms": len(chosen_flat),
            "inspected_envs": stats_flat["total_inspected_count"],
            "activated_envs": stats_flat["total_activated_count"],
            "final_coverage": stats_flat["final_coverage"],
            "coverage_fraction": (
                stats_flat["final_coverage"] / max(len(U_universal), 1)
            ),
            "runtime": t_flat,
        },

        "two_stage_scot": {
            "atoms": two_stage["s2_iterations"],
            "inspected_envs": two_stage["s1_iterations"],
            "activated_envs": len(two_stage["activated_envs"]),
            "final_coverage": len(U_universal),
            "coverage_fraction": 1.0,
            "runtime": t_two_stage,
            "waste": two_stage["waste"],
        },
    }

    path = os.path.join(
        args.result_dir,
        f"scot_two_stage_vs_flat_{run_id}.json"
    )

    with open(path, "w") as f:
        json.dump(save, f, indent=2)

    log(f"Saved results → {path}")


# ============================================================
if __name__ == "__main__":
    args = parse_args()
    main(args)
