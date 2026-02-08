# =============================================================================
# Two-Stage SCOT vs Random (GLOBAL POOL) — FULL EXPERIMENT
# =============================================================================

import argparse
import json
import os
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

# -----------------------------------------------------------------------------
# Imports (UNCHANGED CORE PIPELINE)
# -----------------------------------------------------------------------------
from utils import (
    generate_random_gridworld_envs,
    compute_successor_features_family,
    derive_constraints_from_q_family,
    derive_constraints_from_atoms,
    compute_Q_from_weights_with_VI,
    remove_redundant_constraints,
    parallel_value_iteration,
    recover_constraints_and_coverage,
    GenerationSpec,
    DemoSpec,
    FeedbackSpec,
)
from utils.successor_features import max_q_sa_pairs
from utils.common_helper import calculate_expected_value_difference
from utils.feedback_budgeting import generate_candidate_atoms_for_scot
from reward_learning.multi_env_atomic_birl import MultiEnvAtomicBIRL
from gridworld_env_layout import GridWorldMDPFromLayoutEnv

from teaching.two_stage_scot import two_stage_scot

# =============================================================================
# Ground-truth reward generator
# =============================================================================
def generate_w_true(d, seed=None):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    return w / np.linalg.norm(w)

# =============================================================================
# BIRL → Q helper
# =============================================================================
def _compute_Q_wrapper(args):
    env, w, vi_eps = args
    return compute_Q_from_weights_with_VI(env, w, vi_epsilon=vi_eps)

def birl_atomic_to_Q_lists(
    envs,
    atoms_flat,
    *,
    beta=10.0,
    samples=5000,
    stepsize=0.1,
    burn_frac=0.2,
    skip_rate=10,
    vi_epsilon=1e-6,
):
    birl = MultiEnvAtomicBIRL(
        envs,
        atoms_flat,
        beta_demo=beta,
        beta_pairwise=beta,
        beta_estop=beta,
        beta_improvement=beta,
        
    )

    birl.run_mcmc(samples=samples, stepsize=stepsize, adaptive=False, normalize=True)

    w_map = birl.get_map_solution()
    w_mean = birl.get_mean_solution(burn_frac=burn_frac, skip_rate=skip_rate)

    with ProcessPoolExecutor() as ex:
        Q_map = list(ex.map(_compute_Q_wrapper, [(e, w_map, vi_epsilon) for e in envs]))
        Q_mean = list(ex.map(_compute_Q_wrapper, [(e, w_mean, vi_epsilon) for e in envs]))

    return Q_map, Q_mean

# =============================================================================
# Regret
# =============================================================================
def regrets_from_Q(envs, Q_list, epsilon=1e-4):
    regrets = []
    for env, Q in zip(envs, Q_list):
        pi = max_q_sa_pairs(env, Q)
        r = calculate_expected_value_difference(
            env=env,
            eval_policy=pi,
            epsilon=epsilon,
            normalize_with_random_policy=False,
        )
        regrets.append(float(r))
    return np.asarray(regrets)

# =============================================================================
# RANDOM BASELINE — GLOBAL ATOM POOL
# =============================================================================
def sample_random_atoms_global_pool(
    candidates_per_env,
    n_to_pick,
    seed=None,
):
    rng = np.random.default_rng(seed)

    pool = [
        (env_idx, atom)
        for env_idx, atoms in enumerate(candidates_per_env)
        for atom in atoms
    ]

    idxs = rng.choice(len(pool), size=n_to_pick, replace=False)
    return [pool[i] for i in idxs]

def run_random_trials(
    envs,
    candidates_per_env,
    n_to_pick,
    seed,
    *,
    trials,
    birl_kwargs,
    SFs,
    U_universal,
):
    all_regrets = []
    mdp_counts = []
    constraint_counts = []
    coverages = []

    for sd in range(trials):
        chosen_rand = sample_random_atoms_global_pool(
            candidates_per_env,
            n_to_pick,
            seed=sd + seed,
        )

        # --- MDP count
        used_envs = {env_idx for env_idx, _ in chosen_rand}
        mdp_counts.append(len(used_envs))

        # --- Constraints + coverage
        n_c, cov = recover_constraints_and_coverage(
            chosen_rand,
            SFs,
            envs,
            U_universal,
        )
        constraint_counts.append(n_c)
        coverages.append(cov)

        # --- Regret
        Q_map, _ = birl_atomic_to_Q_lists(
            envs,
            chosen_rand,
            **birl_kwargs,
        )

        reg = regrets_from_Q(envs, Q_map)
        all_regrets.append(reg)

    return {
        "regrets": np.vstack(all_regrets),
        "mdp_counts": mdp_counts,
        "constraint_counts": constraint_counts,
        "coverages": coverages,
    }

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================
def run_experiment(
    *,
    n_envs=30,
    mdp_size=10,
    feature_dim=2,
    feedback_count=50,
    random_trials=10,
    seed=0,
    result_dir="results_two_stage",
    feedback=("demo", "pairwise", "estop", "improvement"),
    demo_env_fraction=1.0,
    total_budget=50,
    **birl_kwargs,
):
    os.makedirs(result_dir, exist_ok=True)

    print("\n================= EXPERIMENT START =================\n")

    # --------------------------------------------------
    # 1. True reward
    # --------------------------------------------------
    W_TRUE = generate_w_true(feature_dim, seed=seed)

    # --------------------------------------------------
    # 2. Environments
    # --------------------------------------------------
    color_to_feature_map = {
        f"f{i}": [1 if j == i else 0 for j in range(feature_dim)]
        for i in range(feature_dim)
    }

    envs, _ = generate_random_gridworld_envs(
        n_envs=n_envs,
        rows=mdp_size,
        cols=mdp_size,
        color_to_feature_map=color_to_feature_map,
        palette=list(color_to_feature_map.keys()),
        p_color_range={c: (0.3, 0.8) for c in color_to_feature_map},
        terminal_policy=dict(kind="random_k", k_min=1, k_max=1),
        gamma_range=(0.99, 0.99),
        noise_prob_range=(0.0, 0.0),
        w_mode="fixed",
        W_fixed=W_TRUE,
        seed=seed,
        GridEnvClass=GridWorldMDPFromLayoutEnv,
    )
    # --------------------------------------------------
    # 3. Optimal Q
    # --------------------------------------------------
    Q_list = parallel_value_iteration(envs, epsilon=1e-10)

    # --------------------------------------------------
    # 4. Successor features
    # --------------------------------------------------
    SFs = compute_successor_features_family(
        envs,
        Q_list,
        convention="entering",
        zero_terminal_features=True,
    )

    # --------------------------------------------------
    # 5. CONSTRAINT + ATOM GENERATION (SPEC-BASED)
    # --------------------------------------------------
    print("GENERATING CONSTRAINTS")

    # Q-based constraints
    U_per_env_q, U_q = derive_constraints_from_q_family(
        SFs,
        Q_list,
        envs,
        skip_terminals=True,
        normalize=True,
    )

    enabled = set(feedback)

    spec = GenerationSpec(
        seed=seed,

        demo=DemoSpec(
            enabled=("demo" in enabled),
            env_fraction=1.0,
            max_steps=1,
            state_fraction=demo_env_fraction,
            alloc_method="uniform",
        ),

        pairwise=FeedbackSpec(
            enabled=("pairwise" in enabled),
            total_budget=total_budget if "pairwise" in enabled else 0,
            alloc_method="uniform",
            # alloc_method="dirichlet",
            # alloc_params={"alpha": 0.3},    
        ),

        estop=FeedbackSpec(
            enabled=("estop" in enabled),
            total_budget=total_budget if "estop" in enabled else 0,
            alloc_method="uniform",
            # alloc_method="dirichlet",
            # alloc_params={"alpha": 0.3},
        ),

        improvement=FeedbackSpec(
            enabled=("improvement" in enabled),
            total_budget=total_budget if "improvement" in enabled else 0,
            alloc_method="uniform",
            # alloc_method="dirichlet",
            # alloc_params={"alpha": 0.3},
        ),
    )

    candidates_per_env = generate_candidate_atoms_for_scot(
        envs,
        Q_list,
        spec=spec,
    )

    atom_counts = [len(a) for a in candidates_per_env]
    print(
        f"Atoms per env: mean={np.mean(atom_counts):.1f},"
        f"total={sum(atom_counts)}"
    )

    U_per_env_atoms, U_atoms = derive_constraints_from_atoms(
        candidates_per_env,
        SFs,
        envs,
    )

    # --- Deduplicate Q-only constraints
    U_q_unique = remove_redundant_constraints(U_q)

    # --- Deduplicate Q + Atom constraints
    #if U_atoms is not None and len(U_atoms) > 0:
    U_union_unique = remove_redundant_constraints(
        np.vstack([U_q, U_atoms])
    )
    # else:
    #     U_union_unique = U_q_unique

    # --- Log diagnostic info
    print(f"|U_q| raw            = {len(U_q)}")
    print(f"|U_q| unique         = {len(U_q_unique)}")
    print(f"|U_atoms| raw        = {0 if U_atoms is None else len(U_atoms)}")
    print(f"|U_q ∪ U_atoms| uniq = {len(U_union_unique)}")
    print(f"Atom-implied uniques = {len(U_union_unique) - len(U_q_unique)}")

    # --- Use union as final universal set
    U_universal = U_union_unique

    print("universal constrainsts")
    for i in U_universal:
        print(i)
    
    # --------------------------------------------------
    # 6. TWO-STAGE SCOT
    # --------------------------------------------------
    out = two_stage_scot(
        U_universal=U_universal,
        U_per_env_atoms=U_per_env_atoms,
        #U_per_env_q=U_per_env_q,
        U_per_env_q=None,
        candidates_per_env=candidates_per_env,
        SFs=SFs,
        envs=envs,
    )

    chosen_two_stage = out["chosen"]
    print(f"TWO-STAGE selected {len(chosen_two_stage)} atoms")
   # print(chosen_two_stage)
   # --- TWO-STAGE constraint recovery
   #print()
    print(chosen_two_stage)
    ts_n_constraints, ts_coverage = recover_constraints_and_coverage(
        chosen_two_stage,
        SFs,
        envs,
        U_universal,
    )

    print(f"TWO-STAGE unique constraints recovered: {ts_n_constraints}")
    print(f"TWO-STAGE constraint coverage: {100*ts_coverage:.2f}%")

    # Number of unique environments actually used
    used_envs = sorted({env_idx for env_idx, _ in chosen_two_stage})
    num_used_envs = len(used_envs)

    print(f"TWO-STAGE used {num_used_envs}/{n_envs} environments")

    # for i in used_envs:
    #     envs[i].print_mdp_info()
    #     envs[i].print_optimal_policy()
        
    # --------------------------------------------------
    # 7. Regret — TWO-STAGE
    # --------------------------------------------------
    Q_ts, _ = birl_atomic_to_Q_lists(
        envs,
        chosen_two_stage,
        **birl_kwargs,
    )

    reg_ts = regrets_from_Q(envs, Q_ts)
    print(f"TWO-STAGE mean regret: {reg_ts.mean():.4f}")

    # --------------------------------------------------
    # 8. Regret — RANDOM
    # --------------------------------------------------
    rand_out = run_random_trials(
        envs,
        candidates_per_env,
        n_to_pick=len(chosen_two_stage),
        seed=args.seed,
        trials=random_trials,
        birl_kwargs=birl_kwargs,
        SFs=SFs,
        U_universal=U_universal,
    )

    reg_rand = rand_out["regrets"]

    print(f"RANDOM mean regret: {reg_rand.mean():.4f}")
    print(f"RANDOM mean regret: {reg_rand.mean():.4f}")
    print(f"RANDOM mean #MDPs selected: {np.mean(rand_out['mdp_counts']):.2f}")
    print(f"RANDOM mean unique constraints: {np.mean(rand_out['constraint_counts']):.2f}")
    print(f"RANDOM mean constraint coverage: {100*np.mean(rand_out['coverages']):.2f}%")

    # --------------------------------------------------
    # 9. Save
    # --------------------------------------------------
    results = {
        # --------------------
        # Core results
        # --------------------
        "two_stage_regret": reg_ts.tolist(),
        "random_regret": reg_rand.tolist(),
        "two_stage_mean": float(reg_ts.mean()),
        "random_mean": float(reg_rand.mean()),

        # --------------------
        # SCOT statistics
        # --------------------
        "num_atoms_selected": len(chosen_two_stage),
        "num_envs_used": num_used_envs,
        "used_envs": used_envs,

        # --------------------
        # Constraint statistics
        # --------------------
        "U_q_raw": len(U_q),
        "U_q_unique": len(U_q_unique),
        "U_atoms_raw": 0 if U_atoms is None else len(U_atoms),
        "U_union_unique": len(U_universal),
        "atom_implied_unique": len(U_universal) - len(U_q_unique),

        # --------------------
        # Experiment config (reproducibility)
        # --------------------
        "config": {
            "seed": seed,
            "n_envs": n_envs,
            "mdp_size": mdp_size,
            "feature_dim": feature_dim,
            "feedback": list(feedback),
            "demo_env_fraction": demo_env_fraction,
            "total_budget": total_budget,
            "random_trials": random_trials,
            "birl": {
                "beta": birl_kwargs["beta"],
                "samples": birl_kwargs["samples"],
                "stepsize": birl_kwargs["stepsize"],
            },
        },
        # --------------------
        # Constraint recovery
        # --------------------
        "two_stage_constraints": {
            "unique_constraints": ts_n_constraints,
            "coverage": ts_coverage,
        },

        "random_constraints": {
            "mdp_counts": rand_out["mdp_counts"],
            "constraint_counts": rand_out["constraint_counts"],
            "coverages": rand_out["coverages"],
            "mean_mdp_count": float(np.mean(rand_out["mdp_counts"])),
            "mean_unique_constraints": float(np.mean(rand_out["constraint_counts"])),
            "mean_coverage": float(np.mean(rand_out["coverages"])),
        },
    }

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    exp_name = (
        f"two_stage_vs_random_"
        f"env{n_envs}_"
        f"size{mdp_size}_"
        f"fd{feature_dim}_"
        f"budget{total_budget}_"
        f"seed{seed}_"
        f"{timestamp}.json"
    )

    out_path = os.path.join(result_dir, exp_name)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {out_path}")
    print("\n================= EXPERIMENT END =================\n")

# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=30)
    parser.add_argument("--mdp_size", type=int, default=10)
    parser.add_argument("--feature_dim", type=int, default=2)

    parser.add_argument("--feedback", nargs="+",
                        default=["demo", "pairwise", "estop", "improvement"])
    parser.add_argument("--demo_env_fraction", type=float, default=1.0)
    parser.add_argument("--total_budget", type=int, default=50)

    parser.add_argument("--random_trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result_dir", type=str, default="results_two_stage")

    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--stepsize", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=10.0)

    args = parser.parse_args()

    birl_kwargs = dict(
        beta=args.beta,
        samples=args.samples,
        stepsize=args.stepsize,
    )

    run_experiment(
        n_envs=args.n_envs,
        mdp_size=args.mdp_size,
        feature_dim=args.feature_dim,
        random_trials=args.random_trials,
        seed=args.seed,
        result_dir=args.result_dir,
        feedback=args.feedback,
        demo_env_fraction=args.demo_env_fraction,
        total_budget=args.total_budget,
        **birl_kwargs,
    )