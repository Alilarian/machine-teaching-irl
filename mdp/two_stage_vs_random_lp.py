# =============================================================================
# Two-Stage SCOT vs Random (GLOBAL POOL) — FULL EXPERIMENT with LP reward learning
# =============================================================================
import argparse
import json
import os
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# ─── LP solver ───────────────────────────────────────────────────────────────
import pulp

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
from gridworld_env_layout import GridWorldMDPFromLayoutEnv
from teaching.two_stage_scot import two_stage_scot
from teaching.scot import scot_greedy_family_atoms_tracked

# =============================================================================
# Ground-truth reward generator
# =============================================================================
def generate_w_true(d, seed=None):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    return w / np.linalg.norm(w)

# =============================================================================
# Q computation helper (unchanged)
# =============================================================================
def _compute_Q_wrapper(args):
    env, w, vi_eps = args
    return compute_Q_from_weights_with_VI(env, w, vi_epsilon=vi_eps)

# =============================================================================
# LP-based reward inference (replaces BIRL)
# =============================================================================
def lp_atomic_to_Q_lists(
    envs,
    atoms_flat,          # list of (env_idx, atom)
    SFs,                 # successor features — required for constraint derivation
    epsilon=1e-2,        # minimum margin
    vi_epsilon=1e-6,
):
    # Group atoms back per environment (needed by derive_constraints_from_atoms)
    atoms_per_env = [[] for _ in envs]
    for env_idx, atom in atoms_flat:
        atoms_per_env[env_idx].append(atom)

    # Derive constraint matrix from selected atoms
    U_per_env_atoms, U_atoms = derive_constraints_from_atoms(
        atoms_per_env,
        SFs,
        envs,
    )

    if U_atoms is None or len(U_atoms) == 0:
        print("Warning: No constraints from selected atoms → using zero reward")
        w_sol = np.zeros(len(envs[0].feature_map))  # assume feature dim from env
    else:
        U = remove_redundant_constraints(U_atoms)
        U = np.asarray(U, dtype=float)           
        d = U.shape[1]
        n = U.shape[0]

        prob = pulp.LpProblem("MaxMarginRewardLP", pulp.LpMaximize)

        # Reward weights
        w = [pulp.LpVariable(f"w_{j}") for j in range(d)]

        # Objective: maximize sum of margins
        margins = [pulp.lpSum(U[i, j] * w[j] for j in range(d)) for i in range(n)]
        prob += pulp.lpSum(margins)

        # Hard margin constraints
        for m in margins:
            prob += m >= epsilon

        # L1 normalization: ∑ |w_j| = 1
        abs_w = [pulp.LpVariable(f"abs_w_{j}", lowBound=0) for j in range(d)]
        for j in range(d):
            prob += abs_w[j] >= w[j]
            prob += abs_w[j] >= -w[j]
        prob += pulp.lpSum(abs_w) == 1

        # Solve
        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[status] != "Optimal":
            print(f"LP not optimal ({pulp.LpStatus[status]}) → using zero vector")
            w_sol = np.zeros(d)
        else:
            w_sol = np.array([pulp.value(wj) for wj in w])

    # Compute Q for all environments
    with ProcessPoolExecutor() as ex:
        Q_list = list(ex.map(_compute_Q_wrapper, [(e, w_sol, vi_epsilon) for e in envs]))

    return Q_list, None  # keep signature compatible (no mean solution)

# =============================================================================
# Regret computation (unchanged)
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


def _random_trial_worker(args):
    (
        trial_id,
        envs,
        candidates_per_env,
        n_to_pick,
        seed,
        SFs,
        U_universal,
    ) = args

    chosen_rand = sample_random_atoms_global_pool(
        candidates_per_env,
        n_to_pick,
        seed=seed + trial_id,
    )
    used_envs = {env_idx for env_idx, _ in chosen_rand}
    n_c, cov = recover_constraints_and_coverage(
        chosen_rand,
        SFs,
        envs,
        U_universal,
    )
    Q_map, _ = lp_atomic_to_Q_lists(
        envs,
        chosen_rand,
        SFs=SFs,
        epsilon=1e-3,
    )
    reg = regrets_from_Q(envs, Q_map)
    return {
        "regret": reg,
        "mdp_count": len(used_envs),
        "constraint_count": n_c,
        "coverage": cov,
    }


def run_random_trials(
    envs,
    candidates_per_env,
    n_to_pick,
    seed,
    *,
    trials,
    SFs,
    U_universal,
    max_workers=None,
):
    args = [
        (t, envs, candidates_per_env, n_to_pick, seed, SFs, U_universal)
        for t in range(trials)
    ]
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_random_trial_worker, args))
    return {
        "regrets": np.vstack([r["regret"] for r in results]),
        "mdp_counts": [r["mdp_count"] for r in results],
        "constraint_counts": [r["constraint_count"] for r in results],
        "coverages": [r["coverage"] for r in results],
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
    alloc_method="uniform",
    alloc=None,
):
    os.makedirs(result_dir, exist_ok=True)
    print("\n================= EXPERIMENT START =================\n")

    # 1. True reward
    W_TRUE = generate_w_true(feature_dim, seed=seed)

    # 2. Environments
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

    # 3. Optimal Q
    Q_list = parallel_value_iteration(envs, epsilon=1e-10)

    # 4. Successor features
    SFs = compute_successor_features_family(
        envs,
        Q_list,
        convention="entering",
        zero_terminal_features=True,
    )

    # 5. CONSTRAINT + ATOM GENERATION
    print("GENERATING CONSTRAINTS")
    enabled = set(feedback)
    spec = GenerationSpec(
        seed=seed,
        base_max_horizon=200,
        demo=DemoSpec(
            enabled=("demo" in enabled),
            env_fraction=1.0,
            max_steps=1,
            state_fraction=demo_env_fraction,
        ),
        pairwise=FeedbackSpec(
            enabled=("pairwise" in enabled),
            total_budget=total_budget if "pairwise" in enabled else 0,
            alloc_method=alloc_method,
            alloc_params=None if alloc_method == "uniform" else {"alpha": alloc},
        ),
        estop=FeedbackSpec(
            enabled=("estop" in enabled),
            total_budget=total_budget if "estop" in enabled else 0,
            alloc_method=alloc_method,
            alloc_params=None if alloc_method == "uniform" else {"alpha": alloc},
        ),
        improvement=FeedbackSpec(
            enabled=("improvement" in enabled),
            total_budget=total_budget if "improvement" in enabled else 0,
            alloc_method=alloc_method,
            alloc_params=None if alloc_method == "uniform" else {"alpha": alloc},
        ),
    )

    candidates_per_env = generate_candidate_atoms_for_scot(
        envs,
        Q_list,
        spec=spec,
    )

    atom_counts = [len(a) for a in candidates_per_env]
    print(f"Atoms per env: mean={np.mean(atom_counts):.1f}, total={sum(atom_counts)}")

    U_per_env_atoms, U_atoms = derive_constraints_from_atoms(
        candidates_per_env,
        SFs,
        envs,
    )

    U_union_unique = remove_redundant_constraints(U_atoms)
    U_universal = U_union_unique

    print(f"|U_atoms| raw = {0 if U_atoms is None else len(U_atoms)}")
    print(f"|U_union_unique| = {len(U_universal)}")

    # 6. TWO-STAGE SCOT
    out = two_stage_scot(
        U_universal=U_universal,
        U_per_env_atoms=U_per_env_atoms,
        candidates_per_env=candidates_per_env,
        SFs=SFs,
        envs=envs,
    )
    chosen_two_stage = out["chosen"]
    print(f"TWO-STAGE selected {len(chosen_two_stage)} atoms")

    ts_n_constraints, ts_coverage = recover_constraints_and_coverage(
        chosen_two_stage,
        SFs,
        envs,
        U_universal,
    )
    print(f"TWO-STAGE unique constraints: {ts_n_constraints}")
    print(f"TWO-STAGE coverage: {100*ts_coverage:.2f}%")

    used_envs = sorted({env_idx for env_idx, _ in chosen_two_stage})
    num_used_envs = len(used_envs)
    print(f"TWO-STAGE used {num_used_envs}/{n_envs} environments")

    # 7. Regret — TWO-STAGE
    Q_ts, _ = lp_atomic_to_Q_lists(
        envs,
        chosen_two_stage,
        SFs=SFs,
        epsilon=1e-3,
    )
    reg_ts = regrets_from_Q(envs, Q_ts)
    print(f"TWO-STAGE mean regret: {reg_ts.mean():.4f}")

    # 8. Regret — RANDOM
    rand_out = run_random_trials(
        envs=envs,
        candidates_per_env=candidates_per_env,
        n_to_pick=len(chosen_two_stage),
        seed=seed,
        trials=random_trials,
        SFs=SFs,
        U_universal=U_universal,
    )
    reg_rand = rand_out["regrets"]
    print(f"RANDOM mean regret: {reg_rand.mean():.4f}")
    print(f"RANDOM mean #MDPs: {np.mean(rand_out['mdp_counts']):.2f}")
    print(f"RANDOM mean unique constraints: {np.mean(rand_out['constraint_counts']):.2f}")
    print(f"RANDOM mean coverage: {100*np.mean(rand_out['coverages']):.2f}%")

    # 9. Save results
    results = {
        "methods": {
            "two_stage": {
                "regret": reg_ts.tolist(),
                "mean_regret": float(reg_ts.mean()),
                "selection_stats": {
                    "num_atoms_selected": len(chosen_two_stage),
                    "num_envs_used": num_used_envs,
                    "used_envs": used_envs,
                },
                "constraint_stats": {
                    "unique_constraints": ts_n_constraints,
                    "coverage": ts_coverage,
                },
            },
            "random": {
                "regret": reg_rand.tolist(),
                "mean_regret": float(reg_rand.mean()),
                "selection_stats": {
                    "mdp_counts": rand_out["mdp_counts"],
                    "mean_mdp_count": float(np.mean(rand_out["mdp_counts"])),
                },
                "constraint_stats": {
                    "constraint_counts": rand_out["constraint_counts"],
                    "coverages": rand_out["coverages"],
                    "mean_unique_constraints": float(np.mean(rand_out["constraint_counts"])),
                    "mean_coverage": float(np.mean(rand_out["coverages"])),
                },
            },
        },
        "universal_constraints": {
            "U_atoms_raw": 0 if U_atoms is None else len(U_atoms),
            "U_union_unique": len(U_universal),
        },
        "config": {
            "seed": seed,
            "n_envs": n_envs,
            "mdp_size": mdp_size,
            "feature_dim": feature_dim,
            "feedback": list(feedback),
            "demo_env_fraction": demo_env_fraction,
            "total_budget": total_budget,
            "random_trials": random_trials,
            "lp": {
                "epsilon": 1e-3,
            },
        },
    }

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_name = (
        f"two_stage_vs_random_"
        f"env{n_envs}_size{mdp_size}_fd{feature_dim}_"
        f"budget{total_budget}_seed{seed}_{timestamp}.json"
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
    parser = argparse.ArgumentParser(description="Two-Stage SCOT vs Random — LP reward learning")
    parser.add_argument("--n_envs", type=int, default=3)
    parser.add_argument("--mdp_size", type=int, default=6)
    parser.add_argument("--feature_dim", type=int, default=4)
    parser.add_argument("--feedback", nargs="+",
                        default=["demo", "pairwise", "estop", "improvement"])
    parser.add_argument("--demo_env_fraction", type=float, default=1.0)
    parser.add_argument("--total_budget", type=int, default=50)
    parser.add_argument("--random_trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result_dir", type=str, default="results_two_stage")
    parser.add_argument(
        "--alloc_method",
        type=str,
        default="uniform",
        choices=["uniform", "dirichlet"],
    )
    parser.add_argument("--alloc", type=float, default=None)

    args = parser.parse_args()
    if args.alloc_method != "uniform" and args.alloc is None:
        args.alloc = 0.5

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
        alloc_method=args.alloc_method,
        alloc=args.alloc,
    )