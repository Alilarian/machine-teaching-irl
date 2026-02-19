# """
# Two-Stage SCOT vs Random — LP reward learning
# with multiple feedback generations (non-demo only)
# """

# import argparse
# import json
# import os
# import sys
# import time
# import numpy as np
# from concurrent.futures import ProcessPoolExecutor
# import pulp

# # Path setup
# module_path = os.path.abspath(os.path.join(".."))
# if module_path not in sys.path:
#     sys.path.append(module_path)

# # Imports
# from utils import (
#     generate_random_gridworld_envs,
#     compute_successor_features_family,
#     derive_constraints_from_atoms,
#     compute_Q_from_weights_with_VI,
#     remove_redundant_constraints,
#     parallel_value_iteration,
#     recover_constraints_and_coverage,
#     GenerationSpec,
#     DemoSpec,
#     FeedbackSpec,
# )
# from utils.successor_features import max_q_sa_pairs
# from utils.common_helper import calculate_expected_value_difference
# from utils.feedback_budgeting import generate_candidate_atoms_for_scot
# from gridworld_env_layout import GridWorldMDPFromLayoutEnv
# from teaching.two_stage_scot import two_stage_scot

# # Ground-truth reward
# def generate_w_true(d, seed=None):
#     rng = np.random.default_rng(seed)
#     w = rng.normal(size=d)
#     return w / np.linalg.norm(w)

# # Q wrapper
# def _compute_Q_wrapper(args):
#     env, w, vi_eps = args
#     return compute_Q_from_weights_with_VI(env, w, vi_epsilon=vi_eps)

# # Simple fixed LP reward learning (no tuning)
# def lp_atomic_to_Q_lists(
#     envs,
#     atoms_flat,
#     SFs,
#     epsilon=1e-3,
#     vi_epsilon=1e-6,
# ):
#     atoms_per_env = [[] for _ in envs]
#     for env_idx, atom in atoms_flat:
#         atoms_per_env[env_idx].append(atom)

#     U_per_env_atoms, U_atoms = derive_constraints_from_atoms(atoms_per_env, SFs, envs)

#     if U_atoms is None or len(U_atoms) == 0:
#         print("Warning: No constraints → zero reward")
#         w_sol = np.zeros(len(envs[0].feature_map))
#     else:
#         unique = []
#         for v in U_atoms:
#             v = np.asarray(v)
#             v_norm = np.linalg.norm(v)
#             if v_norm == 0:
#                 continue
#             is_close = any(
#                 np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u)) > 1 - 1e-3
#                 for u in unique
#             )
#             if not is_close:
#                 unique.append(v)

#         U = remove_redundant_constraints(unique)
#         U = np.asarray(U, dtype=float)

#         if len(U) == 0:
#             print("No constraints after cleaning → zero reward")
#             w_sol = np.zeros(len(envs[0].feature_map))
#         else:
#             n, d = U.shape
#             print(f" → {n} constraints | {d} dimensions")

#             prob = pulp.LpProblem("MaxMarginRewardLP", pulp.LpMaximize)
#             w_vars = [pulp.LpVariable(f"w_{j}") for j in range(d)]
#             abs_w = [pulp.LpVariable(f"abs_w_{j}", lowBound=0) for j in range(d)]

#             for j in range(d):
#                 prob += abs_w[j] >= w_vars[j]
#                 prob += abs_w[j] >= -w_vars[j]
#             prob += pulp.lpSum(abs_w) == 1

#             margins = [pulp.lpSum(U[i,j] * w_vars[j] for j in range(d)) for i in range(n)]
#             prob += pulp.lpSum(margins)
#             for m in margins:
#                 prob += m >= epsilon

#             status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
#             print(f"LP status: {pulp.LpStatus[status]}")

#             if pulp.LpStatus[status] != "Optimal":
#                 print(f"LP not optimal → zero vector")
#                 w_sol = np.zeros(d)
#             else:
#                 w_sol = np.array([pulp.value(wj) for wj in w_vars])

#     # Compute Q
#     with ProcessPoolExecutor() as ex:
#         Q_list = list(ex.map(_compute_Q_wrapper, [(e, w_sol, vi_epsilon) for e in envs]))

#     return Q_list, None

# # Regret
# def regrets_from_Q(envs, Q_list, epsilon=1e-4):
#     regrets = []
#     for env, Q in zip(envs, Q_list):
#         pi = max_q_sa_pairs(env, Q)
#         r = calculate_expected_value_difference(
#             env=env,
#             eval_policy=pi,
#             epsilon=epsilon,
#             normalize_with_random_policy=False,
#         )
#         regrets.append(float(r))
#     return np.asarray(regrets)

# # Random baseline helpers (unchanged)
# def sample_random_atoms_global_pool(candidates_per_env, n_to_pick, seed=None):
#     rng = np.random.default_rng(seed)
#     pool = [(i, a) for i, atoms in enumerate(candidates_per_env) for a in atoms]
#     idxs = rng.choice(len(pool), size=n_to_pick, replace=False)
#     return [pool[i] for i in idxs]

# def _random_trial_worker(args):
#     trial_id, envs, candidates_per_env, n_to_pick, seed, SFs, U_universal = args
#     chosen = sample_random_atoms_global_pool(candidates_per_env, n_to_pick, seed + trial_id)
#     used = {env_idx for env_idx, _ in chosen}
#     n_c, cov = recover_constraints_and_coverage(chosen, SFs, envs, U_universal)
#     Q_map, _ = lp_atomic_to_Q_lists(envs, chosen, SFs=SFs, epsilon=1e-3)
#     reg = regrets_from_Q(envs, Q_map)
#     return {
#         "regret": reg,
#         "mdp_count": len(used),
#         "constraint_count": n_c,
#         "coverage": cov,
#     }

# def run_random_trials(envs, candidates_per_env, n_to_pick, seed, *, trials, SFs, U_universal, max_workers=None):
#     args = [(t, envs, candidates_per_env, n_to_pick, seed, SFs, U_universal) for t in range(trials)]
#     with ProcessPoolExecutor(max_workers=max_workers) as ex:
#         results = list(ex.map(_random_trial_worker, args))
#     return {
#         "regrets": np.vstack([r["regret"] for r in results]),
#         "mdp_counts": [r["mdp_count"] for r in results],
#         "constraint_counts": [r["constraint_count"] for r in results],
#         "coverages": [r["coverage"] for r in results],
#     }

# # =============================================================================
# # MAIN EXPERIMENT
# # =============================================================================
# def run_experiment(
#     n_envs=30,
#     mdp_size=10,
#     feature_dim=2,
#     random_trials=10,
#     seed=0,
#     result_dir="results_two_stage",
#     feedback=("demo", "pairwise", "estop", "improvement"),
#     demo_env_fraction=1.0,
#     total_budget=50,
#     alloc_method="uniform",
#     alloc=None,
#     feedback_generations=1,
#     lp_epsilon=1e-3,
# ):
#     os.makedirs(result_dir, exist_ok=True)
#     print("\n================= EXPERIMENT START =================\n")

#     W_TRUE = generate_w_true(feature_dim, seed=seed)

#     color_to_feature_map = {f"f{i}": [1 if j == i else 0 for j in range(feature_dim)] for i in range(feature_dim)}
#     envs, _ = generate_random_gridworld_envs(
#         n_envs=n_envs,
#         rows=mdp_size,
#         cols=mdp_size,
#         color_to_feature_map=color_to_feature_map,
#         palette=list(color_to_feature_map.keys()),
#         p_color_range={c: (0.3, 0.8) for c in color_to_feature_map},
#         terminal_policy=dict(kind="random_k", k_min=1, k_max=1),
#         gamma_range=(0.99, 0.99),
#         noise_prob_range=(0.0, 0.0),
#         w_mode="fixed",
#         W_fixed=W_TRUE,
#         seed=seed,
#         GridEnvClass=GridWorldMDPFromLayoutEnv,
#     )

#     Q_list = parallel_value_iteration(envs, epsilon=1e-10)

#     SFs = compute_successor_features_family(
#         envs, Q_list, convention="entering", zero_terminal_features=True,
#     )

#     print("GENERATING CONSTRAINTS (multiple generations for non-demo feedback)")
#     enabled = set(feedback)
#     demo_enabled = "demo" in enabled
#     non_demo_types = enabled - {"demo"}

#     best_mean_regret = float('inf')
#     best_chosen = None
#     best_reg_values = None
#     best_n_constraints = None
#     best_coverage = None
#     best_used_envs = None
#     best_candidates_per_env = None
#     best_U_universal = None

#     for gen in range(feedback_generations):
#         print(f"\nGeneration {gen+1}/{feedback_generations}")
#         spec_seed = seed + gen if feedback_generations > 1 else seed

#         spec = GenerationSpec(
#             seed=spec_seed,
#             base_max_horizon=50,
#             demo=DemoSpec(
#                 enabled=demo_enabled,
#                 env_fraction=1.0,
#                 max_steps=1,
#                 state_fraction=demo_env_fraction,
#             ),
#             pairwise=FeedbackSpec(
#                 enabled=("pairwise" in non_demo_types),
#                 total_budget=total_budget if "pairwise" in non_demo_types else 0,
#                 alloc_method=alloc_method,
#                 alloc_params=None if alloc_method == "uniform" else {"alpha": alloc},
#             ),
#             estop=FeedbackSpec(
#                 enabled=("estop" in non_demo_types),
#                 total_budget=total_budget if "estop" in non_demo_types else 0,
#                 alloc_method=alloc_method,
#                 alloc_params=None if alloc_method == "uniform" else {"alpha": alloc},
#             ),
#             improvement=FeedbackSpec(
#                 enabled=("improvement" in non_demo_types),
#                 total_budget=total_budget if "improvement" in non_demo_types else 0,
#                 alloc_method=alloc_method,
#                 alloc_params=None if alloc_method == "uniform" else {"alpha": alloc},
#             ),
#         )

#         candidates_per_env = generate_candidate_atoms_for_scot(envs, Q_list, spec=spec)
#         print(f"Atoms per env: mean={np.mean([len(a) for a in candidates_per_env]):.1f}, total={sum(len(a) for a in candidates_per_env)}")

#         U_per_env_atoms, U_atoms = derive_constraints_from_atoms(candidates_per_env, SFs, envs)
#         U_union_unique = remove_redundant_constraints(U_atoms)
#         U_universal = U_union_unique

#         print(f"|U_atoms| raw = {0 if U_atoms is None else len(U_atoms)}")
#         print(f"|U_union_unique| = {len(U_universal)}")

#         out = two_stage_scot(
#             U_universal=U_universal,
#             U_per_env_atoms=U_per_env_atoms,
#             candidates_per_env=candidates_per_env,
#             SFs=SFs,
#             envs=envs,
#         )
#         chosen = out["chosen"]
#         print(f"TWO-STAGE selected {len(chosen)} atoms")

#         n_c, cov = recover_constraints_and_coverage(chosen, SFs, envs, U_universal)
#         used_envs = sorted({env_idx for env_idx, _ in chosen})
#         num_used = len(used_envs)

#         print(f"Unique constraints: {n_c}")
#         print(f"Coverage: {100*cov:.2f}%")
#         print(f"Used {num_used}/{n_envs} environments")

#         Q_ts, _ = lp_atomic_to_Q_lists(envs, chosen, SFs=SFs, epsilon=lp_epsilon)
#         reg_ts = regrets_from_Q(envs, Q_ts)
#         mean_reg = reg_ts.mean()
#         print(f"Mean regret: {mean_reg:.4f}")

#         if mean_reg < best_mean_regret:
#             best_mean_regret = mean_reg
#             best_chosen = chosen
#             best_reg_values = reg_ts
#             best_n_constraints = n_c
#             best_coverage = cov
#             best_used_envs = used_envs
#             best_candidates_per_env = candidates_per_env
#             best_U_universal = U_universal

#     print(f"\nBest generation selected (mean regret: {best_mean_regret:.4f})")

#     # Random baseline using best candidates
#     rand_out = run_random_trials(
#         envs=envs,
#         candidates_per_env=best_candidates_per_env,
#         n_to_pick=len(best_chosen),
#         seed=seed,
#         trials=random_trials,
#         SFs=SFs,
#         U_universal=best_U_universal,
#     )
#     reg_rand = rand_out["regrets"]

#     print(f"RANDOM mean regret: {reg_rand.mean():.4f}")
#     print(f"RANDOM mean #MDPs: {np.mean(rand_out['mdp_counts']):.2f}")
#     print(f"RANDOM mean unique constraints: {np.mean(rand_out['constraint_counts']):.2f}")
#     print(f"RANDOM mean coverage: {100*np.mean(rand_out['coverages']):.2f}%")

#     # Save
#     results = {
#         "methods": {
#             "two_stage": {
#                 "regret": best_reg_values.tolist(),
#                 "mean_regret": float(best_mean_regret),
#                 "selection_stats": {
#                     "num_atoms_selected": len(best_chosen),
#                     "num_envs_used": len(best_used_envs),
#                     "used_envs": best_used_envs,
#                 },
#                 "constraint_stats": {
#                     "unique_constraints": best_n_constraints,
#                     "coverage": best_coverage,
#                 },
#             },
#             "random": {
#                 "regret": reg_rand.tolist(),
#                 "mean_regret": float(reg_rand.mean()),
#                 "selection_stats": {
#                     "mdp_counts": rand_out["mdp_counts"],
#                     "mean_mdp_count": float(np.mean(rand_out["mdp_counts"])),
#                 },
#                 "constraint_stats": {
#                     "constraint_counts": rand_out["constraint_counts"],
#                     "coverages": rand_out["coverages"],
#                     "mean_unique_constraints": float(np.mean(rand_out["constraint_counts"])),
#                     "mean_coverage": float(np.mean(rand_out["coverages"])),
#                 },
#             },
#         },
#         "universal_constraints": {
#             "U_atoms_raw": 0 if U_atoms is None else len(U_atoms),
#             "U_union_unique": len(U_universal),
#         },
#         "config": {
#             "seed": seed,
#             "n_envs": n_envs,
#             "mdp_size": mdp_size,
#             "feature_dim": feature_dim,
#             "feedback": list(feedback),
#             "demo_env_fraction": demo_env_fraction,
#             "total_budget": total_budget,
#             "random_trials": random_trials,
#             "feedback_generations": feedback_generations,
#             "lp_epsilon": lp_epsilon,
#         },
#     }

#     timestamp = time.strftime("%Y%m%d-%H%M%S")
#     exp_name = f"two_stage_vs_random_env{n_envs}_size{mdp_size}_fd{feature_dim}_budget{total_budget}_seed{seed}_{timestamp}.json"
#     out_path = os.path.join(result_dir, exp_name)
#     with open(out_path, "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"\nSaved to {out_path}")
#     print("\n================= EXPERIMENT END =================\n")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Two-Stage SCOT vs Random baseline with multiple feedback generations",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser.add_argument("--n_envs", type=int, default=30, help="Number of MDPs")
#     parser.add_argument("--mdp_size", type=int, default=10, help="Grid side length")
#     parser.add_argument("--feature_dim", type=int, default=2, help="Reward feature dimension")
#     parser.add_argument("--feedback", nargs="+", default=["demo", "pairwise", "estop", "improvement"],
#                         help="Feedback types to include")
#     parser.add_argument("--demo_env_fraction", type=float, default=1.0,
#                         help="Fraction of envs for demos")
#     parser.add_argument("--total_budget", type=int, default=50,
#                         help="Feedback budget (pairwise/estop/improvement)")
#     parser.add_argument("--random_trials", type=int, default=10,
#                         help="Number of random baseline trials")
#     parser.add_argument("--seed", type=int, default=0, help="Random seed")
#     parser.add_argument("--result_dir", type=str, default="results_two_stage",
#                         help="Output directory")
#     parser.add_argument("--alloc_method", type=str, default="uniform",
#                         choices=["uniform", "dirichlet"], help="Allocation method")
#     parser.add_argument("--alloc", type=float, default=None,
#                         help="Dirichlet alpha (if alloc_method=dirichlet)")

#     # New parameters
#     parser.add_argument("--feedback-generations", type=int, default=10,
#                         help="How many times to regenerate non-demo feedback")
#     parser.add_argument("--lp-epsilon", type=float, default=1e-6,
#                         help="Fixed margin ε for LP reward learning")

#     args = parser.parse_args()

#     if args.alloc_method != "uniform" and args.alloc is None:
#         args.alloc = 0.5

#     run_experiment(
#         n_envs=args.n_envs,
#         mdp_size=args.mdp_size,
#         feature_dim=args.feature_dim,
#         random_trials=args.random_trials,
#         seed=args.seed,
#         result_dir=args.result_dir,
#         feedback=args.feedback,
#         demo_env_fraction=args.demo_env_fraction,
#         total_budget=args.total_budget,
#         alloc_method=args.alloc_method,
#         alloc=args.alloc,
#         feedback_generations=args.feedback_generations,
#         lp_epsilon=args.lp_epsilon,
#     )



"""
Two-Stage SCOT vs Random — LP reward learning
with multiple feedback generations (non-demo only)
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import pulp

# Path setup
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

# Imports
from utils import (
    generate_random_gridworld_envs,
    compute_successor_features_family,
    derive_constraints_from_atoms,  # still used elsewhere (LP learning)
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

# NEW: cached two-stage
from teaching.two_stage_scot import (
    build_atom_constraint_cache,
    two_stage_scot_cached,
)

# Ground-truth reward
def generate_w_true(d, seed=None):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    return w / np.linalg.norm(w)

# Q wrapper
def _compute_Q_wrapper(args):
    env, w, vi_eps = args
    return compute_Q_from_weights_with_VI(env, w, vi_epsilon=vi_eps)

# Simple fixed LP reward learning (no tuning)
def lp_atomic_to_Q_lists(
    envs,
    atoms_flat,
    SFs,
    epsilon=1e-3,
    vi_epsilon=1e-6,
):
    atoms_per_env = [[] for _ in envs]
    for env_idx, atom in atoms_flat:
        atoms_per_env[env_idx].append(atom)

    U_per_env_atoms, U_atoms = derive_constraints_from_atoms(atoms_per_env, SFs, envs)

    if U_atoms is None or len(U_atoms) == 0:
        print("Warning: No constraints → zero reward")
        w_sol = np.zeros(len(envs[0].feature_map))
    else:
        unique = []
        for v in U_atoms:
            v = np.asarray(v, dtype=float)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                continue
            is_close = any(
                np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u)) > 1 - 1e-3
                for u in unique
            )
            if not is_close:
                unique.append(v)

        U = remove_redundant_constraints(unique)
        U = np.asarray(U, dtype=float)

        if len(U) == 0:
            print("No constraints after cleaning → zero reward")
            w_sol = np.zeros(len(envs[0].feature_map))
        else:
            n, d = U.shape
            print(f" → {n} constraints | {d} dimensions")

            prob = pulp.LpProblem("MaxMarginRewardLP", pulp.LpMaximize)
            w_vars = [pulp.LpVariable(f"w_{j}") for j in range(d)]
            abs_w = [pulp.LpVariable(f"abs_w_{j}", lowBound=0) for j in range(d)]

            for j in range(d):
                prob += abs_w[j] >= w_vars[j]
                prob += abs_w[j] >= -w_vars[j]
            prob += pulp.lpSum(abs_w) == 1

            margins = [pulp.lpSum(U[i, j] * w_vars[j] for j in range(d)) for i in range(n)]
            prob += pulp.lpSum(margins)
            for m in margins:
                prob += m >= epsilon

            status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
            print(f"LP status: {pulp.LpStatus[status]}")

            if pulp.LpStatus[status] != "Optimal":
                print("LP not optimal → zero vector")
                w_sol = np.zeros(d)
            else:
                w_sol = np.array([pulp.value(wj) for wj in w_vars], dtype=float)

    # Compute Q
    with ProcessPoolExecutor() as ex:
        Q_list = list(ex.map(_compute_Q_wrapper, [(e, w_sol, vi_epsilon) for e in envs]))

    return Q_list, None

# Regret
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

# Random baseline helpers (unchanged)
def sample_random_atoms_global_pool(candidates_per_env, n_to_pick, seed=None):
    rng = np.random.default_rng(seed)
    pool = [(i, a) for i, atoms in enumerate(candidates_per_env) for a in atoms]
    idxs = rng.choice(len(pool), size=n_to_pick, replace=False)
    return [pool[i] for i in idxs]

def _random_trial_worker(args):
    trial_id, envs, candidates_per_env, n_to_pick, seed, SFs, U_universal = args
    chosen = sample_random_atoms_global_pool(candidates_per_env, n_to_pick, seed + trial_id)
    used = {env_idx for env_idx, _ in chosen}
    n_c, cov = recover_constraints_and_coverage(chosen, SFs, envs, U_universal)
    Q_map, _ = lp_atomic_to_Q_lists(envs, chosen, SFs=SFs, epsilon=1e-3)
    reg = regrets_from_Q(envs, Q_map)
    return {
        "regret": reg,
        "mdp_count": len(used),
        "constraint_count": n_c,
        "coverage": cov,
    }

def run_random_trials(envs, candidates_per_env, n_to_pick, seed, *, trials, SFs, U_universal, max_workers=None):
    args = [(t, envs, candidates_per_env, n_to_pick, seed, SFs, U_universal) for t in range(trials)]
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
    n_envs=30,
    mdp_size=10,
    feature_dim=2,
    random_trials=10,
    seed=0,
    result_dir="results_two_stage",
    feedback=("demo", "pairwise", "estop", "improvement"),
    demo_env_fraction=1.0,
    total_budget=50,
    alloc_method="uniform",
    alloc=None,
    feedback_generations=1,
    lp_epsilon=1e-3,
):
    os.makedirs(result_dir, exist_ok=True)
    print("\n================= EXPERIMENT START =================\n")

    W_TRUE = generate_w_true(feature_dim, seed=seed)

    color_to_feature_map = {f"f{i}": [1 if j == i else 0 for j in range(feature_dim)] for i in range(feature_dim)}
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

    Q_list = parallel_value_iteration(envs, epsilon=1e-10)

    SFs = compute_successor_features_family(
        envs, Q_list, convention="entering", zero_terminal_features=True,
    )

    print("GENERATING CONSTRAINTS (multiple generations for non-demo feedback)")
    enabled = set(feedback)
    demo_enabled = "demo" in enabled
    non_demo_types = enabled - {"demo"}

    best_mean_regret = float("inf")
    best_chosen = None
    best_reg_values = None
    best_n_constraints = None
    best_coverage = None
    best_used_envs = None
    best_candidates_per_env = None
    best_U_universal = None

    for gen in range(feedback_generations):
        print(f"\nGeneration {gen+1}/{feedback_generations}")
        spec_seed = seed + gen if feedback_generations > 1 else seed

        spec = GenerationSpec(
            seed=spec_seed,
            base_max_horizon=150,
            demo=DemoSpec(
                enabled=demo_enabled,
                env_fraction=1.0,
                max_steps=1,
                state_fraction=demo_env_fraction,
            ),
            pairwise=FeedbackSpec(
                enabled=("pairwise" in non_demo_types),
                total_budget=total_budget if "pairwise" in non_demo_types else 0,
                alloc_method=alloc_method,
                alloc_params=None if alloc_method == "uniform" else {"alpha": alloc},
            ),
            estop=FeedbackSpec(
                enabled=("estop" in non_demo_types),
                total_budget=total_budget if "estop" in non_demo_types else 0,
                alloc_method=alloc_method,
                alloc_params=None if alloc_method == "uniform" else {"alpha": alloc},
            ),
            improvement=FeedbackSpec(
                enabled=("improvement" in non_demo_types),
                total_budget=total_budget if "improvement" in non_demo_types else 0,
                alloc_method=alloc_method,
                alloc_params=None if alloc_method == "uniform" else {"alpha": alloc},
            ),
        )

        candidates_per_env = generate_candidate_atoms_for_scot(envs, Q_list, spec=spec)
        print(f"Atoms per env: mean={np.mean([len(a) for a in candidates_per_env]):.1f}, total={sum(len(a) for a in candidates_per_env)}")

        # =========================
        # ✅ FIX: Build cache ONCE
        # =========================
        atom_constraints_per_env, U_atoms = build_atom_constraint_cache(
            candidates_per_env, SFs, envs
        )

        U_universal = remove_redundant_constraints(U_atoms)

        print(f"|U_atoms| raw = {0 if U_atoms is None else len(U_atoms)}")
        print(f"|U_universal| = {len(U_universal)}")

        out = two_stage_scot_cached(
            U_universal=U_universal,
            candidates_per_env=candidates_per_env,
            atom_constraints_per_env=atom_constraints_per_env,
        )
        chosen = out["chosen"]
        print(f"TWO-STAGE (cached) selected {len(chosen)} atoms")

        n_c, cov = recover_constraints_and_coverage(chosen, SFs, envs, U_universal)
        used_envs = sorted({env_idx for env_idx, _ in chosen})
        num_used = len(used_envs)

        print(f"Unique constraints: {n_c}")
        print(f"Coverage: {100 * cov:.2f}%")
        print(f"Used {num_used}/{n_envs} environments")

        Q_ts, _ = lp_atomic_to_Q_lists(envs, chosen, SFs=SFs, epsilon=lp_epsilon)
        reg_ts = regrets_from_Q(envs, Q_ts)
        mean_reg = reg_ts.mean()
        print(f"Mean regret: {mean_reg:.4f}")

        if mean_reg < best_mean_regret:
            best_mean_regret = mean_reg
            best_chosen = chosen
            best_reg_values = reg_ts
            best_n_constraints = n_c
            best_coverage = cov
            best_used_envs = used_envs
            best_candidates_per_env = candidates_per_env
            best_U_universal = U_universal

    print(f"\nBest generation selected (mean regret: {best_mean_regret:.4f})")

    # Random baseline using best candidates
    rand_out = run_random_trials(
        envs=envs,
        candidates_per_env=best_candidates_per_env,
        n_to_pick=len(best_chosen),
        seed=seed,
        trials=random_trials,
        SFs=SFs,
        U_universal=best_U_universal,
    )
    reg_rand = rand_out["regrets"]

    print(f"RANDOM mean regret: {reg_rand.mean():.4f}")
    print(f"RANDOM mean #MDPs: {np.mean(rand_out['mdp_counts']):.2f}")
    print(f"RANDOM mean unique constraints: {np.mean(rand_out['constraint_counts']):.2f}")
    print(f"RANDOM mean coverage: {100 * np.mean(rand_out['coverages']):.2f}%")

    # Save
    results = {
        "methods": {
            "two_stage_cached": {
                "regret": best_reg_values.tolist(),
                "mean_regret": float(best_mean_regret),
                "selection_stats": {
                    "num_atoms_selected": len(best_chosen),
                    "num_envs_used": len(best_used_envs),
                    "used_envs": best_used_envs,
                },
                "constraint_stats": {
                    "unique_constraints": best_n_constraints,
                    "coverage": best_coverage,
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
            "U_union_unique": len(best_U_universal),
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
            "feedback_generations": feedback_generations,
            "lp_epsilon": lp_epsilon,
        },
    }

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_name = f"two_stage_vs_random_cached_env{n_envs}_size{mdp_size}_fd{feature_dim}_budget{total_budget}_seed{seed}_{timestamp}.json"
    out_path = os.path.join(result_dir, exp_name)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {out_path}")
    print("\n================= EXPERIMENT END =================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-Stage SCOT vs Random baseline with cached constraints (no Stage-2 recomputation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n_envs", type=int, default=30, help="Number of MDPs")
    parser.add_argument("--mdp_size", type=int, default=10, help="Grid side length")
    parser.add_argument("--feature_dim", type=int, default=2, help="Reward feature dimension")
    parser.add_argument("--feedback", nargs="+", default=["demo", "pairwise", "estop", "improvement"],
                        help="Feedback types to include")
    parser.add_argument("--demo_env_fraction", type=float, default=1.0,
                        help="Fraction of envs for demos")
    parser.add_argument("--total_budget", type=int, default=50,
                        help="Feedback budget (pairwise/estop/improvement)")
    parser.add_argument("--random_trials", type=int, default=10,
                        help="Number of random baseline trials")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--result_dir", type=str, default="results_two_stage",
                        help="Output directory")
    parser.add_argument("--alloc_method", type=str, default="uniform",
                        choices=["uniform", "dirichlet"], help="Allocation method")
    parser.add_argument("--alloc", type=float, default=None,
                        help="Dirichlet alpha (if alloc_method=dirichlet)")
    parser.add_argument("--feedback-generations", type=int, default=10,
                        help="How many times to regenerate non-demo feedback")
    parser.add_argument("--lp-epsilon", type=float, default=1e-6,
                        help="Fixed margin ε for LP reward learning")

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
        feedback_generations=args.feedback_generations,
        lp_epsilon=args.lp_epsilon,
    )
