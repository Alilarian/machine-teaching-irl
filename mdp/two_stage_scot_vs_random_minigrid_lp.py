# =============================================================================
# Two-Stage SCOT + Reward Learning (MiniGrid LavaWorld) — LP version
# Full CLI-driven pipeline with feedback selection + baselines
# =============================================================================
import os
import sys
import json
import time
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# ──── NEW: LP solver ─────────────────────────────────────────────────────────
import pulp

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from utils.feedback_budgeting_minigrid import (
    GenerationSpec_minigrid,
    DemoSpec_minigrid,
    FeedbackSpec_minigrid,
)
from utils.minigrid_lava_generator import generate_lavaworld, enumerate_states
from utils import (
    value_iteration_next_state_multi,
    compute_successor_features_multi,
    generate_demos_from_policies_multi,
    constraints_from_demos_next_state_multi,
    generate_candidate_atoms_for_scot_minigrid,
    constraints_from_atoms_multi_env,
    remove_redundant_constraints,
    expected_value_difference_next_state_multi,
)
from teaching.two_stage_scot_minigrid import (
    two_stage_scot,
    scot_greedy_family_atoms_tracked,
)

def coverage_report_key_based(
    U_universal,
    U_per_env_envlevel,
    selected_envs,
    *,
    normalize=True,
    round_decimals=12,
):
    from teaching.two_stage_scot_minigrid import make_key_for
    key_for = make_key_for(normalize=normalize, round_decimals=round_decimals)
    key_to_uid = {}
    for u in U_universal:
        k = key_for(u)
        if k not in key_to_uid:
            key_to_uid[k] = len(key_to_uid)
    universe = set(key_to_uid.values())
    def covered_by(env_ids):
        covered = set()
        for e in env_ids:
            H = U_per_env_envlevel[e]
            if H is None or len(H) == 0:
                continue
            for row in np.atleast_2d(H):
                uid = key_to_uid.get(key_for(row))
                if uid is not None:
                    covered.add(uid)
        return covered
    cov_selected = covered_by(selected_envs)
    cov_all = covered_by(range(len(U_per_env_envlevel)))
    return {
        "universe_size": len(universe),
        "covered_by_selected": len(cov_selected),
        "covered_by_all_envs": len(cov_all),
        "coverage_frac_selected": len(cov_selected) / max(len(universe), 1),
        "coverage_frac_all_envs": len(cov_all) / max(len(universe), 1),
    }

# =============================================================================
# Utility Functions
# =============================================================================
def regrets_from_Q(mdps, Q_list, theta_true):
    pi_list = [np.argmax(Q, axis=1) for Q in Q_list]
    reg = expected_value_difference_next_state_multi(
        eval_policies=pi_list,
        mdps=mdps,
        theta=theta_true,
        normalize_with_random_policy=False,
        include_terminal_in_mean=False,
    )
    return np.asarray(reg, dtype=float)

def scot_output_to_atoms_flat(out):
    atoms_flat = []
    for item in out["chosen"]:
        if isinstance(item, tuple):
            atoms_flat.append((int(item[0]), item[1]))
        elif hasattr(item, "env_id"):
            atoms_flat.append((int(item.env_id), item))
        else:
            raise TypeError("Unknown atom format")
    return atoms_flat

# ──── REPLACEMENT: LP instead of BIRL ────────────────────────────────────────
def lp_atomic_to_Q_and_wmap(mdps, atoms_flat, Psi_sa_list, args):
    """
    LP-based reward inference with max-margin + L1 normalization.
    """
    # Group atoms per environment
    atoms_per_env = [[] for _ in mdps]
    for env_idx, atom in atoms_flat:
        atoms_per_env[env_idx].append(atom)

    # Derive constraints
    U_atoms = constraints_from_atoms_multi_env(
        atoms_per_env=atoms_per_env,
        mdps=mdps,
        Psi_sa_list=Psi_sa_list,
        terminal_mask_list=[mdp["terminal"] for mdp in mdps],
        normalize=True,
        n_jobs=args.n_jobs,
    )

    def flatten_constraints(constraints_per_env_per_atom):
        flat = []
        for env_list in constraints_per_env_per_atom:
            for atom_list in env_list:
                for c in atom_list:
                    flat.append(np.asarray(c, dtype=float))
        return np.vstack(flat)  # shape (N, d)

    U_atoms = flatten_constraints(U_atoms)

    print(U_atoms[0])
    print(len(U_atoms))


    print(U_atoms.shape)

    if U_atoms is None or len(U_atoms) == 0:
        print("Warning: No constraints from selected atoms → using zero reward")
        w_sol = np.zeros(len(mdps[0]["true_w"]))
    else:
        U = remove_redundant_constraints(np.vstack(U_atoms))
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
            prob += m >= args.epsilon

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

    # Compute Q-values
    _, Q_list, _ = value_iteration_next_state_multi(
        mdps=mdps,
        theta=w_sol,
        gamma=args.gamma,
        n_jobs=1,
    )

    return Q_list, w_sol

# =============================================================================
# Baselines
# =============================================================================
# def random_atom_trial(args_tuple):
#     trial_id, mdps, atoms_per_env, k_atoms, args, enabled_feedback, Psi_sa_list = args_tuple
#     rng = np.random.default_rng(args.seed + trial_id)
#     pool = [
#         (env_idx, atom)
#         for env_idx, atoms in enumerate(atoms_per_env)
#         for atom in atoms
#     ]
#     idxs = rng.choice(len(pool), size=k_atoms, replace=False)
#     chosen = [pool[i] for i in idxs]
#     Q_list, _ = lp_atomic_to_Q_and_wmap(
#         mdps, chosen, Psi_sa_list, args
#     )
#     return regrets_from_Q(mdps, Q_list, mdps[0]["true_w"])

def random_atom_trial(args_tuple):
    trial_id, mdps, atoms_per_env, k_atoms, args, enabled_feedback, Psi_sa_list, heldout_mdps = args_tuple
    rng = np.random.default_rng(args.seed + trial_id)
    pool = [
        (env_idx, atom)
        for env_idx, atoms in enumerate(atoms_per_env)
        for atom in atoms
    ]
    idxs = rng.choice(len(pool), size=k_atoms, replace=False)
    chosen = [pool[i] for i in idxs]
    train_Q_list, w_sol = lp_atomic_to_Q_and_wmap(
        mdps, chosen, Psi_sa_list, args
    )
    train_reg = regrets_from_Q(mdps, train_Q_list, mdps[0]["true_w"])
    if heldout_mdps:
        _, heldout_Q_list, _ = value_iteration_next_state_multi(
            mdps=heldout_mdps,
            theta=w_sol,
            gamma=args.gamma,
            n_jobs=1,  # Use 1 job per process to avoid nesting issues
        )
        heldout_reg = regrets_from_Q(heldout_mdps, heldout_Q_list, mdps[0]["true_w"])
    else:
        heldout_reg = None
    return train_reg, heldout_reg

# (other baseline functions remain commented out — unchanged)

# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main(args):
    enabled_feedback = set(args.feedback)

    # 1) Generate Environments
    envs, mdps, _ = generate_lavaworld(
        n_envs=args.n_envs,
        size=args.grid_size,
        seed=args.seed,
        gamma=args.gamma,
    )


    rng = np.random.default_rng(args.seed)
    n_envs = args.n_envs
    if args.heldout_frac > 0:
        n_train = int(n_envs * (1 - args.heldout_frac))
        train_idxs = rng.choice(n_envs, n_train, replace=False)
        train_idxs = sorted(train_idxs)
        heldout_idxs = sorted(set(range(n_envs)) - set(train_idxs))
        train_envs = [envs[i] for i in train_idxs]
        heldout_envs = [envs[i] for i in heldout_idxs]
        train_mdps = [mdps[i] for i in train_idxs]
        heldout_mdps = [mdps[i] for i in heldout_idxs]
    else:
        train_envs = envs
        heldout_envs = []
        train_mdps = mdps
        heldout_mdps = []
    
    theta_true = train_mdps[0]["true_w"]


    # 2) Oracle Value Iteration
    _, train_Q_list, train_pi_list = value_iteration_next_state_multi(
        mdps=train_mdps,
        theta=theta_true,
        gamma=args.gamma,
        n_jobs=args.n_jobs,
    )

    # ... (Psi computation)
    train_Psi_sa_list, train_Psi_s_list = compute_successor_features_multi(
        mdps=train_mdps,
        Q_list=train_Q_list,
        gamma=args.gamma,
        n_jobs=args.n_jobs,
    )

    d = train_Psi_s_list[0].shape[1]

    # 4) Feedback Atom Generation
    GEN_SPEC = GenerationSpec_minigrid(
        seed=args.seed,
        demo=None if "demo" not in enabled_feedback else DemoSpec_minigrid(
            enabled=True,
            env_fraction=1.0,
            state_fraction=args.state_fraction,
        ),
        pairwise=None if "pairwise" not in enabled_feedback else FeedbackSpec_minigrid(
            enabled=True,
            total_budget=args.total_budget,
            alloc_method=args.alloc_method,
            alloc_params=None if args.alloc_method == "uniform"
            else {"alpha": args.alloc},
        ),
        estop=None if "estop" not in enabled_feedback else FeedbackSpec_minigrid(
            enabled=True,
            total_budget=args.total_budget,
            alloc_method=args.alloc_method,
            alloc_params=None if args.alloc_method == "uniform"
            else {"alpha": args.alloc},
        ),
        improvement=None if "improvement" not in enabled_feedback else FeedbackSpec_minigrid(
            enabled=True,
            total_budget=args.total_budget,
            alloc_method=args.alloc_method,
            alloc_params=None if args.alloc_method == "uniform"
            else {"alpha": args.alloc},
        ),
    )

    # 4) Feedback Atom Generation
    # ... (GEN_SPEC unchanged)
    atoms_per_env = generate_candidate_atoms_for_scot_minigrid(
        mdps=train_mdps,
        pi_list=train_pi_list,
        spec=GEN_SPEC,
        enumerate_states=enumerate_states,
        max_horizon=400,
    )

    U_atoms_per_env = constraints_from_atoms_multi_env(
        atoms_per_env=atoms_per_env,
        mdps=train_mdps,
        Psi_sa_list=train_Psi_sa_list,
        terminal_mask_list=[mdp["terminal"] for mdp in train_mdps],
        normalize=True,
        n_jobs=args.n_jobs,
    )


    U_atoms_flat = [c for env in U_atoms_per_env for atom_cs in env for c in atom_cs]
    U_atoms_unique = remove_redundant_constraints(np.vstack(U_atoms_flat))
    U_universal = remove_redundant_constraints(U_atoms_unique)

    # 5) Two-Stage SCOT
    U_atoms_envlevel = [
        np.vstack([c for atom_cs in env for c in atom_cs])
        for env in U_atoms_per_env
    ]
    out = two_stage_scot(
        U_universal=U_universal,
        U_per_env_atoms_envlevel=U_atoms_envlevel,
        constraints_per_env_per_atom=U_atoms_per_env,
        candidates_per_env=atoms_per_env,
        normalize=True,
        round_decimals=12,
    )
    atoms_flat = scot_output_to_atoms_flat(out)
    U_atoms_envlevel = [
        np.vstack([c for atom_cs in env for c in atom_cs])
        if len(env) > 0 else np.zeros((0, d))
        for env in U_atoms_per_env
    ]
    cov = coverage_report_key_based(
        U_universal=U_universal,
        U_per_env_envlevel=U_atoms_envlevel,
        selected_envs=out["selected_mdps"],
    )

    # 6) Reward Learning — LP
    train_Q_learned, w_map = lp_atomic_to_Q_and_wmap(
        mdps=train_mdps, atoms_flat=atoms_flat, Psi_sa_list=train_Psi_sa_list, args=args
    )
    train_reg_scot = regrets_from_Q(train_mdps, train_Q_learned, theta_true)

    if heldout_mdps:
        _, heldout_Q_learned, _ = value_iteration_next_state_multi(
            mdps=heldout_mdps,
            theta=w_map,
            gamma=args.gamma,
            n_jobs=args.n_jobs,
        )
        heldout_reg_scot = regrets_from_Q(heldout_mdps, heldout_Q_learned, theta_true)
    else:
        heldout_reg_scot = None

    # 7) Baselines
    k_atoms = len(out["chosen"])
    used_envs = sorted(set(out["selected_mdps"]))

    with ProcessPoolExecutor() as ex:
        results_list = list(ex.map(
            random_atom_trial,
            [
                (t, train_mdps, atoms_per_env, k_atoms, args, enabled_feedback, train_Psi_sa_list, heldout_mdps)
                for t in range(args.random_trials)
            ]
        ))
    reg_rand = np.vstack([r[0] for r in results_list])
    if heldout_mdps:
        heldout_reg_rand = np.vstack([r[1] for r in results_list])
    else:
        heldout_reg_rand = None

    # (other baselines remain commented)

    # 8) Save Results
    cov_ts = coverage_report_key_based(
        U_universal=U_universal,
        U_per_env_envlevel=U_atoms_envlevel,
        selected_envs=out["selected_mdps"],
    )
    ts_n_constraints = cov_ts["covered_by_selected"]
    ts_coverage = cov_ts["coverage_frac_selected"]

    rand_constraint_counts = []
    rand_coverages = []
    rand_mdp_counts = []
    for t in range(args.random_trials):
        rng = np.random.default_rng(args.seed + t)
        pool = [
            (env_idx, atom)
            for env_idx, atoms in enumerate(atoms_per_env)
            for atom in atoms
        ]
        idxs = rng.choice(len(pool), size=len(out["chosen"]), replace=False)
        chosen = [pool[i] for i in idxs]
        used_envs_rand = sorted({e for e, _ in chosen})
        cov_rand = coverage_report_key_based(
            U_universal=U_universal,
            U_per_env_envlevel=U_atoms_envlevel,
            selected_envs=used_envs_rand,
        )
        rand_constraint_counts.append(cov_rand["covered_by_selected"])
        rand_coverages.append(cov_rand["coverage_frac_selected"])
        rand_mdp_counts.append(len(used_envs_rand))

    results = {
        "methods": {
            "two_stage": {
                # Training set (what was originally called "regret")
                "regret": train_reg_scot.tolist(),
                "mean_regret": float(np.mean(train_reg_scot)),
                
                # Held-out set (new)
                "heldout_regret": heldout_reg_scot.tolist() if heldout_reg_scot is not None else None,
                "mean_heldout_regret": float(np.mean(heldout_reg_scot)) if heldout_reg_scot is not None else None,
                
                # The rest stays exactly the same
                "selection_stats": {
                    "num_atoms_selected": int(len(out["chosen"])),
                    "num_envs_used": int(len(out["selected_mdps"])),
                    "used_envs": list(out["selected_mdps"]),
                },
                "constraint_stats": {
                    "unique_constraints": int(ts_n_constraints),
                    "coverage": float(ts_coverage),
                },
            },
            "random": {
                # Training set regrets — shape: (n_random_trials, n_train_mdps)
                "regret": reg_rand.tolist(),
                "mean_regret": float(np.mean(reg_rand)),
                
                # Held-out set regrets — shape: (n_random_trials, n_heldout_mdps) or None
                "heldout_regret": heldout_reg_rand.tolist() if heldout_reg_rand is not None else None,
                "mean_heldout_regret": float(np.mean(heldout_reg_rand)) if heldout_reg_rand is not None else None,
                
                # The rest stays the same (these are training-set based stats)
                "selection_stats": {
                    "mdp_counts": rand_mdp_counts,
                    "mean_mdp_count": float(np.mean(rand_mdp_counts)),
                },
                "constraint_stats": {
                    "constraint_counts": rand_constraint_counts,
                    "coverages": rand_coverages,
                    "mean_unique_constraints": float(np.mean(rand_constraint_counts)),
                    "mean_coverage": float(np.mean(rand_coverages)),
                },
            },
            # If you later uncomment other baselines, apply similar pattern:
            # e.g. "active", "diversity", etc. would also get heldout_* keys
        },
        
        "universal_constraints": {
            "U_atoms_unique": int(len(U_atoms_unique)),
            "U_union_unique": int(len(U_universal)),
        },
        
        "config": {
            "seed": args.seed,
            "n_envs": args.n_envs,
            "grid_size": args.grid_size,
            "gamma": args.gamma,
            "feedback": list(enabled_feedback),
            "state_fraction": args.state_fraction,
            "total_budget": args.total_budget,
            "random_trials": args.random_trials,
            "alloc_method": args.alloc_method,
            "alloc_alpha": args.alloc,
            "lp": {
                "epsilon": args.epsilon,
            },
            # ─── New field added for the held-out experiment ───
            "heldout_frac": args.heldout_frac,
            # You can also add these if useful for analysis:
            # "n_train": len(train_mdps),
            # "n_heldout": len(heldout_mdps),
        },
    }


    os.makedirs(args.result_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(args.result_dir, f"minigrid_lp_run_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {path}\n")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=2)
    parser.add_argument("--grid_size", type=int, default=6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--state_fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=125)
    parser.add_argument("--n_jobs", type=int, default=None)
    parser.add_argument("--feedback",
                        nargs="+",
                        default=["pairwise"],
                        choices=["demo", "pairwise", "estop", "improvement"])
    parser.add_argument("--total_budget", type=int, default=1000)
    parser.add_argument("--alloc_method",
                        type=str,
                        default="uniform",
                        choices=["uniform", "dirichlet"])
    parser.add_argument("--alloc", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=1e-6,
                        help="Minimum margin for LP hard constraints")
    parser.add_argument("--random_trials", type=int, default=10)
    parser.add_argument("--result_dir", type=str, default="results_minigrid_lp")
    # In the CLI section (after other parser.add_argument calls)
    parser.add_argument("--heldout_frac", type=float, default=0.2,
                    help="Fraction of MDPs to hold out for evaluation (0.0 to disable)")

    args = parser.parse_args()
    if args.alloc_method != "uniform" and args.alloc is None:
        args.alloc = 0.5

    main(args)