

# =============================================================================
# Two-Stage SCOT + Reward Learning (MiniGrid LavaWorld)
# Full CLI-driven pipeline with feedback selection + baselines
# =============================================================================

import os
import sys
import json
import time
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

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
from reward_learning.multi_env_atomic_birl_minigrid import (
    MultiEnvAtomicBIRL_MiniGrid,
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

def birl_atomic_to_Q_and_wmap(mdps, atoms_flat, args, enabled_feedback):
    birl = MultiEnvAtomicBIRL_MiniGrid(
        mdps=mdps,
        atoms_flat=atoms_flat,
        beta_demo=args.beta if "demo" in enabled_feedback else 0.0,
        beta_pairwise=args.beta if "pairwise" in enabled_feedback else 0.0,
        beta_estop=args.beta if "estop" in enabled_feedback else 0.0,
        beta_improvement=args.beta if "improvement" in enabled_feedback else 0.0,
        epsilon=1e-8,
    )

    # IMPORTANT: seed always None
    birl.run_mcmc(
        samples=args.samples,
        stepsize=args.stepsize,
        normalize=False,
        seed=None,
    )

    w_map = birl.get_map_solution()

    _, Q_list, _ = value_iteration_next_state_multi(
        mdps=mdps,
        theta=w_map,
        gamma=args.gamma,
        n_jobs=1,
    )

    return Q_list, w_map

# =============================================================================
# Baselines
# =============================================================================
def random_atom_trial(args_tuple):
    trial_id, mdps, atoms_per_env, k_atoms, args, enabled_feedback = args_tuple
    rng = np.random.default_rng(args.seed + trial_id)

    pool = [
        (env_idx, atom)
        for env_idx, atoms in enumerate(atoms_per_env)
        for atom in atoms
    ]

    idxs = rng.choice(len(pool), size=k_atoms, replace=False)
    chosen = [pool[i] for i in idxs]

    Q_list, _ = birl_atomic_to_Q_and_wmap(
        mdps, chosen, args, enabled_feedback
    )

    return regrets_from_Q(mdps, Q_list, mdps[0]["true_w"])

def random_mdp_scot_trial(args_tuple):
    (
        trial_id,
        mdps,
        atoms_per_env,
        constraints_per_env_per_atom,
        U_universal,
        n_mdps_to_pick,
        k_atoms,
        args,
        enabled_feedback,
    ) = args_tuple

    rng = np.random.default_rng(args.seed + trial_id)

    selected_envs = rng.choice(len(mdps), size=n_mdps_to_pick, replace=False)

    atoms_subset = [atoms_per_env[i] for i in selected_envs]
    constraints_subset = [constraints_per_env_per_atom[i] for i in selected_envs]

    chosen_local, _, _ = scot_greedy_family_atoms_tracked(
        U_universal,
        atoms_subset,
        constraints_subset,
        normalize=True,
        round_decimals=12,
    )

    chosen_atoms = [
        (int(selected_envs[int(e)]), atom)
        for e, atom in chosen_local
    ][:k_atoms]

    Q_list, _ = birl_atomic_to_Q_and_wmap(
        mdps, chosen_atoms, args, enabled_feedback
    )

    return regrets_from_Q(mdps, Q_list, mdps[0]["true_w"])

def same_mdp_random_atom_trial(args_tuple):
    (
        trial_id,
        mdps,
        atoms_per_env,
        selected_envs,     # from two_stage
        k_atoms,
        args,
        enabled_feedback,
    ) = args_tuple

    rng = np.random.default_rng(args.seed + trial_id)

    # Build pool ONLY from SCOT-selected MDPs
    pool = [
        (env_idx, atom)
        for env_idx in selected_envs
        for atom in atoms_per_env[env_idx]
    ]

    # If pool smaller than k_atoms, clip safely
    k = min(k_atoms, len(pool))

    idxs = rng.choice(len(pool), size=k, replace=False)
    chosen = [pool[i] for i in idxs]

    Q_list, _ = birl_atomic_to_Q_and_wmap(
        mdps, chosen, args, enabled_feedback
    )

    return regrets_from_Q(mdps, Q_list, mdps[0]["true_w"])

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main(args):

    enabled_feedback = set(args.feedback)

    # -------------------------------------------------------------------------
    # 1) Generate Environments
    # -------------------------------------------------------------------------
    envs, mdps, _ = generate_lavaworld(
        n_envs=args.n_envs,
        size=args.grid_size,
        seed=args.seed,
        gamma=args.gamma,
    )

    theta_true = mdps[0]["true_w"]

    # -------------------------------------------------------------------------
    # 2) Oracle Value Iteration
    # -------------------------------------------------------------------------
    _, Q_list, pi_list = value_iteration_next_state_multi(
        mdps=mdps,
        theta=theta_true,
        gamma=args.gamma,
        n_jobs=args.n_jobs,
    )

    Psi_sa_list, Psi_s_list = compute_successor_features_multi(
        mdps=mdps,
        Q_list=Q_list,
        gamma=args.gamma,
        n_jobs=args.n_jobs,
    )

    d = Psi_s_list[0].shape[1]

    # -------------------------------------------------------------------------
    # 3) Oracle Demo Constraints
    # -------------------------------------------------------------------------
    # demos_list = generate_demos_from_policies_multi(
    #     mdps=mdps,
    #     pi_list=pi_list,
    #     n_jobs=args.n_jobs,
    # )

    # U_demo_per_env = constraints_from_demos_next_state_multi(
    #     demos_list=demos_list,
    #     Psi_sa_list=Psi_sa_list,
    #     terminal_mask_list=[mdp["terminal"] for mdp in mdps],
    #     normalize=True,
    #     n_jobs=args.n_jobs,
    # )

    # U_demo = np.vstack([c for env in U_demo_per_env for c in env])
    # U_demo_unique = remove_redundant_constraints(U_demo)

    # -------------------------------------------------------------------------
    # 4) Feedback Atom Generation
    # -------------------------------------------------------------------------
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

    atoms_per_env = generate_candidate_atoms_for_scot_minigrid(
        mdps=mdps,
        pi_list=pi_list,
        spec=GEN_SPEC,
        enumerate_states=enumerate_states,
        max_horizon=400,
    )


    U_atoms_per_env = constraints_from_atoms_multi_env(
        atoms_per_env=atoms_per_env,
        mdps=mdps,
        Psi_sa_list=Psi_sa_list,
        terminal_mask_list=[mdp["terminal"] for mdp in mdps],
        normalize=True,
        n_jobs=args.n_jobs,
    )

    U_atoms_flat = [c for env in U_atoms_per_env for atom_cs in env for c in atom_cs]
    U_atoms_unique = remove_redundant_constraints(np.vstack(U_atoms_flat))

    # U_universal = remove_redundant_constraints(
    #     np.vstack([U_demo_unique, U_atoms_unique])
    # )
    U_universal = remove_redundant_constraints(U_atoms_unique)


    # -------------------------------------------------------------------------
    # 5) Two-Stage SCOT
    # -------------------------------------------------------------------------
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


    # -------------------------------------------------------------------------
    # 6) Reward Learning (BIRL)
    # -------------------------------------------------------------------------
    Q_learned, w_map = birl_atomic_to_Q_and_wmap(
        mdps, atoms_flat, args, enabled_feedback
    )

    reg_scot = regrets_from_Q(mdps, Q_learned, theta_true)

    # -------------------------------------------------------------------------
    # 7) Baselines
    # -------------------------------------------------------------------------
    k_atoms = len(out["chosen"])
    used_envs = sorted(set(out["selected_mdps"]))
    n_mdps_to_pick = len(used_envs)

    with ProcessPoolExecutor() as ex:
        reg_rand = list(ex.map(
            random_atom_trial,
            [
                (t, mdps, atoms_per_env, k_atoms, args, enabled_feedback)
                for t in range(args.random_trials)
            ]
        ))

    with ProcessPoolExecutor() as ex:
        reg_rand_mdp = list(ex.map(
            random_mdp_scot_trial,
            [
                (t, mdps, atoms_per_env, U_atoms_per_env,
                 U_universal, n_mdps_to_pick,
                 k_atoms, args, enabled_feedback)
                for t in range(args.random_trials)
            ]
        ))

    reg_rand = np.vstack(reg_rand)
    reg_rand_mdp = np.vstack(reg_rand_mdp)

    with ProcessPoolExecutor() as ex:
        reg_same_mdp_rand = list(ex.map(
            same_mdp_random_atom_trial,
            [
                (
                    t,
                    mdps,
                    atoms_per_env,
                    used_envs,   # SCOT-selected MDPs
                    k_atoms,
                    args,
                    enabled_feedback,
                )
                for t in range(args.random_trials)
            ]
        ))

    reg_same_mdp_rand = np.vstack(reg_same_mdp_rand)


    # -------------------------------------------------------------------------
    # 8) Save Results
    # -------------------------------------------------------------------------
    
    # Two-stage coverage
    # ----------------------------
    cov_ts = coverage_report_key_based(
        U_universal=U_universal,
        U_per_env_envlevel=U_atoms_envlevel,
        selected_envs=out["selected_mdps"],
    )

    ts_n_constraints = cov_ts["covered_by_selected"]
    ts_coverage = cov_ts["coverage_frac_selected"]

    # ----------------------------
    # Random-Atom coverage per trial
    # ----------------------------
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

    # ----------------------------
    # Random-MDP→Greedy coverage per trial
    # ----------------------------
    rand_mdp_constraint_counts = []
    rand_mdp_coverages = []
    rand_mdp_counts = []

    for t in range(args.random_trials):
        rng = np.random.default_rng(args.seed + t)

        selected_envs = rng.choice(
            len(mdps),
            size=len(out["selected_mdps"]),
            replace=False,
        )

        cov_rand_mdp = coverage_report_key_based(
            U_universal=U_universal,
            U_per_env_envlevel=U_atoms_envlevel,
            selected_envs=selected_envs,
        )

        rand_mdp_constraint_counts.append(cov_rand_mdp["covered_by_selected"])
        rand_mdp_coverages.append(cov_rand_mdp["coverage_frac_selected"])
        rand_mdp_counts.append(len(selected_envs))



    # ----------------------------
    # Same-MDP Random-Atom coverage per trial
    # ----------------------------
    same_mdp_rand_constraint_counts = []
    same_mdp_rand_coverages = []

    for t in range(args.random_trials):
        rng = np.random.default_rng(args.seed + t)

        pool = [
            (env_idx, atom)
            for env_idx in used_envs
            for atom in atoms_per_env[env_idx]
        ]

        k = min(len(out["chosen"]), len(pool))
        idxs = rng.choice(len(pool), size=k, replace=False)
        chosen = [pool[i] for i in idxs]

        cov_same = coverage_report_key_based(
            U_universal=U_universal,
            U_per_env_envlevel=U_atoms_envlevel,
            selected_envs=used_envs,
        )

        same_mdp_rand_constraint_counts.append(cov_same["covered_by_selected"])
        same_mdp_rand_coverages.append(cov_same["coverage_frac_selected"])

    results = {
        # ============================================================
        # Core Regret (per method)
        # ============================================================
        "methods": {

            # --------------------------------------------------------
            # Two-Stage SCOT
            # --------------------------------------------------------
            "two_stage": {
                "regret": reg_scot.tolist(),
                "mean_regret": float(np.mean(reg_scot)),

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

            # --------------------------------------------------------
            # Same-MDP Random
            # --------------------------------------------------------
            "same_mdp_random": {
                "regret": reg_same_mdp_rand.tolist(),
                "mean_regret": float(np.mean(reg_same_mdp_rand)),

                "selection_stats": {
                    "mdp_counts": None,  # not applicable
                },

                "constraint_stats": {
                    "constraint_counts": same_mdp_rand_constraint_counts,
                    "coverages": same_mdp_rand_coverages,
                    "mean_unique_constraints": float(np.mean(same_mdp_rand_constraint_counts)),
                    "mean_coverage": float(np.mean(same_mdp_rand_coverages)),
                },
            },

            # --------------------------------------------------------
            # Random (global random atoms)
            # --------------------------------------------------------
            "random": {
                "regret": reg_rand.tolist(),
                "mean_regret": float(np.mean(reg_rand)),

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

            # --------------------------------------------------------
            # Random MDP + SCOT within MDP
            # --------------------------------------------------------
            "random_mdp_scot": {
                "regret": reg_rand_mdp.tolist(),
                "mean_regret": float(np.mean(reg_rand_mdp)),

                "selection_stats": {
                    "mdp_counts": rand_mdp_counts,
                    "mean_mdp_count": float(np.mean(rand_mdp_counts)),
                },

                "constraint_stats": {
                    "constraint_counts": rand_mdp_constraint_counts,
                    "coverages": rand_mdp_coverages,
                    "mean_unique_constraints": float(np.mean(rand_mdp_constraint_counts)),
                    "mean_coverage": float(np.mean(rand_mdp_coverages)),
                },
            },
        },

        # ============================================================
        # Universal Constraint Diagnostics
        # ============================================================
        "universal_constraints": {
            #"U_demo_unique": int(len(U_demo_unique)),
            "U_atoms_unique": int(len(U_atoms_unique)),
            "U_union_unique": int(len(U_universal)),
            #"atom_implied_unique": int(len(U_universal) - len(U_demo_unique)),
        },

        # ============================================================
        # Experiment Config (Reproducibility)
        # ============================================================
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
            "birl": {
                "beta": args.beta,
                "samples": args.samples,
                "stepsize": args.stepsize,
            },
        },
    }



    os.makedirs(args.result_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(args.result_dir, f"minigrid_run_{timestamp}.json")

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

    parser.add_argument("--total_budget", type=int, default=10000)
    parser.add_argument("--alloc_method",
                        type=str,
                        default="uniform",
                        choices=["uniform", "dirichlet"])
    parser.add_argument("--alloc", type=float, default=None)

    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--stepsize", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=1.0)

    parser.add_argument("--random_trials", type=int, default=10)
    parser.add_argument("--result_dir", type=str, default="results_minigrid")

    args = parser.parse_args()

    if args.alloc_method != "uniform" and args.alloc is None:
        args.alloc = 0.5

    main(args)
