
# # =============================================================================
# # Two-Stage SCOT + Reward Learning (MultiEnvAtomicBIRL) — MiniGrid LavaWorld
# # =============================================================================

# import os
# import sys
# import numpy as np

# # -----------------------------------------------------------------------------
# # Path setup
# # -----------------------------------------------------------------------------
# module_path = os.path.abspath(os.path.join(".."))
# if module_path not in sys.path:
#     sys.path.append(module_path)

# # -----------------------------------------------------------------------------
# # Imports (your existing pipeline pieces)
# # -----------------------------------------------------------------------------
# from utils.feedback_budgeting_minigrid import (
#     GenerationSpec_minigrid,
#     DemoSpec_minigrid,
#     FeedbackSpec_minigrid,
# )
# from utils.minigrid_lava_generator import generate_lavaworld, enumerate_states
# from utils import (
#     value_iteration_next_state_multi,
#     compute_successor_features_multi,
#     generate_demos_from_policies_multi,
#     constraints_from_demos_next_state_multi,
#     generate_candidate_atoms_for_scot_minigrid,
#     constraints_from_atoms_multi_env,
#     remove_redundant_constraints,

# )
# from teaching.two_stage_scot_minigrid import (
#     two_stage_scot,
#     make_key_for,
# )
# # -----------------------------------------------------------------------------
# # Import the MiniGrid BIRL you created
# # Change this path to wherever you put the class file
# # -----------------------------------------------------------------------------
# from reward_learning.multi_env_atomic_birl_minigrid import MultiEnvAtomicBIRL_MiniGrid


# # -----------------------------------------------------------------------------
# # Configuration
# # -----------------------------------------------------------------------------
# SEED      = 125
# N_ENVS    = 2
# GRID_SIZE = 6
# GAMMA     = 0.99
# N_JOBS    = None  # use all cores

# # -----------------------------------------------------------------------------
# # Atom generation spec (budgeted, selectable)
# # -----------------------------------------------------------------------------
# GEN_SPEC = GenerationSpec_minigrid(
#     seed=SEED,
#     # demo=DemoSpec_minigrid(
#     #     enabled=True,
#     #     env_fraction=1.0,
#     #     state_fraction=1.0,
#     # ),
#     pairwise=FeedbackSpec_minigrid(
#         enabled=True,
#         total_budget=10000,
#         alloc_method="uniform",
#         alloc_params={},
#     ),
#     # estop=None,
#     # improvement=None,
# )

# # -----------------------------------------------------------------------------
# # Coverage helper (KEY-SPACE)
# # -----------------------------------------------------------------------------
# def coverage_report_key_based(
#     U_universal,
#     U_per_env_envlevel,
#     selected_envs,
#     *,
#     normalize=True,
#     round_decimals=12,
# ):
#     key_for = make_key_for(normalize=normalize, round_decimals=round_decimals)

#     key_to_uid = {}
#     for u in U_universal:
#         k = key_for(u)
#         if k not in key_to_uid:
#             key_to_uid[k] = len(key_to_uid)
#     universe = set(key_to_uid.values())

#     def covered_by(env_ids):
#         covered = set()
#         for e in env_ids:
#             H = U_per_env_envlevel[e]
#             if H is None or len(H) == 0:
#                 continue
#             for row in np.atleast_2d(H):
#                 uid = key_to_uid.get(key_for(row))
#                 if uid is not None:
#                     covered.add(uid)
#         return covered

#     cov_selected = covered_by(selected_envs)
#     cov_all = covered_by(range(len(U_per_env_envlevel)))

#     return {
#         "universe_size": len(universe),
#         "covered_by_selected": len(cov_selected),
#         "covered_by_all_envs": len(cov_all),
#         "coverage_frac_selected": len(cov_selected) / max(len(universe), 1),
#         "coverage_frac_all_envs": len(cov_all) / max(len(universe), 1),
#     }


# # -----------------------------------------------------------------------------
# # Helper: flatten SCOT chosen atoms to atoms_flat format
# # -----------------------------------------------------------------------------
# def scot_output_to_atoms_flat(scot_out):
#     """
#     Converts out["chosen"] to atoms_flat = [(env_idx, Atom), ...]
#     Robust to chosen being:
#       - list[Atom]                    (Atom has env_id)
#       - list[(env_id, Atom)]          (already paired)
#       - list[dict] with {"env_id":..,"atom":..} (just in case)
#     """
#     chosen = scot_out["chosen"]
#     atoms_flat = []

#     for item in chosen:
#         if isinstance(item, tuple) and len(item) == 2:
#             env_id, atom = item
#             atoms_flat.append((int(env_id), atom))
#         elif hasattr(item, "env_id"):
#             atoms_flat.append((int(item.env_id), item))
#         elif isinstance(item, dict) and "env_id" in item and "atom" in item:
#             atoms_flat.append((int(item["env_id"]), item["atom"]))
#         else:
#             raise TypeError(f"Unrecognized chosen atom format: {type(item)}")

#     return atoms_flat


# # -----------------------------------------------------------------------------
# # MAIN PIPELINE
# # -----------------------------------------------------------------------------
# def main():
#     rng = np.random.default_rng(SEED)

#     print("\n=== 1) Generating environments ===")
#     envs, mdps, meta = generate_lavaworld(
#         n_envs=N_ENVS,
#         size=GRID_SIZE,
#         seed=SEED,
#         gamma=GAMMA,
#     )

#     phi = mdps[0]["Phi"]
#     print("Phi from MDP")
#     print(len(phi))
#     print(phi.shape)
#     print(phi[0])
    

#     # -------------------------------------------------------------------------
#     print("\n=== 2) Value Iteration (oracle theta_true for teacher side) ===")
#     theta_true = mdps[0]["true_w"]
#     V_list, Q_list, pi_list = value_iteration_next_state_multi(
#         mdps=mdps,
#         theta=theta_true,
#         gamma=GAMMA,
#         n_jobs=N_JOBS,
#     )

#     # -------------------------------------------------------------------------
#     print("\n=== 3) Successor Features (oracle side) ===")
#     Psi_sa_list, Psi_s_list = compute_successor_features_multi(
#         mdps=mdps,
#         Q_list=Q_list,
#         gamma=GAMMA,
#         n_jobs=N_JOBS,
#     )
#     d = Psi_s_list[0].shape[1]

#     print("Shape of Psi_s")
#     print(len(Psi_s_list))
#     print(Psi_s_list[0].shape)

#     # =========================================================================
#     # ORACLE UNIVERSE CONSTRUCTION (FULL DEMOS)
#     # =========================================================================
#     print("\n=== 4) Oracle demo constraints (FULL information) ===")
#     demos_list = generate_demos_from_policies_multi(
#         mdps=mdps,
#         pi_list=pi_list,
#         n_jobs=N_JOBS,
#     )

#     U_demo_per_env = constraints_from_demos_next_state_multi(
#         demos_list=demos_list,
#         Psi_sa_list=Psi_sa_list,
#         terminal_mask_list=[mdp["terminal"] for mdp in mdps],
#         normalize=True,
#         n_jobs=N_JOBS,
#     )

#     U_demo = np.vstack([c for env in U_demo_per_env for c in env]) if len(U_demo_per_env) else np.zeros((0, d))
#     U_demo_unique = remove_redundant_constraints(U_demo)
#     print(f"|U_demo| unique = {len(U_demo_unique)}")

#     # =========================================================================
#     # SELECTABLE ATOMS (BUDGETED)
#     # =========================================================================
#     print("\n=== 5) Candidate atom generation ===")
#     atoms_per_env = generate_candidate_atoms_for_scot_minigrid(
#         mdps=mdps,
#         pi_list=pi_list,
#         spec=GEN_SPEC,
#         enumerate_states=enumerate_states,
#         max_horizon=400,
#     )
#     print("Atoms per env:", [len(a) for a in atoms_per_env])

#     # -------------------------------------------------------------------------
#     print("\n=== 6) Constraints from atoms (STRICT per-atom) ===")
#     # U_atoms_per_env = constraints_from_atoms_multi_env(
#     #     atoms_per_env=atoms_per_env,
#     #     #Psi_s_list=Psi_s_list,
#     #     Psi_s_list=pi_list,
#     #     Psi_sa_list=Psi_sa_list,  # required for demo atoms
#     #     idx_of_list=[mdp["idx_of"] for mdp in mdps],
#     #     terminal_mask_list=[mdp["terminal"] for mdp in mdps],
#     #     gamma=GAMMA,
#     #     normalize=True,
#     # )
#     U_atoms_per_env = constraints_from_atoms_multi_env(
#         atoms_per_env=atoms_per_env,
#         mdps=mdps,
#         Psi_sa_list=Psi_sa_list,  # only needed if demo atoms exist
#         terminal_mask_list=[mdp["terminal"] for mdp in mdps],
#         normalize=True,
#         n_jobs=N_JOBS,
#     )


#     U_atoms_flat = [c for env in U_atoms_per_env for atom_cs in env for c in atom_cs]
#     U_atoms = np.vstack(U_atoms_flat) if len(U_atoms_flat) else np.zeros((0, d))
#     U_atoms_unique = remove_redundant_constraints(U_atoms)
#     print(f"|U_atoms| unique = {len(U_atoms_unique)}")

#     # =========================================================================
#     # UNIVERSAL SET = ORACLE DEMOS ∪ ATOM CONSTRAINTS
#     # =========================================================================
#     print("\n=== 7) Universal constraint set ===")
#     U_universal = remove_redundant_constraints(np.vstack([U_demo_unique, U_atoms_unique]))
#     print(f"|U_universal| = {len(U_universal)}")

#     # =========================================================================
#     # ENV-LEVEL AGGREGATION (FOR STAGE-1)
#     # =========================================================================
#     U_atoms_envlevel = []
#     for env_constraints in U_atoms_per_env:
#         flat = [c for atom_cs in env_constraints for c in atom_cs]
#         U_atoms_envlevel.append(np.vstack(flat) if len(flat) else np.zeros((0, d)))

#     # =========================================================================
#     # TWO-STAGE SCOT
#     # =========================================================================
#     print("\n=== 8) Two-Stage SCOT ===")
#     out = two_stage_scot(
#         U_universal=U_universal,
#         U_per_env_atoms_envlevel=U_atoms_envlevel,
#         constraints_per_env_per_atom=U_atoms_per_env,
#         candidates_per_env=atoms_per_env,
#         normalize=True,
#         round_decimals=12,
#     )

#     cov = coverage_report_key_based(
#         U_universal=U_universal,
#         U_per_env_envlevel=U_atoms_envlevel,
#         selected_envs=out["selected_mdps"],
#     )

#     print("\n--- SCOT Summary ---")
#     print("Selected MDPs:", out["selected_mdps"])
#     print("Activated envs:", out["activated_envs"])
#     print("Selected atoms:", len(out["chosen"]))
#     print("Waste:", out["waste"])
#     print("Coverage (selected):", cov["coverage_frac_selected"])
#     print("Coverage (all envs):", cov["coverage_frac_all_envs"])

#     # =========================================================================
#     # REWARD LEARNING (BIRL) USING SCOT OUTPUT
#     # =========================================================================
#     print("\n=== 9) Reward Learning: MultiEnvAtomicBIRL on SCOT chosen atoms ===")

#     atoms_flat = scot_output_to_atoms_flat(out)


#     birl = MultiEnvAtomicBIRL_MiniGrid(
#         mdps=mdps,
#         atoms_flat=atoms_flat,
#         beta_demo=10.0,
#         beta_pairwise=10.0,
#         beta_estop=10.0,
#         beta_improvement=10.0,
#         epsilon=1e-8,
#     )


#     # Run MCMC (tune these however you want)
#     birl.run_mcmc(
#         samples=4000,
#         stepsize=0.3,
#         normalize=False,
#         seed=None,
#     )

#     w_map = birl.get_map_solution()
#     w_mean = birl.get_mean_solution(burn_frac=0.2, skip_rate=5)

#     # Report
#     print("\n--- Reward Learning Summary ---")
#     print("True w:", mdps[0]["true_w"])
#     print("MAP  w:", w_map/np.linalg.norm(w_map))
#     print("Mean w:", w_mean/np.linalg.norm((w_mean)))
#     print("Accept rate:", birl.accept_rate)

#     # (Optional) quick cosine similarities
#     def cos(a, b):
#         a = np.asarray(a); b = np.asarray(b)
#         na = np.linalg.norm(a); nb = np.linalg.norm(b)
#         if na == 0 or nb == 0:
#             return 0.0
#         return float(a @ b / (na * nb))

#     print("\nCosine(true, MAP):", cos(mdps[0]["true_w"], w_map))
#     print("Cosine(true, mean):", cos(mdps[0]["true_w"], w_mean))

#     print("\nPipeline finished successfully.\n")


# # -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     main()

# =============================================================================
# Two-Stage SCOT + Reward Learning (MultiEnvAtomicBIRL) — MiniGrid LavaWorld
# + Baselines (parallel) + Regret + Result handling
# =============================================================================

import os
import sys
import json
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
# Imports (your existing pipeline pieces)
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
    make_key_for,
    scot_greedy_family_atoms_tracked
)

# -----------------------------------------------------------------------------
# Import the MiniGrid BIRL you created
# -----------------------------------------------------------------------------
from reward_learning.multi_env_atomic_birl_minigrid import MultiEnvAtomicBIRL_MiniGrid



# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SEED      = 125
N_ENVS    = 2
GRID_SIZE = 6
GAMMA     = 0.99
N_JOBS    = None  # use all cores (inside main pipeline pieces)

RANDOM_TRIALS = 10
RESULT_DIR = "results_minigrid"


# -----------------------------------------------------------------------------
# Atom generation spec (budgeted, selectable)
# -----------------------------------------------------------------------------
GEN_SPEC = GenerationSpec_minigrid(
    seed=SEED,
    # demo=DemoSpec_minigrid(
    #     enabled=True,
    #     env_fraction=1.0,
    #     state_fraction=1.0,
    # ),
    pairwise=FeedbackSpec_minigrid(
        enabled=True,
        total_budget=10000,
        alloc_method="uniform",
        alloc_params={},
    ),
    # estop=None,
    # improvement=None,
)

# -----------------------------------------------------------------------------
# Coverage helper (KEY-SPACE)
# -----------------------------------------------------------------------------
def coverage_report_key_based(
    U_universal,
    U_per_env_envlevel,
    selected_envs,
    *,
    normalize=True,
    round_decimals=12,
):
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


# -----------------------------------------------------------------------------
# Helper: flatten SCOT chosen atoms to atoms_flat format
# -----------------------------------------------------------------------------
def scot_output_to_atoms_flat(scot_out):
    """
    Converts out["chosen"] to atoms_flat = [(env_idx, Atom), ...]
    Robust to chosen being:
      - list[Atom]                    (Atom has env_id)
      - list[(env_id, Atom)]          (already paired)
      - list[dict] with {"env_id":..,"atom":..} (just in case)
    """
    chosen = scot_out["chosen"]
    atoms_flat = []

    for item in chosen:
        if isinstance(item, tuple) and len(item) == 2:
            env_id, atom = item
            atoms_flat.append((int(env_id), atom))
        elif hasattr(item, "env_id"):
            atoms_flat.append((int(item.env_id), item))
        elif isinstance(item, dict) and "env_id" in item and "atom" in item:
            atoms_flat.append((int(item["env_id"]), item["atom"]))
        else:
            raise TypeError(f"Unrecognized chosen atom format: {type(item)}")

    return atoms_flat


# =============================================================================
# ADDITIONS: Regret + Baselines + Result handling
# =============================================================================

def regrets_from_Q(mdps, Q_list, theta_true):
    """
    Regret per env:
      mean(V_opt(theta_true)) - mean(V_pi(theta_true))
    where pi is greedy wrt Q_list (learned).
    """
    pi_list = [np.argmax(Q, axis=1) for Q in Q_list]
    reg = expected_value_difference_next_state_multi(
        eval_policies=pi_list,
        mdps=mdps,
        theta=theta_true,
        normalize_with_random_policy=False,
        include_terminal_in_mean=False,
    )
    return np.asarray(reg, dtype=float)


def birl_atomic_to_Q_and_wmap(mdps, atoms_flat, *, seed):
    """
    Run BIRL on atoms_flat and return (Q_list, w_map).
    Note: inside baseline workers we keep VI single-process to avoid nested pools.
    """
    birl = MultiEnvAtomicBIRL_MiniGrid(
        mdps=mdps,
        atoms_flat=atoms_flat,
        beta_demo=10.0,
        beta_pairwise=10.0,
        beta_estop=10.0,
        beta_improvement=10.0,
        epsilon=1e-8,
    )

    birl.run_mcmc(
        samples=4000,
        stepsize=0.3,
        normalize=False,
        seed=None,
    )

    w_map = birl.get_map_solution()

    # IMPORTANT: value_iteration_next_state_multi returns (V_list, Q_list, pi_list)
    V_list, Q_list, pi_list = value_iteration_next_state_multi(
        mdps=mdps,
        theta=w_map,
        gamma=GAMMA,
        n_jobs=1,  # avoid nested multiprocessing in baseline workers
    )

    return Q_list, w_map


def sample_random_atoms_global_pool(candidates_per_env, n_to_pick, seed):
    rng = np.random.default_rng(seed)
    pool = [
        (env_idx, atom)
        for env_idx, atoms in enumerate(candidates_per_env)
        for atom in atoms
    ]
    if len(pool) < n_to_pick:
        raise ValueError(f"Global pool too small: {len(pool)} < {n_to_pick}")
    idxs = rng.choice(len(pool), size=n_to_pick, replace=False)
    return [pool[i] for i in idxs]


def random_atom_trial(args):
    """
    Random Atom baseline:
      choose K atoms uniformly from global pool (across envs), run BIRL, compute regret.
    """
    trial_id, mdps, candidates_per_env, k_atoms, seed = args
    theta_true = mdps[0]["true_w"]

    chosen = sample_random_atoms_global_pool(
        candidates_per_env=candidates_per_env,
        n_to_pick=k_atoms,
        seed=seed + trial_id,
    )

    Q_list, w_map = birl_atomic_to_Q_and_wmap(
        mdps=mdps,
        atoms_flat=chosen,
        seed=seed + 10000 + trial_id,
    )

    reg = regrets_from_Q(mdps, Q_list, theta_true)
    return reg


# def random_mdp_scot_trial(args):
#     """
#     Random-MDP → SCOT baseline:
#       pick M envs at random, run SCOT on that subset (stage-2 only is fine), lift chosen to global env ids,
#       run BIRL, compute regret.
#     """
#     (
#         trial_id,
#         mdps,
#         candidates_per_env,
#         U_universal,
#         n_mdps_to_pick,
#         seed,
#     ) = args

#     theta_true = mdps[0]["true_w"]
#     rng = np.random.default_rng(seed + trial_id)

#     selected_envs = rng.choice(
#         len(mdps),
#         size=n_mdps_to_pick,
#         replace=False,
#     )

#     atoms_subset = [candidates_per_env[i] for i in selected_envs]

#     out = two_stage_scot(
#         U_universal=U_universal,
#         U_per_env_atoms_envlevel=None,
#         constraints_per_env_per_atom=None,
#         candidates_per_env=atoms_subset,
#         normalize=True,
#         round_decimals=12,
#     )

#     # out["chosen"] are in subset index space -> map back to original env ids
#     chosen_atoms = []
#     for env_idx_subset, atom in out["chosen"]:
#         chosen_atoms.append((int(selected_envs[int(env_idx_subset)]), atom))

#     Q_list, w_map = birl_atomic_to_Q_and_wmap(
#         mdps=mdps,
#         atoms_flat=chosen_atoms,
#         seed=seed + 20000 + trial_id,
#     )

#     reg = regrets_from_Q(mdps, Q_list, theta_true)
#     return reg
def random_mdp_scot_trial(args):
    """
    Random-MDP → Greedy-Atoms baseline:

    1) Randomly select M environments
    2) From those envs, run atom-level greedy selection
    3) Force it to pick exactly K atoms (same as two-stage)
    4) Run BIRL
    5) Compute regret
    """
    (
        trial_id,
        mdps,
        candidates_per_env,
        constraints_per_env_per_atom,  # STRICT per-atom constraints
        U_universal,
        n_mdps_to_pick,
        k_atoms,                       # <-- must match two-stage atom count
        seed,
    ) = args

    theta_true = mdps[0]["true_w"]
    rng = np.random.default_rng(seed + trial_id)

    # --------------------------------------------------
    # 1) Randomly select M environments
    # --------------------------------------------------
    selected_envs = rng.choice(
        len(mdps),
        size=n_mdps_to_pick,
        replace=False,
    )

    # Restrict to subset
    atoms_subset = [candidates_per_env[i] for i in selected_envs]
    constraints_subset = [constraints_per_env_per_atom[i] for i in selected_envs]

    # --------------------------------------------------
    # 2) Run greedy atom selection (NOT two-stage)
    # --------------------------------------------------
    greedy_out = scot_greedy_family_atoms_tracked(
        U_universal=U_universal,
        constraints_per_env_per_atom=constraints_subset,
        candidates_per_env=atoms_subset,
        max_atoms=k_atoms,             # <-- force same number as two-stage
        normalize=True,
        round_decimals=12,
    )

    # --------------------------------------------------
    # 3) Map subset env indices back to global indices
    # --------------------------------------------------
    chosen_atoms = []
    for env_idx_subset, atom in greedy_out["chosen"]:
        global_env_idx = int(selected_envs[int(env_idx_subset)])
        chosen_atoms.append((global_env_idx, atom))

    # --------------------------------------------------
    # 4) Run BIRL
    # --------------------------------------------------
    Q_list, w_map = birl_atomic_to_Q_and_wmap(
        mdps=mdps,
        atoms_flat=chosen_atoms,
        seed=seed + 20000 + trial_id,
    )

    # --------------------------------------------------
    # 5) Compute regret
    # --------------------------------------------------
    reg = regrets_from_Q(mdps, Q_list, theta_true)
    return reg



def save_results_json(results, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(result_dir, f"minigrid_scot_birl_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {out_path}")


# -----------------------------------------------------------------------------
# MAIN PIPELINE (UNCHANGED) + additions appended at end
# -----------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)

    print("\n=== 1) Generating environments ===")
    envs, mdps, meta = generate_lavaworld(
        n_envs=N_ENVS,
        size=GRID_SIZE,
        seed=SEED,
        gamma=GAMMA,
    )

    phi = mdps[0]["Phi"]
    print("Phi from MDP")
    print(len(phi))
    print(phi.shape)
    print(phi[0])

    # -------------------------------------------------------------------------
    print("\n=== 2) Value Iteration (oracle theta_true for teacher side) ===")
    theta_true = mdps[0]["true_w"]
    V_list, Q_list, pi_list = value_iteration_next_state_multi(
        mdps=mdps,
        theta=theta_true,
        gamma=GAMMA,
        n_jobs=N_JOBS,
    )

    # -------------------------------------------------------------------------
    print("\n=== 3) Successor Features (oracle side) ===")
    Psi_sa_list, Psi_s_list = compute_successor_features_multi(
        mdps=mdps,
        Q_list=Q_list,
        gamma=GAMMA,
        n_jobs=N_JOBS,
    )
    d = Psi_s_list[0].shape[1]

    print("Shape of Psi_s")
    print(len(Psi_s_list))
    print(Psi_s_list[0].shape)

    # =========================================================================
    # ORACLE UNIVERSE CONSTRUCTION (FULL DEMOS)
    # =========================================================================
    print("\n=== 4) Oracle demo constraints (FULL information) ===")
    demos_list = generate_demos_from_policies_multi(
        mdps=mdps,
        pi_list=pi_list,
        n_jobs=N_JOBS,
    )

    U_demo_per_env = constraints_from_demos_next_state_multi(
        demos_list=demos_list,
        Psi_sa_list=Psi_sa_list,
        terminal_mask_list=[mdp["terminal"] for mdp in mdps],
        normalize=True,
        n_jobs=N_JOBS,
    )

    U_demo = np.vstack([c for env in U_demo_per_env for c in env]) if len(U_demo_per_env) else np.zeros((0, d))
    U_demo_unique = remove_redundant_constraints(U_demo)
    print(f"|U_demo| unique = {len(U_demo_unique)}")

    # =========================================================================
    # SELECTABLE ATOMS (BUDGETED)
    # =========================================================================
    print("\n=== 5) Candidate atom generation ===")
    atoms_per_env = generate_candidate_atoms_for_scot_minigrid(
        mdps=mdps,
        pi_list=pi_list,
        spec=GEN_SPEC,
        enumerate_states=enumerate_states,
        max_horizon=400,
    )
    print("Atoms per env:", [len(a) for a in atoms_per_env])

    # -------------------------------------------------------------------------
    print("\n=== 6) Constraints from atoms (STRICT per-atom) ===")
    U_atoms_per_env = constraints_from_atoms_multi_env(
        atoms_per_env=atoms_per_env,
        mdps=mdps,
        Psi_sa_list=Psi_sa_list,  # only needed if demo atoms exist
        terminal_mask_list=[mdp["terminal"] for mdp in mdps],
        normalize=True,
        n_jobs=N_JOBS,
    )

    U_atoms_flat = [c for env in U_atoms_per_env for atom_cs in env for c in atom_cs]
    U_atoms = np.vstack(U_atoms_flat) if len(U_atoms_flat) else np.zeros((0, d))
    U_atoms_unique = remove_redundant_constraints(U_atoms)
    print(f"|U_atoms| unique = {len(U_atoms_unique)}")

    # =========================================================================
    # UNIVERSAL SET = ORACLE DEMOS ∪ ATOM CONSTRAINTS
    # =========================================================================
    print("\n=== 7) Universal constraint set ===")
    U_universal = remove_redundant_constraints(np.vstack([U_demo_unique, U_atoms_unique]))
    print(f"|U_universal| = {len(U_universal)}")

    # =========================================================================
    # ENV-LEVEL AGGREGATION (FOR STAGE-1)
    # =========================================================================
    U_atoms_envlevel = []
    for env_constraints in U_atoms_per_env:
        flat = [c for atom_cs in env_constraints for c in atom_cs]
        U_atoms_envlevel.append(np.vstack(flat) if len(flat) else np.zeros((0, d)))

    # =========================================================================
    # TWO-STAGE SCOT
    # =========================================================================
    print("\n=== 8) Two-Stage SCOT ===")
    out = two_stage_scot(
        U_universal=U_universal,
        U_per_env_atoms_envlevel=U_atoms_envlevel,
        constraints_per_env_per_atom=U_atoms_per_env,
        candidates_per_env=atoms_per_env,
        normalize=True,
        round_decimals=12,
    )

    cov = coverage_report_key_based(
        U_universal=U_universal,
        U_per_env_envlevel=U_atoms_envlevel,
        selected_envs=out["selected_mdps"],
    )

    print("\n--- SCOT Summary ---")
    print("Selected MDPs:", out["selected_mdps"])
    print("Activated envs:", out["activated_envs"])
    print("Selected atoms:", len(out["chosen"]))
    print("Waste:", out["waste"])
    print("Coverage (selected):", cov["coverage_frac_selected"])
    print("Coverage (all envs):", cov["coverage_frac_all_envs"])

    # =========================================================================
    # REWARD LEARNING (BIRL) USING SCOT OUTPUT
    # =========================================================================
    print("\n=== 9) Reward Learning: MultiEnvAtomicBIRL on SCOT chosen atoms ===")

    atoms_flat = scot_output_to_atoms_flat(out)

    birl = MultiEnvAtomicBIRL_MiniGrid(
        mdps=mdps,
        atoms_flat=atoms_flat,
        beta_demo=10.0,
        beta_pairwise=10.0,
        beta_estop=10.0,
        beta_improvement=10.0,
        epsilon=1e-8,
    )

    # Run MCMC (tune these however you want)
    birl.run_mcmc(
        samples=4000,
        stepsize=0.3,
        normalize=False,
        seed=None,
    )

    w_map = birl.get_map_solution()
    w_mean = birl.get_mean_solution(burn_frac=0.2, skip_rate=5)

    # Report
    print("\n--- Reward Learning Summary ---")
    print("True w:", mdps[0]["true_w"])
    print("MAP  w:", w_map / np.linalg.norm(w_map))
    print("Mean w:", w_mean / np.linalg.norm((w_mean)))
    print("Accept rate:", birl.accept_rate)

    # (Optional) quick cosine similarities
    def cos(a, b):
        a = np.asarray(a); b = np.asarray(b)
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(a @ b / (na * nb))

    print("\nCosine(true, MAP):", cos(mdps[0]["true_w"], w_map))
    print("Cosine(true, mean):", cos(mdps[0]["true_w"], w_mean))

    # =========================================================================
    # 10) REGRET + BASELINES (ADDED)
    # =========================================================================
    print("\n=== 10) Regret computation + baselines ===")

    # Learned Q under MAP theta (greedy policy evaluated under TRUE theta)
    _, Q_learned_list, _ = value_iteration_next_state_multi(
        mdps=mdps,
        theta=w_map,
        gamma=GAMMA,
        n_jobs=N_JOBS,
    )
    reg_scot = regrets_from_Q(mdps, Q_learned_list, theta_true)
    print("SCOT+BIRL regret per env:", reg_scot)
    print("SCOT+BIRL mean regret:", float(np.mean(reg_scot)))

    # Baselines: parallel trials
    k_atoms = len(out["chosen"])
    used_envs = sorted(set(out["selected_mdps"])) if "selected_mdps" in out else sorted({e for e, _ in atoms_flat})
    n_mdps_to_pick = max(1, len(used_envs))

    print(f"\nRunning baselines with RANDOM_TRIALS={RANDOM_TRIALS}")
    print(f"Random-Atom picks K={k_atoms}")
    print(f"Random-MDP-SCOT picks M={n_mdps_to_pick}")

    # Random Atom baseline (parallel)
    with ProcessPoolExecutor() as ex:
        reg_rand = list(ex.map(
            random_atom_trial,
            [
                (t, mdps, atoms_per_env, k_atoms, SEED)
                for t in range(RANDOM_TRIALS)
            ]
        ))
    reg_rand = np.vstack(reg_rand)
    print("Random-Atom mean regret:", float(np.mean(reg_rand)))

    # Random-MDP→SCOT baseline (parallel)
    k_atoms = len(out["chosen"])

    with ProcessPoolExecutor() as ex:
        reg_rand_mdp = list(
            ex.map(
                random_mdp_scot_trial,
                [
                    (
                        t,
                        mdps,
                        atoms_per_env,
                        U_atoms_per_env,    # <-- strict per-atom constraints
                        U_universal,
                        n_mdps_to_pick,
                        k_atoms,            # <-- match two-stage
                        SEED,
                    )
                    for t in range(RANDOM_TRIALS)
                ],
            )
        )

    # =========================================================================
    # 11) SAVE RESULTS (ADDED)
    # =========================================================================
    results = {
        "config": {
            "seed": SEED,
            "n_envs": N_ENVS,
            "grid_size": GRID_SIZE,
            "gamma": GAMMA,
            "n_jobs": N_JOBS,
            "random_trials": RANDOM_TRIALS,
            "pairwise_total_budget": getattr(GEN_SPEC.pairwise, "total_budget", None) if GEN_SPEC.pairwise else None,
        },
        "scot": {
            "selected_mdps": list(out.get("selected_mdps", [])),
            "activated_envs": list(out.get("activated_envs", [])),
            "num_selected_atoms": int(len(out.get("chosen", []))),
            "waste": float(out.get("waste", 0.0)) if out.get("waste", None) is not None else None,
            "coverage": cov,
        },
        "reward_learning": {
            "true_w": theta_true.tolist(),
            "w_map": (w_map / np.linalg.norm(w_map)).tolist() if np.linalg.norm(w_map) > 0 else w_map.tolist(),
            "w_mean": (w_mean / np.linalg.norm(w_mean)).tolist() if np.linalg.norm(w_mean) > 0 else w_mean.tolist(),
            "accept_rate": float(getattr(birl, "accept_rate", np.nan)),
            "cos_true_map": cos(theta_true, w_map),
            "cos_true_mean": cos(theta_true, w_mean),
        },
        "regret": {
            "scot_birl_per_env": reg_scot.tolist(),
            "scot_birl_mean": float(np.mean(reg_scot)),
            "random_atom_trials": reg_rand.tolist(),          # shape [T, E]
            "random_atom_mean": float(np.mean(reg_rand)),
            "random_mdp_scot_trials": reg_rand_mdp.tolist(),  # shape [T, E]
            "random_mdp_scot_mean": float(np.mean(reg_rand_mdp)),
        },
    }

    save_results_json(results, RESULT_DIR)

    print("\nPipeline finished successfully.\n")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
