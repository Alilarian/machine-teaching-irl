# # =============================================================================
# # Two-Stage SCOT vs Random (GLOBAL POOL) — FULL EXPERIMENT (KEY-BASED, ORACLE)
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
# # Imports
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
# # Configuration
# # -----------------------------------------------------------------------------
# SEED      = 125
# N_ENVS    = 2
# GRID_SIZE = 10
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
#         total_budget=50000,
#         alloc_method="uniform",
#         alloc_params={},
# ),
#     #estop=None,
#     #improvement=None,
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

#     # -------------------------------------------------------------------------
#     print("\n=== 2) Value Iteration ===")
#     theta_true = mdps[0]["true_w"]
#     V_list, Q_list, pi_list = value_iteration_next_state_multi(
#         mdps=mdps,
#         theta=theta_true,
#         gamma=GAMMA,
#         n_jobs=N_JOBS,
#     )

#     # -------------------------------------------------------------------------
#     print("\n=== 3) Successor Features ===")
#     Psi_sa_list, Psi_s_list = compute_successor_features_multi(
#         mdps=mdps,
#         Q_list=Q_list,
#         gamma=GAMMA,
#         n_jobs=N_JOBS,
#     )

#     d = Psi_s_list[0].shape[1]

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

#     U_demo = np.vstack(
#         [c for env in U_demo_per_env for c in env]
#     ) if len(U_demo_per_env) else np.zeros((0, d))

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
#     )

#     print("Atoms per env:", [len(a) for a in atoms_per_env])

#     # -------------------------------------------------------------------------
#     print("\n=== 6) Constraints from atoms (STRICT per-atom) ===")

#     U_atoms_per_env = constraints_from_atoms_multi_env(
#         atoms_per_env=atoms_per_env,
#         Psi_s_list=Psi_s_list,
#         Psi_sa_list=Psi_sa_list,  # required for demo atoms
#         idx_of_list=[mdp["idx_of"] for mdp in mdps],
#         terminal_mask_list=[mdp["terminal"] for mdp in mdps],
#         gamma=GAMMA,
#         normalize=True,
#     )

#     # flatten atom constraints
#     U_atoms_flat = [
#         c for env in U_atoms_per_env
#           for atom_cs in env
#           for c in atom_cs
#     ]
#     U_atoms = np.vstack(U_atoms_flat) if len(U_atoms_flat) else np.zeros((0, d))
#     U_atoms_unique = remove_redundant_constraints(U_atoms)

#     print(f"|U_atoms| unique = {len(U_atoms_unique)}")

#     # =========================================================================
#     # UNIVERSAL SET = ORACLE DEMOS ∪ ATOM CONSTRAINTS
#     # =========================================================================
#     print("\n=== 7) Universal constraint set ===")

#     # U_universal = remove_redundant_constraints(
#     #     np.vstack([U_demo_unique, U_atoms_unique])
#     # ) if (len(U_demo_unique) + len(U_atoms_unique)) else np.zeros((0, d))

#     U_universal = np.vstack([U_demo_unique, U_atoms_unique])
 
#     print(f"|U_universal| = {len(U_universal)}")

#     # =========================================================================
#     # ENV-LEVEL AGGREGATION (FOR STAGE-1)
#     # =========================================================================
#     U_atoms_envlevel = []
#     for env_constraints in U_atoms_per_env:
#         flat = [c for atom_cs in env_constraints for c in atom_cs]
#         U_atoms_envlevel.append(
#             np.vstack(flat) if len(flat) else np.zeros((0, d))
#         )

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

#     print("\n--- Summary ---")
#     print("Selected MDPs:", out["selected_mdps"])
#     print("Activated envs:", out["activated_envs"])
#     print("Selected atoms:", len(out["chosen"]))
#     print("Waste:", out["waste"])
#     print("Coverage (selected):", cov["coverage_frac_selected"])
#     print("Coverage (all envs):", cov["coverage_frac_all_envs"])

#     print("\nPipeline finished successfully.\n")

# # -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     main()

# =============================================================================
# Two-Stage SCOT + Reward Learning (MultiEnvAtomicBIRL) — MiniGrid LavaWorld
# =============================================================================

import os
import sys
import numpy as np

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
)
from teaching.two_stage_scot_minigrid import (
    two_stage_scot,
    make_key_for,
)
# -----------------------------------------------------------------------------
# Import the MiniGrid BIRL you created
# Change this path to wherever you put the class file
# -----------------------------------------------------------------------------
from reward_learning.multi_env_atomic_birl_minigrid import MultiEnvAtomicBIRL_MiniGrid


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SEED      = 125
N_ENVS    = 2
GRID_SIZE = 10
GAMMA     = 0.99
N_JOBS    = None  # use all cores

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
        total_budget=5000,
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


# -----------------------------------------------------------------------------
# MAIN PIPELINE
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
    )
    print("Atoms per env:", [len(a) for a in atoms_per_env])

    # -------------------------------------------------------------------------
    print("\n=== 6) Constraints from atoms (STRICT per-atom) ===")
    U_atoms_per_env = constraints_from_atoms_multi_env(
        atoms_per_env=atoms_per_env,
        Psi_s_list=Psi_s_list,
        Psi_sa_list=Psi_sa_list,  # required for demo atoms
        idx_of_list=[mdp["idx_of"] for mdp in mdps],
        terminal_mask_list=[mdp["terminal"] for mdp in mdps],
        gamma=GAMMA,
        normalize=True,
    )

    U_atoms_flat = [c for env in U_atoms_per_env for atom_cs in env for c in atom_cs]
    U_atoms = np.vstack(U_atoms_flat) if len(U_atoms_flat) else np.zeros((0, d))
    U_atoms_unique = remove_redundant_constraints(U_atoms)
    print(f"|U_atoms| unique = {len(U_atoms_unique)}")

    # =========================================================================
    # UNIVERSAL SET = ORACLE DEMOS ∪ ATOM CONSTRAINTS
    # =========================================================================
    print("\n=== 7) Universal constraint set ===")
    U_universal = np.vstack([U_demo_unique, U_atoms_unique])
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

    print(atoms_flat)


    birl = MultiEnvAtomicBIRL_MiniGrid(
        mdps=mdps,
        atoms_flat=atoms_flat,
        beta_demo=10.0,
        beta_pairwise=10.0,
        beta_estop=10.0,
        beta_improvement=10.0,
        gamma=GAMMA,
        epsilon=1e-8,
    )

    # Run MCMC (tune these however you want)
    birl.run_mcmc(
        samples=4000,
        stepsize=0.6,
        normalize=True,
        seed=SEED,
    )

    w_map = birl.get_map_solution()
    w_mean = birl.get_mean_solution(burn_frac=0.2, skip_rate=5)

    # Report
    print("\n--- Reward Learning Summary ---")
    print("True w:", mdps[0]["true_w"])
    print("MAP  w:", w_map)
    print("Mean w:", w_mean)
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

    print("\nPipeline finished successfully.\n")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
