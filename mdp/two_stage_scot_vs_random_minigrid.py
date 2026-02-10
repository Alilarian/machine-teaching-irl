# =============================================================================
# Two-Stage SCOT vs Random (GLOBAL POOL) — FULL EXPERIMENT (KEY-BASED)
# =============================================================================

import os
import sys
import time
import numpy as np

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

# ---------------------------
# Imports from your codebase
# ---------------------------
from utils.feedback_budgeting_minigrid import GenerationSpec_minigrid, DemoSpec_minigrid
from utils.minigrid_lava_generator import generate_lavaworld, enumerate_states
from utils import (
    value_iteration_next_state_multi,
    compute_successor_features_multi,
    generate_demos_from_policies_multi,
    constraints_from_demos_next_state_multi,
    remove_redundant_constraints,
    FeedbackSpec_minigrid,
    generate_candidate_atoms_for_scot_minigrid,
    constraints_from_atoms_multi_env,
)

# ---------------------------
# KEY-BASED Two-Stage SCOT
# ---------------------------
from teaching.two_stage_scot_minigrid import (
    two_stage_scot,
    make_key_for,
)

# =====================================================
# Pipeline configuration
# =====================================================
SEED = 124
N_ENVS = 5
GRID_SIZE = 10
GAMMA = 0.99
N_JOBS = None  # use all cores

# -----------------------------------------------------
# Atom generation spec
# -----------------------------------------------------
GEN_SPEC = GenerationSpec_minigrid(
    seed=SEED,
    
    demo=DemoSpec_minigrid(
    enabled=True,
    env_fraction=1,
    state_fraction=1

    ),

    # pairwise=FeedbackSpec_minigrid(
    #     enabled=True,
    #     total_budget=50000,
    #     alloc_method="uniform",
    #     alloc_params={},
    # ),
    estop=None,
    improvement=None,
)

# =====================================================
# Helper: KEY-based coverage report (matches Stage-1 & Stage-2)
# =====================================================
def coverage_report_key_based(
    U_universal,
    U_per_env_envlevel,
    selected_envs,
    *,
    normalize=True,
    round_decimals=12,
):
    """
    Coverage report in canonical KEY space (the same identity used by key-based SCOT).
    """
    key_for = make_key_for(normalize=normalize, round_decimals=round_decimals)

    # Universe keys
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
            H = np.asarray(H, dtype=float)
            if H.ndim == 1:
                H = H[None, :]
            for row in H:
                k = key_for(row)
                uid = key_to_uid.get(k, None)
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

# =====================================================
# MAIN PIPELINE
# =====================================================
def main():
    rng = np.random.default_rng(SEED)

    print("\n==============================")
    print("1) Generating LavaWorld envs")
    print("==============================")

    envs, mdps, meta = generate_lavaworld(
        n_envs=N_ENVS,
        size=GRID_SIZE,
        seed=SEED,
        gamma=GAMMA,
    )

    print(f"Generated {len(mdps)} environments")
    print("Sample goal positions:", meta["goals"][: min(3, len(meta.get('goals', [])))])

    # --------------------------------------------------
    print("\n==============================")
    print("2) Value Iteration (parallel)")
    print("==============================")

    theta_true = mdps[0]["true_w"]

    V_list, Q_list, pi_list = value_iteration_next_state_multi(
        mdps=mdps,
        theta=theta_true,
        gamma=GAMMA,
        n_jobs=N_JOBS,
    )

    print("VI done.")
    print("Sample policy actions:", pi_list[0][:10])

    # --------------------------------------------------
    print("\n==============================")
    print("3) Successor Features")
    print("==============================")

    Psi_sa_list, Psi_s_list = compute_successor_features_multi(
        mdps=mdps,
        Q_list=Q_list,
        gamma=GAMMA,
        n_jobs=N_JOBS,
    )

    print("SF computed.")
    print("Psi_sa shape env0:", Psi_sa_list[0].shape)

    # --------------------------------------------------
    print("\n==============================")
    print("4) Optimal demos (all states)")
    print("==============================")

    demos_list = generate_demos_from_policies_multi(
        mdps=mdps,
        pi_list=pi_list,
        n_jobs=N_JOBS,
    )

    print("Demos generated.")
    print("Demo count per env:", [len(d) for d in demos_list])

    # --------------------------------------------------
    print("\n==============================")
    print("5) Constraints from demos")
    print("==============================")

    U_q_per_env = constraints_from_demos_next_state_multi(
        demos_list=demos_list,
        Psi_sa_list=Psi_sa_list,
        terminal_mask_list=[mdp["terminal"] for mdp in mdps],
        normalize=True,
        n_jobs=N_JOBS,
    )

    U_q = np.vstack([c for env in U_q_per_env for c in env]) if len(U_q_per_env) else np.zeros((0, Psi_s_list[0].shape[1]))
    print(f"Total demo constraints: {len(U_q)}")

    # --------------------------------------------------
    print("\n==============================")
    print("6) Candidate atom generation")
    print("==============================")

    atoms_per_env = generate_candidate_atoms_for_scot_minigrid(
        mdps=mdps,
        pi_list=pi_list,
        spec=GEN_SPEC,
        enumerate_states=enumerate_states,
    )

    atom_counts = [len(a) for a in atoms_per_env]
    print("Atoms per env:", atom_counts)
    print("Total atoms:", sum(atom_counts))

    # --------------------------------------------------
    print("\n==============================")
    print("7) Constraints from atoms (STRICT per-atom lists)")
    print("==============================")

    U_atoms_per_env = constraints_from_atoms_multi_env(
        atoms_per_env=atoms_per_env,
        Psi_s_list=Psi_s_list,
        idx_of_list=[mdp["idx_of"] for mdp in mdps],
        gamma=GAMMA,
    )

    print("U_atoms_per_env")
    print(U_atoms_per_env)

    # Sanity check: strict alignment must hold
    for e in range(len(atoms_per_env)):
        if len(atoms_per_env[e]) != len(U_atoms_per_env[e]):
            raise RuntimeError(
                f"Env {e}: atoms/constraints mismatch "
                f"({len(atoms_per_env[e])} atoms vs {len(U_atoms_per_env[e])} constraint-lists)"
            )

    # Env-level aggregation for Stage-1 and coverage report (NO numeric dedup here)
    d = Psi_s_list[0].shape[1]
    U_atoms_envlevel = []
    for env_constraints_per_atom in U_atoms_per_env:
        flat = []
        for atom_cs in env_constraints_per_atom:
            for c in atom_cs:
                flat.append(np.asarray(c, dtype=float))
        if len(flat):
            U_atoms_envlevel.append(np.vstack(flat))
        else:
            U_atoms_envlevel.append(np.zeros((0, d)))

    # Flatten for logging only
    U_atoms_flat = [c for env in U_atoms_per_env for atom_cs in env for c in atom_cs]
    U_atoms = np.vstack(U_atoms_flat) if len(U_atoms_flat) else np.zeros((0, d))
    print("|U_atoms| raw =", len(U_atoms))

    # --------------------------------------------------
    print("\n==============================")
    print("8) Deduplication (building universal set)")
    print("==============================")

    # Note: remove_redundant_constraints is numeric; acceptable for making U_universal compact.
    # SCOT itself uses KEY identity, so any remaining numeric duplicates won't break correctness.
    U_atom_unique = remove_redundant_constraints(U_atoms) if len(U_atoms) else np.zeros((0, d))
    U_q_unique = remove_redundant_constraints(U_q) if len(U_q) else np.zeros((0, d))

    U_union_unique = remove_redundant_constraints(
        np.vstack([U_q_unique, U_atom_unique])
    ) if (len(U_q_unique) + len(U_atom_unique)) else np.zeros((0, d))

    print(f"|U_q| raw            = {len(U_q)}")
    print(f"|U_q| unique         = {len(U_q_unique)}")
    print(f"|U_atoms| raw        = {len(U_atoms)}")
    print(f"|U_atoms| unique     = {len(U_atom_unique)}")
    print(f"|U_q ∪ U_atoms| uniq = {len(U_union_unique)}")
    print(f"Atom-implied uniques = {max(len(U_union_unique) - len(U_q_unique), 0)}")

    # --------------------------------------------------
    print("\n==============================")
    print("9) Final universal set ready")
    print("==============================")

    U_universal = np.asarray(U_union_unique, dtype=float)
    if U_universal.ndim == 1:
        U_universal = U_universal[None, :]
    print("Final |U| =", len(U_universal))

    # ----------------------------------------------
    print("\n==============================")
    print("10) Two-Stage SCOT (KEY-BASED, NO LEAK)")
    print("==============================")

    two_stage_out = two_stage_scot(
        U_universal=U_universal,
        U_per_env_atoms_envlevel=U_atoms_envlevel,     # Stage-1 env-level
        constraints_per_env_per_atom=U_atoms_per_env,  # Stage-2 per-atom (STRICT)
        candidates_per_env=atoms_per_env,
        normalize=True,
        round_decimals=12,
    )

    selected_mdps = two_stage_out["selected_mdps"]
    chosen = two_stage_out["chosen"]
    activated_envs = two_stage_out["activated_envs"]

    cov = coverage_report_key_based(
        U_universal=U_universal,
        U_per_env_envlevel=U_atoms_envlevel,
        selected_envs=selected_mdps,
        normalize=True,
        round_decimals=12,
    )

    # --------------------------------------------------
    print("\n--- Two-Stage Summary ---")
    print(f"Selected MDPs (Stage-1): {len(selected_mdps)} / {len(mdps)}")
    print("Selected MDP ids:", selected_mdps)
    print(f"Activated envs (Stage-2): {len(activated_envs)}")
    print("Activated env ids:", activated_envs)
    print(f"Selected atoms (Stage-2): {len(chosen)}")
    print(f"Waste (selected - activated): {two_stage_out['waste']}")

    print("\n--- Coverage (KEY-SPACE) ---")
    print(f"Universe size (unique keys): {cov['universe_size']}")
    print(f"Covered by selected envs:    {cov['covered_by_selected']} "
          f"({cov['coverage_frac_selected']:.3f})")
    print(f"Covered by ALL envs:         {cov['covered_by_all_envs']} "
          f"({cov['coverage_frac_all_envs']:.3f})")

    s2_final_cov = two_stage_out["s2_stats"].get("final_coverage", None)
    if s2_final_cov is not None:
        print(f"\nStage-2 reported final coverage (key-space): {s2_final_cov}")

    print("\nPipeline finished successfully.\n")


# =====================================================
if __name__ == "__main__":
    main()
