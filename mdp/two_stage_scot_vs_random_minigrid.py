# =============================================================================
# Two-Stage SCOT vs Random (GLOBAL POOL) — FULL EXPERIMENT
# =============================================================================
import argparse
import json
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

from utils.minigrid_lava_generator import generate_lavaworld, enumerate_states
from utils import (
    value_iteration_next_state_multi,
    compute_successor_features_multi,
    generate_demos_from_policies_multi,
    constraints_from_demos_next_state_multi,
    remove_redundant_constraints,
    GenerationSpec,
    DemoSpec,
    FeedbackSpec,
    generate_candidate_atoms_for_scot_minigrid,
    constraints_from_atoms_multi_env,
)

# IMPORTANT: use the no-leak two-stage implementation
# (adjust import path to wherever you placed it)
from teaching.two_stage_scot_minigrid import two_stage_scot_minigrid

# =====================================================
# Pipeline configuration
# =====================================================
SEED = 1230
N_ENVS = 10
GRID_SIZE = 10
GAMMA = 0.99
N_JOBS = None  # use all cores
# -----------------------------------------------------
# Atom generation spec (example – tune freely)
# -----------------------------------------------------
GEN_SPEC = GenerationSpec(
    seed=SEED,
    # demo=DemoSpec(
    #     enabled=True,
    #     env_fraction=1.0,
    #     state_fraction=1,
    # ),
    demo=None,
    pairwise=FeedbackSpec(
        enabled=True,
        total_budget=4000,
        alloc_method="uniform",
        alloc_params={},
    ),
    #pairwise=None,
    estop=None,
    improvement=None,
)

# =====================================================
# Helper: key-based coverage report (matches stage-2)
# =====================================================
def _make_key_for(normalize=True, round_decimals=12):
    def key_for(v):
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        if n == 0.0 or not np.isfinite(n):
            return ("ZERO",)
        vv = (v / n) if normalize else v
        return tuple(np.round(vv, round_decimals))
    return key_for

def coverage_report(U_universal, U_per_env, selected_envs, normalize=True, round_decimals=12):
    """
    Returns coverage stats in the same canonical key space as SCOT.
    """
    key_for = _make_key_for(normalize=normalize, round_decimals=round_decimals)

    # Universe keys
    U_keys = [key_for(u) for u in np.asarray(U_universal)]
    U_keyset = set(U_keys)

    # Coverage for any set of env ids
    def covered_by(env_ids):
        covered = set()
        for k in env_ids:
            H = U_per_env[k]
            if H is None:
                continue
            H = np.asarray(H, dtype=float)
            if H.size == 0:
                continue
            if H.ndim == 1:
                H = H[None, :]
            for row in H:
                covered.add(key_for(row))
        return covered

    cov_selected = covered_by(selected_envs)
    cov_all = covered_by(range(len(U_per_env)))

    # Intersect with universe (should already match, but keep clean)
    cov_selected &= U_keyset
    cov_all &= U_keyset

    return {
        "universe_size_keys": len(U_keyset),
        "covered_by_selected": len(cov_selected),
        "covered_by_all_envs": len(cov_all),
        "coverage_frac_selected": (len(cov_selected) / max(len(U_keyset), 1)),
        "coverage_frac_all_envs": (len(cov_all) / max(len(U_keyset), 1)),
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
    print("Sample goal positions:", meta["goals"][:3])

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

    U_q = np.vstack([c for env in U_q_per_env for c in env])
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
    print("7) Constraints from atoms")
    print("==============================")

    U_atoms_per_env = constraints_from_atoms_multi_env(
        atoms_per_env=atoms_per_env,
        Psi_s_list=Psi_s_list,
        idx_of_list=[mdp["idx_of"] for mdp in mdps],
        gamma=GAMMA,
    )

    # # Env-level aggregation for Stage-1 and coverage_report
    # d = Psi_s_list[0].shape[1]
    # U_atoms_envlevel = []
    # for env in U_atoms_per_env:
    #     flat = [c for atom_cs in env for c in atom_cs]
    #     U_atoms_envlevel.append(np.vstack(flat) if len(flat) else np.zeros((0, d)))

    # Flatten for logging / optional union
    U_atoms_flat = [c for env in U_atoms_per_env for atom_cs in env for c in atom_cs]
    U_atoms = np.vstack(U_atoms_flat) if len(U_atoms_flat) > 0 else None

    print("|U_atoms| raw =", 0 if U_atoms is None else len(U_atoms))

    # --------------------------------------------------
    print("\n==============================")
    print("8) Deduplication")
    print("==============================")

    U_union_unique = remove_redundant_constraints(np.vstack([U_q, U_atoms]))

    U_q_unique = remove_redundant_constraints(U_q)
    #U_union_unique = U_q_unique  # (you chose no-union for now)

    print(f"|U_q| raw            = {len(U_q)}")
    print(f"|U_q| unique         = {len(U_q_unique)}")
    print(f"|U_atoms| raw        = {0 if U_atoms is None else len(U_atoms)}")
    print(f"|U_q ∪ U_atoms| uniq = {len(U_union_unique)}")
    print(f"Atom-implied uniques = {len(U_union_unique) - len(U_q_unique)}")

    # --------------------------------------------------
    print("\n==============================")
    print("9) Final universal set ready")
    print("==============================")

    U_universal = U_union_unique
    print("Final |U| =", len(U_universal))

    # --------------------------------------------------
    print("\n==============================")
    print("10) Two-Stage SCOT (NO LEAK)")
    print("==============================")

    two_stage_out = two_stage_scot_minigrid(
        U_universal=U_universal,
        U_per_env_atoms_envlevel=U_atoms_envlevel,  # Stage-1
        constraints_per_env_atoms=U_atoms_per_env,  # Stage-2
        candidates_per_env=atoms_per_env,
        SFs=Psi_s_list,
        envs=mdps,
        normalize=True,
        round_decimals=12,
        verbose=True,
    )

    selected_mdps = two_stage_out["selected_mdps"]
    chosen = two_stage_out["chosen"]

    cov = coverage_report(
        U_universal=U_universal,
        U_per_env=U_atoms_envlevel,   # env-level
        selected_envs=selected_mdps,
        normalize=True,
        round_decimals=12,
    )


    # Atom stats
    num_selected_mdps = len(selected_mdps)
    num_selected_atoms = len(chosen)
    activated_envs = two_stage_out["activated_envs"]

    print("\n--- Two-Stage Summary ---")
    print(f"Selected MDPs (Stage-1): {num_selected_mdps} / {len(mdps)}")
    print("Selected MDP ids:", selected_mdps)
    print(f"Activated envs (Stage-2): {len(activated_envs)}")
    print("Activated env ids:", activated_envs)
    print(f"Selected atoms (Stage-2): {num_selected_atoms}")
    print(f"Waste (selected - activated): {two_stage_out['waste']}")

    print("\n--- Coverage (key-space) ---")
    print(f"Universe size (unique keys): {cov['universe_size_keys']}")
    print(f"Covered by selected envs:    {cov['covered_by_selected']} "
          f"({cov['coverage_frac_selected']:.3f})")
    print(f"Covered by ALL envs:         {cov['covered_by_all_envs']} "
          f"({cov['coverage_frac_all_envs']:.3f})")

    # If stage-2 provides final coverage in its own stats, print it too
    s2_final_cov = two_stage_out["s2_stats"].get("final_coverage", None)
    if s2_final_cov is not None:
        print(f"\nStage-2 reported final coverage: {s2_final_cov}")

    print("\nPipeline finished successfully.\n")


# =====================================================
if __name__ == "__main__":
    main() 