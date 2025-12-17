import numpy as np
import sys
import os

import numpy as np
import os
import sys

# ----------------------------
# Project imports
# ----------------------------
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)
# ------------------------------------------------------------
# Imports from your repo
# ------------------------------------------------------------
from mdp.gridworld_env_layout import GridWorldMDPFromLayoutEnv
from utils import (
    generate_random_gridworld_envs,
    parallel_value_iteration,
    compute_successor_features_family,
    derive_constraints_from_q_family,
    derive_constraints_from_atoms,
    generate_candidate_atoms_for_scot,
    remove_redundant_constraints,
)
from teaching.two_stage_scot import two_stage_scot_no_cost
from teaching import scot_greedy_family_atoms_tracked


# ============================================================
# Pretty logging
# ============================================================

def log(msg):
    print(msg, flush=True)


def summarize(title, values):
    log(
        f"{title}: min / mean / max = "
        f"{min(values)} / {np.mean(values):.2f} / {max(values)}"
    )


# ============================================================
# MAIN TEST
# ============================================================

def run_two_stage_scot_test():
    log("\n==============================================")
    log(" TWO-STAGE SCOT — FULL DIAGNOSTIC TEST")
    log("==============================================\n")

    # --------------------------------------------------------
    # 1. Small test configuration
    # --------------------------------------------------------
    n_envs = 5
    mdp_size = 4
    feature_dim = 3
    seed = 0

    rng = np.random.default_rng(seed)
    W_TRUE = rng.normal(size=feature_dim)
    W_TRUE /= np.linalg.norm(W_TRUE)

    log(f"[SETUP] n_envs={n_envs}, grid={mdp_size}x{mdp_size}, d={feature_dim}")
    log(f"[SETUP] W_TRUE = {W_TRUE}\n")

    # --------------------------------------------------------
    # 2. Generate environments
    # --------------------------------------------------------
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
        p_color_range={c: (0.3, 0.7) for c in color_to_feature_map},
        terminal_policy=dict(kind="random_k", k_min=1, k_max=1, p_no_terminal=0.0),
        gamma_range=(0.99, 0.99),
        noise_prob_range=(0.0, 0.0),
        w_mode="fixed",
        W_fixed=W_TRUE,
        seed=seed,
        GridEnvClass=GridWorldMDPFromLayoutEnv,
    )

    # --------------------------------------------------------
    # 3. Value iteration
    # --------------------------------------------------------
    Q_list = parallel_value_iteration(envs, epsilon=1e-10)

    # --------------------------------------------------------
    # 4. Successor features
    # --------------------------------------------------------
    SFs = compute_successor_features_family(
        envs,
        Q_list,
        convention="entering",
        zero_terminal_features=True,
        tol=1e-10,
        max_iters=10000,
    )

    # --------------------------------------------------------
    # 5. Q-based constraints
    # --------------------------------------------------------
    U_per_env_q, U_q_global = derive_constraints_from_q_family(
        SFs,
        Q_list,
        envs,
        skip_terminals=True,
        normalize=True,
    )

    # --------------------------------------------------------
    # 6. Candidate atoms
    # --------------------------------------------------------
    candidates_per_env = generate_candidate_atoms_for_scot(
        envs,
        Q_list,
        use_q_demos=False,
        num_q_rollouts_per_state=0,
        q_demo_max_steps=0,
        use_pairwise=True,
        use_estop=False,
        use_improvement=False,
        n_pairwise=5000,
        n_estops=0,
        n_improvements=0,
    )
    #print("candidates_per_env")
    #print(candidates_per_env)

    # --------------------------------------------------------
    # 7. Atom constraints
    # --------------------------------------------------------
    U_per_env_atoms, U_atoms_global = derive_constraints_from_atoms(
        candidates_per_env,
        SFs,
        envs,
    )

    # --------------------------------------------------------
    # 8. Universal constraint set
    # --------------------------------------------------------
    U_universal = remove_redundant_constraints(
        np.vstack([U_q_global, U_atoms_global]),
        epsilon=1e-4,
    )
    # U_universal = remove_redundant_constraints(
    #     U_q_global,
    #     epsilon=1e-4,
    # )

    log(f"[UNIVERSE] |U| = {len(U_universal)}\n")

    # --------------------------------------------------------
    # 9. Per-env constraint stats
    # --------------------------------------------------------
    counts_q = [len(H) for H in U_per_env_q]
    counts_atoms = [len(H) for H in U_per_env_atoms]
    counts_total = [
        len(Hq) + len(Ha)
        for Hq, Ha in zip(U_per_env_q, U_per_env_atoms)
    ]

    log("[STATS] Per-MDP constraint counts:")
    for i in range(n_envs):
        log(
            f"  env {i}: Q={counts_q[i]}, "
            f"Atoms={counts_atoms[i]}, "
            f"Total={counts_total[i]}"
        )

    summarize("Q-only", counts_q)
    summarize("Atom-only", counts_atoms)
    summarize("Combined", counts_total)
    log("")

    # --------------------------------------------------------
    # 10. TWO-STAGE SCOT
    # --------------------------------------------------------
    log("[RUN] TWO-STAGE SCOT\n")

    two_stage = two_stage_scot_no_cost(
        U_universal=U_universal,
        U_per_env_atoms=U_per_env_atoms,
        U_per_env_q=U_per_env_q,
        candidates_per_env=candidates_per_env,
        SFs=SFs,
        envs=envs,
    )

    log(f"  Stage-1 selected MDPs: {two_stage['selected_mdps']}")
    log(f"  Stage-2 activated MDPs: {two_stage['activated_envs']}")
    log(f"  #atoms selected: {len(two_stage['chosen_atoms'])}\n")

    # --------------------------------------------------------
    # 11. FLAT SCOT
    # --------------------------------------------------------
    log("[RUN] FLAT SCOT\n")

    chosen_flat, flat_stats, _ = scot_greedy_family_atoms_tracked(
        U_universal,
        candidates_per_env,
        SFs,
        envs,
    )

    flat_envs = sorted({i for i, _ in chosen_flat})

    log(f"  Flat SCOT activated MDPs: {flat_envs}")
    log(f"  Flat SCOT #atoms selected: {len(chosen_flat)}\n")

    # --------------------------------------------------------
    # 12. Summary
    # --------------------------------------------------------
    log("==============================================")
    log(" SUMMARY")
    log("==============================================")
    log(f"  Universal constraints: {len(U_universal)}")
    log(f"  Two-stage activated envs: {len(two_stage['activated_envs'])}")
    log(f"  Flat SCOT activated envs: {len(flat_envs)}")
    log(f"  Two-stage atoms: {len(two_stage['chosen_atoms'])}")
    log(f"  Flat SCOT atoms: {len(chosen_flat)}")
    log("==============================================\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_two_stage_scot_test()
