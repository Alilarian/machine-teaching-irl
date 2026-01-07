import numpy as np
from teaching import scot_greedy_family_atoms_tracked

# ------------------------------------------------------------
# Stage 0: Build MDP-level coverage
# ------------------------------------------------------------

def build_mdp_coverage_from_constraints(U_per_env, U_universal, epsilon=1e-4):
    """
    Maps which indices of the Universal set are covered by each MDP.
    """
    mdp_cov = []
    for H_k in U_per_env:
        indices = set()
        if len(H_k) > 0:
            for row in H_k:
                diff = np.linalg.norm(U_universal - row, axis=1)
                matches = np.where(diff < epsilon)[0]
                indices.update(matches.tolist())
        mdp_cov.append(indices)
    return mdp_cov

# ------------------------------------------------------------
# Stage 1: Unweighted greedy MDP selection
# ------------------------------------------------------------

def greedy_select_mdps_unweighted(mdp_cov, universe_size):
    """
    Greedy set cover over MDPs with NO cost.
    Selects the MDP that covers the largest number of uncovered constraints.
    """
    universe = set(range(universe_size))
    covered = set()
    selected = []

    # Metrics
    s1_iterations = 0
    s1_shallow_checks = 0

    while covered != universe:
        best_gain = -1
        best_k = None
        best_new = None
        s1_iterations += 1

        for k, cov_k in enumerate(mdp_cov):
            if k in selected:
                continue

            s1_shallow_checks += 1
            new_elements = cov_k - covered
            gain = len(new_elements)

            if gain > best_gain:
                best_gain = gain
                best_k = k
                best_new = new_elements

        if best_k is None or best_gain == 0:
            break

        selected.append(best_k)
        covered |= best_new

    return selected, {
        "s1_iterations": s1_iterations,
        "s1_shallow_checks": s1_shallow_checks,
    }

# ------------------------------------------------------------
# Two-stage SCOT (cost-free)
# ------------------------------------------------------------

def two_stage_scot(
    *,
    U_universal,
    U_per_env_atoms,
    U_per_env_q,
    candidates_per_env,
    SFs,
    envs,
):
    # --------------------------------------------------------
    # Stage 0: Constraint aggregation per MDP (ROBUST)
    # --------------------------------------------------------
    n_envs = len(envs)
    
    d = np.array(U_universal).shape[1]

    U_per_env = []

    for k in range(n_envs):
        # Handle possibly-missing atom constraints
        if U_per_env_atoms is not None and k < len(U_per_env_atoms):
            H_a = U_per_env_atoms[k]
        else:
            H_a = np.zeros((0, d))

        # Handle possibly-missing Q constraints
        if U_per_env_q is not None and k < len(U_per_env_q):
            H_q = U_per_env_q[k]
        else:
            H_q = np.zeros((0, d))

        # Combine safely
        if len(H_a) > 0 and len(H_q) > 0:
            combined = np.vstack([H_a, H_q])
        elif len(H_a) > 0:
            combined = H_a
        elif len(H_q) > 0:
            combined = H_q
        else:
            combined = np.zeros((0, d))

        U_per_env.append(combined)

    mdp_cov = build_mdp_coverage_from_constraints(U_per_env, U_universal)

    # --------------------------------------------------------
    # Stage 1: Unweighted MDP pool selection
    # --------------------------------------------------------
    selected_mdps, s1_stats = greedy_select_mdps_unweighted(
        mdp_cov,
        len(U_universal),
    )

    # --------------------------------------------------------
    # Stage 2: Atomic SCOT restricted to selected MDPs
    # --------------------------------------------------------
    pool_atoms = [candidates_per_env[k] for k in selected_mdps]
    pool_SFs   = [SFs[k] for k in selected_mdps]
    pool_envs  = [envs[k] for k in selected_mdps]

    chosen_local, pool_stats, _ = scot_greedy_family_atoms_tracked(
        U_universal,
        pool_atoms,
        pool_SFs,
        pool_envs,
    )

    # Map back to global MDP indices
    chosen_global = []
    for local_env_idx, atom in chosen_local:
        global_env_idx = selected_mdps[local_env_idx]
        chosen_global.append((global_env_idx, atom))

    activated_envs = sorted({k for k, _ in chosen_global})
    waste = len(selected_mdps) - len(activated_envs)

    return {
        "chosen": chosen_global,
        "s1_iterations": s1_stats["s1_iterations"],
        "s1_checks": s1_stats["s1_shallow_checks"],
        "s2_iterations": len(chosen_global),
        #"s2_deep_checks": pool_stats["total_deep_atom_evals"],
        "activated_envs": activated_envs,
        "waste": waste,
    }