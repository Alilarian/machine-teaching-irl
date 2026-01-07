import numpy as np
from teaching import scot_greedy_family_atoms_tracked

def build_mdp_metadata(candidates_per_env):
    """
    Computes costs for each MDP based on the number and diversity of atoms.
    """
    metadata = []
    for atoms in candidates_per_env:
        # A 'type' is a category like 'demo', 'estop', etc.
        types = set(a.feedback_type for a in atoms)
        num_types = max(1, len(types))
        num_atoms = len(atoms)
        
        # Heuristic: Cost = (Atoms + Setup) / Diversity
        # Environments with multiple feedback types become 'cheaper' relative to their gain.
        cost = (num_atoms + 1) / num_types
        
        metadata.append({
            "cost": cost,
            "num_atoms": num_atoms,
            "types": types,
            "num_types": num_types
        })
    return metadata

def build_mdp_coverage_from_constraints(U_per_env, U_universal, epsilon=1e-4):
    """Maps which indices of the Universal set are covered by each MDP."""
    mdp_cov = []
    for H_k in U_per_env:
        indices = set()
        if len(H_k) > 0:
            # Vectorized comparison against the universal set
            for row in H_k:
                diff = np.linalg.norm(U_universal - row, axis=1)
                matches = np.where(diff < epsilon)[0]
                indices.update(matches.tolist())
        mdp_cov.append(indices)
    return mdp_cov

def greedy_select_mdps_weighted(mdp_cov, universe_size, mdp_metadata):
    universe = set(range(universe_size))
    covered = set()
    selected = []
    
    # METRICS
    s1_iterations = 0  # Number of MDPs added to the pool
    s1_shallow_checks = 0 # Number of times we looked at an MDP's summary
    
    while covered != universe:
        best_efficiency = -1
        best_k = None
        best_new = None
        s1_iterations += 1

        for k, cov_k in enumerate(mdp_cov):
            if k in selected: continue
            
            s1_shallow_checks += 1 # We are "checking" this MDP summary
            new_elements = cov_k - covered
            gain = len(new_elements)
            
            if gain > 0:
                efficiency = gain / mdp_metadata[k]["cost"]
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_k = k
                    best_new = new_elements

        if best_k is None: break
        selected.append(best_k)
        covered |= best_new

    return selected, {"s1_iterations": s1_iterations, "s1_shallow_checks": s1_shallow_checks}

def two_stage_scot_weighted(
    *,
    U_universal,
    U_per_env_atoms,
    U_per_env_q,
    candidates_per_env,
    SFs,
    envs,
):
    # --- STAGE 0: Constraint Mapping ---
    U_per_env = []
    for H_a, H_q in zip(U_per_env_atoms, U_per_env_q):
        if len(H_a) > 0 and len(H_q) > 0:
            combined = np.vstack([H_a, H_q])
        elif len(H_a) > 0:
            combined = H_a
        elif len(H_q) > 0:
            combined = H_q
        else:
            combined = np.zeros((0, U_universal.shape[1]))
        U_per_env.append(combined)

    mdp_cov = build_mdp_coverage_from_constraints(U_per_env, U_universal)
    mdp_metadata = build_mdp_metadata(candidates_per_env)

    # --- STAGE 1: Weighted Pool Selection ---
    # This selects the 'drawers' we are going to open.
    selected_mdps, s1_stats = greedy_select_mdps_weighted(
        mdp_cov, 
        len(U_universal), 
        mdp_metadata
    )

    # --- STAGE 2: Atomic Selection (Restricted) ---
    # IMPORTANT: We only pass the data for selected_mdps to Stage 2.
    # This ensures scot_greedy_family_atoms_tracked only 'inspects' this pool.
    pool_atoms = [candidates_per_env[k] for k in selected_mdps]
    pool_SFs   = [SFs[k] for k in selected_mdps]
    pool_envs  = [envs[k] for k in selected_mdps]

    # The tracker now only sees the pool, so inspection_count = len(selected_mdps)
    chosen_local, pool_stats, _ = scot_greedy_family_atoms_tracked(
        U_universal, 
        pool_atoms, 
        pool_SFs, 
        pool_envs
    )

    # Map local Stage-2 indices back to original global MDP indices
    chosen_global = []
    for local_env_idx, atom_idx in chosen_local:
        global_env_idx = selected_mdps[local_env_idx]
        chosen_global.append((global_env_idx, atom_idx))

    # Identify final activated environments (those that actually contributed atoms)
    activated_envs = sorted({c[0] for c in chosen_global})
    
    # Calculate the 'Waste' in the pool
    # i.e., MDPs we picked in Stage 1 but didn't actually need in Stage 2
    inspected_count = len(selected_mdps)
    selection_count = len(activated_envs)
    waste = inspected_count - selection_count

    return {
            "s1_iterations": s1_stats["s1_iterations"], # How many MDPs were chosen for pool
            "s1_checks": s1_stats["s1_shallow_checks"], # Total summary comparisons
            "s2_iterations": len(chosen_global),         # How many atoms were picked
            "s2_deep_checks": pool_stats["total_deep_atom_evals"], # (Add this to your SCOT func)
            "activated_envs": activated_envs,
            "waste": len(selected_mdps) - len(activated_envs)
        }