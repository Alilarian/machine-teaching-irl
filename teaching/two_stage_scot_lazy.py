import numpy as np
import heapq
from teaching import scot_greedy_family_atoms_tracked

def build_mdp_metadata(candidates_per_env):
    """
    Computes costs for each MDP based on the number and diversity of atoms.
    """
    metadata = []
    for atoms in candidates_per_env:
        types = set(a.feedback_type for a in atoms)
        num_types = max(1, len(types))
        num_atoms = len(atoms)

        # Cost heuristic
        cost = (num_atoms + 1) / num_types

        metadata.append({
            "cost": cost,
            "num_atoms": num_atoms,
            "types": types,
            "num_types": num_types
        })
    return metadata

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

def greedy_select_mdps_weighted_lazy(mdp_cov, universe_size, mdp_metadata):
    """
    Lazy greedy selection of MDPs using (marginal gain / cost).

    Returns:
        selected_mdps
        stats: inspection metrics
    """
    universe = set(range(universe_size))
    covered = set()
    selected = []

    # Max-heap: (-upper_bound, mdp_idx, last_seen_covered_size)
    heap = []

    # ---- Initial optimistic bounds (unavoidable one pass) ----
    initial_inspections = 0
    for k, cov_k in enumerate(mdp_cov):
        gain = len(cov_k)
        cost = mdp_metadata[k]["cost"]
        ub = gain / cost if gain > 0 else 0.0
        heapq.heappush(heap, (-ub, k, 0))
        initial_inspections += 1

    recomputations = 0

    # ---- Lazy greedy loop ----
    while covered != universe and heap:
        neg_ub, k, seen_covered_size = heapq.heappop(heap)

        if k in selected:
            continue

        # Coverage changed → recompute true marginal gain
        if seen_covered_size != len(covered):
            new_elements = mdp_cov[k] - covered
            gain = len(new_elements)
            cost = mdp_metadata[k]["cost"]
            true_eff = gain / cost if gain > 0 else 0.0

            heapq.heappush(heap, (-true_eff, k, len(covered)))
            recomputations += 1
            continue

        # Fresh & best → select
        new_elements = mdp_cov[k] - covered
        if not new_elements:
            continue

        selected.append(k)
        covered |= new_elements

    stats = {
        "stage1_initial_inspections": initial_inspections,
        "stage1_recomputations": recomputations,
        "stage1_selected_count": len(selected),
    }

    return selected, stats

def two_stage_scot_weighted_lazy(
    *,
    U_universal,
    U_per_env_atoms,
    U_per_env_q,
    candidates_per_env,
    SFs,
    envs,
):
    # --------------------------------------------------------
    # STAGE 0: Combine constraint sources per MDP
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # STAGE 1: Lazy weighted MDP selection
    # --------------------------------------------------------
    selected_mdps, stage1_stats = greedy_select_mdps_weighted_lazy(
        mdp_cov,
        len(U_universal),
        mdp_metadata
    )

    # --------------------------------------------------------
    # STAGE 2: SCOT over restricted pool
    # --------------------------------------------------------
    pool_atoms = [candidates_per_env[k] for k in selected_mdps]
    pool_SFs   = [SFs[k] for k in selected_mdps]
    pool_envs  = [envs[k] for k in selected_mdps]

    chosen_local, pool_stats, _ = scot_greedy_family_atoms_tracked(
        U_universal,
        pool_atoms,
        pool_SFs,
        pool_envs
    )

    # Map back to global indices
    chosen_global = []
    for local_env_idx, atom_idx in chosen_local:
        global_env_idx = selected_mdps[local_env_idx]
        chosen_global.append((global_env_idx, atom_idx))

    activated_envs = sorted({e for e, _ in chosen_global})

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    inspected_count = len(selected_mdps)        # drawers opened
    activated_count = len(activated_envs)       # drawers actually used
    waste = inspected_count - activated_count

    return {
        "selected_mdps": selected_mdps,
        "activated_envs": activated_envs,
        "chosen_atoms": chosen_global,
        "inspection_count": inspected_count,
        "activation_count": activated_count,
        "waste": waste,
        "stage1_stats": stage1_stats,
        "stage2_stats": pool_stats,
        "mdp_metadata": mdp_metadata,
    }
