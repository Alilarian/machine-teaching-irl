from .derive_constraints import atom_to_constraints
from .lp_redundancy import remove_redundant_constraints
import numpy as np


# ============================================================
# A. Atom Weighting Based on Number of Non-Redundant Constraints
# ============================================================

def compute_atom_weight(atom, mu_sa, env, lp_epsilon=1e-4):
    """
    Return weight = number of non-redundant constraints this atom implies.

    - demo → multiple constraints (pruned)
    - optimal_sa → like demo but single-step
    - pairwise / estop / improvement → exactly 1 constraint
    """

    # -------------- Simple cases: 1 constraint --------------
    if atom.feedback_type in ("pairwise", "estop", "improvement"):
        return 1

    # -------------- Demonstrations or Q-based demos --------------
    cons = atom_to_constraints(atom, mu_sa, env)

    if len(cons) == 0:
        return 0

    # Remove redundancy to get *true* info content
    cons = remove_redundant_constraints(cons, epsilon=lp_epsilon)

    return len(cons)
# ============================================================
# B. Build Global Pool + Constraint-Based Weights
# ============================================================

def build_weighted_pool(envs, candidates_per_env, SFs, lp_epsilon=1e-4):
    """
    Returns:
        flat_pool:  [(env_idx, atom), ...]
        weights:    normalized numpy array of weights aligned with flat_pool
    """
    flat_pool = []
    weights = []

    for env_idx, (atom_list, sf, env) in enumerate(zip(candidates_per_env, SFs, envs)):
        mu_sa = sf[0]  # successor features per (s,a)

        for atom in atom_list:
            flat_pool.append((env_idx, atom))

            w = compute_atom_weight(atom, mu_sa, env, lp_epsilon=lp_epsilon)

            # Ensure minimum weight = 1, so "zero constraint" atoms still appear
            weights.append(max(w, 1))

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()  # normalize weights for sampling
    return flat_pool, weights

# ============================================================
# C. Weighted Random Baseline Matching SCOT Sample Size
# ============================================================

def sample_weighted_atoms_like_scot_baseline(envs, candidates_per_env, SFs, chosen_scot, seed=None):
    """
    Weighted baseline:
    - Create a global atom pool
    - Compute weight per atom = # of non-redundant constraints
    - Sample the same TOTAL number of atoms SCOT selected
    """

    if seed is not None:
        np.random.seed(seed)

    total_scot_atoms = len(chosen_scot)

    # Build weighted global pool
    flat_pool, weights = build_weighted_pool(envs, candidates_per_env, SFs)

    if len(flat_pool) == 0:
        return []

    replace = len(flat_pool) < total_scot_atoms

    idxs = np.random.choice(
        len(flat_pool),
        size=total_scot_atoms,
        replace=replace,
        p=weights
    )

    return [flat_pool[i] for i in idxs]

def sample_random_atoms_like_scot_baseline(candidates_per_env, chosen_scot, seed=None):
    """
    New baseline:
    - Build one global pool of atoms across ALL environments.
    - Sample the same TOTAL number of atoms as SCOT selected.
    """

    if seed is not None:
        np.random.seed(seed)

    # --- 1. How many atoms did SCOT select? ---
    total_scot_atoms = len(chosen_scot)

    # --- 2. Build a global pool of (env_idx, atom) ---
    global_pool = []
    for env_idx, atom_list in enumerate(candidates_per_env):
        for atom in atom_list:
            global_pool.append((env_idx, atom))

    if len(global_pool) == 0:
        return []

    # --- 3. Sample from the global pool ---
    replace = len(global_pool) < total_scot_atoms
    idxs = np.random.choice(
        len(global_pool),
        size=total_scot_atoms,
        replace=replace
    )

    # --- 4. Gather chosen atoms ---
    out = [global_pool[i] for i in idxs]

    return out