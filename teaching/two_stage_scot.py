import numpy as np
import mdp

import numpy as np
import time
from utils import atom_to_constraints


# ============================================================
# 1️⃣ Canonical Constraint Identity (GLOBAL)
# ============================================================

def make_key_for(*, normalize=True, round_decimals=12):
    """
    Canonical identity for constraints.
    Direction-only identity.
    """
    def key_for(v):
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)

        if n == 0.0 or not np.isfinite(n):
            return ("ZERO",)

        vv = v / n if normalize else v

        # collapse sign (allow sign flip)
        if vv[0] < 0:
            vv = -vv

        return tuple(np.round(vv, round_decimals))

    return key_for


# ============================================================
# 2️⃣ Build Universal Key Index
# ============================================================

def build_universal_key_index(
    U_universal,
    *,
    normalize=True,
    round_decimals=12,
):
    """
    Converts numeric universe into canonical key universe.
    """
    key_for = make_key_for(
        normalize=normalize,
        round_decimals=round_decimals,
    )

    key_to_uid = {}
    uid_to_key = []

    for v in np.asarray(U_universal):
        k = key_for(v)
        if k not in key_to_uid:
            key_to_uid[k] = len(uid_to_key)
            uid_to_key.append(k)

    return key_to_uid, uid_to_key


# ============================================================
# 3️⃣ Stage-1: Build MDP Coverage (KEY-BASED)
# ============================================================

def build_mdp_coverage_from_constraints_keys(
    U_per_env,
    key_to_uid,
    *,
    normalize=True,
    round_decimals=12,
):
    """
    Per-MDP coverage in canonical key-space.
    """

    key_for = make_key_for(
        normalize=normalize,
        round_decimals=round_decimals,
    )

    mdp_cov = []

    for H in U_per_env:
        cov = set()

        H = np.asarray(H)

        if H.size != 0:
            for row in H:
                k = key_for(row)
                uid = key_to_uid.get(k, None)
                if uid is not None:
                    cov.add(uid)

        mdp_cov.append(cov)

    return mdp_cov


# ============================================================
# 4️⃣ Greedy MDP Set Cover
# ============================================================

def greedy_select_mdps_unweighted(mdp_cov, universe_size):

    universe = set(range(universe_size))
    covered = set()

    selected = []
    selected_set = set()

    while universe - covered:

        best_gain = 0
        best_k = None
        best_new = None

        for k, cov_k in enumerate(mdp_cov):

            if k in selected_set:
                continue

            new_elements = cov_k - covered
            gain = len(new_elements)

            if gain > best_gain:
                best_gain = gain
                best_k = k
                best_new = new_elements

        if best_k is None or best_gain == 0:
            break

        selected.append(best_k)
        selected_set.add(best_k)
        covered |= best_new

    return selected


# ============================================================
# 5️⃣ Stage-2: Atomic SCOT (KEY-BASED)
# ============================================================

def scot_greedy_family_atoms_tracked(
    U_universal,
    atoms_per_env,
    SFs,
    envs,
    *,
    normalize=True,
    round_decimals=12,
):

    key_to_uid, uid_to_key = build_universal_key_index(
        U_universal,
        normalize=normalize,
        round_decimals=round_decimals,
    )

    universe = set(range(len(uid_to_key)))
    covered = set()

    key_for = make_key_for(
        normalize=normalize,
        round_decimals=round_decimals,
    )

    # -----------------------------------------
    # Precompute atom coverage
    # -----------------------------------------
    cov = []

    for env_idx, (atom_list, sf, env) in enumerate(
        zip(atoms_per_env, SFs, envs)
    ):
        mu_sa = sf[0]
        cov_i = []

        for atom in atom_list:

            constraints = atom_to_constraints(atom, mu_sa, env)

            atom_cov = set()

            for v in constraints:
                k = key_for(v)
                uid = key_to_uid.get(k, None)
                if uid is not None:
                    atom_cov.add(uid)

            cov_i.append(atom_cov)

        cov.append(cov_i)

    # -----------------------------------------
    # Greedy Set Cover (Atoms)
    # -----------------------------------------
    chosen = []

    while universe - covered:

        best_gain = 0
        best_atom = None
        best_new = None

        for env_idx in range(len(atoms_per_env)):

            for atom_idx, atom_cov in enumerate(cov[env_idx]):

                new_cover = atom_cov - covered
                gain = len(new_cover)

                if gain > best_gain:
                    best_gain = gain
                    best_atom = (env_idx, atom_idx)
                    best_new = new_cover

        if best_atom is None or best_gain == 0:
            break

        env_idx, atom_idx = best_atom

        chosen.append((env_idx, atoms_per_env[env_idx][atom_idx]))
        covered |= best_new

    return chosen


# ============================================================
# 6️⃣ Two-Stage SCOT (KEY-BASED)
# ============================================================

def two_stage_scot(
    *,
    U_universal,
    U_per_env_atoms,
    candidates_per_env,
    SFs,
    envs,
    normalize=True,
    round_decimals=12,
):

    # -----------------------------------------
    # Stage 1
    # -----------------------------------------

    key_to_uid, uid_to_key = build_universal_key_index(
        U_universal,
        normalize=normalize,
        round_decimals=round_decimals,
    )

    mdp_cov = build_mdp_coverage_from_constraints_keys(
        U_per_env_atoms,
        key_to_uid,
        normalize=normalize,
        round_decimals=round_decimals,
    )

    selected_mdps = greedy_select_mdps_unweighted(
        mdp_cov,
        universe_size=len(uid_to_key),
    )

    # -----------------------------------------
    # Stage 2
    # -----------------------------------------

    pool_atoms = [candidates_per_env[k] for k in selected_mdps]
    pool_SFs   = [SFs[k] for k in selected_mdps]
    pool_envs  = [envs[k] for k in selected_mdps]

    chosen_local = scot_greedy_family_atoms_tracked(
        U_universal,
        pool_atoms,
        pool_SFs,
        pool_envs,
        normalize=normalize,
        round_decimals=round_decimals,
    )

    # Map back to global indices
    chosen_global = []
    for local_env_idx, atom in chosen_local:
        global_env_idx = selected_mdps[local_env_idx]
        chosen_global.append((global_env_idx, atom))

    return {
        "chosen": chosen_global,
        "selected_mdps": selected_mdps,
        "universe_size": len(uid_to_key),
    }
