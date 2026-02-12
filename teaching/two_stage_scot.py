import numpy as np
import mdp
from teaching import scot_greedy_family_atoms_tracked

# ---------------------------------------------------------------------
# Canonicalization: MUST match Stage-2 (scot_greedy_family_atoms_tracked)
# ---------------------------------------------------------------------
# def make_key_for(*, normalize=True, round_decimals=12):
#     def key_for(v):
#         v = np.asarray(v)
#         n = np.linalg.norm(v)
#         if n == 0.0 or not np.isfinite(n):
#             return ("ZERO",)
#         vv = (v / n) if normalize else v
#         return tuple(np.round(vv, round_decimals))
#     return key_for

# ---------------------------------------------------------------------
# Stage-1 coverage using canonical keys (FIXED)
# ---------------------------------------------------------------------

# def build_mdp_coverage_from_constraints_keys(
#     U_per_env,
#     U_universal,
#     *,
#     normalize=True,
#     round_decimals=12,
# ):
#     """
#     Build MDP coverage sets in the SAME constraint identity space as Stage-2:

#     - Canonicalize each universal constraint into a key (direction-only, rounded).
#     - Canonicalize each env constraint into a key.
#     - MDP covers a universal element iff keys match.

#     Returns:
#         mdp_cov: list[set[int]] where each set contains "unique universal key ids"
#         key_to_uid: dict[key -> uid]
#         uid_to_key: list[key]
#     """
#     key_for = make_key_for(normalize=normalize, round_decimals=round_decimals)

#     print("Universal insode the build map:########################################################## ")
#     print(len(U_per_env))
#     print(U_per_env[0].shape)

#     # Unique universe in key-space (collapses duplicates properly)
#     key_to_uid = {}
#     uid_to_key = []
#     for u in np.asarray(U_universal):
#         k = key_for(u)
#         if k not in key_to_uid:
#             key_to_uid[k] = len(uid_to_key)
#             uid_to_key.append(k)

#     # print("Inside the build_mdp_coverage_from_constraints_keys: ")
#     # print()
#     # print(key_to_uid)

    
#     # Per-env coverage in uid-space
#     mdp_cov = []
#     for H_k in U_per_env:
#         cov_uids = set()
#         H_k = np.asarray(H_k)
#         if H_k.size != 0:
#             for row in H_k:
#                 #print("ROWWWWWWWWWW")
#                 #print(row)
#                 kk = key_for(row)
#                 #print("KKKKK")
#                 #print(kk)
#                 uid = key_to_uid.get(kk, None)
#                 if uid is not None:
#                     cov_uids.add(uid)
#         mdp_cov.append(cov_uids)

#     print("Inside the build_mdp_coverage_from_constraints_keys: ")
#     print()
#     print(mdp_cov)
    
#     return mdp_cov, key_to_uid, uid_to_key




# def normalize_constraints(X, *, normalize=True):
#     """
#     Normalize constraint vectors to unit length.
#     Zero / invalid vectors are removed.

#     Returns:
#         Xn : (k, d) normalized constraints
#         mask : boolean mask of kept rows
#     """
#     X = np.asarray(X, dtype=float)
#     if X.size == 0:
#         return X.reshape(0, X.shape[-1]), np.zeros(0, dtype=bool)

#     if not normalize:
#         return X, np.ones(len(X), dtype=bool)

#     norms = np.linalg.norm(X, axis=1)
#     good = (norms > 0) & np.isfinite(norms)
#     Xn = X[good] / norms[good][:, None]
#     return Xn, good

# def build_universal_index(
#     U_universal,
#     *,
#     eps=1e-6,
#     normalize=True,
#     allow_sign_flip=True,
# ):
#     """
#     Deduplicate universal constraints numerically using cosine similarity.

#     Returns:
#         Uu : (M, d) unique normalized constraints
#     """
#     U, _ = normalize_constraints(U_universal, normalize=normalize)
#     if U.size == 0:
#         return U

#     uniq = []
#     for u in U:
#         if not uniq:
#             uniq.append(u)
#             continue

#         sims = np.dot(np.vstack(uniq), u)
#         if allow_sign_flip:
#             sims = np.abs(sims)

#         if np.max(sims) < 1.0 - eps:
#             uniq.append(u)

#     return np.vstack(uniq)

# def build_mdp_coverage_from_constraints_numpy(
#     U_per_env,
#     U_universal_unique,
#     *,
#     eps=1e-6,
#     normalize=True,
#     allow_sign_flip=True,
# ):
#     """
#     Compute per-MDP coverage over universal constraints using cosine similarity.

#     Returns:
#         mdp_cov : list[set[int]]
#     """
#     Uu = np.asarray(U_universal_unique, dtype=float)
#     if Uu.ndim != 2:
#         raise ValueError("U_universal_unique must be 2D")

#     # Normalize universe once
#     Uu, _ = normalize_constraints(Uu, normalize=normalize)
#     M = Uu.shape[0]

#     mdp_cov = []

#     for H in U_per_env:
#         H = np.asarray(H, dtype=float)
#         cov = set()

#         if H.size == 0:
#             mdp_cov.append(cov)
#             continue

#         Hn, _ = normalize_constraints(H, normalize=normalize)
#         if Hn.size == 0:
#             mdp_cov.append(cov)
#             continue

#         sims = Hn @ Uu.T
#         if allow_sign_flip:
#             sims = np.abs(sims)

#         # any env constraint that matches a universe element
#         hits = sims >= (1.0 - eps)
#         covered = np.any(hits, axis=0)

#         cov.update(np.nonzero(covered)[0].tolist())
#         mdp_cov.append(cov)

#     return mdp_cov




# def greedy_select_mdps_unweighted(mdp_cov, universe_size):
#     """
#     Greedy set cover over MDPs with NO cost (unweighted).
#     Universe elements are integers [0..universe_size-1].

#     FIX:
#       - Terminate on uncovered (universe - covered), not covered == universe.
#       - Uses sets for uniqueness (already correct) + selected_set for speed.
#     """
#     universe = set(range(universe_size))
#     covered = set()
#     selected = []
#     selected_set = set()

#     s1_iterations = 0
#     s1_shallow_checks = 0

#     while (universe - covered):
#         best_gain = 0
#         best_k = None
#         best_new = None
#         s1_iterations += 1

#         for k, cov_k in enumerate(mdp_cov):
#             if k in selected_set:
#                 continue

#             s1_shallow_checks += 1
#             new_elements = cov_k - covered
#             gain = len(new_elements)

#             if gain > best_gain:
#                 best_gain = gain
#                 best_k = k
#                 best_new = new_elements

#         # No progress possible
#         if best_k is None or best_gain == 0:
#             break

#         selected.append(best_k)
#         selected_set.add(best_k)
#         covered |= best_new

#     return selected, {
#         "s1_iterations": s1_iterations,
#         "s1_shallow_checks": s1_shallow_checks,
#         "s1_final_coverage": len(covered),
#         "s1_universe_size": universe_size,
#     }

# # ------------------------------------------------------------
# # Two-stage SCOT (cost-free) — FIXED Stage-1 coverage
# # ------------------------------------------------------------
# def two_stage_scot(
#     *,
#     U_universal,
#     U_per_env_atoms,
#     U_per_env_q,
#     candidates_per_env,
#     SFs,
#     envs,
#     normalize=True,
#     round_decimals=12,
# ):
#     # --------------------------------------------------------
#     # Stage 0: Constraint aggregation per MDP (ROBUST)
#     # --------------------------------------------------------
#     n_envs = len(envs)
#     U_universal = np.asarray(U_universal)
#     d = U_universal.shape[1]

#     U_per_env = []
#     for k in range(n_envs):
#         # Atom constraints for env k
#         if U_per_env_atoms is not None and k < len(U_per_env_atoms) and len(U_per_env_atoms[k]) > 0:
#             H_a = np.asarray(U_per_env_atoms[k])
#         else:
#             H_a = np.zeros((0, d))

#         # # Q constraints for env k
#         # if U_per_env_q is not None and k < len(U_per_env_q) and len(U_per_env_q[k]) > 0:
#         #     H_q = np.asarray(U_per_env_q[k])
#         # else:
#         #     H_q = np.zeros((0, d))

#         # Combine safely
#         #if H_a.shape[0] > 0 and H_q.shape[0] > 0:
#         #    combined = np.vstack([H_a, H_q])
#         #elif H_a.shape[0] > 0:
#         #    combined = H_a
#         #elif H_q.shape[0] > 0:
#         #    combined = H_q
#         #else:
#         #    combined = np.zeros((0, d))

#         U_per_env.append(H_a)

#     # --------------------------------------------------------
#     # Stage 1: Build MDP coverage in CANONICAL key space (FIXED)
#     # --------------------------------------------------------
    
#     ############################################################################################## replacing this block with numpy based instead of key
    
#     # mdp_cov, key_to_uid, uid_to_key = build_mdp_coverage_from_constraints_keys(
#     #     U_per_env,
#     #     U_universal,
#     #     normalize=normalize,
#     #     round_decimals=round_decimals,
#     # )

#     # print("MDP coverages (uid-space):")
#     # print(mdp_cov)
#     # print(f"Stage-1 unique-universe size (after key collapse): {len(uid_to_key)}")

#     # selected_mdps, s1_stats = greedy_select_mdps_unweighted(
#     #     mdp_cov,
#     #     len(uid_to_key),
#     # )

#     ############################################################################################## replacing this block with numpy based instead of key
    
#     # Deduplicate universe numerically
#     U_universal_unique = build_universal_index(
#         U_universal,
#         eps=1e-6,
#         normalize=normalize,
#         allow_sign_flip=True,
#     )

#     # Build MDP coverage numerically
#     mdp_cov = build_mdp_coverage_from_constraints_numpy(
#         U_per_env,
#         U_universal_unique,
#         eps=1e-6,
#         normalize=normalize,
#         allow_sign_flip=True,
#     )

#     selected_mdps, s1_stats = greedy_select_mdps_unweighted(
#         mdp_cov,
#         universe_size=len(U_universal_unique),
#     )

#     print("Selected MDPs inside the two-stage:")
#     print(selected_mdps)
#     print("Stage-1 stats:", s1_stats)

#     # --------------------------------------------------------
#     # Stage 2: Atomic SCOT restricted to selected MDPs
#     # --------------------------------------------------------
#     pool_atoms = [candidates_per_env[k] for k in selected_mdps]
#     pool_SFs   = [SFs[k] for k in selected_mdps]
#     pool_envs  = [envs[k] for k in selected_mdps]

#     #### Here we are pssing U_universal

#     chosen_local, pool_stats, _ = scot_greedy_family_atoms_tracked(
#         U_universal,
#         pool_atoms,
#         pool_SFs,
#         pool_envs,
#         normalize=normalize,
#         round_decimals=round_decimals,
#     )

#     print("Inside the two-stage (local chosen):")
#     print(chosen_local)

#     # Map back to global MDP indices
#     chosen_global = []
#     for local_env_idx, atom in chosen_local:
#         global_env_idx = selected_mdps[local_env_idx]
#         chosen_global.append((global_env_idx, atom))

#     activated_envs = sorted({k for k, _ in chosen_global})
#     waste = len(selected_mdps) - len(activated_envs)

#     return {
#         "chosen": chosen_global,

#         # Stage-1 bookkeeping
#         "s1_iterations": s1_stats["s1_iterations"],
#         "s1_checks": s1_stats["s1_shallow_checks"],
#         "s1_unique_universe_size": s1_stats["s1_universe_size"],
#         "s1_final_coverage": s1_stats["s1_final_coverage"],
#         "selected_mdps": selected_mdps,

#         # Stage-2 bookkeeping
#         "s2_iterations": len(chosen_global),
#         "activated_envs": activated_envs,
#         "waste": waste,

#         # Optional: keep pool_stats if you want more stage-2 timings/coverage
#         "s2_stats": {
#             "total_precompute_time": pool_stats.get("total_precompute_time", None),
#             "total_greedy_time": pool_stats.get("total_greedy_time", None),
#             "final_coverage": pool_stats.get("final_coverage", None),
#             "total_iterations": pool_stats.get("total_iterations", None),
#             "total_inspected_count": pool_stats.get("total_inspected_count", None),
#             "total_activated_count": pool_stats.get("total_activated_count", None),
#             "activated_env_indices_local": pool_stats.get("activated_env_indices", None),
#         },
#     }

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
