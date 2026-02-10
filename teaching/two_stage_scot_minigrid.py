
################################################################ n
# ============================================================
# Helpers
# ============================================================

# def normalize_constraints(X, *, normalize=True):
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
#     Numerically deduplicate universal constraints via cosine similarity.
#     Returns unique normalized (if normalize=True) representative rows.
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

#         # treat as duplicate if cosine similarity is ~1
#         if np.max(sims) < 1.0 - eps:
#             uniq.append(u)

#     return np.vstack(uniq)


# def build_mdp_coverage_from_constraints_numpy(
#     U_per_env_envlevel,
#     U_universal_unique,
#     *,
#     eps=1e-6,
#     normalize=True,
#     allow_sign_flip=True,
# ):
#     """
#     Build per-env coverage sets over the numeric universal constraint set.

#     U_per_env_envlevel[e] : (n_e, d) constraints (env-level)
#     U_universal_unique    : (M, d) constraints (universe reps)
#     """
#     Uu, _ = normalize_constraints(U_universal_unique, normalize=normalize)
#     mdp_cov = []

#     for H in U_per_env_envlevel:
#         H = np.asarray(H, dtype=float)
#         cov = set()

#         if H.size == 0:
#             mdp_cov.append(cov)
#             continue

#         if H.ndim == 1:
#             H = H[None, :]

#         Hn, _ = normalize_constraints(H, normalize=normalize)
#         if Hn.size == 0:
#             mdp_cov.append(cov)
#             continue

#         sims = Hn @ Uu.T
#         if allow_sign_flip:
#             sims = np.abs(sims)

#         hits = sims >= (1.0 - eps)  # (n_e, M) boolean
#         covered = np.any(hits, axis=0)  # (M,)
#         cov.update(np.nonzero(covered)[0].tolist())
#         mdp_cov.append(cov)

#     return mdp_cov


# def greedy_select_mdps_unweighted(mdp_cov, universe_size):
#     universe = set(range(universe_size))
#     covered = set()
#     selected = []
#     selected_set = set()

#     iters = checks = 0

#     while universe - covered:
#         best_gain = 0
#         best_k = None
#         best_new = None
#         iters += 1

#         for k, cov_k in enumerate(mdp_cov):
#             if k in selected_set:
#                 continue

#             checks += 1
#             new = cov_k - covered
#             if len(new) > best_gain:
#                 best_gain = len(new)
#                 best_k = k
#                 best_new = new

#         if best_k is None or best_gain == 0:
#             break

#         selected.append(best_k)
#         selected_set.add(best_k)
#         covered |= best_new

#     return selected, {
#         "s1_iterations": iters,
#         "s1_shallow_checks": checks,
#         "s1_final_coverage": len(covered),
#         "s1_universe_size": universe_size,
#     }


# # ============================================================
# # Stage-2: Atom-level SCOT using STRICT per-atom constraints
# # ============================================================

# def scot_greedy_family_atoms_tracked(
#     U_universal_unique,
#     atoms_per_env,
#     constraints_per_env_per_atom,
#     *,
#     eps=1e-6,
#     normalize=True,
#     allow_sign_flip=True,
# ):
#     """
#     Greedy SCOT selection over (env, atom) using precomputed per-atom constraints.

#     constraints_per_env_per_atom[e][a] must be a list of constraint vectors (d,)
#     for atom a in env e.
#     """
#     Uu, _ = normalize_constraints(U_universal_unique, normalize=normalize)
#     universe = set(range(len(Uu)))
#     covered = set()

#     n_envs = len(atoms_per_env)

#     # ----- hard contract checks -----
#     if len(constraints_per_env_per_atom) != n_envs:
#         raise ValueError(
#             "constraints_per_env_per_atom must have same length as atoms_per_env "
#             f"({len(constraints_per_env_per_atom)} vs {n_envs})"
#         )
#     for e in range(n_envs):
#         if len(constraints_per_env_per_atom[e]) != len(atoms_per_env[e]):
#             raise ValueError(
#                 f"Env {e}: mismatch (#atoms={len(atoms_per_env[e])} vs "
#                 f"#constraint-lists={len(constraints_per_env_per_atom[e])})"
#             )

#     chosen = []
#     chosen_constraints = []
#     inspected_envs = set()

#     env_stats = {
#         i: {
#             "atoms": [],
#             "indices": [],
#             "coverage_counts": [],
#             "total_coverage": 0,
#             "was_inspected": False,
#             "precompute_time": 0.0,
#         }
#         for i in range(n_envs)
#     }

#     # ---------------- Precompute atom coverage ----------------
#     cov = []
#     t0 = time.time()

#     for e, atom_constraints_list in enumerate(constraints_per_env_per_atom):
#         t_env = time.time()
#         cov_e = []

#         for C_list in atom_constraints_list:
#             atom_cov = set()

#             if C_list is not None and len(C_list) > 0:
#                 C = np.asarray(C_list, dtype=float)

#                 if C.ndim == 1:
#                     C = C[None, :]

#                 Cn, _ = normalize_constraints(C, normalize=normalize)
#                 if Cn.size > 0:
#                     sims = Cn @ Uu.T
#                     if allow_sign_flip:
#                         sims = np.abs(sims)

#                     hits = sims >= (1.0 - eps)
#                     atom_cov.update(np.nonzero(np.any(hits, axis=0))[0].tolist())

#             cov_e.append(atom_cov)

#         cov.append(cov_e)
#         env_stats[e]["precompute_time"] = time.time() - t_env

#     env_stats["total_precompute_time"] = time.time() - t0

#     # ---------------- Greedy SCOT ----------------
#     it = 0
#     t1 = time.time()

#     while universe - covered:
#         best_gain = 0
#         best = None
#         best_new = None

#         for e in range(n_envs):
#             if atoms_per_env[e]:
#                 inspected_envs.add(e)
#                 env_stats[e]["was_inspected"] = True

#             for a, atom_cov in enumerate(cov[e]):
#                 new = atom_cov - covered
#                 if len(new) > best_gain:
#                     best_gain = len(new)
#                     best = (e, a)
#                     best_new = new

#         if best is None or best_gain == 0:
#             break

#         e, a = best
#         atom = atoms_per_env[e][a]

#         chosen.append((e, atom))
#         # reuse already-computed constraints
#         chosen_constraints.extend(constraints_per_env_per_atom[e][a])

#         covered |= best_new

#         env_stats[e]["atoms"].append(atom)
#         env_stats[e]["indices"].append(it)
#         env_stats[e]["coverage_counts"].append(len(best_new))
#         env_stats[e]["total_coverage"] += len(best_new)
#         it += 1

#     env_stats["total_greedy_time"] = time.time() - t1
#     env_stats["final_coverage"] = len(covered)
#     env_stats["total_iterations"] = it
#     env_stats["total_inspected_count"] = len(inspected_envs)
#     env_stats["total_activated_count"] = len({e for e, _ in chosen})
#     env_stats["activated_env_indices"] = sorted({e for e, _ in chosen})

#     if chosen_constraints:
#         chosen_constraints = np.asarray(chosen_constraints, dtype=float)
#         if chosen_constraints.ndim == 1:
#             chosen_constraints = chosen_constraints[None, :]
#     else:
#         chosen_constraints = np.zeros((0, Uu.shape[1]))

#     return chosen, env_stats, chosen_constraints


# # ============================================================
# # Two-stage SCOT (contract-correct)
# # ============================================================

# def two_stage_scot(
#     *,
#     U_universal,
#     U_per_env_atoms_envlevel,
#     constraints_per_env_per_atom,
#     candidates_per_env,
#     eps=1e-6,
#     normalize=True,
# ):
#     """
#     Stage-1: select envs using env-level constraints U_per_env_atoms_envlevel
#     Stage-2: select atoms using per-atom constraints constraints_per_env_per_atom

#     Required inputs:
#       - U_universal: (M,d)
#       - U_per_env_atoms_envlevel: list[(n_e,d)]
#       - constraints_per_env_per_atom: list[list[list[d]]]
#       - candidates_per_env: list[list[Atom]]
#     """
#     U_universal = np.asarray(U_universal, dtype=float)
#     if U_universal.ndim != 2 or U_universal.shape[0] == 0:
#         raise ValueError(f"U_universal must be non-empty 2D array, got {U_universal.shape}")
#     d = U_universal.shape[1]

#     n_envs = len(candidates_per_env)

#     # ----- hard contract checks -----
#     if not (
#         len(U_per_env_atoms_envlevel) ==
#         len(constraints_per_env_per_atom) ==
#         n_envs
#     ):
#         raise ValueError(
#             "Length mismatch among U_per_env_atoms_envlevel / "
#             "constraints_per_env_per_atom / candidates_per_env "
#             f"({len(U_per_env_atoms_envlevel)}, {len(constraints_per_env_per_atom)}, {n_envs})"
#         )

#     for e in range(n_envs):
#         if len(constraints_per_env_per_atom[e]) != len(candidates_per_env[e]):
#             raise ValueError(
#                 f"Env {e}: constraints_per_env_per_atom[e] must align with candidates_per_env[e] "
#                 f"(got {len(constraints_per_env_per_atom[e])} vs {len(candidates_per_env[e])})"
#             )

#     # -------- Stage 0: normalize env-level constraints container --------
#     U_per_env = []
#     for H in U_per_env_atoms_envlevel:
#         if H is None or len(H) == 0:
#             U_per_env.append(np.zeros((0, d)))
#         else:
#             H = np.asarray(H, dtype=float)
#             if H.ndim == 1:
#                 H = H[None, :]
#             if H.ndim != 2 or H.shape[1] != d:
#                 raise ValueError(f"Env-level constraint shape mismatch: got {H.shape}, expected (*,{d})")
#             U_per_env.append(H)

#     # -------- Stage 1: numeric universe + MDP selection --------
#     U_unique = build_universal_index(
#         U_universal,
#         eps=eps,
#         normalize=normalize,
#         allow_sign_flip=True,
#     )

#     mdp_cov = build_mdp_coverage_from_constraints_numpy(
#         U_per_env,
#         U_unique,
#         eps=eps,
#         normalize=normalize,
#         allow_sign_flip=True,
#     )

#     selected_mdps, s1_stats = greedy_select_mdps_unweighted(
#         mdp_cov,
#         universe_size=len(U_unique),
#     )

#     # -------- Stage 2: atom-level SCOT --------
#     pool_atoms = [candidates_per_env[k] for k in selected_mdps]
#     pool_constraints = [constraints_per_env_per_atom[k] for k in selected_mdps]

#     chosen_local, s2_stats, _ = scot_greedy_family_atoms_tracked(
#         U_unique,
#         pool_atoms,
#         pool_constraints,
#         eps=eps,
#         normalize=normalize,
#         allow_sign_flip=True,
#     )

#     # Map chosen back to global env ids
#     chosen_global = [(selected_mdps[e], atom) for e, atom in chosen_local]

#     activated_envs = sorted({e for e, _ in chosen_global})
#     waste = len(selected_mdps) - len(activated_envs)

#     return {
#         "chosen": chosen_global,
#         "selected_mdps": selected_mdps,

#         "s1_iterations": s1_stats["s1_iterations"],
#         "s1_checks": s1_stats["s1_shallow_checks"],
#         "s1_unique_universe_size": s1_stats["s1_universe_size"],
#         "s1_final_coverage": s1_stats["s1_final_coverage"],

#         "s2_iterations": len(chosen_global),
#         "activated_envs": activated_envs,
#         "waste": waste,
#         "s2_stats": s2_stats,
#     }




import numpy as np
import time

# ============================================================
# Canonical constraint identity (SINGLE source of truth)
# ============================================================

def make_key_for(*, normalize=True, round_decimals=12):
    """
    Returns a function that maps a constraint vector -> canonical key.
    """
    def key_for(v):
        v = np.asarray(v, dtype=float)
        v = np.atleast_1d(v)

        n = np.linalg.norm(v)
        if n == 0.0 or not np.isfinite(n):
            return ("ZERO",)

        vv = (v / n) if normalize else v
        vv = np.round(vv, round_decimals)
        return tuple(vv.tolist())

    return key_for


def as_constraint_list(x):
    """
    Normalize constraint container into list[np.ndarray(d,)].
    """
    if x is None:
        return []

    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            v = np.asarray(v, dtype=float)
            if v.ndim == 1:
                out.append(v)
            elif v.ndim == 2:
                out.extend(v[i] for i in range(v.shape[0]))
            else:
                raise ValueError(f"Invalid constraint shape {v.shape}")
        return out

    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return [x]
    if x.ndim == 2:
        return [x[i] for i in range(x.shape[0])]

    raise ValueError(f"Invalid constraint array shape {x.shape}")


# ============================================================
# Stage-1: env-level coverage in KEY space
# ============================================================

def build_mdp_coverage_from_constraints_keys(
    U_per_env_envlevel,
    U_universal,
    *,
    normalize=True,
    round_decimals=12,
):
    """
    Stage-1 coverage using canonical keys.

    Returns:
      mdp_cov : list[set[int]]
      key_to_uid
      uid_to_key
    """
    key_for = make_key_for(normalize=normalize, round_decimals=round_decimals)

    key_to_uid = {}
    uid_to_key = []

    for u in U_universal:
        k = key_for(u)
        if k not in key_to_uid:
            key_to_uid[k] = len(uid_to_key)
            uid_to_key.append(k)

    mdp_cov = []
    for H in U_per_env_envlevel:
        cov = set()
        for v in as_constraint_list(H):
            k = key_for(v)
            if k in key_to_uid:
                cov.add(key_to_uid[k])
        mdp_cov.append(cov)

    return mdp_cov, key_to_uid, uid_to_key


def greedy_select_mdps_unweighted(mdp_cov, universe_size):
    universe = set(range(universe_size))
    covered = set()
    selected = []
    selected_set = set()

    iters = checks = 0

    while universe - covered:
        best_gain = 0
        best_k = None
        best_new = None
        iters += 1

        for k, cov_k in enumerate(mdp_cov):
            if k in selected_set:
                continue

            checks += 1
            new = cov_k - covered
            if len(new) > best_gain:
                best_gain = len(new)
                best_k = k
                best_new = new

        if best_k is None or best_gain == 0:
            break

        selected.append(best_k)
        selected_set.add(best_k)
        covered |= best_new

    return selected, {
        "s1_iterations": iters,
        "s1_shallow_checks": checks,
        "s1_final_coverage": len(covered),
        "s1_universe_size": universe_size,
    }


# ============================================================
# Stage-2: atom-level SCOT in SAME KEY space
# ============================================================

def scot_greedy_family_atoms_tracked(
    U_universal,
    atoms_per_env,
    constraints_per_env_per_atom,
    *,
    normalize=True,
    round_decimals=12,
):
    """
    STRICT key-based SCOT over (env, atom).
    """
    key_for = make_key_for(normalize=normalize, round_decimals=round_decimals)

    key_to_uid = {}
    for u in U_universal:
        k = key_for(u)
        if k not in key_to_uid:
            key_to_uid[k] = len(key_to_uid)

    universe = set(range(len(key_to_uid)))
    covered = set()

    n_envs = len(atoms_per_env)

    # ----- contract checks -----
    if len(constraints_per_env_per_atom) != n_envs:
        raise ValueError("constraints_per_env_per_atom length mismatch")

    for e in range(n_envs):
        if len(constraints_per_env_per_atom[e]) != len(atoms_per_env[e]):
            raise ValueError(f"Env {e}: atom/constraint mismatch")

    cov = []
    t0 = time.time()

    for e in range(n_envs):
        cov_e = []
        for atom_constraints in constraints_per_env_per_atom[e]:
            atom_cov = set()
            for v in as_constraint_list(atom_constraints):
                k = key_for(v)
                if k in key_to_uid:
                    atom_cov.add(key_to_uid[k])
            cov_e.append(atom_cov)
        cov.append(cov_e)

    precompute_time = time.time() - t0

    chosen = []
    chosen_constraints = []
    inspected_envs = set()

    env_stats = {
        i: {
            "atoms": [],
            "indices": [],
            "coverage_counts": [],
            "total_coverage": 0,
            "was_inspected": False,
        }
        for i in range(n_envs)
    }

    it = 0
    t1 = time.time()

    while universe - covered:
        best_gain = 0
        best = None
        best_new = None

        for e in range(n_envs):
            if atoms_per_env[e]:
                inspected_envs.add(e)
                env_stats[e]["was_inspected"] = True

            for a, atom_cov in enumerate(cov[e]):
                new = atom_cov - covered
                if len(new) > best_gain:
                    best_gain = len(new)
                    best = (e, a)
                    best_new = new

        if best is None or best_gain == 0:
            break

        e, a = best
        chosen.append((e, atoms_per_env[e][a]))
        chosen_constraints.extend(as_constraint_list(constraints_per_env_per_atom[e][a]))

        covered |= best_new

        env_stats[e]["atoms"].append(atoms_per_env[e][a])
        env_stats[e]["indices"].append(it)
        env_stats[e]["coverage_counts"].append(len(best_new))
        env_stats[e]["total_coverage"] += len(best_new)
        it += 1

    greedy_time = time.time() - t1

    if chosen_constraints:
        chosen_constraints = np.vstack(chosen_constraints)
    else:
        chosen_constraints = np.zeros((0, len(U_universal[0])))

    env_stats.update({
        "total_precompute_time": precompute_time,
        "total_greedy_time": greedy_time,
        "final_coverage": len(covered),
        "total_iterations": it,
        "total_inspected_count": len(inspected_envs),
        "total_activated_count": len({e for e, _ in chosen}),
        "activated_env_indices": sorted({e for e, _ in chosen}),
    })

    return chosen, env_stats, chosen_constraints


# ============================================================
# Two-Stage SCOT (KEY-BASED, CONTRACT-CORRECT)
# ============================================================

def two_stage_scot(
    *,
    U_universal,
    U_per_env_atoms_envlevel,
    constraints_per_env_per_atom,
    candidates_per_env,
    normalize=True,
    round_decimals=12,
):
    """
    KEY-BASED two-stage SCOT.

    Stage-1: env selection in canonical key space
    Stage-2: atom selection in SAME key space
    """
    n_envs = len(candidates_per_env)

    if not (
        len(U_per_env_atoms_envlevel) ==
        len(constraints_per_env_per_atom) ==
        n_envs
    ):
        raise ValueError("Input length mismatch in two_stage_scot")

    mdp_cov, key_to_uid, uid_to_key = build_mdp_coverage_from_constraints_keys(
        U_per_env_atoms_envlevel,
        U_universal,
        normalize=normalize,
        round_decimals=round_decimals,
    )

    selected_mdps, s1_stats = greedy_select_mdps_unweighted(
        mdp_cov,
        universe_size=len(uid_to_key),
    )

    pool_atoms = [candidates_per_env[k] for k in selected_mdps]
    pool_constraints = [constraints_per_env_per_atom[k] for k in selected_mdps]

    chosen_local, s2_stats, _ = scot_greedy_family_atoms_tracked(
        U_universal,
        pool_atoms,
        pool_constraints,
        normalize=normalize,
        round_decimals=round_decimals,
    )

    chosen_global = [(selected_mdps[e], atom) for e, atom in chosen_local]

    activated_envs = sorted({e for e, _ in chosen_global})
    waste = len(selected_mdps) - len(activated_envs)

    return {
        "chosen": chosen_global,
        "selected_mdps": selected_mdps,

        "s1_iterations": s1_stats["s1_iterations"],
        "s1_checks": s1_stats["s1_shallow_checks"],
        "s1_unique_universe_size": s1_stats["s1_universe_size"],
        "s1_final_coverage": s1_stats["s1_final_coverage"],

        "s2_iterations": len(chosen_global),
        "activated_envs": activated_envs,
        "waste": waste,
        "s2_stats": s2_stats,
    }
