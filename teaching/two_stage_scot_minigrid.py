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
