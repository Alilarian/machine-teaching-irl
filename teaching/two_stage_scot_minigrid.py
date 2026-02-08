

import numpy as np
import time


# ---------------------------------------------------------------------
# Canonicalization helpers (single source of truth)
# ---------------------------------------------------------------------
def make_key_for(*, normalize=True, round_decimals=12):
    def key_for(v):
        v = np.asarray(v, dtype=float)
        v = np.atleast_1d(v)

        n = np.linalg.norm(v)
        if n == 0.0 or not np.isfinite(n):
            return ("ZERO",)

        vv = (v / n) if normalize else v
        vv = np.atleast_1d(vv)
        return tuple(np.round(vv, round_decimals))
    return key_for


def _as_constraint_list(atom_constraints):
    """
    Normalize a single atom's constraints to: list[np.ndarray(d,)].

    Accepts:
      - None -> []
      - (d,) -> [v]
      - (n,d) -> [row0, row1, ...]
      - list of vectors -> list(vectors)
    """
    if atom_constraints is None:
        return []

    # If it's already a python list/tuple, normalize each element
    if isinstance(atom_constraints, (list, tuple)):
        out = []
        for v in atom_constraints:
            v = np.asarray(v, dtype=float)
            if v.ndim == 0:
                out.append(np.atleast_1d(v))
            elif v.ndim == 1:
                out.append(v)
            elif v.ndim == 2:
                out.extend([v[i, :] for i in range(v.shape[0])])
            else:
                raise ValueError(f"Unsupported constraint ndim={v.ndim}, shape={v.shape}")
        return out

    # Otherwise interpret as ndarray-like
    x = np.asarray(atom_constraints, dtype=float)

    if x.ndim == 0:
        return [np.atleast_1d(x)]
    if x.ndim == 1:
        return [x]
    if x.ndim == 2:
        return [x[i, :] for i in range(x.shape[0])]

    raise ValueError(f"Unsupported constraint array ndim={x.ndim}, shape={x.shape}")


# ---------------------------------------------------------------------
# Stage-2: Greedy SCOT over (env, atom) pairs with STRICT per-atom constraints
# ---------------------------------------------------------------------
def scot_greedy_family_atoms_tracked_minigrid(
    U_global,
    atoms_per_env,
    constraints_per_env_atoms,   # STRICT: constraints_per_env_atoms[e][i] = list of vectors for atom i
    *,
    normalize=True,
    round_decimals=12,
):
    """
    MiniGrid-compatible SCOT greedy selection (STRICT).

    Parameters
    ----------
    U_global : (M, d) array
        Universal constraint set (deduplicated)
    atoms_per_env : list[list[Atom]]
        Candidate atoms per env
    constraints_per_env_atoms : list[list[list[np.ndarray]]]
        constraints_per_env_atoms[e][i] = list of constraint vectors (d,)
        induced by atom i in env e
    """

    U_global = np.asarray(U_global, dtype=float)
    if U_global.ndim != 2 or U_global.shape[0] == 0:
        raise ValueError(f"U_global must be non-empty 2D array, got shape={U_global.shape}")

    d = U_global.shape[1]
    n_envs = len(atoms_per_env)

    if len(constraints_per_env_atoms) != n_envs:
        raise ValueError(
            "Length mismatch: constraints_per_env_atoms must have same length as atoms_per_env "
            f"({len(constraints_per_env_atoms)} vs {n_envs})"
        )

    key_for = make_key_for(normalize=normalize, round_decimals=round_decimals)

    # -----------------------------
    # Map U_global constraint keys -> universe indices
    # -----------------------------
    key_to_uix = {}
    for idx, v in enumerate(U_global):
        k = key_for(v)
        key_to_uix.setdefault(k, []).append(idx)

    universe = set(range(len(U_global)))
    covered = set()

    chosen = []
    chosen_constraints_list = []
    inspected_env_indices = set()

    # -----------------------------
    # Stats
    # -----------------------------
    env_stats = {
        i: {
            "atoms": [],
            "indices": [],
            "coverage_counts": [],
            "total_coverage": 0,
            "was_inspected": False,
            "precompute_time": 0.0,
        }
        for i in range(n_envs)
    }

    # -----------------------------
    # Precompute coverage per atom (STRICT)
    # -----------------------------
    precompute_start = time.time()
    cov = []

    for env_idx in range(n_envs):
        env_precompute_start = time.time()

        atoms = atoms_per_env[env_idx]
        per_atom_constraints = constraints_per_env_atoms[env_idx]

        # STRICT contract check
        if len(per_atom_constraints) != len(atoms):
            raise ValueError(
                f"Env {env_idx}: constraints_per_env_atoms[e] must have one entry per atom. "
                f"Got {len(per_atom_constraints)} constraint-lists for {len(atoms)} atoms."
            )

        cov_i = []
        for atom_idx, atom_constraints in enumerate(per_atom_constraints):
            atom_cov = set()
            for v in _as_constraint_list(atom_constraints):
                k = key_for(v)
                if k in key_to_uix:
                    atom_cov.update(key_to_uix[k])
            cov_i.append(atom_cov)

        cov.append(cov_i)
        env_stats[env_idx]["precompute_time"] = time.time() - env_precompute_start

    env_stats["total_precompute_time"] = time.time() - precompute_start

    # -----------------------------
    # Greedy set cover over (env, atom)
    # -----------------------------
    greedy_start = time.time()
    iter_count = 0

    while True:
        uncovered = universe - covered
        if not uncovered:
            break

        best_gain = 0
        best_choice = None
        best_new = None

        for env_idx in range(n_envs):
            if atoms_per_env[env_idx]:
                inspected_env_indices.add(env_idx)
                env_stats[env_idx]["was_inspected"] = True

            for atom_idx, atom_cov in enumerate(cov[env_idx]):
                if not atom_cov:
                    continue
                new_cover = uncovered & atom_cov
                gain = len(new_cover)

                if gain > best_gain:
                    best_gain = gain
                    best_choice = (env_idx, atom_idx)
                    best_new = new_cover

        if best_choice is None or best_gain == 0:
            break

        env_idx, atom_idx = best_choice
        atom = atoms_per_env[env_idx][atom_idx]

        chosen.append((env_idx, atom))

        # Add actual constraint vectors for this chosen atom
        chosen_atom_constraints = constraints_per_env_atoms[env_idx][atom_idx]
        chosen_constraints_list.extend(_as_constraint_list(chosen_atom_constraints))

        covered |= best_new

        env_stats[env_idx]["atoms"].append(atom)
        env_stats[env_idx]["indices"].append(iter_count)
        env_stats[env_idx]["coverage_counts"].append(len(best_new))
        env_stats[env_idx]["total_coverage"] += len(best_new)

        iter_count += 1

    env_stats["total_greedy_time"] = time.time() - greedy_start

    # -----------------------------
    # Final outputs
    # -----------------------------
    if chosen_constraints_list:
        chosen_constraints = np.vstack([np.atleast_1d(v) for v in chosen_constraints_list])
        if chosen_constraints.ndim == 1:
            chosen_constraints = chosen_constraints[None, :]
    else:
        chosen_constraints = np.zeros((0, d))

    env_stats["total_inspected_count"] = len(inspected_env_indices)
    env_stats["total_activated_count"] = len({c[0] for c in chosen})
    env_stats["activated_env_indices"] = sorted({c[0] for c in chosen})
    env_stats["total_iterations"] = iter_count
    env_stats["final_coverage"] = len(covered)

    return chosen, env_stats, chosen_constraints


# ---------------------------------------------------------------------
# Stage-1 coverage using canonical keys (ENV-LEVEL constraints)
# ---------------------------------------------------------------------
def build_mdp_coverage_from_constraints_keys(
    U_per_env_envlevel,
    U_universal,
    *,
    normalize=True,
    round_decimals=12,
):
    """
    Stage-1 coverage in canonical key space.

    Parameters
    ----------
    U_per_env_envlevel : list[np.ndarray]
        Each entry is (n_constraints, d) or empty (0, d)
    U_universal : (M, d) array

    Returns
    -------
    mdp_cov : list[set[int]]
        Each set contains universal key-ids covered by that env
    key_to_uid : dict[key -> uid]
    uid_to_key : list[key]
    """
    key_for = make_key_for(normalize=normalize, round_decimals=round_decimals)

    U_universal = np.asarray(U_universal, dtype=float)
    if U_universal.ndim != 2 or U_universal.shape[0] == 0:
        raise ValueError("U_universal must be non-empty 2D array")

    # Unique universe in key-space
    key_to_uid = {}
    uid_to_key = []
    for u in U_universal:
        k = key_for(u)
        if k not in key_to_uid:
            key_to_uid[k] = len(uid_to_key)
            uid_to_key.append(k)

    # Per-env coverage in uid-space
    mdp_cov = []
    for env_idx, H_k in enumerate(U_per_env_envlevel):
        cov_uids = set()
        if H_k is None:
            mdp_cov.append(cov_uids)
            continue

        H_k = np.asarray(H_k, dtype=float)
        if H_k.size == 0:
            mdp_cov.append(cov_uids)
            continue

        if H_k.ndim == 1:
            H_k = H_k[None, :]

        if H_k.ndim != 2:
            raise ValueError(f"Env {env_idx}: expected (n,d) constraints, got shape={H_k.shape}")

        for row in H_k:
            kk = key_for(row)
            uid = key_to_uid.get(kk, None)
            if uid is not None:
                cov_uids.add(uid)

        mdp_cov.append(cov_uids)

    return mdp_cov, key_to_uid, uid_to_key


def greedy_select_mdps_unweighted(mdp_cov, universe_size):
    universe = set(range(universe_size))
    covered = set()
    selected = []
    selected_set = set()

    s1_iterations = 0
    s1_shallow_checks = 0

    while (universe - covered):
        best_gain = 0
        best_k = None
        best_new = None
        s1_iterations += 1

        for k, cov_k in enumerate(mdp_cov):
            if k in selected_set:
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
        selected_set.add(best_k)
        covered |= best_new

    return selected, {
        "s1_iterations": s1_iterations,
        "s1_shallow_checks": s1_shallow_checks,
        "s1_final_coverage": len(covered),
        "s1_universe_size": universe_size,
    }


# ------------------------------------------------------------
# Two-stage SCOT (NO-LEAK): Stage-1 env selection, Stage-2 atom selection
# ------------------------------------------------------------
def two_stage_scot_minigrid(
    *,
    U_universal,
    U_per_env_atoms_envlevel,    # Stage-1 input: list[np.ndarray (n,d)]
    constraints_per_env_atoms,   # Stage-2 input: list[list[list[np.ndarray]]]
    candidates_per_env,
    SFs,                         # kept for API parity; not used here
    envs,                        # kept for API parity; not used here
    normalize=True,
    round_decimals=12,
    verbose=True,
):
    """
    Stage-1: selects environments using ONLY atom-implied constraints (env-level aggregation).
    Stage-2: runs atom-level SCOT on selected envs using per-atom constraints.
    """

    n_envs = len(envs)
    if not (
        len(candidates_per_env) == len(SFs) == len(U_per_env_atoms_envlevel) ==
        len(constraints_per_env_atoms) == n_envs
    ):
        raise ValueError(
            "Length mismatch among envs / candidates_per_env / SFs / "
            "U_per_env_atoms_envlevel / constraints_per_env_atoms"
        )

    U_universal = np.asarray(U_universal, dtype=float)
    if U_universal.ndim != 2 or U_universal.shape[0] == 0:
        raise ValueError("U_universal must be non-empty 2D array")

    # -----------------------------
    # Stage-1: coverage in key space (ENV-LEVEL)
    # -----------------------------
    mdp_cov, _, uid_to_key = build_mdp_coverage_from_constraints_keys(
        U_per_env_envlevel=U_per_env_atoms_envlevel,
        U_universal=U_universal,
        normalize=normalize,
        round_decimals=round_decimals,
    )

    selected_mdps, s1_stats = greedy_select_mdps_unweighted(
        mdp_cov,
        len(uid_to_key),
    )

    if verbose:
        print("\n[Two-Stage SCOT | Stage-1]")
        print(f"Unique universe size (key-collapsed): {len(uid_to_key)}")
        print("Selected envs:", selected_mdps)
        print("Stage-1 stats:", s1_stats)

    # -----------------------------
    # Stage-2: SCOT on restricted pool (ATOM-LEVEL)
    # -----------------------------
    pool_atoms = [candidates_per_env[k] for k in selected_mdps]
    pool_constraints = [constraints_per_env_atoms[k] for k in selected_mdps]

    chosen_local, pool_stats, _ = scot_greedy_family_atoms_tracked_minigrid(
        U_global=U_universal,
        atoms_per_env=pool_atoms,
        constraints_per_env_atoms=pool_constraints,
        normalize=normalize,
        round_decimals=round_decimals,
    )

    # Map back to global env ids
    chosen_global = []
    for local_env_idx, atom in chosen_local:
        global_env_idx = selected_mdps[local_env_idx]
        chosen_global.append((global_env_idx, atom))

    activated_envs = sorted({k for k, _ in chosen_global})
    waste = len(selected_mdps) - len(activated_envs)

    if verbose:
        print("\n[Two-Stage SCOT | Stage-2]")
        print("Chosen (global):", chosen_global[:10], "..." if len(chosen_global) > 10 else "")
        print("Activated envs:", activated_envs)
        print("Waste:", waste)

    return {
        "chosen": chosen_global,
        "selected_mdps": selected_mdps,

        # Stage-1 bookkeeping
        "s1_iterations": s1_stats["s1_iterations"],
        "s1_checks": s1_stats["s1_shallow_checks"],
        "s1_unique_universe_size": s1_stats["s1_universe_size"],
        "s1_final_coverage": s1_stats["s1_final_coverage"],

        # Stage-2 bookkeeping
        "s2_iterations": len(chosen_global),
        "activated_envs": activated_envs,
        "waste": waste,
        "s2_stats": {
            "total_precompute_time": pool_stats.get("total_precompute_time", None),
            "total_greedy_time": pool_stats.get("total_greedy_time", None),
            "final_coverage": pool_stats.get("final_coverage", None),
            "total_iterations": pool_stats.get("total_iterations", None),
            "total_inspected_count": pool_stats.get("total_inspected_count", None),
            "total_activated_count": pool_stats.get("total_activated_count", None),
            "activated_env_indices_local": pool_stats.get("activated_env_indices", None),
        },
    }
