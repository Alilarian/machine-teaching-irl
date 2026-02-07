import numpy as np
import mdp
from teaching import scot_greedy_family_atoms_tracked

# ---------------------------------------------------------------------
# Canonicalization: MUST match Stage-2 (scot_greedy_family_atoms_tracked)
# ---------------------------------------------------------------------
def make_key_for(*, normalize=True, round_decimals=12):
    def key_for(v):
        v = np.asarray(v)
        n = np.linalg.norm(v)
        if n == 0.0 or not np.isfinite(n):
            return ("ZERO",)
        vv = (v / n) if normalize else v
        return tuple(np.round(vv, round_decimals))
    return key_for


# ---------------------------------------------------------------------
# Stage-1 coverage using canonical keys (FIXED)
# ---------------------------------------------------------------------
def build_mdp_coverage_from_constraints_keys(
    U_per_env,
    U_universal,
    *,
    normalize=True,
    round_decimals=12,
):
    """
    Build MDP coverage sets in the SAME constraint identity space as Stage-2:

    - Canonicalize each universal constraint into a key (direction-only, rounded).
    - Canonicalize each env constraint into a key.
    - MDP covers a universal element iff keys match.

    Returns:
        mdp_cov: list[set[int]] where each set contains "unique universal key ids"
        key_to_uid: dict[key -> uid]
        uid_to_key: list[key]
    """
    key_for = make_key_for(normalize=normalize, round_decimals=round_decimals)

    # Unique universe in key-space (collapses duplicates properly)
    key_to_uid = {}
    uid_to_key = []
    for u in np.asarray(U_universal):
        k = key_for(u)
        if k not in key_to_uid:
            key_to_uid[k] = len(uid_to_key)
            uid_to_key.append(k)

    # Per-env coverage in uid-space
    mdp_cov = []
    for H_k in U_per_env:
        cov_uids = set()
        H_k = np.asarray(H_k)
        if H_k.size != 0:
            for row in H_k:
                kk = key_for(row)
                uid = key_to_uid.get(kk, None)
                if uid is not None:
                    cov_uids.add(uid)
        mdp_cov.append(cov_uids)

    return mdp_cov, key_to_uid, uid_to_key


def greedy_select_mdps_unweighted(mdp_cov, universe_size):
    """
    Greedy set cover over MDPs with NO cost (unweighted).
    Universe elements are integers [0..universe_size-1].

    FIX:
      - Terminate on uncovered (universe - covered), not covered == universe.
      - Uses sets for uniqueness (already correct) + selected_set for speed.
    """
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

        # No progress possible
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
# Two-stage SCOT (cost-free) — FIXED Stage-1 coverage
# ------------------------------------------------------------
def two_stage_scot(
    *,
    U_universal,
    U_per_env_atoms,
    U_per_env_q,
    candidates_per_env,
    SFs,
    envs,
    normalize=True,
    round_decimals=12,
):
    # --------------------------------------------------------
    # Stage 0: Constraint aggregation per MDP (ROBUST)
    # --------------------------------------------------------
    n_envs = len(envs)
    U_universal = np.asarray(U_universal)
    d = U_universal.shape[1]

    U_per_env = []
    for k in range(n_envs):
        # Atom constraints for env k
        if U_per_env_atoms is not None and k < len(U_per_env_atoms) and len(U_per_env_atoms[k]) > 0:
            H_a = np.asarray(U_per_env_atoms[k])
        else:
            H_a = np.zeros((0, d))

        # Q constraints for env k
        if U_per_env_q is not None and k < len(U_per_env_q) and len(U_per_env_q[k]) > 0:
            H_q = np.asarray(U_per_env_q[k])
        else:
            H_q = np.zeros((0, d))

        # Combine safely
        if H_a.shape[0] > 0 and H_q.shape[0] > 0:
            combined = np.vstack([H_a, H_q])
        elif H_a.shape[0] > 0:
            combined = H_a
        elif H_q.shape[0] > 0:
            combined = H_q
        else:
            combined = np.zeros((0, d))

        U_per_env.append(combined)

    # --------------------------------------------------------
    # Stage 1: Build MDP coverage in CANONICAL key space (FIXED)
    # --------------------------------------------------------
    mdp_cov, key_to_uid, uid_to_key = build_mdp_coverage_from_constraints_keys(
        U_per_env,
        U_universal,
        normalize=normalize,
        round_decimals=round_decimals,
    )

    print("MDP coverages (uid-space):")
    print(mdp_cov)
    print(f"Stage-1 unique-universe size (after key collapse): {len(uid_to_key)}")

    selected_mdps, s1_stats = greedy_select_mdps_unweighted(
        mdp_cov,
        len(uid_to_key),
    )

    print("Selected MDPs inside the two-stage:")
    print(selected_mdps)
    print("Stage-1 stats:", s1_stats)

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
        normalize=normalize,
        round_decimals=round_decimals,
    )

    print("Inside the two-stage (local chosen):")
    print(chosen_local)

    # Map back to global MDP indices
    chosen_global = []
    for local_env_idx, atom in chosen_local:
        global_env_idx = selected_mdps[local_env_idx]
        chosen_global.append((global_env_idx, atom))

    activated_envs = sorted({k for k, _ in chosen_global})
    waste = len(selected_mdps) - len(activated_envs)

    return {
        "chosen": chosen_global,

        # Stage-1 bookkeeping
        "s1_iterations": s1_stats["s1_iterations"],
        "s1_checks": s1_stats["s1_shallow_checks"],
        "s1_unique_universe_size": s1_stats["s1_universe_size"],
        "s1_final_coverage": s1_stats["s1_final_coverage"],
        "selected_mdps": selected_mdps,

        # Stage-2 bookkeeping
        "s2_iterations": len(chosen_global),
        "activated_envs": activated_envs,
        "waste": waste,

        # Optional: keep pool_stats if you want more stage-2 timings/coverage
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