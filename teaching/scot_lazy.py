import numpy as np
import heapq
from utils import atom_to_constraints


def scot_greedy_family_atoms_tracked_lazy(
    U_global,
    atoms_per_env,
    SFs,
    envs,
    *,
    normalize=True,
    round_decimals=12,
):
    """
    Lazy greedy SCOT over atoms across envs.
    Pool-based: only uses the atoms_per_env/envs/SFs you pass in.

    Returns:
        chosen: list[(env_idx, atom_obj)]   # NOTE: atom object, like your current code
        env_stats: dict (includes lazy-specific counters)
        chosen_constraints: np.ndarray
    """
    U_global = np.asarray(U_global)

    # ---------------------------
    # Vector -> key normalization
    # ---------------------------
    def key_for(v):
        n = np.linalg.norm(v)
        if n == 0.0 or not np.isfinite(n):
            return ("ZERO",)
        vv = v / n if normalize else v
        return tuple(np.round(vv, round_decimals))

    # Map each universal constraint vector -> universal index/indices
    key_to_uix = {}
    for idx, v in enumerate(U_global):
        key_to_uix.setdefault(key_for(v), []).append(idx)

    universe = set(range(len(U_global)))
    covered = set()
    chosen = []
    chosen_constraints_list = []

    # ---------------------------
    # Stats
    # ---------------------------
    env_stats = {
        i: {
            "atoms": [],
            "indices": [],
            "coverage_counts": [],
            "total_coverage": 0,
            "was_inspected": False,
            "lazy_atom_recomputations": 0,   # how many times atoms from this env were "checked"
        }
        for i in range(len(atoms_per_env))
    }

    # Global lazy stats
    lazy_stats = {
        "initial_atom_inspections": 0,   # unavoidable one-time construction
        "atom_recomputations": 0,        # meaningful: true marginal gain recomputed
        "heap_pops": 0,
    }

    # ---------------------------
    # Build per-atom coverage once
    # ---------------------------
    # cov[env_idx][atom_idx] = set of universal indices covered by that atom
    cov = []
    for env_idx, (atom_list, sf, env) in enumerate(zip(atoms_per_env, SFs, envs)):
        mu_sa = sf[0]
        cov_i = []
        for atom in atom_list:
            constraints = atom_to_constraints(atom, mu_sa, env)
            atom_cov = set()
            for v in constraints:
                k = key_for(v)
                if k in key_to_uix:
                    atom_cov.update(key_to_uix[k])
            cov_i.append(atom_cov)
            lazy_stats["initial_atom_inspections"] += 1
        cov.append(cov_i)

    # ---------------------------
    # Lazy heap initialization
    # ---------------------------
    # Upper bound for an atom is simply |cov(atom)| because uncovered ⊆ universe.
    # True gain = |cov(atom) ∩ uncovered| ≤ |cov(atom)|.
    #
    # heap items: (-ub, env_idx, atom_idx, seen_covered_size)
    heap = []
    for env_idx in range(len(cov)):
        for atom_idx, atom_cov in enumerate(cov[env_idx]):
            ub = len(atom_cov)
            if ub <= 0:
                continue
            heapq.heappush(heap, (-ub, env_idx, atom_idx, 0))

    # ---------------------------
    # Lazy greedy loop
    # ---------------------------
    iter_count = 0
    while True:
        uncovered = universe - covered
        if not uncovered:
            break
        if not heap:
            break

        neg_score, env_idx, atom_idx, seen_cov_size = heapq.heappop(heap)
        lazy_stats["heap_pops"] += 1

        # If "covered" changed since this atom was last evaluated, recompute its TRUE marginal gain.
        if seen_cov_size != len(covered):
            atom_cov = cov[env_idx][atom_idx]
            true_new = atom_cov & uncovered
            true_gain = len(true_new)

            lazy_stats["atom_recomputations"] += 1
            env_stats[env_idx]["lazy_atom_recomputations"] += 1
            env_stats[env_idx]["was_inspected"] = True

            if true_gain > 0:
                # push back with updated true gain (which becomes the new upper bound until covered changes again)
                heapq.heappush(heap, (-true_gain, env_idx, atom_idx, len(covered)))
            # else: discard this atom (it can never help again because covered only grows)
            continue

        # Fresh score: at this point, neg_score corresponds to a true_gain computed at current covered_size.
        gain = -neg_score
        if gain <= 0:
            continue

        # Select this atom
        atom = atoms_per_env[env_idx][atom_idx]
        mu_sa = SFs[env_idx][0]
        constraints_for_atom = atom_to_constraints(atom, mu_sa, envs[env_idx])
        chosen_constraints_list.append(np.array(constraints_for_atom))

        # Update coverage using exact set (safe)
        best_new = cov[env_idx][atom_idx] & uncovered

        chosen.append((env_idx, atom))
        covered |= best_new

        env_stats[env_idx]["atoms"].append(atom)
        env_stats[env_idx]["indices"].append(iter_count)
        env_stats[env_idx]["coverage_counts"].append(len(best_new))
        env_stats[env_idx]["total_coverage"] += len(best_new)
        env_stats[env_idx]["was_inspected"] = True

        iter_count += 1

    # ---------------------------
    # Pack chosen constraints
    # ---------------------------
    if chosen_constraints_list:
        chosen_constraints = np.vstack(chosen_constraints_list)
    else:
        chosen_constraints = np.zeros((0, U_global.shape[1]))

    # env-level summaries
    env_stats["total_inspected_count"] = sum(
        1 for i in range(len(atoms_per_env)) if env_stats[i]["was_inspected"]
    )
    env_stats["activated_env_indices"] = sorted({c[0] for c in chosen})

    # attach lazy global stats
    env_stats["lazy_global"] = lazy_stats

    return chosen, env_stats, chosen_constraints
