"""
SCOT with Environment-Level Heuristic Selection

Heuristic:
- For each env:
    * Sample K atoms (prioritize sample_atom if provided)
    * Extract constraints from sampled atoms
    * Remove redundant constraints
    * Score env by number of unique constraints
- Select top X% most informative envs
- Run SCOT greedy only on selected envs
"""

from typing import Sequence, Any, Tuple, List, Dict
import numpy as np
import random
import time

from utils import atom_to_constraints, remove_redundant_constraints

EnvIdx = int
AtomIdx = int
ConstraintKey = Tuple
HeuristicScore = float


# ============================================================
# Constraint hashing
# ============================================================

def _constraint_key(v: np.ndarray, *, normalize: bool, round_decimals: int) -> ConstraintKey:
    n = np.linalg.norm(v)
    if n == 0.0 or not np.isfinite(n):
        return ("ZERO",)
    vv = v / n if normalize else v
    return tuple(np.round(vv, round_decimals))


# ============================================================
# ENV HEURISTIC
# ============================================================

def rank_envs_by_constraint_informativeness(
    atoms_per_env: Sequence[Sequence[Any]],
    SFs: Sequence[Any],
    envs: Sequence[Any],
    *,
    K: int,
    sample_atom_per_env: Sequence[Any] | None = None,
    normalize: bool = True,
    round_decimals: int = 12,
) -> Tuple[List[EnvIdx], Dict]:
    """
    Returns env indices sorted by informativeness (descending).
    """

    stats = {}
    scores = []

    t0 = time.time()

    for env_idx, (atoms, sf, env) in enumerate(zip(atoms_per_env, SFs, envs)):
        mu_sa = sf[0]

        if len(atoms) == 0:
            scores.append((env_idx, 0))
            stats[env_idx] = {"unique_constraints": 0}
            continue

        # ----------------------------
        # Sample atoms
        # ----------------------------
        sampled = []

        if sample_atom_per_env is not None and sample_atom_per_env[env_idx] is not None:
            sampled.append(sample_atom_per_env[env_idx])

        remaining = [a for a in atoms if a not in sampled]
        if remaining:
            sampled.extend(random.sample(remaining, min(K - len(sampled), len(remaining))))

        # ----------------------------
        # Extract constraints
        # ----------------------------
        constraints = []
        for atom in sampled:
            constraints.extend(atom_to_constraints(atom, mu_sa, env))

        if len(constraints) == 0:
            scores.append((env_idx, 0))
            stats[env_idx] = {"unique_constraints": 0}
            continue

        # ----------------------------
        # Remove redundant constraints
        # ----------------------------
        U = remove_redundant_constraints(
            np.asarray(constraints),
        )

        score = len(U)
        scores.append((env_idx, score))
        stats[env_idx] = {
            "unique_constraints": score,
            "sampled_atoms": len(sampled),
        }

    total_time = time.time() - t0
    stats["heuristic_time"] = total_time

    # Sort descending by score
    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_envs = [i for i, _ in scores]

    return ranked_envs, stats


# ============================================================
# SCOT GREEDY (restricted to selected envs)
# ============================================================

def scot_greedy_family_atoms_with_env_filter(
    U_global,
    atoms_per_env,
    SFs,
    envs,
    *,
    selected_envs,
    normalize=True,
    round_decimals=12,
):
    U_global = np.asarray(U_global)

    # -------- build universe --------
    key_to_uix = {}
    for i, v in enumerate(U_global):
        k = _constraint_key(v, normalize=normalize, round_decimals=round_decimals)
        key_to_uix.setdefault(k, []).append(i)

    universe = set(range(len(U_global)))
    covered = set()

    chosen = []
    chosen_constraints = []

    # -------- stats init --------
    per_env = {
        i: {
            "was_inspected": False,
            "atoms_chosen": 0,
            "total_coverage": 0,
            "coverage_counts": [],
        }
        for i in range(len(atoms_per_env))
    }

    inspected_envs = set()

    # -------- precompute --------
    t0 = time.time()
    cov = {}

    for env_idx in selected_envs:
        mu_sa = SFs[env_idx][0]
        env = envs[env_idx]
        cov[env_idx] = []

        for atom in atoms_per_env[env_idx]:
            atom_cov = set()
            for v in atom_to_constraints(atom, mu_sa, env):
                k = _constraint_key(v, normalize=normalize, round_decimals=round_decimals)
                if k in key_to_uix:
                    atom_cov.update(key_to_uix[k])
            cov[env_idx].append(atom_cov)

    precompute_time = time.time() - t0

    # -------- greedy loop --------
    t1 = time.time()
    iter_count = 0

    while True:
        uncovered = universe - covered
        if not uncovered:
            break

        best_gain = 0
        best = None
        best_new = None

        for env_idx in selected_envs:
            per_env[env_idx]["was_inspected"] = True
            inspected_envs.add(env_idx)

            for atom_idx, atom_cov in enumerate(cov[env_idx]):
                gain = len(uncovered & atom_cov)
                if gain > best_gain:
                    best_gain = gain
                    best = (env_idx, atom_idx)
                    best_new = uncovered & atom_cov

        if best is None or best_gain == 0:
            break

        env_idx, atom_idx = best
        atom = atoms_per_env[env_idx][atom_idx]

        chosen.append((env_idx, atom))
        chosen_constraints.append(
            np.asarray(atom_to_constraints(atom, SFs[env_idx][0], envs[env_idx]))
        )

        covered |= best_new

        per_env[env_idx]["atoms_chosen"] += 1
        per_env[env_idx]["total_coverage"] += len(best_new)
        per_env[env_idx]["coverage_counts"].append(len(best_new))

        iter_count += 1

    greedy_time = time.time() - t1

    # -------- finalize --------
    activated_envs = sorted({e for e, _ in chosen})

    stats = {
        "total_precompute_time": precompute_time,
        "total_greedy_time": greedy_time,
        "heuristic_computation_time": 0.0,

        "final_coverage": len(covered),
        "total_iterations": iter_count,

        "total_inspected_count": len(inspected_envs),
        "total_activated_count": len(activated_envs),
        "activated_env_indices": activated_envs,

        "per_env": per_env,
    }

    if chosen_constraints:
        chosen_constraints = np.vstack(chosen_constraints)
    else:
        chosen_constraints = np.zeros((0, U_global.shape[1]))

    return chosen, stats, chosen_constraints


# ============================================================
# FULL PIPELINE
# ============================================================

def scot_with_env_heuristic(
    U_global,
    atoms_per_env,
    SFs,
    envs,
    *,
    K: int = 5,
    top_frac: float = 0.1,
    sample_atom_per_env=None,
    normalize=True,
    round_decimals=12,
):
    """
    Full pipeline:
    1) Rank envs by heuristic
    2) Keep top frac
    3) Run SCOT on filtered envs
    """

    ranked_envs, heuristic_stats = rank_envs_by_constraint_informativeness(
        atoms_per_env,
        SFs,
        envs,
        K=K,
        sample_atom_per_env=sample_atom_per_env,
        normalize=normalize,
        round_decimals=round_decimals,
    )

    n_keep = max(1, int(len(ranked_envs) * top_frac))
    selected_envs = ranked_envs[:n_keep]

    chosen, stats, chosen_constraints = (
        scot_greedy_family_atoms_with_env_filter(
            U_global,
            atoms_per_env,
            SFs,
            envs,
            selected_envs=selected_envs,
            normalize=normalize,
            round_decimals=round_decimals,
        )
    )

    stats["heuristic"] = heuristic_stats

    return chosen, stats, chosen_constraints

