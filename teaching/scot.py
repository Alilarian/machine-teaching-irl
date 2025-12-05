import numpy as np
import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from utils import atom_to_constraints

def scot_greedy_family_atoms_tracked(
    U_global,
    atoms_per_env,
    SFs,
    envs,
    *,
    normalize=True,
    round_decimals=12,
):
    """
    SCOT greedy selection over atoms with full environment tracking.

    Returns:
        chosen_atoms: list of (env_idx, Atom)
        env_stats: {
            env_idx: {
                'atoms': [Atom, Atom, ...],
                'indices': [0, 5, 9, ...],   # SCOT iteration numbers
                'coverage_counts': [12, 4, ...],  # new constraints covered each time
                'total_coverage': int
            }
        }
    """

    # ---------- Utility ----------
    def key_for(v):
        n = np.linalg.norm(v)
        if n == 0.0 or not np.isfinite(n):
            return ("ZERO",)
        vv = v / n if normalize else v
        return tuple(np.round(vv, round_decimals))

    # ---------- Build U_global dictionary ----------
    key_to_uix = {}
    for idx, v in enumerate(U_global):
        key_to_uix.setdefault(key_for(v), []).append(idx)

    universe = set(range(len(U_global)))
    covered  = set()
    chosen   = []

    # ---------- Tracking state ----------
    env_stats = {
        i: {
            "atoms": [],
            "indices": [],
            "coverage_counts": [],
            "total_coverage": 0,
        }
        for i in range(len(atoms_per_env))
    }

    # ---------- Precompute coverage for each atom ----------

    cov = []
    mu_sa_list = [sf[0] for sf in SFs]

    for env_idx, (atoms, sf, env) in enumerate(zip(atoms_per_env, SFs, envs)):
        mu_sa = sf[0]
        cov_i = []

        for atom in atoms:
            constraints = atom_to_constraints(atom, mu_sa, env)

            covered_set = set()
            for v in constraints:
                k = key_for(v)
                if k in key_to_uix:
                    covered_set.update(key_to_uix[k])

            cov_i.append(covered_set)

        cov.append(cov_i)

    # ---------- Greedy Loop ----------
    iter_count = 0

    while True:
        uncovered = universe - covered
        if not uncovered:
            break

        best_gain = 0
        best_atom = None
        best_new  = None

        for i in range(len(atoms_per_env)):
            for j, covered_by_atom in enumerate(cov[i]):
                if not covered_by_atom:
                    continue

                new_cover = uncovered & covered_by_atom
                gain = len(new_cover)

                if gain > best_gain:
                    best_gain = gain
                    best_atom = (i, j)
                    best_new = new_cover

        if best_atom is None:
            break

        i, j = best_atom
        atom = atoms_per_env[i][j]

        # Add to chosen list
        chosen.append((i, atom))
        covered |= best_new

        # ---------- Update env_stats ----------
        env_stats[i]["atoms"].append(atom)
        env_stats[i]["indices"].append(iter_count)
        env_stats[i]["coverage_counts"].append(len(best_new))
        env_stats[i]["total_coverage"] += len(best_new)

        iter_count += 1

    return chosen, env_stats

## how many envs got activated
## each env 
## we can provide some examples whereE-stop is more informative than others. specifically env without terminals
## but in expectation story is different