import numpy as np
from scot import scot_greedy_family_atoms_tracked


def _make_key(v, decimals=12):
    n = np.linalg.norm(v)
    if n == 0 or not np.isfinite(n):
        return ("ZERO",)
    return tuple(np.round(v / n, decimals))

def build_mdp_coverage_from_constraints(
    U_per_env,
    U_global,
    *,
    decimals=12,
):
    """
    U_per_env[k] : list/array of constraint vectors for env k
    U_global     : global constraint array

    Returns:
        mdp_cov[k] : set of indices into U_global
    """
    key_to_uix = {}
    for i, v in enumerate(U_global):
        key_to_uix.setdefault(_make_key(v, decimals), []).append(i)

    mdp_cov = []
    for H_k in U_per_env:
        cov_k = set()
        for v in H_k:
            k = _make_key(v, decimals)
            if k in key_to_uix:
                cov_k.update(key_to_uix[k])
        mdp_cov.append(cov_k)

    return mdp_cov

def greedy_select_mdps_no_cost(mdp_cov, universe_size):
    """
    Select MDPs greedily by marginal constraint coverage.

    Returns:
        selected_mdps : list[int]
        stats         : dict
    """
    universe = set(range(universe_size))
    covered = set()
    selected = []

    stats = {
        "order": [],
        "per_mdp": {},
        "final_covered": 0,
    }

    while covered != universe:
        best_gain = 0
        best_k = None
        best_new = None

        for k, cov_k in enumerate(mdp_cov):
            if k in selected:
                continue
            new = cov_k - covered
            if len(new) > best_gain:
                best_gain = len(new)
                best_k = k
                best_new = new

        if best_k is None or best_gain == 0:
            break

        selected.append(best_k)
        covered |= best_new

        stats["order"].append(best_k)
        stats["per_mdp"][best_k] = {
            "gain": best_gain,
            "cumulative": len(covered),
        }

    stats["final_covered"] = len(covered)
    return selected, stats


def two_stage_scot_no_cost(
    *,
    U_universal,
    U_per_env_atoms,
    U_per_env_q,
    candidates_per_env,
    SFs,
    envs,
):
    """
    Full two-stage SCOT:
      1) MDP selection via constraint coverage
      2) SCOT greedy restricted to selected MDPs
    """

    # --------------------------------------------------
    # Stage 0: combine per-env constraints
    # --------------------------------------------------
    U_per_env = []
    for H_a, H_q in zip(U_per_env_atoms, U_per_env_q):
        if len(H_a) and len(H_q):
            U_per_env.append(np.vstack([H_a, H_q]))
        elif len(H_a):
            U_per_env.append(H_a)
        elif len(H_q):
            U_per_env.append(H_q)
        else:
            U_per_env.append(np.zeros((0, U_universal.shape[1])))

    # --------------------------------------------------
    # Stage 1: MDP selection
    # --------------------------------------------------
    mdp_cov = build_mdp_coverage_from_constraints(
        U_per_env,
        U_universal,
    )

    selected_mdps, mdp_stats = greedy_select_mdps_no_cost(
        mdp_cov,
        universe_size=len(U_universal),
    )

    # Fallback safety
    if len(selected_mdps) == 0:
        selected_mdps = list(range(len(envs)))

    # --------------------------------------------------
    # Stage 2: SCOT on selected MDPs
    # --------------------------------------------------
    cand_sel = [candidates_per_env[k] for k in selected_mdps]
    SFs_sel  = [SFs[k] for k in selected_mdps]
    envs_sel = [envs[k] for k in selected_mdps]

    chosen_local, env_stats_local, chosen_constraints = scot_greedy_family_atoms_tracked(
        U_universal,
        cand_sel,
        SFs_sel,
        envs_sel,
    )

    # Remap env indices back to original
    chosen_global = [
        (selected_mdps[env_idx], atom)
        for env_idx, atom in chosen_local
    ]

    activated_envs = sorted({i for i, _ in chosen_global})

    return {
        "selected_mdps": selected_mdps,
        "activated_envs": activated_envs,
        "chosen_atoms": chosen_global,
        "env_stats": env_stats_local,
        "chosen_constraints": chosen_constraints,
        "mdp_stats": mdp_stats,
    }
