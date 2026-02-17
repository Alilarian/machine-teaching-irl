# ============================================================
# derive_constraints.py  (FULL FIXED VERSION)
# ============================================================

import numpy as np
from .successor_features import compute_successor_features_iterative_from_q
from .lp_redundancy import remove_redundant_constraints
from concurrent.futures import ProcessPoolExecutor
import itertools
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# 1. Successor Features Family Wrapper
# ============================================================

def _sf_worker(args):
    env, q, kw = args
    return compute_successor_features_iterative_from_q(env, q, **kw)

def compute_successor_features_family(
    envs,
    Q_list,
    *,
    n_jobs=None,
    **kw,
):
    """
    Parallel version of compute_successor_features_family.
    Uses process pool — safe & fast for CPU-heavy SF computations.
    """

    worker_args = [(env, q, kw) for env, q in zip(envs, Q_list)]

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # results = [(mu_sa, mu_s, Phi, P_pi), ...]
        results = list(executor.map(_sf_worker, worker_args))

    return results


# ============================================================
# 2. Q-based Optimality Constraints (State-Action Level)
# ============================================================

def derive_constraints_from_q_ties(
    mu_sa,
    q_values,
    env,
    tie_eps=1e-10,
    skip_terminals=True,
    normalize=True,
    tol=1e-12,
):
    S, A, d = mu_sa.shape
    q = np.asarray(q_values, float)

    m = np.max(q, axis=1, keepdims=True)
    argmax_mask = np.abs(q - m) <= tie_eps

    if skip_terminals and getattr(env, "terminal_states", None) is not None:
        terms = np.array(env.terminal_states, dtype=int)
        argmax_mask[terms] = False

    constraints = []
    for s in range(S):
        A_star = np.where(argmax_mask[s])[0]
        if A_star.size == 0: continue
        B = np.where(~argmax_mask[s])[0]
        if B.size == 0: continue

        psi_s = mu_sa[s]
        for a_star in A_star:
            diffs = psi_s[a_star][None, :] - psi_s[B]
            norms = np.linalg.norm(diffs, axis=1)

            for i, b in enumerate(B):
                if norms[i] <= tol: continue
                v = diffs[i] / norms[i] if normalize else diffs[i]
                constraints.append((v, s, a_star, b))

    return constraints

# ============================================================
# 3. Demo Constraints
# ============================================================

def constraints_from_demo(traj, mu_sa, env=None, normalize=True, tol=1e-12):
    """
    Demo: [(s,a), ...]
    Builds constraints psi(s,a*) - psi(s,a) for all a!=a*.
    """
    mu_sa = np.asarray(mu_sa)
    S, A, d = mu_sa.shape
    constraints = []

    if traj is None:
        return []

    for s, a_star in traj:
        if a_star is None:
            continue
        s, a_star = int(s), int(a_star)
        if not (0 <= s < S) or not (0 <= a_star < A):
            continue

        psi_s = mu_sa[s]
        others = np.arange(A)[np.arange(A) != a_star]
        diffs = psi_s[a_star][None, :] - psi_s[others]
        norms = np.linalg.norm(diffs, axis=1)

        for i, b in enumerate(others):
            if norms[i] <= tol: continue
            v = diffs[i] / norms[i] if normalize else diffs[i]
            constraints.append(v)

    return constraints


# ============================================================
# 4. Pairwise, E-stop, Improvement Constraints
# ============================================================

def constraints_from_pairwise(atom_data, env):
    preferred, other = atom_data

    # extract state indices only
    pref_states = [s for s, _ in preferred]
    other_states = [s for s, _ in other]

    preferred_feats = env.state_features[pref_states].sum(axis=0)
    other_feats = env.state_features[other_states].sum(axis=0)

    diff = preferred_feats - other_feats
    norm = np.linalg.norm(diff)

    if norm == 0:
        return []

    return [diff / norm]

def constraints_from_estop(atom_data, env):
    traj, t_stop = atom_data

    states_full = [s for s, _ in traj]
    states_partial = states_full[: t_stop + 1]

    feats_up_to_t = env.state_features[states_partial].sum(axis=0)
    full_feats = env.state_features[states_full].sum(axis=0)

    diff = feats_up_to_t - full_feats
    norm = np.linalg.norm(diff)

    if norm == 0.0:
        return []

    return [diff / norm]


def constraints_from_improvement(atom_data, env):
    improved, original = atom_data

    imp_states = [s for s, _ in improved]
    org_states = [s for s, _ in original]

    imp_feats = env.state_features[imp_states].sum(axis=0)
    org_feats = env.state_features[org_states].sum(axis=0)

    diff = imp_feats - org_feats
    norm = np.linalg.norm(diff)

    if norm == 0.0:
        #return [np.zeros_like(diff)]
        return []

    return [diff / norm]

# def constraints_from_pairwise(atom_data, env, gamma=1):
#     preferred, other = atom_data
    
#     pref_states = [s for s, _ in preferred]
#     other_states = [s for s, _ in other]
    
#     # lengths may differ → we usually discount from the beginning of each traj
#     n_pref = len(pref_states)
#     n_other = len(other_states)
    
#     discounts_pref = np.power(gamma, np.arange(n_pref))
#     discounts_other = np.power(gamma, np.arange(n_other))
    
#     preferred_feats = (env.state_features[pref_states] * discounts_pref[:, None]).sum(axis=0)
#     other_feats     = (env.state_features[other_states] * discounts_other[:, None]).sum(axis=0)
    
#     diff = preferred_feats - other_feats
#     norm = np.linalg.norm(diff)
    
#     #if norm < 1e-9:           # slightly safer threshold
#     #    return []
#     return [diff / norm]       # ← or [diff] if you follow the "no unit norm" advice

# def constraints_from_estop(atom_data, env, gamma=1):
#     traj, t_stop = atom_data
#     states_full = [s for s, _ in traj]
    
#     n = len(states_full)
#     discounts = np.power(gamma, np.arange(n))
    
#     feats_weighted = env.state_features[states_full] * discounts[:, None]
    
#     sum_up_to_t   = feats_weighted[:t_stop+1].sum(axis=0)
#     sum_full      = feats_weighted.sum(axis=0)
    
#     diff = sum_up_to_t - sum_full
#     norm = np.linalg.norm(diff)
#     #if norm < 1e-9:
#     #    return []
#     return [diff / norm]

# def constraints_from_improvement(atom_data, env, gamma=1):
#     improved, original = atom_data
    
#     imp_states = [s for s, _ in improved]
#     org_states = [s for s, _ in original]
    
#     n_imp = len(imp_states)
#     n_org = len(org_states)
    
#     disc_imp = np.power(gamma, np.arange(n_imp))
#     disc_org = np.power(gamma, np.arange(n_org))
    
#     imp_feats = (env.state_features[imp_states] * disc_imp[:, None]).sum(axis=0)
#     org_feats = (env.state_features[org_states] * disc_org[:, None]).sum(axis=0)
    
#     diff = imp_feats - org_feats
#     norm = np.linalg.norm(diff)
#     #if norm < 1e-9:
#     #    return []
#     return [diff / norm]



# ============================================================
# 5. Atom → Constraint Dispatcher
# ============================================================

def atom_to_constraints(atom, mu_sa, env):
    t = atom.feedback_type
    data = atom.data

    if t in ("demo", "random_traj"):
        return constraints_from_demo(data, mu_sa, env=env)

    if t == "optimal_sa":
        if isinstance(data, tuple):
            data = [data]    # convert (s,a) to [(s,a)]
        return constraints_from_demo(data, mu_sa, env=env)

    if t == "pairwise":
        return constraints_from_pairwise(data, env)

    if t == "estop":
        return constraints_from_estop(data, env)

    if t == "improvement":
        return constraints_from_improvement(data, env)

    raise ValueError(f"Unknown atom type: {t}")

# ============================================================
# 6. Atom-based Constraint Builder (GLOBAL + PER-ENV)
# ============================================================


### try to make this threaded
def _derive_constraints_one_env(args):
    atoms, sf, env, precision = args
    mu_sa = sf[0]
    env_constraints = []

    for atom in atoms:
        env_constraints.extend(atom_to_constraints(atom, mu_sa, env))

    return env_constraints

def derive_constraints_from_atoms(
    atoms_per_env,
    SFs,
    envs,
    *,
    precision=1e-3,
    lp_epsilon=1e-4,
    max_workers=None,
):
    all_constraints = []
    U_per_env = []

    tasks = list(zip(atoms_per_env, SFs, envs, [precision] * len(envs)))

    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     results = list(executor.map(_derive_constraints_one_env, tasks))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_derive_constraints_one_env, tasks))

    # ----------------------------
    # Collect per-env constraints
    # ----------------------------
    for env_constraints, sf in zip(results, SFs):
        if len(env_constraints) == 0:
            d = sf[0].shape[-1]
            U_per_env.append(np.zeros((0, d)))
            continue
        
        #U_per_env.append(np.asarray(env_constraints))
        
        U_per_env.append(remove_redundant_constraints(np.asarray(env_constraints)))
        all_constraints.extend(env_constraints)

    # ----------------------------
    # Global uniqueness (serial)
    # ----------------------------
    unique = []
    for v in all_constraints:
        v = np.asarray(v)
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            continue

        is_close = any(
            np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u)) > 1 - precision
            for u in unique
        )
        if not is_close:
            unique.append(v)

    U_global = np.array(
        remove_redundant_constraints(unique, epsilon=lp_epsilon)
    )

    return U_per_env, U_global

# ============================================================
# 7. Q-only Constraint Builder
# ============================================================

def _derive_constraints_from_q_one_env(args):
    mu_sa, q, env, tie_eps, skip_terminals, normalize, tol = args

    cons = derive_constraints_from_q_ties(
        mu_sa,
        q,
        env,
        tie_eps=tie_eps,
        skip_terminals=skip_terminals,
        normalize=normalize,
        tol=tol,
    )

    # H_i = list of constraint vectors
    return [c[0] for c in cons]


## change this to process and cpu instead of thread
def derive_constraints_from_q_family(
    SFs,
    Q_list,
    envs,
    *,
    tie_eps=1e-10,
    skip_terminals=True,
    normalize=True,
    tol=1e-12,
    precision=1e-3,
    lp_epsilon=1e-4,
    max_workers=None,
):
    U_per_mdp = []
    all_H = []

    tasks = [
        (
            sf[0],      # mu_sa
            q,
            env,
            tie_eps,
            skip_terminals,
            normalize,
            tol,
        )
        for sf, q, env in zip(SFs, Q_list, envs)
    ]

    # ---------------------------
    # Parallel per-env extraction
    # ---------------------------
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_derive_constraints_from_q_one_env, tasks))

    # ---------------------------
    # Collect results
    # ---------------------------
    for H_i in results:
        U_per_mdp.append(H_i)
        all_H.extend(H_i)

    # ---------------------------
    # Global uniqueness (serial)
    # ---------------------------
    pre = []
    for v in all_H:
        v = np.asarray(v)
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            continue

        is_close = any(
            np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u)) > 1 - precision
            for u in pre
        )
        if not is_close:
            pre.append(v)

    if not pre:
        d = SFs[0][0].shape[-1]
        return U_per_mdp, np.zeros((0, d))

    U_global = np.array(
        remove_redundant_constraints(pre, epsilon=lp_epsilon)
    )

    return U_per_mdp, U_global

def recover_constraints_and_coverage(
    chosen_atoms,
    SFs,
    envs,
    U_universal,
):
    """
    Returns:
      - n_unique_constraints
      - coverage_fraction
    """
    if len(chosen_atoms) == 0:
        return 0, 0.0
    
    num_envs = len(envs)
    atoms_per_env = [[] for _ in range(num_envs)]

    for env_idx, atom in chosen_atoms:
        if env_idx < 0 or env_idx >= num_envs:
            raise ValueError(f"Invalid env_idx={env_idx} in atoms_flat.")
        atoms_per_env[env_idx].append(atom)

    _, U_chosen = derive_constraints_from_atoms(
        atoms_per_env,
        SFs,
        envs,
    )

    if U_chosen is None or len(U_chosen) == 0:
        return 0, 0.0

    U_chosen_unique = remove_redundant_constraints(U_chosen)

    # union test for coverage
    union = remove_redundant_constraints(
        np.vstack([U_universal, U_chosen_unique])
    )

    n_unique = len(U_chosen_unique)
    coverage = n_unique / len(U_universal)

    return n_unique, coverage

