# import numpy as np
# from .successor_features import compute_successor_features_iterative_from_q
# from .lp_redundancy import remove_redundant_constraints




# # ============================================================
# # 1. Successor Features: Family Wrapper
# # ============================================================

# def compute_successor_features_family(envs, Q_list, **kw):
#     """
#     Compute successor features for each env from its Q-function.

#     Assumes you have:
#         compute_successor_features_iterative_from_q(env, q, **kw)
#     implemented elsewhere.
#     """
#     SFs = []
#     for env, q in zip(envs, Q_list):
#         mu_sa, mu_s, Phi, P_pi = compute_successor_features_iterative_from_q(env, q, **kw)
#         SFs.append((mu_sa, mu_s, Phi, P_pi))
#     return SFs


# # ============================================================
# # 2. DEMO Constraints (State-action, successor-feature-based)
# # ============================================================

# def derive_constraints_from_q_ties(
#     mu_sa,
#     q_values,
#     env,
#     tie_eps=1e-10,
#     skip_terminals=True,
#     normalize=True,
#     tol=1e-12,
# ):
#     S, A, d = mu_sa.shape
#     q = np.asarray(q_values, float)

#     m = np.max(q, axis=1, keepdims=True)
#     argmax_mask = np.abs(q - m) <= tie_eps

#     if skip_terminals and getattr(env, "include_terminal", False):
#         terms = np.array(env.terminal_states or [], dtype=int)
#         if len(terms) > 0:
#             argmax_mask[terms] = False

#     constraints = []
#     for s in range(S):
#         A_star = np.where(argmax_mask[s])[0]
#         if A_star.size == 0:
#             continue
#         B = np.where(~argmax_mask[s])[0]
#         if B.size == 0:
#             continue

#         psi_s = mu_sa[s]
#         for a_star in A_star:
#             diffs = psi_s[a_star][None, :] - psi_s[B]
#             norms = np.linalg.norm(diffs, axis=1)
#             for i, b in enumerate(B):
#                 if norms[i] <= tol:
#                     continue
#                 v = diffs[i] / norms[i] if normalize else diffs[i]
#                 constraints.append((v, s, a_star, b))

#     return constraints


# # ============================================================
# # 3. Pairwise Preference Constraints (Trajectory-level)
# # ============================================================

# def constraints_from_pairwise(atom_data, env):
#     """
#     Pairwise preference constraint (preferred_traj, other_traj).
#     Uses feature sums φ(s) from env.get_state_feature(s).
#     """
#     preferred, other = atom_data

#     preferred_feats = [env.get_state_feature(s) for s, _ in preferred]
#     other_feats     = [env.get_state_feature(s) for s, _ in other]

#     preferred_sum = np.sum(preferred_feats, axis=0)
#     other_sum     = np.sum(other_feats, axis=0)

#     diff = preferred_sum - other_sum
#     norm = np.linalg.norm(diff)

#     return [diff / norm if norm != 0 else np.zeros_like(diff)]


# # ============================================================
# # 4. E-stop Constraints (Trajectory-level)
# # ============================================================

# def constraints_from_estop(atom_data, env):
#     """
#     E-stop constraint (trajectory, t_stop).
#     diff = Σ_{k<=t_stop} φ(s_k) - Σ_{k} φ(s_k)
#     """
#     traj, t_stop = atom_data

#     feats_up_to_t = [env.get_state_feature(s) for s, _ in traj[:t_stop+1]]
#     full_feats    = [env.get_state_feature(s) for s, _ in traj]

#     sum_up_to_t = np.sum(feats_up_to_t, axis=0)
#     full_sum    = np.sum(full_feats, axis=0)

#     diff = sum_up_to_t - full_sum
#     norm = np.linalg.norm(diff)

#     return [diff / norm if norm != 0 else np.zeros_like(diff)]

# # ============================================================
# # 5. Improvement Constraints (Trajectory-level)
# # ============================================================

# def constraints_from_improvement(atom_data, env):
#     """
#     Improvement / Correction constraint (improved_traj, original_traj).
#     diff = Σ φ(s)_improved - Σ φ(s)_original
#     """
#     improved, original = atom_data

#     feats_imp = [env.get_state_feature(s) for s, _ in improved]
#     feats_org = [env.get_state_feature(s) for s, _ in original]

#     imp_sum = np.sum(feats_imp, axis=0)
#     org_sum = np.sum(feats_org, axis=0)

#     diff = imp_sum - org_sum
#     norm = np.linalg.norm(diff)

#     return [diff / norm if norm != 0 else np.zeros_like(diff)]


# # ============================================================
# # 6. Dispatcher for Atom → Constraint Vectors
# # ============================================================

# def atom_to_constraints(atom, mu_sa, env):
#     """
#     Returns list of constraint vectors (R^d) for this atom.
#     """
#     t = atom.feedback_type
#     d = atom.data

#     if t in ("demo", "random_traj"):
#         return constraints_from_demo(d, mu_sa)

#     if t == "pairwise":
#         return constraints_from_pairwise(d, env)

#     if t == "estop":
#         return constraints_from_estop(d, env)

#     if t == "improvement":
#         return constraints_from_improvement(d, env)

#     raise ValueError(f"Unknown feedback type: {t}")


# # ============================================================
# # 7. Global/Per-env Constraint Builder
# # ============================================================

# def derive_constraints_from_atoms(
#     atoms_per_env,
#     SFs,
#     envs,
#     *,
#     precision=1e-3,
#     lp_epsilon=1e-4,
# ):
#     """
#     atoms_per_env: List[List[Atom]]
#     SFs:           List[(mu_sa, mu_s, Phi, P_pi)]
#     envs:          List[Env]

#     Returns:
#         U_per_env:  list of np.array (k_i × d), constraint vectors per env
#         U_global:   np.array (K × d), deduped global constraint set
#     """
#     all_constraints = []
#     U_per_env = []

#     for atoms, sf, env in zip(atoms_per_env, SFs, envs):
#         mu_sa = sf[0]  # (S, A, d)
#         env_constraints = []

#         # 1) Extract constraints for all atoms
#         for atom in atoms:
#             env_constraints.extend(atom_to_constraints(atom, mu_sa, env))

#         # 2) Remove LP redundancy inside each env
#         if len(env_constraints) == 0:
#             d = mu_sa.shape[-1]
#             U_per_env.append(np.zeros((0, d)))
#             continue

#         env_constraints = np.array(remove_redundant_constraints(env_constraints, epsilon=lp_epsilon))
#         U_per_env.append(env_constraints)

#         for v in env_constraints:
#             all_constraints.append(v)

#     if not all_constraints:
#         # no constraints anywhere
#         d = SFs[0][0].shape[-1]
#         return U_per_env, np.zeros((0, d))

#     # ============================================================
#     # 3. Global spherical deduplication
#     # ============================================================
#     unique = []
#     for v in all_constraints:
#         v_norm = np.linalg.norm(v)
#         if v_norm == 0:
#             continue

#         is_close = False
#         for u in unique:
#             cos = np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u))
#             if cos > 1 - precision:
#                 is_close = True
#                 break

#         if not is_close:
#             unique.append(v)

#     U_global = np.array(unique)
#     return U_per_env, U_global

# ============================================================
# derive_constraints.py  (FULL FIXED VERSION)
# ============================================================

import numpy as np
from .successor_features import compute_successor_features_iterative_from_q
from .lp_redundancy import remove_redundant_constraints


# ============================================================
# 1. Successor Features Family Wrapper
# ============================================================

def compute_successor_features_family(envs, Q_list, **kw):
    SFs = []
    for env, q in zip(envs, Q_list):
        mu_sa, mu_s, Phi, P_pi = compute_successor_features_iterative_from_q(env, q, **kw)
        SFs.append((mu_sa, mu_s, Phi, P_pi))
    return SFs


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
    preferred_feats = np.sum([env.get_state_feature(s) for s, _ in preferred], axis=0)
    other_feats = np.sum([env.get_state_feature(s) for s, _ in other], axis=0)
    diff = preferred_feats - other_feats
    norm = np.linalg.norm(diff)
    return [diff / norm if norm != 0 else np.zeros_like(diff)]


def constraints_from_estop(atom_data, env):
    traj, t_stop = atom_data
    feats_up_to_t = np.sum([env.get_state_feature(s) for s, _ in traj[:t_stop+1]], axis=0)
    full_feats = np.sum([env.get_state_feature(s) for s, _ in traj], axis=0)
    diff = feats_up_to_t - full_feats
    norm = np.linalg.norm(diff)
    return [diff / norm if norm != 0 else np.zeros_like(diff)]


def constraints_from_improvement(atom_data, env):
    improved, original = atom_data
    imp = np.sum([env.get_state_feature(s) for s, _ in improved], axis=0)
    org = np.sum([env.get_state_feature(s) for s, _ in original], axis=0)
    diff = imp - org
    norm = np.linalg.norm(diff)
    return [diff / norm if norm != 0 else np.zeros_like(diff)]


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

def derive_constraints_from_atoms(
    atoms_per_env,
    SFs,
    envs,
    *,
    precision=1e-3,
    lp_epsilon=1e-4,
):
    all_constraints = []
    U_per_env = []

    for atoms, sf, env in zip(atoms_per_env, SFs, envs):
        mu_sa = sf[0]
        env_constraints = []

        for atom in atoms:
            env_constraints.extend(atom_to_constraints(atom, mu_sa, env))

        if len(env_constraints) == 0:
            d = mu_sa.shape[-1]
            U_per_env.append(np.zeros((0, d)))
            continue

        env_constraints = np.array(remove_redundant_constraints(env_constraints, epsilon=lp_epsilon))
        U_per_env.append(env_constraints)

        for v in env_constraints:
            all_constraints.append(v)

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

    U_global = np.array(unique)
    return U_per_env, U_global


# ============================================================
# 7. Q-only Constraint Builder
# ============================================================

def derive_constraints_from_q_family(
    SFs,
    Q_list,
    envs,
    tie_eps=1e-10,
    skip_terminals=True,
    normalize=True,
    tol=1e-12,
    precision=1e-3,
    lp_epsilon=1e-4,
):
    U_per_mdp = []
    all_H = []

    for (mu_sa, _, _, _), q, env in zip(SFs, Q_list, envs):
        cons = derive_constraints_from_q_ties(
            mu_sa, q, env,
            tie_eps=tie_eps,
            skip_terminals=skip_terminals,
            normalize=normalize,
            tol=tol,
        )
        H_i = [c[0] for c in cons]
        U_per_mdp.append(H_i)
        all_H.extend(H_i)

    pre = []
    for v in all_H:
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

    U_global = np.array(remove_redundant_constraints(pre, epsilon=lp_epsilon))
    return U_per_mdp, U_global
