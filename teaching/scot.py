import numpy as np
from utils import atom_to_constraints
import time






def normalize_constraints(X, *, normalize=True):
    """
    Normalize constraint vectors to unit length.
    Zero / invalid vectors are removed.

    Returns:
        Xn : (k, d) normalized constraints
        mask : boolean mask of kept rows
    """
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return X.reshape(0, X.shape[-1]), np.zeros(0, dtype=bool)

    if not normalize:
        return X, np.ones(len(X), dtype=bool)

    norms = np.linalg.norm(X, axis=1)
    good = (norms > 0) & np.isfinite(norms)
    Xn = X[good] / norms[good][:, None]
    return Xn, good

def build_universal_index(
    U_universal,
    *,
    eps=1e-6,
    normalize=True,
    allow_sign_flip=True,
):
    """
    Deduplicate universal constraints numerically using cosine similarity.

    Returns:
        Uu : (M, d) unique normalized constraints
    """
    U, _ = normalize_constraints(U_universal, normalize=normalize)
    if U.size == 0:
        return U

    uniq = []
    for u in U:
        if not uniq:
            uniq.append(u)
            continue

        sims = np.dot(np.vstack(uniq), u)
        if allow_sign_flip:
            sims = np.abs(sims)

        if np.max(sims) < 1.0 - eps:
            uniq.append(u)

    return np.vstack(uniq)

def build_mdp_coverage_from_constraints_numpy(
    U_per_env,
    U_universal_unique,
    *,
    eps=1e-6,
    normalize=True,
    allow_sign_flip=True,
):
    """
    Compute per-MDP coverage over universal constraints using cosine similarity.

    Returns:
        mdp_cov : list[set[int]]
    """
    Uu = np.asarray(U_universal_unique, dtype=float)
    if Uu.ndim != 2:
        raise ValueError("U_universal_unique must be 2D")

    # Normalize universe once
    Uu, _ = normalize_constraints(Uu, normalize=normalize)
    M = Uu.shape[0]

    mdp_cov = []

    for H in U_per_env:
        H = np.asarray(H, dtype=float)
        cov = set()

        if H.size == 0:
            mdp_cov.append(cov)
            continue

        Hn, _ = normalize_constraints(H, normalize=normalize)
        if Hn.size == 0:
            mdp_cov.append(cov)
            continue

        sims = Hn @ Uu.T
        if allow_sign_flip:
            sims = np.abs(sims)

        # any env constraint that matches a universe element
        hits = sims >= (1.0 - eps)
        covered = np.any(hits, axis=0)

        cov.update(np.nonzero(covered)[0].tolist())
        mdp_cov.append(cov)

    return mdp_cov



def scot_greedy_family_atoms_tracked(
    U_global,
    atoms_per_env,
    SFs,
    envs,
    *,
    normalize=True,
    round_decimals=12,
):
    U_global = np.asarray(U_global)
    
    def key_for(v):
        n = np.linalg.norm(v)
        if n == 0.0 or not np.isfinite(n):
            return ("ZERO",)
        vv = v / n if normalize else v
        return tuple(np.round(vv, round_decimals))
    
    ##################################### replace this block with:
    
    # key_to_uix = {}
    # for idx, v in enumerate(U_global):
    #     key_to_uix.setdefault(key_for(v), []).append(idx)
    
    # universe = set(range(len(U_global)))
    # covered = set()
    # --- Build canonical UID universe (MATCHES Stage-1) ---
    
    
    
    
    # key_to_uid = {}
    # uid_to_key = []

    # for v in U_global:
    #     k = key_for(v)
    #     if k not in key_to_uid:
    #         key_to_uid[k] = len(uid_to_key)
    #         uid_to_key.append(k)

    # universe = set(range(len(uid_to_key)))
    # covered = set()
    # Build numeric canonical universe (MATCHES Stage-1)
    # Numeric universal constraint index (MATCHES Stage-1)
    U_unique = build_universal_index(
        U_global,
        eps=1e-6,
        normalize=normalize,
        allow_sign_flip=True,
    )

    universe = set(range(len(U_unique)))
    covered = set()



    ####################################

    chosen = []
    chosen_constraints_list = []
    inspected_env_indices = set()
    
    n_envs = len(atoms_per_env)
    env_stats = {
        i: {
            "atoms": [],
            "indices": [],
            "coverage_counts": [],
            "total_coverage": 0,
            "was_inspected": False,
            "precompute_time": 0.0,
            "heuristic_score": None,
        }
        for i in range(n_envs)
    }
    
    # Precompute coverage sets with timing
    precompute_start = time.time()
    cov = []
    for env_idx, (atom_list, sf, env) in enumerate(zip(atoms_per_env, SFs, envs)):
        mu_sa = sf[0]
        env_precompute_start = time.time()
        cov_i = []
        for atom in atom_list:
            constraints = atom_to_constraints(atom, mu_sa, env)
            
            ############################# replace this block
            #atom_cov = set()
            # for v in constraints:
            #     k = key_for(v)
            #     if k in key_to_uix:
            #         atom_cov.update(key_to_uix[k])
            #             atom_cov = set()
            
            # for v in constraints:
            #     k = key_for(v)
            #     uid = key_to_uid.get(k, None)
            #     if uid is not None:
            #         atom_cov.add(uid)
            atom_cov = set()

            C = np.asarray(constraints, dtype=float)
            if C.size != 0:
                Cn, _ = normalize_constraints(C, normalize=normalize)
                if Cn.size != 0:
                    sims = Cn @ U_unique.T
                    sims = np.abs(sims)  # allow sign flip

                    hits = sims >= (1.0 - 1e-6)
                    covered_uids = np.nonzero(np.any(hits, axis=0))[0]
                    atom_cov.update(covered_uids.tolist())


            ############################# replace this block
            
            cov_i.append(atom_cov)
        cov.append(cov_i)
        env_stats[env_idx]["precompute_time"] = time.time() - env_precompute_start
    
    env_stats["total_precompute_time"] = time.time() - precompute_start
    
    # SCOT greedy set cover
    greedy_start = time.time()
    iter_count = 0
    while True:
        uncovered = universe - covered
        if not uncovered:
            break
        best_gain = 0
        best_atom = None
        best_new = None
        for env_idx in range(len(atoms_per_env)):
            # Inspection logic
            if len(atoms_per_env[env_idx]) > 0:
                inspected_env_indices.add(env_idx)
                env_stats[env_idx]["was_inspected"] = True
            for atom_idx, atom_cov in enumerate(cov[env_idx]):
                if not atom_cov:
                    continue
                new_cover = uncovered & atom_cov
                gain = len(new_cover)
                if gain > best_gain:
                    best_gain = gain
                    best_atom = (env_idx, atom_idx)
                    best_new = new_cover
        if best_atom is None:
            break
        env_idx, atom_idx = best_atom
        atom = atoms_per_env[env_idx][atom_idx]
        mu_sa = SFs[env_idx][0]
        constraints_for_atom = atom_to_constraints(atom, mu_sa, envs[env_idx])
        chosen_constraints_list.append(np.array(constraints_for_atom))
        chosen.append((env_idx, atom))
        covered |= best_new
        env_stats[env_idx]["atoms"].append(atom)
        env_stats[env_idx]["indices"].append(iter_count)
        env_stats[env_idx]["coverage_counts"].append(len(best_new))
        env_stats[env_idx]["total_coverage"] += len(best_new)
        iter_count += 1
    
    env_stats["total_greedy_time"] = time.time() - greedy_start
    
    if chosen_constraints_list:
        chosen_constraints = np.vstack(chosen_constraints_list)
    else:
        chosen_constraints = np.zeros((0, U_global.shape[1]))
    
    # Global stats
    env_stats["total_inspected_count"] = len(inspected_env_indices)
    env_stats["total_activated_count"] = len({c[0] for c in chosen})
    env_stats["activated_env_indices"] = sorted({c[0] for c in chosen})
    env_stats["total_iterations"] = iter_count
    env_stats["final_coverage"] = len(covered)
    env_stats["heuristic_computation_time"] = 0.0  # Naive has no heuristic
    
    return chosen, env_stats, chosen_constraints