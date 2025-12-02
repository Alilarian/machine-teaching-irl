import numpy as np
from scipy.optimize import linprog
from utils.common_helper import calculate_expected_value_difference

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
# ---------- small helpers ----------


def build_Pi_from_q(env, q_values, tie_eps=1e-10):
    """
    Return Π(s,a) from Q(s,a): uniform over all a achieving max Q(s,·).
    Terminal rows are left zero (P_pi fallback makes them self-loops).
    """
    S, A = env.get_num_states(), env.get_num_actions()
    Pi = np.zeros((S, A), dtype=float)

    terminals = set(getattr(env, "terminal_states", []) or [])
    for s in range(S):
        if s in terminals:
            continue
        row = np.asarray(q_values[s], dtype=float)
        m = np.max(row)
        mask = np.abs(row - m) < tie_eps
        k = int(mask.sum())
        if k > 0:
            Pi[s, mask] = 1.0 / k
        else:
            # extremely rare: all -inf or nan -> fallback to uniform
            Pi[s, :] = 1.0 / A
    return Pi

def compute_successor_features_iterative_from_q(
    env,
    q_values,
    convention: str = "entering"
    "",
    zero_terminal_features: bool = True,
    tol: float = 1e-10,
    max_iters: int = 10000
):
    S, A, d = env.get_num_states(), env.get_num_actions(), env.num_features
    gamma = env.get_discount_factor()

    # Φ and T (your env already normalizes; if you prefer, use it directly)
    Phi = np.asarray(env.grid_features, float).reshape(S, d)
    if zero_terminal_features and getattr(env, "include_terminal", False):
        for t in (env.terminal_states or []):
            Phi[t] = 0.0
    T = np.asarray(env.transitions, float)

    # π from Q with ties handled
    Pi = build_Pi_from_q(env, q_values, tie_eps=1e-10)
    print("Pi from q: ", Pi)

    # P_π(s'|s) = sum_a Π(s,a) T(s'|s,a), with fallback self-loop if a row is zero
    P_pi = np.zeros((S, S), dtype=float)
    for s in range(S):
        P_pi[s] = Pi[s].dot(T[s])
        rs = P_pi[s].sum()
        if rs == 0.0:
            P_pi[s, s] = 1.0
        else:
            P_pi[s] /= rs

    # Iterative policy evaluation for μ(s)
    mu_s = np.zeros((S, d), dtype=float)
    use_enter = convention.lower().startswith("enter")

    for _ in range(max_iters):
        mu_old = mu_s.copy()
        for s in range(S):
            exp_mu_next = np.zeros(d)
            exp_phi_next = np.zeros(d) if use_enter else None
            for a in range(A):
                p_next = T[s, a]
                w = Pi[s, a]
                if w == 0.0:
                    continue
                exp_mu_next += w * (p_next @ mu_old)
                if use_enter:
                    exp_phi_next += w * (p_next @ Phi)
            mu_s[s] = (exp_phi_next if use_enter else Phi[s]) + gamma * exp_mu_next
        if np.max(np.abs(mu_s - mu_old)) < tol:
            break

    # ψ(s,a)
    mu_sa = np.zeros((S, A, d), dtype=float)
    for s in range(S):
        for a in range(A):
            p_next = T[s, a]
            exp_mu_next = p_next @ mu_s
            if use_enter:
                exp_phi_next = p_next @ Phi
                mu_sa[s, a] = exp_phi_next + gamma * exp_mu_next
            else:
                mu_sa[s, a] = Phi[s] + gamma * exp_mu_next

    return mu_sa, mu_s, Phi, P_pi
# ============================================================
# 1) Iterative policy evaluation (until tolerance)
# ============================================================



# ============================================================
# 2) Linear-solver version (direct, exact up to solver precision)
# ============================================================

# ============================================================
# 3) Deriving constraints
# ============================================================

def derive_constraints_from_q_ties(
    mu_sa: np.ndarray,        # (S, A, d) action-SFs ψ(s,a)
    q_values: np.ndarray,     # (S, A)     action values used to decide optimal set(s)
    env,
    tie_eps: float = 1e-10,
    skip_terminals: bool = True,
    normalize: bool = True,
    tol: float = 1e-12,
):
    """
    Build constraints v = ψ(s,a*) - ψ(s,b) for every state s, for every co-optimal a* in argmax Q(s,·),
    and for every non-optimal b ∉ argmax Q(s,·). This preserves all ties.

    Returns: list of (v, s, a_star, b) with v in R^d.
    """
    S, A, d = mu_sa.shape
    q = np.asarray(q_values, dtype=float)
    if q.shape != (S, A):
        raise ValueError(f"q_values shape {q.shape} != (S, A)=({S},{A})")

    # argmax set per state with tie tolerance
    m = np.max(q, axis=1, keepdims=True)
    argmax_mask = np.abs(q - m) <= tie_eps   # (S, A) True where action is co-optimal

    # optionally skip terminals
    if skip_terminals and getattr(env, "include_terminal", False):
        terms = np.array(getattr(env, "terminal_states", []) or [], dtype=int)
        if terms.size:
            argmax_mask[terms] = False  # no constraints emitted from terminals

    constraints = []
    for s in range(S):
        A_star = np.where(argmax_mask[s])[0]
        if A_star.size == 0:
            continue  # nothing to do (terminal or undefined)
        B = np.where(~argmax_mask[s])[0]
        if B.size == 0:
            # all actions are tied optimal at s -> no informative inequality constraints
            continue

        psi_s = mu_sa[s]  # (A, d)
        # for each co-optimal a*, build differences against all non-optimal b
        for a_star in A_star:
            diffs = psi_s[a_star][None, :] - psi_s[B, :]   # (|B|, d)
            norms = np.linalg.norm(diffs, axis=1)

            # optionally normalize and drop ~zero vectors
            for i, b in enumerate(B):
                if norms[i] <= tol:
                    continue
                v = diffs[i]
                if normalize:
                    v = v / norms[i]
                constraints.append((v, int(s), int(a_star), int(b)))

    return constraints



# ============================================================
# 4) Removing redundant constraints
# ============================================================

def _normalize_dir(v, tol=1e-12):
    """Normalize v up to positive scaling so duplicates collapse to same key."""
    v = np.asarray(v, dtype=float)
    nrm = np.linalg.norm(v)
    if nrm < tol:
        # Zero vector is not a valid halfspace normal for c^T w >= 0; skip it upstream
        return None
    v = v / nrm
    # Make a canonical sign: first nonzero component positive
    for x in v:
        if abs(x) > tol:
            if x < 0:
                v = -v
            break
    # Round for stable hashing
    return tuple(np.round(v, 12))

def is_redundant_constraint(h, H, epsilon=1e-4):
    """
    Return True if the inequality h^T w >= 0 is redundant given H w >= 0.
    """
    h = np.asarray(h, dtype=float)
    H = np.asarray(H, dtype=float)

    # If there are no other constraints, h cannot be redundant.
    if H.size == 0:
        return False

    # Ensure H is 2D with shape (m, n)
    if H.ndim == 1:
        H = H.reshape(1, -1)
    m, n = H.shape
    assert h.shape == (n,), f"Shape mismatch: h {h.shape} vs H (m={m}, n={n})"

    # Solve: minimize h^T w  s.t.  (-H) w <= 0,  bounds -1 <= w_i <= 1
    b = np.zeros(m)
    res = linprog(h, A_ub=-H, b_ub=b, bounds=[(-1, 1)]*n, method='highs')

    if res.status != 0:
        # Numerical hiccup: be safe and treat as necessary (not redundant).
        return False

    # If we can push h^T w below 0, then h is necessary; otherwise redundant.
    return res.fun >= -epsilon

def remove_redundant_constraints(halfspaces, epsilon=1e-4):
    """
    Return a list of non-redundant halfspace normals h such that h^T w >= 0.
    - Monotone build: test each h only against the set we've already kept.
    - Final cleanup pass removes any that became redundant after later additions.
    - Exact/positively-scaled duplicates are removed upfront.
    """
    halfspaces = [np.asarray(h, dtype=float) for h in halfspaces]

    # Filter out zeros and deduplicate by direction (up to positive scale)
    seen = set()
    unique = []
    for h in halfspaces:
        key = _normalize_dir(h)
        if key is None:
            continue  # skip zero normals
        if key not in seen:
            seen.add(key)
            unique.append(h)

    kept = []
    # First pass: only compare to what we've already kept (prevents "all dropped" bug)
    for h in unique:
        H_keep = np.vstack(kept) if len(kept) else np.empty((0, h.size))
        if not is_redundant_constraint(h, H_keep, epsilon):
            kept.append(h)

    # Final cleanup: check each kept constraint against all others kept
    final = []
    for i, h in enumerate(kept):
        others = [kept[j] for j in range(len(kept)) if j != i]
        H_others = np.vstack(others) if len(others) else np.empty((0, h.size))
        if not is_redundant_constraint(h, H_others, epsilon):
            final.append(h)

    return final

# ============================================================
# 5) Generating trajectories ==> need to move to generate_feedback.py
# ============================================================



def _intersection_polygon_2d(V, box=1.0, tol=1e-12):
    """
    Return vertices (k,2) of the bounded intersection:
        { w : w^T v_i >= 0 for all i }  ∩  { |w1|<=box, |w2|<=box }
    as a convex polygon ordered counter-clockwise.
    """
    V = np.asarray(V, float).reshape(-1, 2)

    # Convert to a_i x + b_i y + c_i <= 0 form (halfspace format)
    # w^T v >= 0  <=>  (-v)^T w <= 0  -> a=-vx, b=-vy, c=0
    half = [(-vx, -vy, 0.0) for (vx, vy) in V]

    # Bounding box (keeps region bounded for plotting)
    half += [( 1, 0, -box), (-1, 0, -box), (0, 1, -box), (0, -1, -box)]

    pts = []
    m = len(half)
    for i in range(m):
        a1, b1, c1 = half[i]
        for j in range(i + 1, m):
            a2, b2, c2 = half[j]
            D = a1 * b2 - a2 * b1
            if abs(D) < tol:
                continue  # parallel lines
            x = (b1 * c2 - b2 * c1) / D
            y = (c1 * a2 - c2 * a1) / D
            # keep if satisfies all halfspaces
            if all(a * x + b * y + c <= tol for (a, b, c) in half):
                pts.append((x, y))

    if not pts:
        return np.empty((0, 2))

    pts = np.unique(np.round(pts, 12), axis=0)  # dedup numerically
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)
    return pts[order]

def plot_halfspace_intersection_2d(
    V,
    *,
    box=1.0,
    colors=None,
    labels=None,
    w_true=None,
    scot_sol=None,
    title="Intersection of half-spaces"
):
    """
    V: (m,2) array of normals; each row v defines w^T v >= 0.
    box: plot window [-box, box]^2 and bounding halfspaces.
    colors/labels: optional per-constraint styling.
    w_true: optional (w1, w2) to mark with a star.
    """
    V = np.asarray(V, float).reshape(-1, 2)
    m = len(V)
    xs = np.linspace(-box, box, 400)

    if colors is None:
        colors = ["#d81b60", "#008080", "#1f77b4", "#ff7f0e"]  # magenta/teal first
    if labels is None:
        labels = [f"Constraint {i+1}" for i in range(m)]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw each boundary v_x x + v_y y = 0
    handles = []
    for i, (vx, vy) in enumerate(V):
        col = colors[i % len(colors)]
        if abs(vy) < 1e-12:
            h = ax.axvline(0, color=col, lw=4, label=labels[i])
        else:
            y_line = -(vx / vy) * xs
            
            h, = ax.plot(xs, y_line, color=col, lw=1, label=labels[i])
        handles.append(h)

    # Compute and shade feasible polygon (hatched)
    poly = _intersection_polygon_2d(V, box=box)
    
    if poly.shape[0] > 0:
        patch = Polygon(
            poly, closed=True, facecolor="#f5bd23", alpha=0.9,
            edgecolor="none", hatch="///"
        )
        ax.add_patch(patch)

    # Axes, limits, labels
    ax.axhline(0, color="k", lw=1)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlim(-box, box)
    ax.set_ylim(-box, box)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")

    # True reward (optional)
    if w_true is not None:
        ax.plot(w_true[0], w_true[1], marker="*", color="k", ms=10, label="True Reward")
        
    if scot_sol:
        ax.plot(scot_sol[0] , scot_sol[1] , marker="*", color="#23f523", ms=10, label="MAP from SCOT")
        
    # Build a clean legend (constraint lines + star if present)
    if w_true is not None:
        ax.legend(loc="upper right")
    else:
        ax.legend(handles=handles, loc="upper right")

    ax.set_title(title)
    plt.show()


# ============================================================
#  6) computing regret - comparing regret
# ============================================================

def regrets_from_Q(envs, Q_list, *, tie_eps=1e-10, epsilon=1e-4, normalize_with_random_policy=False):
    """
    For each env, build a greedy (tie-aware) policy from Q(s,a) and compute regret:
        Regret = V*(true) - V^pi(true)
    Returns: np.ndarray of shape (num_envs,)
    """
    assert len(envs) == len(Q_list), "envs and Q_list must have same length."
    regrets = []
    for env, Q in zip(envs, Q_list):
        pi = build_Pi_from_q(env, Q, tie_eps=tie_eps)  # uniform over argmax ties
        r = calculate_expected_value_difference(
            eval_policy=pi,
            env=env,                                # env must contain the TRUE reward
            epsilon=epsilon,
            normalize_with_random_policy=normalize_with_random_policy,
        )
        regrets.append(float(r))
    return np.asarray(regrets, dtype=float)

def compare_regret_from_Q(envs, Q_scot_list, Q_rand_list, *,
                          tie_eps=1e-10, epsilon=1e-4, normalize_with_random_policy=False):
    """
    Compute per-env and summary regrets for two Q-based methods.
    Returns a dict with per-env arrays and summary stats.
    """
    reg_scot = regrets_from_Q(envs, Q_scot_list,
                              tie_eps=tie_eps,
                              epsilon=epsilon,
                              normalize_with_random_policy=normalize_with_random_policy)
    reg_rand = regrets_from_Q(envs, Q_rand_list,
                              tie_eps=tie_eps,
                              epsilon=epsilon,
                              normalize_with_random_policy=normalize_with_random_policy)

    def _stats(x):
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "median": float(np.median(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }

    return {
        "per_env": {
            "SCOT": reg_scot,
            "RandomOpt": reg_rand,
            "Delta": reg_scot - reg_rand,   # positive => SCOT has higher regret
        },
        "summary": {
            "SCOT": _stats(reg_scot),
            "RandomOpt": _stats(reg_rand),
            "delta_mean": float(np.mean(reg_scot) - np.mean(reg_rand)),
        },
        "stacked_table": np.stack([reg_scot, reg_rand], axis=1),  # columns: [SCOT, RandomOpt]
    }


# ============================================================
#  7) generating demonstration based on SCOT solution across envs ==> need to move to generate_feedback.py
# ============================================================

def sample_optimal_sa_pairs(
    envs, Q_list, n, *,
    tie_eps=1e-10,
    skip_terminals=True,
    seed=None,
    return_shape="flat",   # "flat": [(env_i, s, a)], "scot": [(env_i, [(s,a)])]
):
    rng = np.random.default_rng(seed)
    assert len(envs) == len(Q_list), "envs and Q_list must have same length."

    # Π from Q (already uniform over arg-max ties)
    Pis = [build_Pi_from_q(env, q, tie_eps=tie_eps) for env, q in zip(envs, Q_list)]

    # Eligible states per env (non-terminal if requested) with nonzero Π mass
    eligible_states = []
    for env, Pi in zip(envs, Pis):
        S = env.get_num_states()
        terms = set(getattr(env, "terminal_states", []) or [])
        if skip_terminals:
            mask = np.array([s not in terms for s in range(S)], dtype=bool)
        else:
            mask = np.ones(S, dtype=bool)
        elig = np.flatnonzero(mask & (Pi.sum(axis=1) > 0))
        eligible_states.append(elig)

    env_pool = [i for i, es in enumerate(eligible_states) if es.size > 0]
    if not env_pool:
        raise RuntimeError("No eligible states found in any env (check terminals/Q).")

    out = []
    for _ in range(n):
        i = int(rng.choice(env_pool))              # pick env uniformly
        s = int(rng.choice(eligible_states[i]))    # pick state uniformly within env
        p = Pis[i][s]
        p = p / p.sum()                            # defensive normalize
        a = int(rng.choice(len(p), p=p))           # sample among co-optimal actions

        out.append((i, [(s, a)]) if return_shape == "scot" else (i, s, a))
    return out

def sample_optimal_sa_pairs_like_scot(
    envs, Q_list, chosen_from_scot, *,
    tie_eps=1e-10,
    skip_terminals=True,
    seed=None,
    return_shape="scot",
):
    n = sum(len(traj) for _, traj in chosen_from_scot)
    return sample_optimal_sa_pairs(
        envs, Q_list, n,
        tie_eps=tie_eps,
        skip_terminals=skip_terminals,
        seed=seed,
        return_shape=return_shape,
    )