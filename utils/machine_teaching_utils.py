import numpy as np
from scipy.optimize import linprog

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
# 5) Generating trajectories
# ============================================================

def generate_candidates_from_q(
    env,
    q_values,                      # shape (S, A)
    num_rollouts_per_state=10,
    max_steps=15,
    tie_eps=1e-10,
):
    """
    Generate trajectories as lists of (state, action) pairs by following a greedy
    policy derived from q_values, sampling next states from env.transitions.
    """
    S = env.get_num_states()
    A = env.get_num_actions()
    terminals = set(env.terminal_states or [])
    T = env.transitions   # already row-stochastic per your env

    # Precompute greedy action sets (allowing ties within tie_eps)
    opt_actions = [[] for _ in range(S)]
    for s in range(S):
        if s in terminals:
            continue
        row = q_values[s]
        max_q = np.max(row)
        opt_actions[s] = [a for a in range(A) if abs(row[a] - max_q) < tie_eps]

    #print(opt_actions)
    trajectories = []
    for start_s in range(S):
        if start_s in terminals or not opt_actions[start_s]:
            continue  # skip terminal starts or states with no valid action

        for _ in range(num_rollouts_per_state):
            tau, s, steps = [], int(start_s), 0
            print("starting state: ", s)
            while steps < max_steps and s not in terminals:
                acts = opt_actions[s]
                if not acts:
                    break
                print("optimal actions: ", acts)
                a = int(np.random.choice(acts))           # pick among optimal actions
                tau.append((s, a))
                s = int(np.random.choice(S, p=T[s, a]))   # sample next state from transitions
                steps += 1
            trajectories.append(tau)

    return trajectories