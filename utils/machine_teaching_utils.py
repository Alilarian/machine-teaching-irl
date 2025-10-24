import numpy as np

# ---------- small helpers ----------

def _prep_env_arrays(env, zero_terminal_features: bool):
    """Return S, A, d, gamma, Phi (S,d), T (S,A,S) normalized."""
    S, A, d = env.get_num_states(), env.get_num_actions(), env.num_features
    gamma = env.get_discount_factor()

    # Features Φ(s)
    Phi = np.asarray(env.grid_features, dtype=float).reshape(S, d)
    if zero_terminal_features and getattr(env, "include_terminal", False) and env.terminal_states:
        Phi[np.asarray(env.terminal_states, dtype=int)] = 0.0

    # Transitions T(s'|s,a), row-stochastic with self-loop fallback
    T = np.asarray(env.transitions, dtype=float, copy=True)
    for s in range(S):
        for a in range(A):
            row = T[s, a]
            tot = row.sum()
            if tot > 0.0:
                T[s, a] = row / tot
            else:
                T[s, a] = 0.0
                T[s, a, s] = 1.0
    return S, A, d, gamma, Phi, T

def _build_policy_matrix_from_pairs(pi_star, S, A):
    """Π(s,a): default uniform, (state, action) rows set to one-hot."""
    Pi = np.full((S, A), 1.0 / A, dtype=float)
    for s, a in pi_star:
        Pi[s, :] = 0.0
        Pi[s, int(a)] = 1.0
    return Pi

def _build_Ppi(Pi, T):
    """P_pi(s'|s) = Σ_a Π(s,a) T(s'|s,a), row-normalized, with self-loop fallback."""
    S, A, _ = T.shape
    P_pi = np.zeros((S, S), dtype=float)
    for s in range(S):
        P_pi[s] = Pi[s].dot(T[s])  # (A,) @ (A,S) -> (S,)
        rs = P_pi[s].sum()
        if rs > 0.0:
            P_pi[s] /= rs
        else:
            P_pi[s, :] = 0.0
            P_pi[s, s] = 1.0
    return P_pi

def _build_mu_sa(Phi, T, mu_s, gamma, convention_is_enter):
    """Assemble ψ(s,a) from μ(s)."""
    S, A = T.shape[:2]
    d = Phi.shape[1]
    mu_sa = np.zeros((S, A, d), dtype=float)
    for s in range(S):
        for a in range(A):
            p_next = T[s, a]           # (S,)
            exp_mu_next  = p_next @ mu_s    # (d,)
            if convention_is_enter:
                exp_phi_next = p_next @ Phi # (d,)
                mu_sa[s, a] = exp_phi_next + gamma * exp_mu_next
            else:
                mu_sa[s, a] = Phi[s] + gamma * exp_mu_next
    return mu_sa


# ============================================================
# 1) Iterative policy evaluation (until tolerance)
# ============================================================

def compute_successor_features_iterative(
    env,
    pi_star,                           # list of (state, action)
    convention: str = "on",
    zero_terminal_features: bool = False,
    tol: float = 1e-10,
    max_iters: int = 10000
):
    """
    Iterative evaluation of state successor features μ(s), then builds ψ(s,a).
    Simple, readable, and matches your env + (state, action) policy format.
    """
    S, A, d, gamma, Phi, T = _prep_env_arrays(env, zero_terminal_features)
    Pi = _build_policy_matrix_from_pairs(pi_star, S, A)
    P_pi = _build_Ppi(Pi, T)

    # Initialize μ(s)
    mu_s = np.zeros((S, d), dtype=float)
    use_enter = convention.lower().startswith("enter")

    for _ in range(max_iters):
        mu_old = mu_s.copy()
        # Bellman update per state
        for s in range(S):
            # expected continuation under π
            exp_mu_next = np.zeros(d, dtype=float)
            exp_phi_next = np.zeros(d, dtype=float) if use_enter else None
            for a in range(A):
                if Pi[s, a] == 0.0: 
                    continue
                p_next = T[s, a]                 # (S,)
                exp_mu_next += Pi[s, a] * (p_next @ mu_old)
                if use_enter:
                    exp_phi_next += Pi[s, a] * (p_next @ Phi)
            mu_s[s] = (exp_phi_next if use_enter else Phi[s]) + gamma * exp_mu_next

        # convergence check (∞-norm)
        if np.max(np.abs(mu_s - mu_old)) < tol:
            break

    # Build ψ(s,a)
    mu_sa = _build_mu_sa(Phi, T, mu_s, gamma, use_enter)
    return mu_sa, mu_s, Phi, P_pi


# ============================================================
# 2) Linear-solver version (direct, exact up to solver precision)
# ============================================================

def compute_successor_features_linear(
    env,
    pi_star,                           # list of (state, action)
    convention: str = "on",
    zero_terminal_features: bool = False
):
    """
    Direct solve: (I - γ P_π) μ = Φ   (on-state)
                  (I - γ P_π) μ = P_π Φ (entering-state)
    Then builds ψ(s,a).
    """
    S, A, d, gamma, Phi, T = _prep_env_arrays(env, zero_terminal_features)
    Pi = _build_policy_matrix_from_pairs(pi_star, S, A)
    P_pi = _build_Ppi(Pi, T)

    I = np.eye(S, dtype=float)
    use_enter = convention.lower().startswith("enter")
    RHS = (P_pi @ Phi) if use_enter else Phi        # (S,d)

    # Solve all d columns at once
    mu_s = np.linalg.solve(I - gamma * P_pi, RHS)   # (S,d)

    # Build ψ(s,a)
    mu_sa = _build_mu_sa(Phi, T, mu_s, gamma, use_enter)
    return mu_sa, mu_s, Phi, P_pi

# ============================================================
# 3) Deriving constraints
# ============================================================

def derive_constraints_from_mu_sa_vec(
    mu_sa: np.ndarray,       # (S, A, d)  action-SFs ψ(s,a)
    pi_pairs,                # list[(state, action)]
    env,
    skip_terminals: bool = True,
    normalize: bool = True,
):
    """
    Vectorized: build constraints v = ψ(s,a*) - ψ(s,b) for all b != a*.
    Returns a list[(v, s, a_star, b)].
    """
    S, A, d = mu_sa.shape

    # map (state, action) pairs -> a*(s); unspecified states set to -1
    a_star = np.full(S, -1, dtype=int)
    for s, a in pi_pairs:
        a_star[int(s)] = int(a)

    # skip terminals if requested
    if skip_terminals and getattr(env, "include_terminal", False):
        terms = np.array(env.terminal_states or [], dtype=int)
        if terms.size:
            a_star[terms] = -1

    valid = np.where(a_star >= 0)[0]
    if valid.size == 0:
        return []

    # ψ(s,a*) for valid states (Nv,d)
    psi_star = mu_sa[valid, a_star[valid], :]              # (Nv, d)
    # ψ(s,·) for valid states (Nv,A,d)
    psi_all  = mu_sa[valid, :, :]                          # (Nv, A, d)
    # differences for all b: (Nv, A, d)
    diffs = psi_star[:, None, :] - psi_all

    # mask out b == a*
    mask = np.ones((valid.size, A), dtype=bool)
    mask[np.arange(valid.size), a_star[valid]] = False

    # flatten selections
    v = diffs[mask]                                        # (Ncons, d)
    s_idx = np.repeat(valid, A)[mask.ravel()]              # (Ncons,)
    b_idx = np.tile(np.arange(A), valid.size)[mask.ravel()]
    a_idx = np.repeat(a_star[valid], A)[mask.ravel()]

    nrm = np.linalg.norm(v, axis=1)
    #keep = nrm > tol
    #if not np.any(keep):
    #    return []
    #v, s_idx, a_idx, b_idx, nrm = v[keep], s_idx[keep], a_idx[keep], b_idx[keep], nrm[keep]
    v, s_idx, a_idx, b_idx, nrm = v, s_idx, a_idx, b_idx, nrm

    # normalize if requested
    if normalize:
        v = np.divide(v, nrm[:, None], out=v.copy(), where=nrm[:, None] > 0)

    return [(v[i], int(s_idx[i]), int(a_idx[i]), int(b_idx[i])) for i in range(v.shape[0])]


# ============================================================
# 4) Removing redundant constraints
# ============================================================

def is_redundant_constraint(h, H, epsilon=0.0001):
    #we have a constraint c^w >= 0 we want to see if we can minimize c^T w and get it to go below 0
    # if not then this constraint is satisfied by the constraints in H, if we can, then we need to add c back into H 
    #Thus, we want to minimize c^T w subject to Hw >= 0
    #First we need to change this into the form min c^T x subject to Ax <= b
    #Our problem is equivalent to min c^T w subject to  -H w <= 0
    H = np.array(H) #just to make sure
    m,_ = H.shape
    #H = np.transpose(H)  #get it into correct format

    #c = matrix(h[non_zeros])
    #G = matrix(-H[:,non_zeros])
    b = np.zeros(m)
    sol = linprog(h, A_ub=-H, b_ub = b, bounds=(-1,1), method = 'revised simplex' )
    # print(sol)
    if sol['status'] != 0:
        print("trying interior point method")
        sol = linprog(h, A_ub=-H, b_ub = b, bounds=(-1,1) )
    
    if sol['status'] != 0: #not sure what to do here. Shouldn't ever be infeasible, so probably a numerical issue
        print("LP NOT SOLVABLE")
        print("IGNORING ERROR FOR NOW!!!!!!!!!!!!!!!!!!!")
        #sys.exit()
        return False #let's be safe and assume it's necessary...
    elif sol['fun'] < -epsilon: #if less than zero then constraint is needed to keep c^T w >=0
        return False
    else: #redundant since without constraint c^T w >=0
        #print("redundant")
        return True

def remove_redundant_constraints(halfspaces, epsilon = 0.0001):
    """Return a new array with all redundant halfspaces removed.

       Parameters
       -----------
       halfspaces : list of halfspace normal vectors such that np.dot(halfspaces[i], w) >= 0 for all i

       epsilon : numerical precision for determining if redundant via LP solution 

       Returns
       -----------
       list of non-redundant halfspaces 
    """
    #for each row in halfspaces, check if it is redundant
    #num_vars = len(halfspaces[0]) #size of weight vector
    non_redundant_halfspaces = []
    halfspaces_to_check = halfspaces
    for i,h in enumerate(halfspaces):
        #print("\nchecking", h)
        halfspaces_lp = [h for h in non_redundant_halfspaces] + [h for h in halfspaces_to_check[1:]]
        halfspaces_lp = np.array(halfspaces_lp)

        if not is_redundant_constraint(h, halfspaces_lp, epsilon):

            non_redundant_halfspaces.append(h)
        else:
            pass
            ##print("redundant")
            
        halfspaces_to_check = halfspaces_to_check[1:]
    return non_redundant_halfspaces