import numpy as np

def build_Pi_from_q(env, q_values, tie_eps=1e-10):
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
            Pi[s, :] = 1.0 / A
    return Pi


def compute_successor_features_iterative_from_q(
    env,
    q_values,
    convention="entering",
    zero_terminal_features=True,
    tol=1e-10,
    max_iters=10000
):
    S = env.get_num_states()
    A = env.get_num_actions()
    d = env.num_features
    gamma = env.get_discount_factor()

    Phi = np.asarray(env.grid_features, float).reshape(S, d)
    if zero_terminal_features and getattr(env, "include_terminal", False):
        for t in (env.terminal_states or []):
            Phi[t] = 0.0

    T = np.asarray(env.transitions, float)
    Pi = build_Pi_from_q(env, q_values)

    # Compute P_pi
    P_pi = np.zeros((S, S), dtype=float)
    for s in range(S):
        P_pi[s] = Pi[s].dot(T[s])
        rs = P_pi[s].sum()
        if rs == 0.0:
            P_pi[s, s] = 1.0
        else:
            P_pi[s] /= rs

    mu_s = np.zeros((S, d), dtype=float)
    use_enter = convention.lower().startswith("enter")

    for _ in range(max_iters):
        mu_old = mu_s.copy()
        for s in range(S):
            exp_mu_next = np.zeros(d)
            exp_phi_next = np.zeros(d) if use_enter else None
            for a in range(A):
                w = Pi[s, a]
                if w == 0:
                    continue
                p_next = T[s, a]
                exp_mu_next += w * (p_next @ mu_old)
                if use_enter:
                    exp_phi_next += w * (p_next @ Phi)

            mu_s[s] = (exp_phi_next if use_enter else Phi[s]) + gamma * exp_mu_next

        if np.max(np.abs(mu_s - mu_old)) < tol:
            break

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
