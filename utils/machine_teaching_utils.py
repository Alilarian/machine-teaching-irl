import numpy as np

import numpy as np

def compute_successor_features(env, pi_star):
    S = len(env.states)
    A = len(env.actions)
    d = env.feature_dim

    # Indices
    s_idx = {s:i for i,s in enumerate(env.states)}
    a_idx = {a:i for i,a in enumerate(env.actions)}

    # Features per state
    Phi = np.zeros((S, d))
    for s in env.states:
        Phi[s_idx[s]] = env.phi(s)

    # Transition under policy π
    T_pi = np.zeros((S, S))
    for s in env.states:
        i = s_idx[s]
        if env.is_terminal(s):
            # absorbing terminal with zero features is standard
            T_pi[i, i] = 1.0
            Phi[i, :] = 0.0
            continue
        # Deterministic or argmax from a dict
        a = pi_star[s] if not isinstance(pi_star[s], dict) else max(pi_star[s], key=pi_star[s].get)
        for s_next, p in env.transitions(s, a):
            j = s_idx[s_next]
            T_pi[i, j] += p

    I = np.eye(S)

    # ENTERING-STATE model:
    # (I - γ Tπ) μπ = Tπ Φ  => μπ = solve(I - γ Tπ, Tπ Φ)
    TPhi = T_pi @ Phi
    mu_pi = np.linalg.solve(I - env.gamma * T_pi, TPhi)   # shape: (S, d)

    # Action-conditioned successor features ψ(s,a)
    # ψ(s,a) = E[φ(s')] + γ E[ μπ(s') ], consistent with entering-state features
    mu_sa = np.zeros((S, A, d))
    for s in env.states:
        i = s_idx[s]
        for a in env.actions:
            ai = a_idx[a]
            if env.is_terminal(s):
                # no future features from terminal
                mu_sa[i, ai, :] = 0.0
                continue
            exp_phi_next = np.zeros(d)
            exp_mu_next  = np.zeros(d)
            for s_next, p in env.transitions(s, a):
                j = s_idx[s_next]
                exp_phi_next += p * Phi[j]    # φ(s′)
                exp_mu_next  += p * mu_pi[j]  # μπ(s′)
            mu_sa[i, ai, :] = exp_phi_next + env.gamma * exp_mu_next

    return mu_sa, s_idx, a_idx
