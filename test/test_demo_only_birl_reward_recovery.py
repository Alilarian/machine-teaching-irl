import numpy as np
import os
import sys

# ----------------------------
# Project imports
# ----------------------------
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from agent.q_learning_agent_ import ValueIteration
from mdp.gridworld_env_layout import GridWorldMDPFromLayoutEnv
from utils import generate_random_gridworld_envs
from utils import Atom


# =====================================================
# Helpers: terminals + feature matrix extraction
# =====================================================

def get_terminal_set(env):
    terms = set()
    if hasattr(env, "terminal_states") and env.terminal_states is not None:
        terms = set(int(x) for x in env.terminal_states)
    elif hasattr(env, "terminals") and env.terminals is not None:
        terms = set(int(x) for x in env.terminals)
    return terms


def extract_phi(env, feature_dim=None):
    """
    Try to extract a (S,d) feature matrix Phi from common env APIs.
    Falls back to building Phi[s] by querying a per-state feature getter.
    """
    S = env.get_num_states() if hasattr(env, "get_num_states") else env.num_states

    # Common attribute names
    for name in ["feature_matrix", "state_features", "Phi", "phi", "features"]:
        if hasattr(env, name):
            Phi = np.asarray(getattr(env, name), dtype=np.float64)
            if Phi.ndim == 2 and Phi.shape[0] == S:
                return Phi

    # Common getter methods (you might have one of these)
    getter_candidates = [
        "get_state_features",
        "get_feature_vector",
        "features_of_state",
        "get_features",
        "phi_s",
    ]
    for g in getter_candidates:
        if hasattr(env, g):
            fn = getattr(env, g)
            rows = []
            for s in range(S):
                rows.append(np.asarray(fn(s), dtype=np.float64))
            Phi = np.vstack(rows)
            return Phi

    # Last resort: if you know feature_dim, create Phi by probing env.compute_reward? (not possible)
    raise RuntimeError(
        "Could not extract Phi (state feature matrix) from env. "
        "Expose env.feature_matrix or env.state_features, or add a per-state getter."
    )


# =====================================================
# FIXED: Construct demo atoms using 1-step trajectories
# =====================================================

def make_demo_atoms_one_step(env, Q):
    """
    Produce list[Atom(env_idx, 'demo', traj)]
    where traj is np.array([[s, a_opt]], dtype=int64)
    """
    atoms = []
    env_idx = 0
    terminals = get_terminal_set(env)

    for s in range(env.get_num_states()):
        if s in terminals:
            continue
        a_opt = int(np.argmax(Q[s]))
        traj = np.array([[s, a_opt]], dtype=np.int64)  # (T=1, 2)
        atoms.append(Atom(env_idx, "demo", traj))

    return atoms


# =====================================================
# Demo-only BIRL (feature-based)
# =====================================================

def logsumexp(x):
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))


def value_iteration_q(T, r, gamma, terminal_mask, epsilon=1e-6, max_iters=10000):
    """
    Reward-on-state convention:
      Q(s,a) = r[s] + gamma * sum_{s'} T[s,a,s'] * V[s']
      V(s)   = max_a Q(s,a)
    Terminal clamp:
      V[t]=r[t], Q[t,:]=r[t]
    """
    S, A, S2 = T.shape
    assert S == S2
    r = np.asarray(r, dtype=np.float64).reshape(S)
    V = np.zeros(S, dtype=np.float64)
    V[terminal_mask] = r[terminal_mask]

    for _ in range(max_iters):
        V_old = V.copy()
        Q = r[:, None] + gamma * np.einsum("sas,s->sa", T, V)
        V = np.max(Q, axis=1)
        V[terminal_mask] = r[terminal_mask]
        if np.max(np.abs(V - V_old)) < epsilon:
            break

    Q = r[:, None] + gamma * np.einsum("sas,s->sa", T, V)
    Q[terminal_mask, :] = r[terminal_mask][:, None]
    return Q


class DemoOnlyBIRL:
    """
    Single-env, demo-only BIRL learning feature weights w:
      r(s) = Phi[s] @ w

    Demos are read from Atom objects where Atom.data is either:
      - np.ndarray shape (T,2) with rows [s,a]
      - list of (s,a)
    """

    def __init__(self, env, atoms, Phi, beta=10.0, epsilon=1e-6):
        self.env = env
        self.atoms = atoms
        self.Phi = np.asarray(Phi, dtype=np.float64)
        self.beta = float(beta)
        self.epsilon = float(epsilon)

        self.T = np.asarray(env.transitions, dtype=np.float64)
        self.S, self.A, _ = self.T.shape

        if hasattr(env, "get_discount_factor"):
            self.gamma = float(env.get_discount_factor())
        elif hasattr(env, "gamma"):
            self.gamma = float(env.gamma)
        else:
            raise AttributeError("env must expose get_discount_factor() or env.gamma")

        terminals = get_terminal_set(env)
        self.terminal_mask = np.zeros(self.S, dtype=bool)
        for t in terminals:
            self.terminal_mask[int(t)] = True

        self.demo_sa = self._extract_demo_sa()

    def _extract_demo_sa(self):
        out = []
        for atom in self.atoms:
            if atom.feedback_type != "demo":
                continue

            traj = atom.data

            # Case 1: numpy array (T,2)
            if isinstance(traj, np.ndarray):
                if traj.ndim != 2 or traj.shape[1] != 2:
                    raise ValueError(f"Demo traj ndarray must be (T,2), got {traj.shape}")
                for row in traj:
                    s = int(row[0])
                    a = int(row[1])
                    if a < 0 or self.terminal_mask[s]:
                        continue
                    out.append((s, a))
                continue

            # Case 2: python list of (s,a)
            for (s, a) in traj:
                if a is None:
                    continue
                s = int(s); a = int(a)
                if a < 0 or self.terminal_mask[s]:
                    continue
                out.append((s, a))

        if len(out) == 0:
            raise ValueError("No (s,a) demo steps found in atoms.")
        return out

    def log_likelihood(self, w):
        w = np.asarray(w, dtype=np.float64)
        r = self.Phi @ w
        Q = value_iteration_q(self.T, r, self.gamma, self.terminal_mask, epsilon=self.epsilon)

        ll = 0.0
        for (s, a) in self.demo_sa:
            row = self.beta * Q[s]
            ll += self.beta * Q[s, a] - logsumexp(row)
        return float(ll)

    def run_mcmc(self, samples=4000, stepsize=0.5, normalize=True, seed=0):
        rng = np.random.default_rng(seed)
        d = self.Phi.shape[1]

        cur = rng.normal(size=d)
        if normalize:
            cur = cur / (np.linalg.norm(cur) + 1e-12)

        cur_ll = self.log_likelihood(cur)

        chain = np.zeros((samples, d), dtype=np.float64)
        ll_chain = np.zeros(samples, dtype=np.float64)

        map_w = cur.copy()
        map_ll = cur_ll
        accept = 0

        for i in range(samples):
            prop = cur.copy()
            j = int(rng.integers(0, d))
            prop[j] += float(rng.normal(0.0, stepsize))

            if normalize:
                prop = prop / (np.linalg.norm(prop) + 1e-12)

            prop_ll = self.log_likelihood(prop)

            if (prop_ll > cur_ll) or (rng.random() < np.exp(prop_ll - cur_ll)):
                cur, cur_ll = prop, prop_ll
                accept += 1

            chain[i] = cur
            ll_chain[i] = cur_ll

            if cur_ll > map_ll:
                map_ll = cur_ll
                map_w = cur.copy()

        self.chain = chain
        self.ll_chain = ll_chain
        self.map_w = map_w
        self.map_ll = map_ll
        self.accept_rate = accept / samples

        return map_w

    def get_map_solution(self):
        return self.map_w

    def get_mean_solution(self, burn_frac=0.2, skip_rate=5):
        b = int(len(self.chain) * burn_frac)
        return np.mean(self.chain[b::skip_rate], axis=0)


# =====================================================
# REWARD RECOVERY EXPERIMENT
# =====================================================

def test_reward_recovery(
    mdp_size=8,
    feature_dim=5,
    seed=10200,
    mcmc_samples=4000,
    stepsize=0.5,
    beta=10.0,
):
    print("\n========================================================")
    print("TEST: DemoOnlyBIRL Reward Recovery (Single Env, Full Demos)")
    print("========================================================\n")

    rng = np.random.default_rng(seed)

    # 1) Sample true reward weights
    W_true = rng.normal(size=feature_dim)
    W_true /= np.linalg.norm(W_true)
    print("[INFO] W_true =", W_true, "\n")

    # 2) Create environment
    color_to_feature_map = {
        f"f{i}": [1 if j == i else 0 for j in range(feature_dim)]
        for i in range(feature_dim)
    }
    palette = list(color_to_feature_map.keys())
    p_color_range = {c: (0.3, 0.8) for c in palette}

    envs, _ = generate_random_gridworld_envs(
        n_envs=1,
        rows=mdp_size,
        cols=mdp_size,
        color_to_feature_map=color_to_feature_map,
        palette=palette,
        p_color_range=p_color_range,
        w_mode="fixed",
        W_fixed=W_true,
        gamma_range=(0.95, 0.99),
        noise_prob_range=(0.0, 0.0),
        terminal_policy=dict(kind="random_k", k_min=0, k_max=1, p_no_terminal=0.1),
        seed=seed,
        GridEnvClass=GridWorldMDPFromLayoutEnv,
    )

    env = envs[0]
    print("[INFO] Environment loaded with", env.get_num_states(), "states.\n")

    # 3) Compute optimal Q*
    VI = ValueIteration(env, reward_convention="on")
    V_opt = VI.run_value_iteration(epsilon=1e-12)
    Q_opt = VI.get_q_values(V_opt)

    # 4) Build 1-step demo atoms
    demo_atoms = make_demo_atoms_one_step(env, Q_opt)
    print(f"[INFO] Generated {len(demo_atoms)} demo atoms.\n")

    # 5) Extract Phi and run demo-only BIRL
    Phi = extract_phi(env, feature_dim=feature_dim)
    print("[INFO] Phi shape:", Phi.shape, "\n")

    birl = DemoOnlyBIRL(env, demo_atoms, Phi, beta=beta, epsilon=1e-6)

    print("[INFO] Running MCMC...\n")
    birl.run_mcmc(samples=mcmc_samples, stepsize=stepsize, normalize=True, seed=seed)

    # 6) Results
    w_map = birl.get_map_solution()
    w_mean = birl.get_mean_solution(burn_frac=0.2, skip_rate=5)

    # (Optional) handle sign flip ambiguity (rare, but safe)
    def best_l2(w_hat, w_true):
        return min(np.linalg.norm(w_hat - w_true), np.linalg.norm(-w_hat - w_true))

    print("\n=========== RESULTS ===========\n")
    print("W_true =", W_true)
    print("\nw_map  =", w_map)
    print("\nw_mean =", w_mean)

    print("\nBest L2 error (MAP)  =", best_l2(w_map, W_true))
    print("Best L2 error (mean) =", best_l2(w_mean, W_true))
    print("\nAcceptance rate      =", birl.accept_rate)
    print("\n================================\n")

    return W_true, w_map, w_mean


# =====================================================
# RUN TEST
# =====================================================

if __name__ == "__main__":
    test_reward_recovery()
