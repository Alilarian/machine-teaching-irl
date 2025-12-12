import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
from tqdm import tqdm


# ------------------------------------------------------------
# 1) Utilities
# ------------------------------------------------------------

def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def _get_terminal_mask(env) -> np.ndarray:
    S = int(env.get_num_states()) if hasattr(env, "get_num_states") else int(env.num_states)
    mask = np.zeros(S, dtype=bool)

    # Most common in your codebase: env.terminal_states
    if hasattr(env, "terminal_states") and env.terminal_states is not None:
        for t in env.terminal_states:
            mask[int(t)] = True
        return mask

    # Fallback: env.terminals
    if hasattr(env, "terminals") and env.terminals is not None:
        for t in env.terminals:
            mask[int(t)] = True
        return mask

    # Fallback: env.terminal_mask
    if hasattr(env, "terminal_mask"):
        tm = np.asarray(env.terminal_mask, dtype=bool)
        if tm.shape[0] == S:
            return tm

    return mask


def _extract_demo_sa_from_atoms(env, atoms: List[Any], env_idx: int = 0) -> List[Tuple[int, int]]:
    """
    atoms: list[Atom] where Atom.data is a trajectory [(s,a), ...]
    We flatten to a list of (s,a) and skip:
      - non-demo atoms
      - terminal states
      - a is None
    """
    terminal_mask = _get_terminal_mask(env)
    demo_sa: List[Tuple[int, int]] = []

    for atom in atoms:
        if getattr(atom, "env_idx", env_idx) != env_idx:
            continue
        if getattr(atom, "feedback_type", None) != "demo":
            continue

        traj = atom.data  # list[(s,a)]
        for (s, a) in traj:
            if a is None:
                continue
            s = int(s); a = int(a)
            if terminal_mask[s]:
                continue
            demo_sa.append((s, a))

    return demo_sa


def _value_iteration_q(
    T: np.ndarray,          # (S,A,S)
    r: np.ndarray,          # (S,)
    gamma: float,
    terminal_mask: np.ndarray,
    epsilon: float = 1e-6,
    max_iters: int = 10_000,
) -> np.ndarray:
    """
    Reward-on-state convention:
      Q(s,a) = r[s] + gamma * sum_{s'} T[s,a,s'] * V[s']
      V(s)   = max_a Q(s,a)
    Terminal handling:
      clamp V[t] = r[t], and Q[t,:] = r[t]
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


# ------------------------------------------------------------
# 2) Result container
# ------------------------------------------------------------

@dataclass
class BIRLResult:
    chain: np.ndarray          # (N,D)
    ll_chain: np.ndarray       # (N,)
    accept_rate: float
    map_theta: np.ndarray
    map_ll: float


# ------------------------------------------------------------
# 3) Demo-only BIRL
# ------------------------------------------------------------

class DemoOnlyBIRL:
    """
    Single-env BIRL using ONLY demo atoms:
      log p(D|theta) = sum_{(s,a) in demos} [ beta*Q_theta(s,a) - logsumexp(beta*Q_theta(s,:)) ]

    Theta can be:
      - state rewards r[s]  (D = S)
      - feature weights w  (D = d), if env exposes Phi
    """

    def __init__(self, env, atoms: List[Any], beta: float, epsilon: float = 1e-6, env_idx: int = 0):
        self.env = env
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.env_idx = int(env_idx)

        # Pull MDP pieces
        self.T = np.asarray(env.transitions, dtype=np.float64)
        self.S, self.A, _ = self.T.shape

        if hasattr(env, "get_discount_factor"):
            self.gamma = float(env.get_discount_factor())
        elif hasattr(env, "gamma"):
            self.gamma = float(env.gamma)
        else:
            raise AttributeError("env must expose discount factor as env.get_discount_factor() or env.gamma")

        self.terminal_mask = _get_terminal_mask(env)

        # Extract demos from atoms
        self.demo_sa = _extract_demo_sa_from_atoms(env, atoms, env_idx=self.env_idx)
        if len(self.demo_sa) == 0:
            raise ValueError("No demo steps found. Make sure atoms contain Atom(..., feedback_type='demo', data=traj).")

        # Reward parameterization: auto-detect features
        self.Phi = None
        if hasattr(env, "feature_matrix"):
            self.Phi = np.asarray(env.feature_matrix, dtype=np.float64)  # (S,d)
        elif hasattr(env, "state_features"):
            self.Phi = np.asarray(env.state_features, dtype=np.float64)  # (S,d)

        if self.Phi is not None:
            self.mode = "feature"
            self.D = int(self.Phi.shape[1])
        else:
            self.mode = "state"
            self.D = int(self.S)

    def _theta_to_reward(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64).reshape(self.D)

        if self.mode == "feature":
            r = self.Phi @ theta
            return r.astype(np.float64, copy=False)
        else:
            return theta.astype(np.float64, copy=False)

    def log_likelihood(self, theta: np.ndarray) -> float:
        r = self._theta_to_reward(theta)
        Q = _value_iteration_q(self.T, r, self.gamma, self.terminal_mask, epsilon=self.epsilon)

        ll = 0.0
        for (s, a) in self.demo_sa:
            row = self.beta * Q[s]  # (A,)
            ll += self.beta * Q[s, a] - _logsumexp(row)
        return float(ll)

    def _proposal(self, theta: np.ndarray, step_size: float, rng: np.random.Generator) -> np.ndarray:
        prop = theta.copy()
        j = int(rng.integers(0, self.D))
        prop[j] += float(rng.normal(0.0, step_size))
        return prop

    def run_mcmc(
        self,
        num_samples: int,
        step_size: float = 0.5,
        init_theta: Optional[np.ndarray] = None,
        theta_clip: Optional[float] = None,
        seed: Optional[int] = None,
        show_progress: bool = True,
    ) -> BIRLResult:
        rng = np.random.default_rng(seed)

        if init_theta is None:
            cur = rng.normal(0.0, 1.0, size=self.D).astype(np.float64)
        else:
            cur = np.asarray(init_theta, dtype=np.float64).reshape(self.D).copy()

        if theta_clip is not None:
            cur = np.clip(cur, -theta_clip, theta_clip)

        cur_ll = self.log_likelihood(cur)

        chain = np.zeros((num_samples, self.D), dtype=np.float64)
        ll_chain = np.zeros(num_samples, dtype=np.float64)

        map_theta = cur.copy()
        map_ll = cur_ll
        accept = 0

        it = range(num_samples)
        if show_progress:
            it = tqdm(it, desc=f"DemoBIRL(MCMC) mode={self.mode}, D={self.D}")

        for k in it:
            prop = self._proposal(cur, step_size, rng)
            if theta_clip is not None:
                prop = np.clip(prop, -theta_clip, theta_clip)

            prop_ll = self.log_likelihood(prop)

            if (prop_ll > cur_ll) or (rng.random() < np.exp(prop_ll - cur_ll)):
                cur, cur_ll = prop, prop_ll
                accept += 1

            chain[k] = cur
            ll_chain[k] = cur_ll

            if cur_ll > map_ll:
                map_ll = cur_ll
                map_theta = cur.copy()

        return BIRLResult(
            chain=chain,
            ll_chain=ll_chain,
            accept_rate=accept / num_samples,
            map_theta=map_theta,
            map_ll=map_ll,
        )
