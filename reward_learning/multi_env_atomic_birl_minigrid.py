# # ============================================================
# # Imports
# # ============================================================

# import numpy as np
# from scipy.special import logsumexp
# from tqdm import tqdm
# from dataclasses import dataclass
# from typing import List, Tuple, Any


# # ------------------------------------------------------------
# # Import YOUR existing DP + SF utilities
# # Adjust the path if needed
# # ------------------------------------------------------------
# from utils.minigrid_utils import (
#     value_iteration_next_state,
#     compute_successor_features_from_q_next_state,

# )


# def trajectory_expected_features(traj, mdp):
#     """
#     Φ(τ) = Σ γ^t φ(s_{t+1})
#     """

#     Phi = mdp["Phi"]
#     idx_of = mdp["idx_of"]
#     gamma = mdp["gamma"]

#     D = Phi.shape[1]
#     phi_sum = np.zeros(D, dtype=float)

#     g = 1.0
#     for (_, _, sp) in traj:
#         s_idx = idx_of[sp]
#         phi_sum += g * Phi[s_idx]
#         g *= gamma

#     return phi_sum



# # ============================================================
# # Atom definition
# # ============================================================

# @dataclass(frozen=True)
# class Atom:
#     atom_type: str        # "demo", "pairwise", "estop", "improvement"
#     env_id: int
#     payload: Any


# # ============================================================
# # MultiEnvAtomicBIRL for MiniGrid
# # ============================================================

# class MultiEnvAtomicBIRL_MiniGrid:
#     """
#     Unified Bayesian IRL for MiniGrid tabular MDP dicts.

#     mdps: list of MDP dicts, each with:
#         - "T"         : (S,A,S)
#         - "Phi"       : (S,D)
#         - "terminal"  : (S,)
#         - "idx_of"    : dict state -> index

#     atoms_flat: list of (env_idx, Atom)
#     """

#     # ------------------------------------------------------------
#     # Initialization
#     # ------------------------------------------------------------
#     def __init__(
#         self,
#         mdps: List[dict],
#         atoms_flat: List[Tuple[int, Atom]],
#         *,
#         beta_demo: float = 5.0,
#         beta_pairwise: float = 1.0,
#         beta_estop: float = 1.0,
#         beta_improvement: float = 1.0,
#         gamma: float = 0.99,
#         epsilon: float = 1e-8,
#     ):

#         self.mdps = mdps
#         self.gamma = gamma
#         self.epsilon = epsilon

#         self.beta_demo = beta_demo
#         self.beta_pairwise = beta_pairwise
#         self.beta_estop = beta_estop
#         self.beta_improvement = beta_improvement

#         num_envs = len(mdps)

#         # Convert flat atoms -> per-env atoms
#         self.atoms_per_env = [[] for _ in range(num_envs)]
#         for env_idx, atom in atoms_flat:
#             if not (0 <= env_idx < num_envs):
#                 raise ValueError(f"Invalid env_idx {env_idx}")
#             self.atoms_per_env[env_idx].append(atom)

#         # Feature dimension
#         self.num_mcmc_dims = mdps[0]["Phi"].shape[1]

#         # Determine required computations
#         self.needs_q = [False] * num_envs
#         self.needs_sf = [False] * num_envs

#         for e, atoms in enumerate(self.atoms_per_env):
#             for atom in atoms:
#                 if atom.atom_type == "demo":
#                     self.needs_q[e] = True
#                 #if atom.atom_type in ("pairwise", "estop", "improvement"):
#                 #    self.needs_sf[e] = True

#         self.chain = None
#         self.likelihoods = None
#         self.map_sol = None
#         self.accept_rate = None

#     # ------------------------------------------------------------
#     # Log-likelihood of all feedback atoms
#     # ------------------------------------------------------------
#     def calc_ll(self, w: np.ndarray) -> float:

#         w = np.asarray(w, float)
#         total_ll = 0.0

#         for env_idx, mdp in enumerate(self.mdps):

#             atoms = self.atoms_per_env[env_idx]
#             if not atoms:
#                 continue

#             Q = None
#             Psi_s = None
#             Psi_sa = None

#             # ----------------------------------------------------
#             # Compute Q if needed
#             # ----------------------------------------------------
#             if self.needs_q[env_idx] or self.needs_sf[env_idx]:
#                 _, Q, _ = value_iteration_next_state(
#                     mdp,
#                     w,
#                     self.gamma,
#                     tol=self.epsilon,
#                 )

#             # ----------------------------------------------------
#             # Compute successor features if needed
#             # ----------------------------------------------------
#             if self.needs_sf[env_idx]:
#                 Psi_sa, Psi_s = compute_successor_features_from_q_next_state(
#                     mdp["T"],
#                     mdp["Phi"],
#                     Q,
#                     mdp["terminal"],
#                     self.gamma,
#                 )

#             # ----------------------------------------------------
#             # Evaluate each atom
#             # ----------------------------------------------------
#             for atom in atoms:

#                 if atom.atom_type == "demo":
#                     total_ll += self._ll_demo(mdp, Q, atom.payload)

#                 elif atom.atom_type == "pairwise":
#                     total_ll += self._ll_pairwise(mdp, Psi_s, atom.payload, w)

#                 elif atom.atom_type == "estop":
#                     total_ll += self._ll_estop(mdp, Psi_s, atom.payload, w)

#                 elif atom.atom_type == "improvement":
#                     total_ll += self._ll_improvement(mdp, Psi_s, atom.payload, w)

#                 else:
#                     raise ValueError(f"Unknown atom_type {atom.atom_type}")

#         return float(total_ll)

#     # ------------------------------------------------------------
#     # Likelihood models
#     # ------------------------------------------------------------

#     def _ll_demo(self, mdp, Q, demos):

#         beta = self.beta_demo
#         terminal = mdp["terminal"]

#         log_l = 0.0

#         # demos can be a single (s,a) or list
#         if isinstance(demos, tuple):
#             demos = [demos]

#         for s, a in demos:
#             if terminal[s]:
#                 continue

#             row = beta * Q[s]
#             Z = logsumexp(row)
#             log_l += beta * Q[s, a] - Z

#         return log_l

#     def _ll_pairwise(self, mdp, Psi_s, pair, w):

#         beta = self.beta_pairwise
#         idx_of = mdp["idx_of"]

#         tau_pos, tau_neg = pair

#         psi_pos = trajectory_successor_features(
#             tau_pos, Psi_s, idx_of, self.gamma
#         )
#         psi_neg = trajectory_successor_features(
#             tau_neg, Psi_s, idx_of, self.gamma
#         )

#         r_pos = psi_pos @ w
#         r_neg = psi_neg @ w

#         Z = logsumexp([beta * r_pos, beta * r_neg])
#         return beta * r_pos - Z

#     def _ll_estop(self, mdp, Psi_s, data, w):

#         beta = self.beta_estop
#         idx_of = mdp["idx_of"]

#         traj, t_stop = data
#         prefix = traj[: t_stop + 1]

#         psi_prefix = trajectory_successor_features(
#             prefix, Psi_s, idx_of, self.gamma
#         )
#         psi_full = trajectory_successor_features(
#             traj, Psi_s, idx_of, self.gamma
#         )

#         r_pref = psi_prefix @ w
#         r_full = psi_full @ w

#         Z = logsumexp([beta * r_full, beta * r_pref])
#         return beta * r_pref - Z

#     def _ll_improvement(self, mdp, Psi_s, data, w):

#         beta = self.beta_improvement
#         idx_of = mdp["idx_of"]

#         tau_new, tau_old = data

#         psi_new = trajectory_successor_features(
#             tau_new, Psi_s, idx_of, self.gamma
#         )
#         psi_old = trajectory_successor_features(
#             tau_old, Psi_s, idx_of, self.gamma
#         )

#         r_new = psi_new @ w
#         r_old = psi_old @ w

#         Z = logsumexp([beta * r_new, beta * r_old])
#         return beta * r_new - Z

#     # ------------------------------------------------------------
#     # MCMC
#     # ------------------------------------------------------------

#     def generate_proposal(self, old, stdev, normalize=True):
#         prop = old + stdev * np.random.randn(len(old))
#         if normalize:
#             n = np.linalg.norm(prop)
#             if n > 0:
#                 prop /= n
#         return prop

#     def initial_solution(self):
#         v = np.random.randn(self.num_mcmc_dims)
#         n = np.linalg.norm(v)
#         return v / n if n > 0 else v

#     def run_mcmc(self, samples, stepsize, normalize=True, seed=None):

#         if seed is not None:
#             np.random.seed(seed)

#         T = int(samples)
#         stdev = float(stepsize)
#         accept_cnt = 0

#         self.chain = np.zeros((T, self.num_mcmc_dims))
#         self.likelihoods = np.zeros(T)

#         cur = self.initial_solution()
#         cur_ll = self.calc_ll(cur)

#         map_ll = cur_ll
#         map_sol = cur.copy()

#         pbar = tqdm(range(T), desc="MCMC Sampling")

#         for t in pbar:

#             prop = self.generate_proposal(cur, stdev, normalize)
#             prop_ll = self.calc_ll(prop)

#             accept = (
#                 prop_ll > cur_ll or
#                 np.random.rand() < np.exp(prop_ll - cur_ll)
#             )

#             if accept:
#                 cur, cur_ll = prop, prop_ll
#                 accept_cnt += 1

#                 if cur_ll > map_ll:
#                     map_ll = cur_ll
#                     map_sol = cur.copy()

#             self.chain[t] = cur
#             self.likelihoods[t] = cur_ll

#             pbar.set_postfix({
#                 "LL": f"{cur_ll:.3f}",
#                 "acc": f"{accept_cnt/(t+1):.3f}"
#             })

#         self.accept_rate = accept_cnt / T
#         self.map_sol = map_sol

#     # ------------------------------------------------------------
#     # Results
#     # ------------------------------------------------------------

#     def get_map_solution(self):
#         return self.map_sol

#     def get_mean_solution(self, burn_frac=0.1, skip_rate=1):
#         b = int(len(self.chain) * burn_frac)
#         return np.mean(self.chain[b::skip_rate], axis=0)

# ============================================================
# MultiEnvAtomicBIRL (MiniGrid) — MDP-native Phi implementation
#   - demos use Q (value iteration)
#   - pairwise/estop/improvement use discounted Phi feature counts
# ============================================================

import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional

# ------------------------------------------------------------
# Import YOUR existing DP utility
# ------------------------------------------------------------
from utils.minigrid_utils import value_iteration_next_state


# ============================================================
# Atom definition
# ============================================================

@dataclass(frozen=True)
class Atom:
    atom_type: str        # "demo", "pairwise", "estop", "improvement"
    env_id: int
    payload: Any


# ============================================================
# Trajectory discounted feature counts (MDP-native)
# ============================================================

def trajectory_expected_features(traj, mdp):
    """
    Compute discounted feature count of a trajectory:

        Φ(τ) = Σ_t γ^t φ(s_{t+1})

    Uses:
        mdp["Phi"]   : (S, D)
        mdp["idx_of"]: dict mapping state -> index
        mdp["gamma"] : discount (or falls back to mdp default if present)

    traj: list of (s, a, s_next)
    """
    Phi = mdp["Phi"]
    idx_of = mdp["idx_of"]
    gamma = mdp.get("gamma", 0.99)

    D = Phi.shape[1]
    phi_sum = np.zeros(D, dtype=float)

    g = 1.0
    for (_, _, sp) in traj:
        s_idx = idx_of[sp]
        phi_sum += g * Phi[s_idx]
        g *= gamma

    return phi_sum


# ============================================================
# MultiEnvAtomicBIRL for MiniGrid
# ============================================================

class MultiEnvAtomicBIRL_MiniGrid:
    """
    Unified Bayesian IRL for MiniGrid tabular MDP dicts.

    mdps: list of MDP dicts, each with:
        - "T"         : (S,A,S)
        - "Phi"       : (S,D)
        - "terminal"  : (S,)
        - "idx_of"    : dict state -> index
        - "gamma"     : float (recommended)

    atoms_flat: list of (env_idx, Atom)
        Atom.payload formats assumed:
          - demo:        (s,a) OR list[(s,a)] where s is state-index (int)
          - pairwise:    (tau_pos, tau_neg)
          - estop:       (traj, t_stop)
          - improvement: (tau_new, tau_old)
    """

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    def __init__(
        self,
        mdps: List[dict],
        atoms_flat: List[Tuple[int, Atom]],
        *,
        beta_demo: float = 5.0,
        beta_pairwise: float = 1.0,
        beta_estop: float = 1.0,
        beta_improvement: float = 1.0,
        gamma: Optional[float] = None,
        epsilon: float = 1e-8,
    ):
        self.mdps = mdps
        self.epsilon = float(epsilon)

        # If user passes gamma, override mdp["gamma"] (optional behavior)
        self.gamma_override = gamma

        self.beta_demo = float(beta_demo)
        self.beta_pairwise = float(beta_pairwise)
        self.beta_estop = float(beta_estop)
        self.beta_improvement = float(beta_improvement)

        num_envs = len(mdps)

        # Convert flat atoms -> per-env atoms
        self.atoms_per_env: List[List[Atom]] = [[] for _ in range(num_envs)]
        for env_idx, atom in atoms_flat:
            if not (0 <= int(env_idx) < num_envs):
                raise ValueError(f"Invalid env_idx {env_idx}")
            self.atoms_per_env[int(env_idx)].append(atom)

        # Feature dimension
        self.num_mcmc_dims = int(mdps[0]["Phi"].shape[1])

        # Determine which envs need Q (only if demo atoms exist)
        self.needs_q = [False] * num_envs
        for e, atoms in enumerate(self.atoms_per_env):
            for atom in atoms:
                if atom.atom_type == "demo":
                    self.needs_q[e] = True

        # MCMC storage
        self.chain = None
        self.likelihoods = None
        self.map_sol = None
        self.accept_rate = None

    # ------------------------------------------------------------
    # Internal: resolve gamma for an env
    # ------------------------------------------------------------
    def _gamma(self, mdp: dict) -> float:
        if self.gamma_override is not None:
            return float(self.gamma_override)
        return float(mdp.get("gamma", 0.99))

    # ------------------------------------------------------------
    # Log-likelihood of all feedback atoms
    # ------------------------------------------------------------
    def calc_ll(self, w: np.ndarray) -> float:
        w = np.asarray(w, float)
        total_ll = 0.0

        for env_idx, mdp in enumerate(self.mdps):
            atoms = self.atoms_per_env[env_idx]
            if not atoms:
                continue

            # Set mdp gamma if override is used (so trajectory_expected_features sees it)
            if self.gamma_override is not None:
                mdp = dict(mdp)  # shallow copy
                mdp["gamma"] = self._gamma(mdp)

            Q = None

            # ----------------------------------------------------
            # Compute Q only if demo atoms exist for this env
            # ----------------------------------------------------
            if self.needs_q[env_idx]:
                _, Q, _ = value_iteration_next_state(
                    mdp,
                    w,
                    self._gamma(mdp),
                    tol=self.epsilon,
                )

            # ----------------------------------------------------
            # Evaluate each atom
            # ----------------------------------------------------
            for atom in atoms:
                t = atom.atom_type

                if t == "demo":
                    total_ll += self._ll_demo(mdp, Q, atom.payload)

                elif t == "pairwise":
                    total_ll += self._ll_pairwise(mdp, atom.payload, w)

                elif t == "estop":
                    total_ll += self._ll_estop(mdp, atom.payload, w)

                elif t == "improvement":
                    total_ll += self._ll_improvement(mdp, atom.payload, w)

                else:
                    raise ValueError(f"Unknown atom_type {t}")

        return float(total_ll)

    # ------------------------------------------------------------
    # Likelihood models
    # ------------------------------------------------------------

    def _ll_demo(self, mdp: dict, Q: np.ndarray, demos):
        """
        Softmax demo likelihood:
            log p(a|s) = beta*Q[s,a] - logsumexp(beta*Q[s,:])

        Assumes demos are (s_idx, a_idx) or list thereof.
        """
        if Q is None:
            raise RuntimeError("Demo likelihood requested but Q is None. (needs_q bug)")

        beta = self.beta_demo
        terminal = mdp["terminal"]

        log_l = 0.0

        # demos can be a single (s,a) or list
        if isinstance(demos, tuple):
            demos = [demos]

        for s, a in demos:
            s = int(s)
            a = int(a)

            if terminal[s]:
                continue

            row = beta * Q[s]
            Z = logsumexp(row)
            log_l += beta * Q[s, a] - Z

        return float(log_l)

    def _ll_pairwise(self, mdp: dict, pair, w: np.ndarray):
        """
        Pairwise preference likelihood using trajectory returns:
            P(tau_pos > tau_neg) = softmax(beta * R(tau_pos), beta * R(tau_neg))
        """
        beta = self.beta_pairwise
        tau_pos, tau_neg = pair

        psi_pos = trajectory_expected_features(tau_pos, mdp)
        psi_neg = trajectory_expected_features(tau_neg, mdp)

        r_pos = float(psi_pos @ w)
        r_neg = float(psi_neg @ w)

        Z = logsumexp([beta * r_pos, beta * r_neg])
        return float(beta * r_pos - Z)

    def _ll_estop(self, mdp: dict, data, w: np.ndarray):
        """
        E-stop likelihood:
          compare prefix vs full:
            P(prefix preferred) = softmax(beta * R(prefix), beta * R(full))
        """
        beta = self.beta_estop
        traj, t_stop = data

        prefix = traj[: int(t_stop) + 1]

        psi_prefix = trajectory_expected_features(prefix, mdp)
        psi_full = trajectory_expected_features(traj, mdp)

        r_pref = float(psi_prefix @ w)
        r_full = float(psi_full @ w)

        Z = logsumexp([beta * r_full, beta * r_pref])
        return float(beta * r_pref - Z)

    def _ll_improvement(self, mdp: dict, data, w: np.ndarray):
        """
        Improvement likelihood:
            P(new preferred) = softmax(beta * R(new), beta * R(old))
        """
        beta = self.beta_improvement
        tau_new, tau_old = data

        psi_new = trajectory_expected_features(tau_new, mdp)
        psi_old = trajectory_expected_features(tau_old, mdp)

        r_new = float(psi_new @ w)
        r_old = float(psi_old @ w)

        Z = logsumexp([beta * r_new, beta * r_old])
        return float(beta * r_new - Z)

    # ------------------------------------------------------------
    # MCMC
    # ------------------------------------------------------------

    def generate_proposal(self, old: np.ndarray, stdev: float, normalize: bool = True):
        prop = old + float(stdev) * np.random.randn(len(old))
        if normalize:
            n = np.linalg.norm(prop)
            if n > 0:
                prop /= n
        return prop

    def initial_solution(self):
        v = np.random.randn(self.num_mcmc_dims)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def run_mcmc(self, samples: int, stepsize: float, normalize: bool = True, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(int(seed))

        T = int(samples)
        stdev = float(stepsize)
        accept_cnt = 0

        self.chain = np.zeros((T, self.num_mcmc_dims), dtype=float)
        self.likelihoods = np.zeros(T, dtype=float)

        cur = self.initial_solution()
        cur_ll = self.calc_ll(cur)

        map_ll = cur_ll
        map_sol = cur.copy()

        pbar = tqdm(range(T), desc="MCMC Sampling")

        for t in pbar:
            prop = self.generate_proposal(cur, stdev, normalize)
            prop_ll = self.calc_ll(prop)

            accept = (prop_ll > cur_ll) or (np.random.rand() < np.exp(prop_ll - cur_ll))

            if accept:
                cur, cur_ll = prop, prop_ll
                accept_cnt += 1

                if cur_ll > map_ll:
                    map_ll = cur_ll
                    map_sol = cur.copy()

            self.chain[t] = cur
            self.likelihoods[t] = cur_ll

            pbar.set_postfix({
                "LL": f"{cur_ll:.3f}",
                "acc": f"{accept_cnt/(t+1):.3f}",
            })

        self.accept_rate = accept_cnt / max(T, 1)
        self.map_sol = map_sol

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    def get_map_solution(self):
        return self.map_sol

    def get_mean_solution(self, burn_frac: float = 0.1, skip_rate: int = 1):
        if self.chain is None or len(self.chain) == 0:
            raise RuntimeError("No chain found. Run run_mcmc() first.")
        b = int(len(self.chain) * float(burn_frac))
        return np.mean(self.chain[b::int(skip_rate)], axis=0)
