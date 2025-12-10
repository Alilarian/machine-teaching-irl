import numpy as np
from concurrent.futures import ProcessPoolExecutor

from numba import njit


@njit
def vi_entering_kernel(T, r_enter, gamma, terminal_mask, V, V_prev,
                       thresh, max_iters, terminal_value):
    """
    Numba-accelerated value iteration for 'entering' reward convention.

    T: (S, A, S)
    r_enter: (S,)
    gamma: float
    terminal_mask: (S,) bool array
    V, V_prev: (S,)
    """
    S, A, _ = T.shape
    iters = 0
    delta = 1e9

    while delta > thresh and iters < max_iters:
        delta = 0.0

        # Precompute target[s'] = r_enter[s'] + gamma * V_prev[s']
        target = np.empty(S)
        for sp in range(S):
            target[sp] = r_enter[sp] + gamma * V_prev[sp]

        for s in range(S):
            if terminal_mask[s]:
                new_v = terminal_value
            else:
                best = -1e15
                for a in range(A):
                    tmp = 0.0
                    for sp in range(S):
                        tmp += T[s, a, sp] * target[sp]
                    if tmp > best:
                        best = tmp
                new_v = best

            diff = abs(new_v - V_prev[s])
            if diff > delta:
                delta = diff
            V[s] = new_v

        # copy V -> V_prev
        for i in range(S):
            V_prev[i] = V[i]

        iters += 1

    return V


@njit
def vi_on_kernel(T, r_on, gamma, terminal_mask, V, V_prev,
                 thresh, max_iters, terminal_value):
    """
    Numba-accelerated value iteration for 'on' reward convention.

    T: (S, A, S)
    r_on: (S,)
    gamma: float
    terminal_mask: (S,) bool array
    V, V_prev: (S,)
    """
    S, A, _ = T.shape
    iters = 0
    delta = 1e9

    while delta > thresh and iters < max_iters:
        delta = 0.0

        for s in range(S):
            if terminal_mask[s]:
                new_v = terminal_value
            else:
                best = -1e15
                for a in range(A):
                    tmp = 0.0
                    for sp in range(S):
                        tmp += T[s, a, sp] * V_prev[sp]
                    tmp = r_on[s] + gamma * tmp
                    if tmp > best:
                        best = tmp
                new_v = best

            diff = abs(new_v - V_prev[s])
            if diff > delta:
                delta = diff
            V[s] = new_v

        # copy V -> V_prev
        for i in range(S):
            V_prev[i] = V[i]

        iters += 1

    return V
class ValueIteration:
    """
    Value Iteration for a tabular MDP with flexible reward convention.

    Assumptions about `mdp`:
      - mdp.transitions: numpy array with shape (S, A, S), row-stochastic per (s,a)
      - mdp.get_num_states() -> int
      - mdp.get_num_actions() -> int
      - mdp.get_discount_factor() -> float in [0, 1)
      - mdp.compute_reward(s: int) -> float (state-based reward function)
      - Optional: mdp.terminal_states -> iterable of terminal state indices (absorbing)
    """

    def __init__(
        self,
        mdp,
        reward_convention: str = "entering",
        clamp_terminal_values: bool = True,
        terminal_value: float = 0.0,
    ):
        self.mdp = mdp
        self.S = mdp.get_num_states()
        self.A = mdp.get_num_actions()
        self.gamma = float(mdp.get_discount_factor())
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError("Discount factor γ must be in [0, 1).")

        self.reward_convention = reward_convention.lower().strip()
        if self.reward_convention not in ("on", "entering"):
            raise ValueError("reward_convention must be 'on' or 'entering'.")

        self.clamp_terminal_values = bool(clamp_terminal_values)
        self.terminal_value = float(terminal_value)

        self.T = np.asarray(mdp.transitions, dtype=float)  # (S, A, S)
        if self.T.shape != (self.S, self.A, self.S):
            raise ValueError("mdp.transitions must have shape (S, A, S).")

        terminals = getattr(mdp, "terminal_states", []) or []
        self.terminals = list(terminals)
        self.terminal_mask = np.zeros(self.S, dtype=np.bool_)
        for t in self.terminals:
            if 0 <= t < self.S:
                self.terminal_mask[t] = True

        # Precompute rewards for both conventions
        self.r_on = np.array(
            [mdp.compute_reward(s) for s in range(self.S)],
            dtype=float
        )
        # For entering-state reward we use reward at s'
        self.r_enter = self.r_on.copy()

        # State values
        self.state_values = np.zeros(self.S, dtype=float)

    def _threshold(self, epsilon: float) -> float:
        """Convergence threshold matching contraction bound; special-case gamma=0."""
        if self.gamma == 0.0:
            return epsilon
        return float(epsilon * (1.0 - self.gamma) / self.gamma)

    def run_value_iteration(self, epsilon: float = 1e-10, max_iters: int = 1_000_000):
        """
        Compute optimal V using Bellman optimality equation with the chosen reward convention.

        Returns:
            np.ndarray of shape (S,) with optimal state values.
        """
        thresh = self._threshold(epsilon)
        V = self.state_values.copy()
        V_prev = V.copy()

        if self.reward_convention == "entering":
            V = vi_entering_kernel(
                self.T,
                self.r_enter,
                self.gamma,
                self.terminal_mask,
                V,
                V_prev,
                thresh,
                max_iters,
                self.terminal_value,
            )
        else:  # "on"
            V = vi_on_kernel(
                self.T,
                self.r_on,
                self.gamma,
                self.terminal_mask,
                V,
                V_prev,
                thresh,
                max_iters,
                self.terminal_value,
            )

        self.state_values = V
        return V

    def get_q_values(self, state_values: np.ndarray | None = None) -> np.ndarray:
        """
        Compute optimal (greedy w.r.t. V) Q-values consistent with the selected reward convention.

        Returns:
            qvalues: np.ndarray with shape (S, A)
        """
        if state_values is None:
            state_values = self.run_value_iteration()

        # Vectorized Q computation
        if self.reward_convention == "entering":
            # Q(s,a) = Σ_{s'} T[s,a,s'] * ( r_enter[s'] + γ V[s'] )
            target = self.r_enter + self.gamma * state_values  # (S,)
            # T: (S, A, S), target: (S,) → result: (S, A)
            Q = self.T @ target
        else:
            # Q(s,a) = r_on[s] + γ * Σ_{s'} T[s,a,s'] * V[s']
            # T @ V: (S, A, S) @ (S,) → (S, A)
            Q = self.r_on[:, None] + self.gamma * (self.T @ state_values)

        # Clamp terminals
        if self.clamp_terminal_values and len(self.terminals) > 0:
            Q[self.terminal_mask, :] = 0.0

        return Q

    def get_optimal_policy(
        self,
        tie_eps: float = 1e-12,
        return_matrix: bool = False,
    ):
        """
        Extract a greedy policy w.r.t. the computed V (and Q).

        Args:
            tie_eps: actions within tie_eps of the max-Q are treated as ties (uniformized).
            return_matrix: if True, return Π as (S,A) stochastic matrix; else list of (state, best_action or None).

        Returns:
            If return_matrix:
                Π: np.ndarray (S, A) with uniform mass over argmax Q actions (row-stochastic).
            Else:
                List[(state, best_action or None)]
        """
        if np.allclose(self.state_values, 0.0):
            self.run_value_iteration()

        Q = self.get_q_values(self.state_values)

        if return_matrix:
            Pi = np.zeros_like(Q)
            for s in range(self.S):
                if self.clamp_terminal_values and self.terminal_mask[s]:
                    continue
                row = Q[s]
                m = np.max(row)
                mask = np.abs(row - m) <= tie_eps
                k = int(mask.sum())
                if k > 0:
                    Pi[s, mask] = 1.0 / k
            return Pi

        policy = []
        for s in range(self.S):
            if self.clamp_terminal_values and self.terminal_mask[s]:
                policy.append((s, None))
            else:
                best_action = int(np.argmax(Q[s]))
                policy.append((s, best_action))
        return policy


from numba import njit

@njit
def pe_entering_kernel(T, Pi, r_enter, gamma, terminal_mask,
                       V, V_prev, thresh, max_iters, terminal_value):
    """
    Numba-accelerated policy evaluation kernel (entering reward convention).
    """
    S, A, _ = T.shape
    iters = 0
    delta = 1e9

    while delta > thresh and iters < max_iters:
        delta = 0.0

        # target[s'] = r_enter[s'] + gamma * V_prev[s']
        target = r_enter + gamma * V_prev

        for s in range(S):
            if terminal_mask[s]:
                new_v = terminal_value
            else:
                # exp_sa[a] = T[s,a] @ target
                best = 0.0
                for a in range(A):
                    tmp = 0.0
                    for sp in range(S):
                        tmp += T[s, a, sp] * target[sp]
                    best += Pi[s, a] * tmp
                new_v = best

            diff = abs(new_v - V_prev[s])
            if diff > delta:
                delta = diff

            V[s] = new_v

        # Copy V -> V_prev
        for i in range(S):
            V_prev[i] = V[i]

        iters += 1

    return V


@njit
def pe_on_kernel(T, Pi, r_on, gamma, terminal_mask,
                 V, V_prev, thresh, max_iters, terminal_value):
    """
    Numba-accelerated policy evaluation kernel (on-state reward).
    """
    S, A, _ = T.shape
    iters = 0
    delta = 1e9

    while delta > thresh and iters < max_iters:
        delta = 0.0

        for s in range(S):
            if terminal_mask[s]:
                new_v = terminal_value
            else:
                acc = 0.0
                for a in range(A):
                    tmp = 0.0
                    for sp in range(S):
                        tmp += T[s, a, sp] * V_prev[sp]
                    acc += Pi[s, a] * tmp
                new_v = r_on[s] + gamma * acc

            diff = abs(new_v - V_prev[s])
            if diff > delta:
                delta = diff

            V[s] = new_v

        # Copy V → V_prev
        for i in range(S):
            V_prev[i] = V[i]

        iters += 1

    return V

import numpy as np

class PolicyEvaluation:
    """
    Optimized Policy Evaluation with:
      - Numba-accelerated iterative evaluation
      - Vectorized Q-value computation
      - Same semantics as your original implementation
    """

    def __init__(
        self,
        mdp,
        policy=None,
        uniform_random: bool = False,
        reward_convention: str = "entering",
        clamp_terminal_values: bool = True,
        terminal_value: float = 0.0,
    ):
        self.mdp = mdp
        self.S = mdp.get_num_states()
        self.A = mdp.get_num_actions()
        self.gamma = float(mdp.get_discount_factor())

        self.reward_convention = reward_convention.lower().strip()
        if self.reward_convention not in ("on", "entering"):
            raise ValueError("reward_convention must be 'on' or 'entering'.")

        self.clamp_terminal_values = bool(clamp_terminal_values)
        self.terminal_value = float(terminal_value)

        # --- Transitions and terminals ---
        self.T = np.asarray(mdp.transitions, dtype=float)
        self.terminals = list(getattr(mdp, "terminal_states", []) or [])
        self.terminal_mask = np.zeros(self.S, dtype=np.bool_)
        for t in self.terminals:
            if 0 <= t < self.S:
                self.terminal_mask[t] = True

        # --- Rewards ---
        self.r_on = np.array(
            [mdp.compute_reward(s) for s in range(self.S)],
            dtype=float,
        )
        self.r_enter = self.r_on.copy()

        # --- Policy ---
        self.uniform_random = bool(uniform_random)
        self.Pi = self._build_policy_matrix(policy, self.uniform_random)

        # --- State-value buffer ---
        self.state_values = np.zeros(self.S, dtype=float)

    # ========== Helpers ==========

    def _threshold(self, epsilon: float) -> float:
        if self.gamma == 0.0:
            return epsilon
        return epsilon * (1 - self.gamma) / self.gamma

    def _build_policy_matrix(self, policy, uniform_random):
        Pi = np.zeros((self.S, self.A), dtype=float)

        if uniform_random:
            for s in range(self.S):
                if not self.terminal_mask[s]:
                    Pi[s, :] = 1.0 / self.A
            return Pi

        if policy is None:
            raise ValueError("Provide a policy or set uniform_random=True.")

        # Matrix
        if isinstance(policy, np.ndarray):
            Pi[:] = policy
        else:
            # List: deterministic
            if isinstance(policy, (list, tuple)) and len(policy) == self.S:
                for s, a in enumerate(policy):
                    if self.terminal_mask[s] or a is None:
                        continue
                    Pi[s, int(a)] = 1.0
            else:
                raise ValueError("Unsupported policy format.")

        # Normalize rows and clamp terminals
        for s in range(self.S):
            if self.terminal_mask[s]:
                Pi[s, :] = 0.0
            else:
                rs = Pi[s].sum()
                if rs <= 0:
                    raise ValueError(f"Policy row {s} has zero mass.")
                Pi[s] /= rs

        return Pi

    # ========== MAIN POLICY EVALUATION ==========

    def run_policy_evaluation(self, epsilon=1e-10, max_iters=1_000_000):
        V = self.state_values.copy()
        V_prev = V.copy()
        thresh = self._threshold(epsilon)

        if self.reward_convention == "entering":
            V = pe_entering_kernel(
                self.T,
                self.Pi,
                self.r_enter,
                self.gamma,
                self.terminal_mask,
                V,
                V_prev,
                thresh,
                max_iters,
                self.terminal_value,
            )
        else:
            V = pe_on_kernel(
                self.T,
                self.Pi,
                self.r_on,
                self.gamma,
                self.terminal_mask,
                V,
                V_prev,
                thresh,
                max_iters,
                self.terminal_value,
            )

        self.state_values = V
        return V

    # ========== VECTORIZED Q FUNCTION ==========

    def get_q_values(self, state_values=None):
        if state_values is None:
            state_values = self.state_values
            if np.allclose(state_values, 0.0):
                state_values = self.run_policy_evaluation()

        if self.reward_convention == "entering":
            target = self.r_enter + self.gamma * state_values
            Q = self.T @ target   # (S, A, S) @ (S,) -> (S, A)
        else:
            Q = self.r_on[:, None] + self.gamma * (self.T @ state_values)

        # Clamp terminals
        Q[self.terminal_mask] = 0.0
        return Q
