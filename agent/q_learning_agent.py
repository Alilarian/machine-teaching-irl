import numpy as np

import numpy as np

class ValueIteration:
    """
    Value Iteration for a tabular MDP with flexible reward convention.

    Assumptions about `mdp`:
      - mdp.transitions: numpy array with shape (S, A, S), row-stochastic per (s,a)
      - mdp.get_num_states() -> int
      - mdp.get_num_actions() -> int
      - mdp.get_discount_factor() -> float in [0, 1)
      - mdp.compute_reward(s: int) -> float  (state-based reward function)
      - Optional: mdp.terminal_states -> iterable of terminal state indices (absorbing)
    """

    def __init__(
        self,
        mdp,
        reward_convention: str = "entering",
        clamp_terminal_values: bool = True,
        terminal_value: float = 0.0,
    ):
        """
        Args:
            mdp: MDP object with the interface described above.
            reward_convention: Either "on" or "entering".
                - "on": r(s) is granted at the current state (on-state reward).
                - "entering": r(s') is granted upon transition to next state (entering-state reward).
            clamp_terminal_values: If True, enforce V(t) = terminal_value and Q(t,·) = 0 for terminals.
            terminal_value: The fixed value assigned to terminal states when clamped.
        """
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

        self.terminals = set(getattr(mdp, "terminal_states", []) or [])

        # Precompute rewards for both conventions
        self.r_on = np.array([mdp.compute_reward(s) for s in range(self.S)], dtype=float)
        # For entering-state reward we still use the same compute_reward(s') API
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
        V = self.state_values  # alias

        # Precompute one-shot arrays used in the inner loop
        # For "entering": Q(s,a) = Σ_{s'} T[s,a,s'] * ( r_enter[s'] + γ V[s'] )
        # For "on":       V(s)   = r_on[s] + γ * max_a Σ_{s'} T[s,a,s'] * V[s']
        delta = np.inf
        iters = 0

        while delta > thresh and iters < max_iters:
            V_prev = V.copy()
            delta = 0.0

            if self.reward_convention == "entering":
                # Vector that will be right-multiplied by T[s]
                target = self.r_enter + self.gamma * V_prev  # shape (S,)
                for s in range(self.S):
                    if self.clamp_terminal_values and s in self.terminals:
                        new_v = self.terminal_value
                    else:
                        # For each action, expected return = T[s,a] @ target
                        exp = self.T[s] @ target  # shape (A,)
                        new_v = float(np.max(exp))
                    delta = max(delta, abs(new_v - V_prev[s]))
                    V[s] = new_v

            else:  # "on" convention
                # For each state: V(s) = r_on[s] + γ * max_a ( T[s,a] @ V_prev )
                for s in range(self.S):
                    if self.clamp_terminal_values and s in self.terminals:
                        new_v = self.terminal_value
                    else:
                        exp = self.T[s] @ V_prev  # shape (A,)
                        new_v = float(self.r_on[s] + self.gamma * np.max(exp))
                    delta = max(delta, abs(new_v - V_prev[s]))
                    V[s] = new_v

            iters += 1

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

        Q = np.zeros((self.S, self.A), dtype=float)

        if self.reward_convention == "entering":
            # Q(s,a) = Σ_{s'} T[s,a,s'] * ( r_enter[s'] + γ V[s'] )
            target = self.r_enter + self.gamma * state_values  # shape (S,)
            for s in range(self.S):
                if self.clamp_terminal_values and s in self.terminals:
                    Q[s, :] = 0.0
                else:
                    Q[s, :] = self.T[s] @ target
        else:
            # Q(s,a) = r_on[s] + γ * Σ_{s'} T[s,a,s'] * V[s']
            for s in range(self.S):
                if self.clamp_terminal_values and s in self.terminals:
                    Q[s, :] = 0.0
                else:
                    Q[s, :] = self.r_on[s] + self.gamma * (self.T[s] @ state_values)

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
                if self.clamp_terminal_values and s in self.terminals:
                    # No action at terminal: leave row zeros (consistent with many planners)
                    continue
                row = Q[s]
                m = np.max(row)
                mask = np.abs(row - m) <= tie_eps
                k = int(mask.sum())
                if k > 0:
                    Pi[s, mask] = 1.0 / k
            return Pi

        # Otherwise best single action per state (None for terminals)
        policy = []
        for s in range(self.S):
            if self.clamp_terminal_values and s in self.terminals:
                policy.append((s, None))
            else:
                best_action = int(np.argmax(Q[s]))
                policy.append((s, best_action))
        return policy

class PolicyEvaluation:
    """
    Policy evaluation aligned with ValueIteration:

    Assumptions about `mdp` match ValueIteration:
      - mdp.transitions: numpy array with shape (S, A, S), row-stochastic per (s,a)
      - mdp.get_num_states() -> int
      - mdp.get_num_actions() -> int
      - mdp.get_discount_factor() -> float in [0, 1)
      - mdp.compute_reward(s: int) -> float  (state-based reward function)
      - Optional: mdp.terminal_states -> iterable of terminal state indices (absorbing)

    Reward conventions:
      - "on":       V(s) = r_on[s] + γ * Σ_a π(s,a) Σ_{s'} T[s,a,s'] V(s')
      - "entering": V(s) = Σ_a π(s,a) Σ_{s'} T[s,a,s'] * ( r_enter[s'] + γ V(s') )

    Terminal handling (identical to ValueIteration):
      - If clamp_terminal_values=True: enforce V(t)=terminal_value and leave Π(t,·)=0, Q(t,·)=0.
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
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError("Discount factor γ must be in [0, 1).")

        self.reward_convention = reward_convention.lower().strip()
        if self.reward_convention not in ("on", "entering"):
            raise ValueError("reward_convention must be 'on' or 'entering'.")

        self.clamp_terminal_values = bool(clamp_terminal_values)
        self.terminal_value = float(terminal_value)

        # Transitions and terminals
        self.T = np.asarray(mdp.transitions, dtype=float)  # (S, A, S)
        if self.T.shape != (self.S, self.A, self.S):
            raise ValueError("mdp.transitions must have shape (S, A, S).")
        self.terminals = set(getattr(mdp, "terminal_states", []) or [])

        # Rewards consistent with ValueIteration
        self.r_on = np.array([mdp.compute_reward(s) for s in range(self.S)], dtype=float)
        self.r_enter = self.r_on.copy()  # entering uses compute_reward(s') as well

        # Policy handling
        self.uniform_random = bool(uniform_random)
        self.Pi = self._build_policy_matrix(policy, self.uniform_random)

        # State values buffer (optional persistence)
        self.state_values = np.zeros(self.S, dtype=float)

    # ---- helpers -------------------------------------------------------------

    def _threshold(self, epsilon: float) -> float:
        """Match ValueIteration convergence threshold; special-case γ=0."""
        if self.gamma == 0.0:
            return float(epsilon)
        return float(epsilon * (1.0 - self.gamma) / self.gamma)

    def _build_policy_matrix(self, policy, uniform_random: bool) -> np.ndarray:
        """
        Construct Π as (S,A) row-stochastic matrix.
        Accepts:
          - None + uniform_random=True  -> uniform over actions for non-terminals
          - (S,A) numpy array           -> validated as row-stochastic on non-terminals
          - list/array of actions       -> deterministic Π with 1 on chosen action
          - list of (state, action|None) pairs (from ValueIteration.get_optimal_policy)
        For terminal states: Π(t,·) = 0 (row of zeros), matching ValueIteration behavior.
        """
        Pi = np.zeros((self.S, self.A), dtype=float)

        if uniform_random:
            for s in range(self.S):
                if s in self.terminals:
                    continue
                Pi[s, :] = 1.0 / self.A
            return Pi

        if policy is None:
            raise ValueError("Provide a policy or set uniform_random=True.")

        # If matrix provided
        if isinstance(policy, np.ndarray):
            if policy.shape != (self.S, self.A):
                raise ValueError("Policy matrix must have shape (S, A).")
            Pi[:] = policy
        else:
            # If list/tuple provided
            # Accept either [action, ...] or [(state, action_or_None), ...]
            if isinstance(policy, (list, tuple)) and len(policy) == self.S:
                # [(s, a|None)] or [a|None]
                if isinstance(policy[0], (list, tuple)) and len(policy[0]) == 2:
                    # [(s, a|None)]
                    for s, a in policy:
                        if s in self.terminals or a is None:
                            continue
                        Pi[s, int(a)] = 1.0
                else:
                    # [a|None]
                    for s, a in enumerate(policy):
                        if s in self.terminals or a is None:
                            continue
                        Pi[s, int(a)] = 1.0
            else:
                raise ValueError("Unsupported policy format.")

        # Enforce terminal rows = 0
        if self.terminals:
            Pi[list(self.terminals), :] = 0.0

        # Validate / renormalize non-terminal rows if needed
        for s in range(self.S):
            if s in self.terminals:
                continue
            row_sum = Pi[s].sum()
            if row_sum <= 0:
                raise ValueError(f"Policy row {s} has zero mass for non-terminal state.")
            # Soft renormalize to guard tiny drift
            Pi[s] /= row_sum

        return Pi

    def set_policy(self, policy=None, uniform_random: bool = False):
        """Update Π cleanly (shape/format checks identical to __init__)."""
        self.Pi = self._build_policy_matrix(policy, uniform_random)

    # ---- evaluation ----------------------------------------------------------

    def run_policy_evaluation(self, epsilon: float = 1e-10, max_iters: int = 1_000_000):
        """
        Iterative policy evaluation with the same reward convention and terminal clamping
        semantics as ValueIteration. Returns V as (S,).
        """
        V = self.state_values  # alias
        V[:] = 0.0  # fresh evaluation
        thresh = self._threshold(epsilon)

        delta = np.inf
        iters = 0

        while delta > thresh and iters < max_iters:
            V_prev = V.copy()
            delta = 0.0

            if self.reward_convention == "entering":
                # Vector used for all (s,a):  T[s,a] @ (r_enter + γ V_prev)
                target = self.r_enter + self.gamma * V_prev  # (S,)
                for s in range(self.S):
                    if self.clamp_terminal_values and s in self.terminals:
                        new_v = self.terminal_value
                    else:
                        exp_sa = self.T[s] @ target     # (A,)
                        new_v = float(np.dot(self.Pi[s], exp_sa))
                    delta = max(delta, abs(new_v - V_prev[s]))
                    V[s] = new_v

            else:  # "on"
                for s in range(self.S):
                    if self.clamp_terminal_values and s in self.terminals:
                        new_v = self.terminal_value
                    else:
                        exp_sa = self.T[s] @ V_prev     # (A,)
                        new_v = float(self.r_on[s] + self.gamma * np.dot(self.Pi[s], exp_sa))
                    delta = max(delta, abs(new_v - V_prev[s]))
                    V[s] = new_v

            iters += 1

        self.state_values = V
        return V

    def get_q_values(self, state_values: np.ndarray | None = None) -> np.ndarray:
        """
        Compute Q(s,a) consistent with the selected reward convention (matches ValueIteration).
        Returns array of shape (S, A).
        """
        if state_values is None:
            state_values = self.state_values
            if np.allclose(state_values, 0.0):
                state_values = self.run_policy_evaluation()

        Q = np.zeros((self.S, self.A), dtype=float)

        if self.reward_convention == "entering":
            target = self.r_enter + self.gamma * state_values  # (S,)
            for s in range(self.S):
                if self.clamp_terminal_values and s in self.terminals:
                    Q[s, :] = 0.0
                else:
                    Q[s, :] = self.T[s] @ target
        else:
            for s in range(self.S):
                if self.clamp_terminal_values and s in self.terminals:
                    Q[s, :] = 0.0
                else:
                    Q[s, :] = self.r_on[s] + self.gamma * (self.T[s] @ state_values)

        return Q