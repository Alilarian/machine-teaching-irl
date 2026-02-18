# import gymnasium as gym
# from gym import spaces
# import numpy as np
# import random
# from agent.q_learning_agent import ValueIteration
# #import pygame

# # Action indices (match docstring: 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT)
# UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# class GridWorldMDPFromLayoutEnv(gym.Env):
#     """
#     A custom GridWorld MDP environment created from a predefined layout with noisy transitions and feature-based rewards.
#     The layout stores color names; we map those to numeric feature vectors for learning.
#     """
#     metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

#     def __init__(
#         self,
#         gamma,
#         layout,
#         color_to_feature_map,
#         noise_prob=0,
#         terminal_states=None,
#         custom_feature_weights=None,
#         render_mode=None
#     ):
#         super(GridWorldMDPFromLayoutEnv, self).__init__()

#         # ----------------- geometry & basics -----------------
#         self.layout = layout
#         self.rows = len(layout)
#         self.columns = len(layout[0])
#         self.size = self.columns                  # IMPORTANT: width used by SF's divmod(s, env.size)
#         self.gamma = float(gamma)
#         self.noise_prob = float(noise_prob)

#         # Map color -> feature vector, validate dimensions, set num_features
#         self.colors_to_features = {
#             color: np.array(features, dtype=float)
#             for color, features in color_to_feature_map.items()
#         }
#         self._validate_layout_colors()
#         feat_lens = {v.shape[0] for v in self.colors_to_features.values()}
#         assert len(feat_lens) == 1, "All feature vectors in color_to_feature_map must have the same length."
#         self.num_features = feat_lens.pop()

#         # Keep colors separately for rendering; build numeric feature grid for learning
#         self.grid_colors = np.array(layout)  # (rows, cols) of color names
#         self.grid_features = np.zeros((self.rows, self.columns, self.num_features), dtype=float)
#         for r in range(self.rows):
#             for c in range(self.columns):
#                 color = self.grid_colors[r, c]
#                 self.grid_features[r, c] = self.colors_to_features[color]

        
        
#         # Gym spaces
#         self.action_space = spaces.Discrete(4)

#         # ----------------- state/action/terminals -----------------
#         self.num_states = self.rows * self.columns
#         self.num_actions = 4
#         self.terminal_states = list(terminal_states) if terminal_states else []
#         self.include_terminal = bool(self.terminal_states)  # used by SF to decide zeroing terminal features
#         self.start_location = (0, 0)


#         # ----------------- flat state feature table (CRITICAL FOR SPEED) -----------------
#         self.state_features = self.grid_features.reshape(
#             self.get_num_states(), self.num_features
#         ).astype(np.float32)

#         # Feature weights (normalized)
#         if custom_feature_weights is None:
#             w = np.random.randn(self.num_features)
#         else:
#             w = np.array(custom_feature_weights, dtype=float)
#             assert w.shape[0] == self.num_features, (
#                 f"custom_feature_weights length {w.shape[0]} != num_features {self.num_features}"
#             )
#         self.feature_weights = w / (np.linalg.norm(w) + 1e-12)

#         # ----------------- transitions -----------------
#         self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=float)
#         self.setup_state_transitions()
#         self.apply_terminal_state_behavior()

#         # ----------------- rendering -----------------
#         self.render_mode = render_mode
#         self.window = None
#         self.clock = None
#         self.pix_square_width = getattr(self, "pix_square_width", 40)
#         self.pix_square_height = getattr(self, "pix_square_height", 40)
#         assert render_mode is None or render_mode in self.metadata["render_modes"]

#     # ----------------- helpers -----------------

#     def _validate_layout_colors(self):
#         """Validates that each color in the layout has a corresponding feature vector."""
#         for row in self.layout:
#             for color in row:
#                 assert color in self.colors_to_features, (
#                     f"Color '{color}' in layout not defined in color_to_feature_map."
#                 )

#     def setup_state_transitions(self):
#         """
#         Build transitions with 'slip' to perpendicular neighbors:
#         - Intended move prob = 1 - 2*noise_prob
#         - Two perpendicular slips prob = noise_prob each
#         - Off-grid moves become self-loops
#         Ensures each (s,a) row sums to 1.
#         """
#         S = self.num_states
#         self.transitions[:] = 0.0

#         def idx(r, c): return r * self.columns + c
#         def on_grid(r, c): return 0 <= r < self.rows and 0 <= c < self.columns

#         deltas = {
#             UP:    (-1, 0),
#             DOWN:  ( 1, 0),
#             LEFT:  ( 0,-1),
#             RIGHT: ( 0, 1),
#         }
#         slips = {
#             UP:    [(0,-1), (0, 1)],
#             DOWN:  [(0,-1), (0, 1)],
#             LEFT:  [(-1,0), (1, 0)],
#             RIGHT: [(-1,0), (1, 0)],
#         }

#         p_main = 1.0 - 2.0 * self.noise_prob
#         p_slip = self.noise_prob

#         for s in range(S):
#             r, c = divmod(s, self.columns)
#             for a in (UP, DOWN, LEFT, RIGHT):
#                 dr, dc = deltas[a]
#                 r_int, c_int = r + dr, c + dc
#                 s_int = idx(r_int, c_int) if on_grid(r_int, c_int) else s

#                 (dr1, dc1), (dr2, dc2) = slips[a]
#                 r1, c1 = r + dr1, c + dc1
#                 r2, c2 = r + dr2, c + dc2
#                 s1 = idx(r1, c1) if on_grid(r1, c1) else s
#                 s2 = idx(r2, c2) if on_grid(r2, c2) else s

#                 self.transitions[s, a, s_int] += p_main
#                 self.transitions[s, a, s1]   += p_slip
#                 self.transitions[s, a, s2]   += p_slip

#         # Renormalize each (s,a,:) to exactly 1
#         row_sums = self.transitions.sum(axis=2, keepdims=True)
#         row_sums[row_sums == 0.0] = 1.0
#         self.transitions /= row_sums

#     def apply_terminal_state_behavior(self):
#         """Terminal states are absorbing."""
#         if self.terminal_states:
#             for t in self.terminal_states:
#                 self.transitions[t, :, :] = 0.0
#                 self.transitions[t, :, t] = 1.0

#     def get_num_states(self):
#         return self.num_states

#     def get_num_actions(self):
#         return self.action_space.n

#     def get_discount_factor(self):
#         return self.gamma

#     def step(self, action):
#         row, col = self._agent_location
#         raw_index = row * self.columns + col

#         probs = self.transitions[raw_index, action]
#         p = np.clip(probs, 0.0, None)
#         s = p.sum()
#         p = (np.ones_like(p) / len(p)) if s <= 0 else (p / s).tolist()

#         next_state = random.choices(range(self.num_states), weights=p, k=1)[0]
#         new_row, new_col = divmod(next_state, self.columns)

#         # entering-state reward; 0 if self-loop
#         reward = 0.0 if next_state == raw_index else float(self.compute_reward(next_state))

#         self._agent_location = np.array([new_row, new_col], dtype=np.int32)
#         terminated = next_state in self.terminal_states if self.terminal_states else False

#         obs = self.get_observation()
#         if self.render_mode == "human":
#             self.render_grid_frame()
#         return obs, reward, terminated, False

#     def reset(self, seed=None, fixed_start=False):
#         self.step_count = 0
#         super().reset(seed=seed)

#         if fixed_start:
#             self._agent_location = np.array(self.start_location, dtype=np.int32)
#         else:
#             valid = [
#                 (i, j)
#                 for i in range(self.rows)
#                 for j in range(self.columns)
#                 if (i * self.columns + j) not in self.terminal_states
#             ]
#             self._agent_location = np.array(random.choice(valid), dtype=np.int32)

#         obs = self.get_observation()
#         if self.render_mode == "human":
#             self.render_grid_frame()
#         return obs

#     def get_observation(self):
#         return {"agent": self._agent_location, "terminal states": self.terminal_states}

#     def compute_reward(self, state):
#         r, c = divmod(state, self.columns)
#         f = self.get_cell_features([r, c])
#         return float(np.dot(f, self.feature_weights))

#     def get_state_features(self, s):
#         """
#         Returns the feature vector φ(s) for a given integer state index.
#         Equivalent to get_cell_features((r,c)) but takes a flat state id.
#         """
#         r, c = divmod(int(s), self.columns)
#         return self.grid_features[r, c]

#     def get_cell_features(self, position):
#         # Now returns numeric feature vector directly (matches SF expectation on env.grid_features)
#         return self.grid_features[position[0], position[1]]

#     def set_random_seed(self, seed):
#         np.random.seed(seed)
#         random.seed(seed)

#     def get_feature_weights(self):
#         return self.feature_weights

#     def set_feature_weights(self, weights):
#         w = np.array(weights, dtype=float)
#         self.feature_weights = w / (np.linalg.norm(w))

#     def print_mdp_info(self):
#         """
#         Prints a summary of the MDP structure.
#         """
#         print("\n========== GridWorld MDP Info ==========")
#         print(f"Grid size           : {self.rows} x {self.columns}")
#         print(f"Num states          : {self.num_states}")
#         print(f"Num actions         : {self.num_actions}")
#         print(f"Discount factor γ   : {self.gamma}")
#         print(f"Noise probability  : {self.noise_prob}")
#         print(f"Num features        : {self.num_features}")
#         print(f"Terminal states     : {self.terminal_states}")
#         print(f"Start location      : {self.start_location}")

#         print("\nFeature weights (normalized):")
#         print(np.round(self.feature_weights, 4))

#         print("\nLayout (colors):")
#         for r in range(self.rows):
#             print(" ".join(self.grid_colors[r]))

#         print("\nSample transition check (state 0):")
#         for a in range(self.num_actions):
#             probs = self.transitions[0, a]
#             nz = [(s, round(p, 3)) for s, p in enumerate(probs) if p > 0]
#             print(f"  Action {a}: {nz}")

#         print("=======================================\n")

#     def print_optimal_policy(self, epsilon=1e-6, max_iter=10_000):
#         """
#         Computes and prints the optimal policy using value iteration.
#         """
#         S, A = self.num_states, self.num_actions
#         V = np.zeros(S)

#         # --- Value Iteration ---
#         for _ in range(max_iter):
#             delta = 0.0
#             for s in range(S):
#                 if s in self.terminal_states:
#                     continue

#                 r = self.compute_reward(s)
#                 Q_sa = np.zeros(A)
#                 for a in range(A):
#                     Q_sa[a] = r + self.gamma * np.dot(
#                         self.transitions[s, a], V
#                     )

#                 v_new = np.max(Q_sa)
#                 delta = max(delta, abs(V[s] - v_new))
#                 V[s] = v_new

#             if delta < epsilon:
#                 break

#         # --- Extract policy ---
#         policy = np.full(S, -1, dtype=int)
#         for s in range(S):
#             if s in self.terminal_states:
#                 continue
#             Q_sa = np.zeros(A)
#             for a in range(A):
#                 Q_sa[a] = self.compute_reward(s) + self.gamma * np.dot(
#                     self.transitions[s, a], V
#                 )
#             policy[s] = int(np.argmax(Q_sa))

#         # --- Pretty print ---
#         arrow = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→"}

#         print("\n========== Optimal Policy ==========")
#         for r in range(self.rows):
#             row_syms = []
#             for c in range(self.columns):
#                 s = r * self.columns + c
#                 if s in self.terminal_states:
#                     row_syms.append(" T ")
#                 else:
#                     row_syms.append(f" {arrow[policy[s]]} ")
#             print("".join(row_syms))
#         print("===================================\n")

#         return policy, V


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


# Action indices (0:UP, 1:DOWN, 2:LEFT, 3:RIGHT)
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


class GridWorldMDPFromLayoutEnv(gym.Env):
    """
    GridWorld MDP from layout with stochastic transitions and linear feature-based rewards.
    All names and core functionality preserved.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        gamma,
        layout,
        color_to_feature_map,
        noise_prob=0.0,
        terminal_states=None,
        custom_feature_weights=None,
        render_mode=None
    ):
        super().__init__()
        self.render_mode = render_mode

        # Grid geometry
        self.layout = layout
        self.rows = len(layout)
        self.columns = len(layout[0])
        self.size = self.columns               # ← important for some external code
        self.num_states = self.rows * self.columns
        self.num_actions = 4

        self.gamma = float(gamma)
        self.noise_prob = float(noise_prob)
        self.terminal_states = list(terminal_states) if terminal_states else []

        # Feature setup
        self.colors_to_features = {
            color: np.array(features, dtype=float)
            for color, features in color_to_feature_map.items()
        }
        self._validate_layout_colors()

        # All features must have same dimension
        feat_dims = {v.shape[0] for v in self.colors_to_features.values()}
        assert len(feat_dims) == 1, "All feature vectors must have same length"
        self.num_features = feat_dims.pop()

        # Build color & feature grids
        self.grid_colors = np.array(layout, dtype=object)
        self.grid_features = np.zeros((self.rows, self.columns, self.num_features), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.columns):
                self.grid_features[r, c] = self.colors_to_features[self.grid_colors[r, c]]

        # Flat feature table (used by successor features etc.)
        self.state_features = self.grid_features.reshape(self.num_states, self.num_features)

        # Reward weights (normalized)
        if custom_feature_weights is None:
            w = np.random.randn(self.num_features)
        else:
            w = np.array(custom_feature_weights, dtype=float)
            assert len(w) == self.num_features
        self.feature_weights = w / (np.linalg.norm(w) + 1e-12)

        # Gym spaces
        self.action_space = spaces.Discrete(4)

        # Agent position (will be set in reset)
        self._agent_location = None

        # Transition matrix: P(s'|s,a)
        self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.float32)
        self._build_transitions()
        self._make_terminals_absorbing()

        # Rendering
        self.window = None
        self.clock = None
        self.pix_square_width = 40
        self.pix_square_height = 40


    def _validate_layout_colors(self):
        for row in self.layout:
            for color in row:
                if color not in self.colors_to_features:
                    raise ValueError(f"Color '{color}' in layout not defined in color_to_feature_map")


    def _build_transitions(self):
        """Builds stochastic transitions with perpendicular slip."""
        p_main = 1.0 - 2.0 * self.noise_prob
        p_slip = self.noise_prob

        deltas_main = [(-1, 0), (1, 0), (0, -1), (0, 1)]          # UP DOWN LEFT RIGHT
        deltas_slip = [[(0,-1),(0,1)], [(0,-1),(0,1)], [(-1,0),(1,0)], [(-1,0),(1,0)]]

        def to_state(r, c):
            if 0 <= r < self.rows and 0 <= c < self.columns:
                return r * self.columns + c
            return None  # means stay

        for s in range(self.num_states):
            r, c = divmod(s, self.columns)

            for a in range(4):
                # Intended direction
                dr, dc = deltas_main[a]
                nr, nc = r + dr, c + dc
                next_s = to_state(nr, nc)

                # Slip directions
                slip1, slip2 = deltas_slip[a]
                sr1, sc1 = r + slip1[0], c + slip1[1]
                sr2, sc2 = r + slip2[0], c + slip2[1]
                slip_s1 = to_state(sr1, sc1)
                slip_s2 = to_state(sr2, sc2)

                # Assign probabilities
                if next_s is not None:
                    self.transitions[s, a, next_s] += p_main
                else:
                    self.transitions[s, a, s] += p_main

                if slip_s1 is not None:
                    self.transitions[s, a, slip_s1] += p_slip
                else:
                    self.transitions[s, a, s] += p_slip

                if slip_s2 is not None:
                    self.transitions[s, a, slip_s2] += p_slip
                else:
                    self.transitions[s, a, s] += p_slip

        # Final normalization (usually very close to 1 already)
        sums = self.transitions.sum(axis=2, keepdims=True)
        self.transitions /= np.maximum(sums, 1e-10)


    def _make_terminals_absorbing(self):
        if not self.terminal_states:
            return
        for t in self.terminal_states:
            self.transitions[t] = 0.0
            self.transitions[t, :, t] = 1.0


    def reset(self, seed=None, fixed_start=False):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if fixed_start:
            loc = self.start_location = (0, 0)
        else:
            valid = [
                (i, j) for i in range(self.rows) for j in range(self.columns)
                if (i * self.columns + j) not in self.terminal_states
            ]
            loc = random.choice(valid)
            self.start_location = loc

        self._agent_location = np.array(loc, dtype=int)
        return self.get_observation(), {}


    def step(self, action):
        current_flat = self._agent_location[0] * self.columns + self._agent_location[1]
        probs = self.transitions[current_flat, action]

        next_flat = np.random.choice(self.num_states, p=probs)
        next_r, next_c = divmod(next_flat, self.columns)

        # Reward = dot product only when actually moving to new state
        reward = 0.0 if next_flat == current_flat else np.dot(
            self.grid_features[next_r, next_c], self.feature_weights
        )

        self._agent_location = np.array([next_r, next_c])

        terminated = next_flat in self.terminal_states
        truncated = False

        if self.render_mode == "human":
            self.render_grid_frame()   # assuming this method exists elsewhere

        return self.get_observation(), reward, terminated, truncated, {}


    def get_observation(self):
        return {
            "agent": self._agent_location.copy(),
            "terminal states": self.terminal_states
        }


    def compute_reward(self, state):
        r, c = divmod(state, self.columns)
        return float(np.dot(self.grid_features[r, c], self.feature_weights))


    def get_state_features(self, s):
        r, c = divmod(int(s), self.columns)
        return self.grid_features[r, c]


    def get_cell_features(self, position):
        r, c = position
        return self.grid_features[r, c]


    def get_feature_weights(self):
        return self.feature_weights.copy()


    def set_feature_weights(self, weights):
        w = np.asarray(weights, dtype=float)
        assert len(w) == self.num_features
        self.feature_weights = w / (np.linalg.norm(w) + 1e-12)


    def get_num_states(self):
        return self.num_states


    def get_num_actions(self):
        return self.num_actions


    def get_discount_factor(self):
        return self.gamma


    def set_random_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)


    # ────────────────────────────────────────────────
    #  The two printing/debug methods (unchanged logic)
    # ────────────────────────────────────────────────

    def print_mdp_info(self):
        print("\n========== GridWorld MDP Info ==========")
        print(f"Grid size     : {self.rows} × {self.columns}")
        print(f"States        : {self.num_states}")
        print(f"Actions       : {self.num_actions}")
        print(f"γ             : {self.gamma:.3f}")
        print(f"Noise         : {self.noise_prob:.3f}")
        print(f"Features      : {self.num_features}")
        print(f"Terminals     : {self.terminal_states}")
        print(f"Start         : {self.start_location}")
        print("\nFeature weights (normalized):")
        print(np.round(self.feature_weights, 4))
        print("\nLayout:")
        for row in self.grid_colors:
            print(" ".join(f"{c:>4}" for c in row))
        print("========================================\n")


    def print_optimal_policy(self, epsilon=1e-6, max_iter=10_000):
        V = np.zeros(self.num_states)

        for _ in range(max_iter):
            delta = 0.0
            for s in range(self.num_states):
                if s in self.terminal_states:
                    continue
                Q = np.array([
                    self.compute_reward(s) + self.gamma * np.dot(self.transitions[s, a], V)
                    for a in range(self.num_actions)
                ])
                v_new = Q.max()
                delta = max(delta, abs(V[s] - v_new))
                V[s] = v_new
            if delta < epsilon:
                break

        # Policy extraction
        policy = np.full(self.num_states, -1, dtype=int)
        for s in range(self.num_states):
            if s in self.terminal_states:
                continue
            Q = np.array([
                self.compute_reward(s) + self.gamma * np.dot(self.transitions[s, a], V)
                for a in range(self.num_actions)
            ])
            policy[s] = Q.argmax()

        # Pretty print
        arrow_map = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→"}
        print("\n========== Optimal Policy ==========")
        for r in range(self.rows):
            row = []
            for c in range(self.columns):
                s = r * self.columns + c
                if s in self.terminal_states:
                    row.append(" T ")
                else:
                    row.append(f" {arrow_map[policy[s]]} ")
            print("".join(row))
        print("====================================\n")

        return policy, V