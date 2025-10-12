import gymnasium as gym
from gym import spaces
import numpy as np
import random
import pygame

import gymnasium as gym
from gym import spaces
import numpy as np
import random
import pygame



class GridWorldMDPFromLayoutEnv(gym.Env):
    """
    A custom GridWorld MDP environment created from a predefined layout with noisy transitions and feature-based rewards.
    This class allows defining a grid layout where each cell can have specific features, and the environment can be either
    episodic or continuous based on the presence of terminal states.

    Attributes:
        layout (list of lists): A 2D list representing the grid layout with color names.
        noise_prob (float): Probability of noisy state transitions.
        gamma (float): Discount factor for MDP.
        terminal_states (list): List of terminal state indices.
        custom_feature_weights (list): Custom feature weights for reward calculation.
        render_mode (str): Rendering mode, either 'human' or 'rgb_array'.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, gamma, layout, color_to_feature_map, noise_prob=0.1, terminal_states=None, custom_feature_weights=None, render_mode=None):
        """
        Initializes the GridWorld MDP environment from a predefined layout.

        Args:
            gamma (float): Discount factor for MDP.
            layout (list of lists): Predefined 2D layout of the grid with color names.
            color_to_feature_map (dict): Dictionary mapping color names to feature vectors.
            noise_prob (float): Probability of noisy transitions.
            terminal_states (list): List of terminal state indices.
            custom_feature_weights (list): Custom weights for feature vectors.
            render_mode (str): Rendering mode, either 'human' or 'rgb_array'.
        """
        super(GridWorldMDPFromLayoutEnv, self).__init__()
        
        self.layout = layout
        self.rows = len(layout)
        self.columns = len(layout[0])
        self.gamma = gamma
        self.noise_prob = noise_prob

        # Feature map based on the input
        self.colors_to_features = {color: np.array(features) for color, features in color_to_feature_map.items()}

        # Validate that all layout entries have corresponding feature vectors
        self._validate_layout_colors()

        self.grid_features = np.array(layout)

        # Initialize environment settings
        self.num_states = self.get_num_states()
        self.num_actions = 4  # UP, DOWN, LEFT, RIGHT
        self.num_feat = len(next(iter(self.colors_to_features.values())))
        self.terminal_states = terminal_states if terminal_states else []
        self.feature_weights = custom_feature_weights if custom_feature_weights else np.random.randn(self.num_feat)

        # Initialize transition matrix
        self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.setup_state_transitions()

        # Apply terminal state behavior
        self.apply_terminal_state_behavior()

        # Initialize rendering settings
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]

    def _validate_layout_colors(self):
        """Validates that each color in the layout has a corresponding feature vector."""
        for row in self.layout:
            for color in row:
                assert color in self.colors_to_features, f"Color '{color}' in layout not defined in color_to_feature_map."

    def setup_state_transitions(self):
        """
        Sets up state transition probabilities with noise for UP, DOWN, LEFT, and RIGHT actions.
        """
        RIGHT, UP, LEFT, DOWN = 3, 0, 2, 1
        num_states = self.rows * self.columns

        for s in range(num_states):
            row, col = divmod(s, self.columns)

            # Transitions for UP
            self._add_transition_for_direction(s, row, col, UP, -self.columns)

            # Transitions for DOWN
            self._add_transition_for_direction(s, row, col, DOWN, self.columns)

            # Transitions for LEFT
            self._add_transition_for_direction(s, row, col, LEFT, -1)

            # Transitions for RIGHT
            self._add_transition_for_direction(s, row, col, RIGHT, 1)

    def _add_transition_for_direction(self, state, row, col, direction, step):
        """Helper function to add transitions for a specific direction (UP, DOWN, LEFT, RIGHT)."""
        num_states = self.rows * self.columns
        new_state = state + step

        if 0 <= new_state < num_states:
            self.transitions[state][direction][new_state] = 1.0 - 2 * self.noise_prob

        # Add noise for neighboring states
        if col > 0:
            self.transitions[state][direction][state - 1] = self.noise_prob
        if col < self.columns - 1:
            self.transitions[state][direction][state + 1] = self.noise_prob
        if row > 0:
            self.transitions[state][direction][state - self.columns] = self.noise_prob
        if row < self.rows - 1:
            self.transitions[state][direction][state + self.columns] = self.noise_prob

    def apply_terminal_state_behavior(self):
        """
        Sets the transition behavior for terminal states (self-loops for all terminal states).
        """
        if self.terminal_states:
            for terminal_state in self.terminal_states:
                self.transitions[terminal_state, :, :] = 0
                self.transitions[terminal_state, :, terminal_state] = 1

    def get_num_states(self):
        """Returns the number of states in the grid."""
        return self.rows * self.columns

    def step(self, action):
        """
        Executes one step in the environment and updates the agent's position.

        Args:
            action (int): The action to take (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT).

        Returns:
            observation (dict): The new observation after taking the action.
            reward (float): The reward for taking the action.
            terminated (bool): Whether the episode has terminated (if a terminal state is reached).
            truncated (bool): Whether the episode has been truncated (based on a maximum step count).
        """
        row, col = self._agent_location
        raw_index = row * self.columns + col

        # Sample the next state based on transition probabilities
        next_state = random.choices(range(self.num_states), self.transitions[raw_index][action])[0]
        new_row, new_col = divmod(next_state, self.columns)

        # Compute reward for the new state
        reward = 0 if next_state == raw_index else self.compute_reward(next_state)

        # Update the agent's position
        self._agent_location = np.array([new_row, new_col])

        # Check if we reached a terminal state
        terminated = next_state in self.terminal_states if self.terminal_states else False

        observation = self.get_observation()

        if self.render_mode == "human":
            self.render_grid_frame()

        return observation, reward, terminated, False

    def reset(self, seed=None, fixed_start=False):
        """
        Resets the environment to an initial state.

        Args:
            seed (int): Random seed for reproducibility.
            fixed_start (bool): Whether to reset to a fixed starting state.

        Returns:
            observation (dict): The initial observation of the environment.
        """
        self.step_count = 0
        super().reset(seed=seed)

        # Set the starting position (fixed start or random)
        if fixed_start:
            self._agent_location = np.array(self.start_location)
        else:
            valid_states = [
                (i, j) for i in range(self.size) for j in range(self.size)
                if i * self.size + j not in self.terminal_states
            ]
            chosen_state = random.choice(valid_states)
            self._agent_location = np.array(chosen_state)

        observation = self.get_observation()

        if self.render_mode == "human":
            self.render_grid_frame()

        return observation

    def get_observation(self):
        return {"agent": self._agent_location, "terminal states": self.terminal_states}

    def compute_reward(self, state):
        row, col = divmod(state, self.columns)
        cell_features = self.get_cell_features([row, col])
        return np.dot(cell_features, self.feature_weights)

    def get_cell_features(self, position):
        color = self.grid_features[position[0], position[1]]
        return self.colors_to_features[color]

    def render_grid_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            # Set the window size dynamically based on grid dimensions
            self.window = pygame.display.set_mode((self.columns * self.pix_square_width, self.rows * self.pix_square_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create a canvas with dimensions based on the number of rows and columns
        canvas = pygame.Surface((self.columns * self.pix_square_width, self.rows * self.pix_square_height))
        canvas.fill((255, 255, 255))  # White background

        self._draw_grid(canvas)
        self._draw_agent(canvas)
        self._draw_gridlines(canvas)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def _draw_grid(self, canvas):
        color_map = {
            "blue": (0, 0, 255),       # Blue
            "red": (255, 0, 0),        # Red
            "green": (0, 255, 0),      # Green
            "yellow": (255, 255, 0),   # Yellow
            "purple": (128, 0, 128),   # Purple
            "orange": (255, 165, 0),   # Orange

        }
        for terminal_state in self.terminal_states:
            row, col = divmod(terminal_state, self.columns)
            pygame.draw.rect(canvas, (0, 0, 0), pygame.Rect(self.pix_square_width * col, self.pix_square_height * row, self.pix_square_width, self.pix_square_height))

        for x in range(self.rows):
            for y in range(self.columns):
                if any([x == t_row and y == t_col for t_row, t_col in [divmod(ts, self.columns) for ts in self.terminal_states]]):
                    continue
                color = color_map[self.grid_features[x, y]]
                pygame.draw.rect(canvas, color, pygame.Rect(self.pix_square_width * y, self.pix_square_height * x, self.pix_square_width, self.pix_square_height))

    def _draw_agent(self, canvas):
        pygame.draw.circle(canvas, (42, 42, 42), ((self._agent_location[1] + 0.5) * self.pix_square_width, (self._agent_location[0] + 0.5) * self.pix_square_height), min(self.pix_square_width, self.pix_square_height) / 3)

    def _draw_gridlines(self, canvas):
        for x in range(self.rows + 1):
            pygame.draw.line(canvas, (0, 0, 0), (0, self.pix_square_height * x), (self.columns * self.pix_square_width, self.pix_square_height * x), width=3)
        for y in range(self.columns + 1):
            pygame.draw.line(canvas, (0, 0, 0), (self.pix_square_width * y, 0), (self.pix_square_width * y, self.rows * self.pix_square_height), width=3)

    def set_random_seed(self, seed):
        """Sets the random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)

    def get_discount_factor(self):
        return self.gamma
    
    def get_num_actions(self):
        return self.action_space.n
    
    def get_num_states(self):
        return self.rows * self.columns

    def get_feature_weights(self):
        return self.feature_weights
    
    def set_feature_weights(self, weights):
        """Set and normalize a new weight vector for feature-based rewards."""
        self.feature_weights = weights / np.linalg.norm(weights)


# class NoisyLinearRewardFeaturizedGridWorldEnv(gym.Env):
#     """
#     A custom GridWorld environment with noisy transitions and linear rewards based on feature vectors.
#     Supports both episodic MDPs with terminal states and continuing MDPs without terminal states.
#     """
#     metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

#     def __init__(self, gamma, color_to_feature_map, grid_features, render_mode=None, noise_prob=0.1, terminal_states=None, custom_feature_weights=None, max_steps=None):
#         """
#         Initializes the GridWorld environment.

#         Args:
#             gamma (float): Discount factor for MDP.
#             color_to_feature_map (dict): Mapping of colors to feature vectors.
#             grid_features (list of lists): 2D grid layout with color names.
#             render_mode (str, optional): Rendering mode. Defaults to None.
#             noise_prob (float, optional): Probability of noisy transitions. Defaults to 0.1.
#             terminal_states (list, optional): List of terminal state indices. Defaults to None (no terminal states).
#             custom_feature_weights (list, optional): Weights for feature vectors. Defaults to None (random weights).
#             max_steps (int, optional): Max steps per episode for truncation. Defaults to None (no limit).
#         """
#         super(NoisyLinearRewardFeaturizedGridWorldEnv, self).__init__()
#         self.rows = len(grid_features)
#         self.columns = len(grid_features[0])
#         self.window_size = 512
#         self.noise_prob = noise_prob
#         self.gamma = gamma
#         self.max_steps = max_steps
#         self.step_count = 0

#         self.pix_square_width = self.window_size / self.columns
#         self.pix_square_height = self.window_size / self.rows

#         self.colors_to_features = {color: np.array(features) for color, features in color_to_feature_map.items()}
#         for row in grid_features:
#             for color in row:
#                 assert color in self.colors_to_features, f"Color '{color}' not in color_to_feature_map."
#         self.grid_features = np.array(grid_features)

#         self.observation_space = spaces.Dict({
#             "agent": spaces.Box(0, max(self.rows, self.columns) - 1, shape=(2,), dtype=int),
#         })
#         self.action_space = spaces.Discrete(4)
#         self.num_states = self.rows * self.columns
#         self.num_actions = self.action_space.n
#         self.terminal_states = terminal_states if terminal_states is not None else []
#         self.num_feat = len(next(iter(self.colors_to_features.values())))

#         self.start_location = (0, 0)
#         #if custom_feature_weights:
#         if custom_feature_weights is not None:
#             assert len(custom_feature_weights) == self.num_feat, "Feature weights must match feature vector length."
#             self.feature_weights = np.array(custom_feature_weights)
#         else:
#             self.feature_weights = sorted(np.random.randn(self.num_feat))

#         self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
#         self.initialize_transition_matrix()
#         if self.terminal_states:
#             self._set_terminal_state_transitions()

#         self.render_mode = render_mode
#         self.window = None
#         self.clock = None
#         assert render_mode is None or render_mode in self.metadata["render_modes"]

#     def get_cell_features(self, position):
#         """Returns the feature vector for a given position."""
#         color = self.grid_features[position[0], position[1]]
#         return self.colors_to_features[color]

#     def get_num_states(self):
#          return self.rows * self.columns
    
#     def get_num_actions(self):
#          return self.action_space.n

#     def get_discount_factor(self):
#          return self.gamma

#     def initialize_transition_matrix(self):
#         """Sets up the transition probabilities with noise."""
#         RIGHT, UP, LEFT, DOWN = 3, 0, 2, 1
#         for s in range(self.num_states):
#             row, col = divmod(s, self.columns)

#             # UP
#             if row > 0:
#                 self.transitions[s][UP][s - self.columns] = 1.0 - 2 * self.noise_prob
#             else:
#                 self.transitions[s][UP][s] = 1.0 - 2 * self.noise_prob
#             self.transitions[s][UP][s - 1 if col > 0 else s] += self.noise_prob
#             self.transitions[s][UP][s + 1 if col < self.columns - 1 else s] += self.noise_prob

#             # DOWN
#             if row < self.rows - 1:
#                 self.transitions[s][DOWN][s + self.columns] = 1.0 - 2 * self.noise_prob
#             else:
#                 self.transitions[s][DOWN][s] = 1.0 - 2 * self.noise_prob
#             self.transitions[s][DOWN][s - 1 if col > 0 else s] += self.noise_prob
#             self.transitions[s][DOWN][s + 1 if col < self.columns - 1 else s] += self.noise_prob

#             # LEFT
#             if col > 0:
#                 self.transitions[s][LEFT][s - 1] = 1.0 - 2 * self.noise_prob
#             else:
#                 self.transitions[s][LEFT][s] = 1.0 - 2 * self.noise_prob
#             self.transitions[s][LEFT][s - self.columns if row > 0 else s] += self.noise_prob
#             self.transitions[s][LEFT][s + self.columns if row < self.rows - 1 else s] += self.noise_prob

#             # RIGHT
#             if col < self.columns - 1:
#                 self.transitions[s][RIGHT][s + 1] = 1.0 - 2 * self.noise_prob
#             else:
#                 self.transitions[s][RIGHT][s] = 1.0 - 2 * self.noise_prob
#             self.transitions[s][RIGHT][s - self.columns if row > 0 else s] += self.noise_prob
#             self.transitions[s][RIGHT][s + self.columns if row < self.rows - 1 else s] += self.noise_prob

#     def _set_terminal_state_transitions(self):
#         """Configures terminal states to self-loop with probability 1."""
#         for ts in self.terminal_states:
#             self.transitions[ts, :, :] = 0
#             self.transitions[ts, :, ts] = 1

#     def step(self, action):
#         """Executes one step in the environment."""
#         self.step_count += 1
#         row, col = self._agent_location
#         raw_index = row * self.columns + col

#         next_state = random.choices(range(self.num_states), self.transitions[raw_index][action])[0]
#         new_row, new_col = divmod(next_state, self.columns)

#         reward = 0 if next_state == raw_index else self.compute_reward(next_state)
#         self._agent_location = np.array([new_row, new_col])

#         terminated = next_state in self.terminal_states
#         truncated = self.step_count >= self.max_steps if self.max_steps else False

#         observation = self.get_observation()
#         if self.render_mode == "human":
#             self.render_grid_frame()

#         return observation, reward, terminated, truncated

#     def reset(self, seed=None, fixed_start=False):
#         """Resets the environment to an initial state."""
#         super().reset(seed=seed)
#         self.step_count = 0

#         if fixed_start:
#             self._agent_location = np.array(self.start_location)
#         else:
#             if self.terminal_states:
#                 non_terminal_states = [s for s in range(self.num_states) if s not in self.terminal_states]
#                 chosen_state = random.choice(non_terminal_states)
#             else:
#                 chosen_state = random.randint(0, self.num_states - 1)
#             row, col = divmod(chosen_state, self.columns)
#             self._agent_location = np.array([row, col])

#         observation = self.get_observation()
#         if self.render_mode == "human":
#             self.render_grid_frame()

#         return observation

#     def get_observation(self):
#         """Returns the current observation."""
#         return {"agent": self._agent_location}
    
#     def set_feature_weights(self, weights):
#          """Set and normalize a new weight vector for feature-based rewards."""
#          self.feature_weights = weights / np.linalg.norm(weights)
         
#     def compute_reward(self, state):
#         """Computes the reward for a given state."""
#         row, col = divmod(state, self.columns)
#         cell_features = self.get_cell_features([row, col])
#         return np.dot(cell_features, self.feature_weights)

#     def render_grid_frame(self):
#         """Renders the current frame."""
#         if self.window is None and self.render_mode == "human":
#             pygame.init()
#             pygame.display.init()
#             self.window = pygame.display.set_mode((self.window_size, self.window_size))
#         if self.clock is None and self.render_mode == "human":
#             self.clock = pygame.time.Clock()

#         canvas = pygame.Surface((self.window_size, self.window_size))
#         canvas.fill((255, 255, 255))
#         self._draw_grid(canvas)
#         self._draw_agent(canvas)
#         self._draw_gridlines(canvas)

#         if self.render_mode == "human":
#             self.window.blit(canvas, canvas.get_rect())
#             pygame.event.pump()
#             pygame.display.update()
#             self.clock.tick(self.metadata["render_fps"])

#     def _draw_grid(self, canvas):
#         """Draws the grid with colors and terminal states."""
#         color_map = {
#             "blue": (0, 0, 255), "red": (255, 0, 0), "green": (0, 255, 0),
#             "yellow": (255, 255, 0), "purple": (128, 0, 128), "orange": (255, 165, 0),
#         }
#         for x in range(self.rows):
#             for y in range(self.columns):
#                 state = x * self.columns + y
#                 color = (0, 0, 0) if state in self.terminal_states else color_map[self.grid_features[x, y]]
#                 pygame.draw.rect(canvas, color, pygame.Rect(y * self.pix_square_width, x * self.pix_square_height, self.pix_square_width, self.pix_square_height))

#     def _draw_agent(self, canvas):
#         """Draws the agent on the grid."""
#         pygame.draw.circle(canvas, (42, 42, 42), ((self._agent_location[1] + 0.5) * self.pix_square_width, (self._agent_location[0] + 0.5) * self.pix_square_height), min(self.pix_square_width, self.pix_square_height) / 3)

#     def _draw_gridlines(self, canvas):
#         """Draws gridlines."""
#         for x in range(self.rows + 1):
#             pygame.draw.line(canvas, (0, 0, 0), (0, x * self.pix_square_height), (self.window_size, x * self.pix_square_height), width=3)
#         for y in range(self.columns + 1):
#             pygame.draw.line(canvas, (0, 0, 0), (y * self.pix_square_width, 0), (y * self.pix_square_width, self.window_size), width=3)

#######################################################################################################################################################################################################
###################################################################### The commented env is for MDP with terminal state ###############################################################################
#######################################################################################################################################################################################################
