import gymnasium as gym
from gym import spaces
import numpy as np
import random
#import pygame

import matplotlib.pyplot as plt

class NoisyLinearRewardFeaturizedGridWorldEnv(gym.Env):
    """
    A custom GridWorld environment with noisy transitions and linear rewards based on feature vectors.
    
    The environment supports the following types of MDPs:
    1. GridWorld without terminal state
    2. GridWorld with terminal states that are not goal states
    3. GridWorld with terminal states that are goal states
    
    Attributes:
        size: The size of the grid (e.g., 5x5 grid).
        noise_prob: The probability of noisy transitions.
        gamma: Discount factor for the MDP.
        num_features: Number of features used for reward computation.
        terminal_states: List of terminal states, can be None or manually provided.
        include_terminal: Flag to include terminal states in the environment.
        goal_reaching: Flag to specify if the task is goal-reaching.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, gamma, render_mode=None, size=5, noise_prob=0.1, num_features=4, 
                 terminal_states=None, seed=None, include_terminal=True, goal_reaching=False):
        """
        Initializes the environment with parameters for grid size, noise, reward features, and terminal states.
        
        Args:
            gamma: Discount factor for the MDP.
            render_mode: Rendering mode, can be 'human' or 'rgb_array'.
            size: Size of the grid (e.g., 5 for a 5x5 grid).
            noise_prob: Probability of noise in state transitions.
            num_features: Number of features for reward computation.
            terminal_states: List of terminal states (if any).
            seed: Random seed for reproducibility.
            include_terminal: Flag to include terminal states in the environment.
            goal_reaching: Flag indicating if the task is goal-reaching.
        """
        super(NoisyLinearRewardFeaturizedGridWorldEnv, self).__init__()
        self.size = size
        self.window_size = 512
        self.noise_prob = noise_prob
        self.gamma = gamma
        self.seed = seed
        self.num_features = num_features
        self.goal_reaching = goal_reaching

        self.set_random_seed(self.seed)
        
        # Define the observation and action spaces
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        })
        self.action_space = spaces.Discrete(4)
        
        self.num_states = self.get_num_states()
        self.num_actions = self.get_num_actions()
        self.include_terminal = include_terminal
        
        if terminal_states is None and self.include_terminal:
            self.terminal_states = [self.size * self.size - 1]
        else:
            self.terminal_states = terminal_states or []

        self.start_location = (0, 0)
        self.grid_features = self._initialize_grid_features(size)

        # Linear weight vector for calculating rewards
        self.feature_weights = np.random.randn(self.num_features)
        self.feature_weights /= np.linalg.norm(self.feature_weights)

        if self.goal_reaching:
            # If it's a goal-reaching task, we prioritize terminal states in feature weights
            self.feature_weights = np.sort(self.feature_weights)[::-1]  # Sort in descending order
        else:
            # Normal sorting for non-goal-reaching tasks
            self.feature_weights = np.random.randn(self.num_features)

        self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.initialize_transition_matrix()
        self._set_terminal_state_transitions()

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]

    def initialize_transition_matrix(self):
        """
        Initializes the transition matrix with noisy state transitions.
        """
        RIGHT = 3
        UP = 0
        LEFT = 2
        DOWN = 1
        num_states = self.size * self.size

        for s in range(num_states):
            row, col = divmod(s, self.size)

            # Transitions for UP
            if row > 0:
                self.transitions[s][UP][s - self.size] = 1.0 - (2 * self.noise_prob)
            else:
                self.transitions[s][UP][s] = 1.0 - (2 * self.noise_prob)
            if col > 0:
                self.transitions[s][UP][s - 1] = self.noise_prob
            else:
                self.transitions[s][UP][s] = self.noise_prob
            if col < self.size - 1:
                self.transitions[s][UP][s + 1] = self.noise_prob
            else:
                self.transitions[s][UP][s] = self.noise_prob

            # Handle top-left and top-right corners for UP
            if s < self.size and col == 0:  # Top-left corner
                self.transitions[s][UP][s] = 1.0 - self.noise_prob
            elif s < self.size and col == self.size - 1:  # Top-right corner
                self.transitions[s][UP][s] = 1.0 - self.noise_prob
        
        #for s in range(num_states):
        #    row, col = divmod(s, self.size)
            # Transitions for DOWN
            if row < self.size - 1:
                self.transitions[s][DOWN][s + self.size] = 1.0 - (2 * self.noise_prob)
            else:
                self.transitions[s][DOWN][s] = 1.0 - (2 * self.noise_prob)
            if col > 0:
                self.transitions[s][DOWN][s - 1] = self.noise_prob
            else:
                self.transitions[s][DOWN][s] = self.noise_prob
            if col < self.size - 1:
                self.transitions[s][DOWN][s + 1] = self.noise_prob
            else:
                self.transitions[s][DOWN][s] = self.noise_prob

            # Handle bottom-left and bottom-right corners for DOWN
            if s >= (self.size - 1) * self.size and col == 0:  # Bottom-left corner
                self.transitions[s][DOWN][s] = 1.0 - self.noise_prob
            elif s >= (self.size - 1) * self.size and col == self.size - 1:  # Bottom-right corner
                self.transitions[s][DOWN][s] = 1.0 - self.noise_prob
        
        #for s in range(num_states):
        #    row, col = divmod(s, self.size)
            # Transitions for LEFT
            if col > 0:
                self.transitions[s][LEFT][s - 1] = 1.0 - (2 * self.noise_prob)
            else:
                self.transitions[s][LEFT][s] = 1.0 - (2 * self.noise_prob)
            if row > 0:
                self.transitions[s][LEFT][s - self.size] = self.noise_prob
            else:
                self.transitions[s][LEFT][s] = self.noise_prob
            if row < self.size - 1:
                self.transitions[s][LEFT][s + self.size] = self.noise_prob
            else:
                self.transitions[s][LEFT][s] = self.noise_prob

            # Handle top-left and bottom-left corners for LEFT
            if s < self.size and col == 0:  # Top-left corner
                self.transitions[s][LEFT][s] = 1.0 - self.noise_prob
            elif s >= (self.size - 1) * self.size and col == 0:  # Bottom-left corner
                self.transitions[s][LEFT][s] = 1.0 - self.noise_prob

        #for s in range(num_states):
        #    row, col = divmod(s, self.size)
            # Transitions for RIGHT
            if col < self.size - 1:
                self.transitions[s][RIGHT][s + 1] = 1.0 - (2 * self.noise_prob)
            else:
                self.transitions[s][RIGHT][s] = 1.0 - (2 * self.noise_prob)
            if row > 0:
                self.transitions[s][RIGHT][s - self.size] = self.noise_prob
            else:
                self.transitions[s][RIGHT][s] = self.noise_prob
            if row < self.size - 1:
                self.transitions[s][RIGHT][s + self.size] = self.noise_prob
            else:
                self.transitions[s][RIGHT][s] = self.noise_prob

            # Handle top-right and bottom-right corners for RIGHT
            if s < self.size and col == self.size - 1:  # Top-right corner
                self.transitions[s][RIGHT][s] = 1.0 - self.noise_prob
            elif s >= (self.size - 1) * self.size and col == self.size - 1:  # Bottom-right corner
                self.transitions[s][RIGHT][s] = 1.0 - self.noise_prob

    def _initialize_grid_features(self, size):
            """
            Initializes the grid features with one-hot encoding for the given number of features.
            This method supports different types of MDPs:
            - GridWorld without terminal state: Random one-hot encoding for all states.
            - GridWorld with terminal state (not goal): Random encoding, with terminal states set to a specific feature vector.
            - GridWorld with terminal state (goal): The terminal state gets a specific feature vector, other cells are random.

            Args:
                size: The grid size (e.g., 5 for a 5x5 grid).
            
            Returns:
                grid_features: A 3D array of shape (size, size, num_features) representing the one-hot encoded grid.
            """
            grid_features = np.zeros((size, size, self.num_features))

            if not self.terminal_states:
                # If no terminal states are provided, randomly assign one-hot encoded vectors
                for i in range(size):
                    for j in range(size):
                        feature_idx = np.random.randint(self.num_features)
                        grid_features[i, j, feature_idx] = 1
            else:
                # If terminal states are provided, assign [0, 0, 0, 0, 1] for terminal states
                for i in range(size):
                    for j in range(size):
                        if (i * size + j) in self.terminal_states:
                            grid_features[i, j] = np.array([0] * (self.num_features - 1) + [1])
                        else:
                            feature_idx = np.random.randint(self.num_features)
                            grid_features[i, j, feature_idx] = 1

            return grid_features
    
    def _set_terminal_state_transitions(self):
        """
        Sets the transition behavior for terminal states (self-loops for all terminal states).
        """
        if self.include_terminal:
            for terminal_state in self.terminal_states:
                self.transitions[terminal_state, :, :] = 0
                self.transitions[terminal_state, :, terminal_state] = 1

    def step(self, action):
        """
        Executes the given action and updates the agent's position with noisy transitions.
        """
        row, col = self._agent_location
        raw_index = row * self.size + col

        # Sample the next state based on transition probabilities
        next_state = random.choices(range(self.size * self.size), self.transitions[raw_index][action])[0]
        new_row, new_col = divmod(next_state, self.size)

        # Check if the agent stayed in the same cell
        if next_state == raw_index:
            reward = 0  # Assign a reward of 0 if the agent stays in the same state
        else:
            reward = self.compute_reward(next_state)

        # Update the agent's position
        self._agent_location = np.array([new_row, new_col])

        # Check if we reached a terminal state
        terminated = next_state in self.terminal_states if self.include_terminal else False

        observation = self.get_observation()

        if self.render_mode == "human":
            self.render_grid_frame()

        return observation, reward, terminated, False

    def reset(self, seed=None, fixed_start=False):
        """
        Resets the environment to the initial state based on a distribution and returns the initial observation.
        """
        if fixed_start:
            self._agent_location = self.start_location
        else:
            # Sample from a start state distribution: uniform over all non-terminal states
            valid_states = [
                (i, j)
                for i in range(self.size)
                for j in range(self.size)
                if i * self.size + j not in self.terminal_states or not self.include_terminal
            ]
            chosen = random.choice(valid_states)
            self._agent_location = np.array(chosen)

        observation = self.get_observation()

        if self.render_mode == "human":
            self.render_grid_frame()

        return observation

    def get_observation(self):
        """Returns the current observation (agent and target positions)."""
        return {"agent": self._agent_location, "terminal states": self.terminal_states}

    def compute_reward(self, state):
        """
        Computes the reward for a given state based on its feature vector and the feature weights.
        """
        row, col = divmod(state, self.size)
        cell_features = self.get_cell_features([row, col])
        return np.dot(cell_features, self.feature_weights)

    def get_cell_features(self, position):
        """
        Returns the feature vector of the grid cell at the given position.
        """
        color = self.grid_features[position[0], position[1]]
        return self.colors_to_features[color]

    def set_random_seed(self, seed):

        """Sets the random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)

    def get_discount_factor(self):
        return self.gamma
    
    def get_num_actions(self):
        return self.action_space.n
    
    def get_num_states(self):
        return self.size * self.size

    def get_feature_weights(self):
        return self.feature_weights
    
    def set_feature_weights(self, weights):
        """Set and normalize a new weight vector for feature-based rewards."""
        self.feature_weights = weights / np.linalg.norm(weights)
    
    def get_cell_features(self, position):
        """
        Returns the feature vector of the grid cell at the given position.
        Now directly returns the one-hot feature vector from grid_features.
        """
        return self.grid_features[position[0], position[1]]


