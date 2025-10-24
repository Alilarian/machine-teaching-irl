import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
#import pygame
from mdp.gridworld_env_layout import GridWorldMDPFromLayoutEnv  # Adjust path based on your structure

# Sample layout and feature map for testing
sample_layout = [
    ["red", "blue", "green"],
    ["yellow", "purple", "orange"],
    ["blue", "red", "yellow"]
]
sample_color_to_feature_map = {
    "red": [1, 0, 0],
    "blue": [0, 1, 0],
    "green": [0, 0, 1],
    "yellow": [1, 1, 0],
    "purple": [0, 1, 1],
    "orange": [1, 0, 1]
}

def test_environment_initialization():
    """Test environment initialization with a sample layout."""
    env = GridWorldMDPFromLayoutEnv(
        gamma=0.9,
        layout=sample_layout,
        color_to_feature_map=sample_color_to_feature_map,
        noise_prob=0.1,
        terminal_states=[0, 5],  # Terminal states at (0,0) and (1,2)
        render_mode="rgb_array"
    )
    assert env.rows == 3
    assert env.columns == 3
    assert env.num_states == 9
    assert env.num_actions == 4
    assert env.terminal_states == [0, 5]
    assert len(env.feature_weights) == 3
    assert env.transitions.shape == (9, 4, 9)
    assert env.render_mode == "rgb_array"

def test_validate_layout_colors():
    """Test that _validate_layout_colors raises an assertion error for invalid colors."""
    invalid_layout = [["red", "invalid_color", "green"]]
    with pytest.raises(AssertionError):
        GridWorldMDPFromLayoutEnv(
            gamma=0.9,
            layout=invalid_layout,
            color_to_feature_map=sample_color_to_feature_map
        )

def test_step():
    """Test the step functionality with noisy transitions."""
    env = GridWorldMDPFromLayoutEnv(
        gamma=0.9,
        layout=sample_layout,
        color_to_feature_map=sample_color_to_feature_map,
        noise_prob=0.1,
        terminal_states=[5],
        render_mode=None
    )
    env._agent_location = np.array([0, 0])  # Start at (0,0)
    obs, reward, terminated, truncated = env.step(1)  # Move DOWN
    assert "agent" in obs
    assert obs["agent"].shape == (2,)
    assert isinstance(reward, float)
    assert terminated == (obs["agent"][0] * env.columns + obs["agent"][1] == 5)
    assert not truncated

def test_reset():
    """Test the reset functionality with fixed and random starts."""
    env = GridWorldMDPFromLayoutEnv(
        gamma=0.9,
        layout=sample_layout,
        color_to_feature_map=sample_color_to_feature_map,
        terminal_states=[5]
    )
    # Test fixed start
    obs = env.reset(fixed_start=True)
    assert np.array_equal(obs["agent"], np.array([0, 0]))  # Default start_location

    # Test random start (ensure not in terminal state)
    obs = env.reset(fixed_start=False)
    state = obs["agent"][0] * env.columns + obs["agent"][1]
    assert state not in env.terminal_states

def test_compute_reward():
    """Test reward computation based on feature weights."""
    env = GridWorldMDPFromLayoutEnv(
        gamma=0.9,
        layout=sample_layout,
        color_to_feature_map=sample_color_to_feature_map,
        custom_feature_weights=[1.0, 2.0, 3.0]  # Custom weights for testing
    )
    reward = env.compute_reward(0)  # State (0,0) is "red" -> [1, 0, 0]
    expected_reward = np.dot([1, 0, 0], [1.0, 2.0, 3.0])
    assert np.isclose(reward, expected_reward)

def test_transition_matrix():
    """Test the transition matrix setup with noise."""
    env = GridWorldMDPFromLayoutEnv(
        gamma=0.9,
        layout=sample_layout,
        color_to_feature_map=sample_color_to_feature_map,
        noise_prob=0.1
    )
    state = 0  # (0,0)
    action = 1  # DOWN
    transition_probs = env.transitions[state, action, :]
    assert np.sum(transition_probs) == 1.0  # Probabilities should sum to 1
    assert transition_probs[3] == 0.8  # Expected transition to (1,0) with 1 - 2*0.1
    assert np.count_nonzero(transition_probs > 0) > 1  # Noise adds other transitions

def test_render():
    """Test rendering functionality."""
    env = GridWorldMDPFromLayoutEnv(
        gamma=0.9,
        layout=sample_layout,
        color_to_feature_map=sample_color_to_feature_map,
        render_mode="rgb_array"
    )
    env._agent_location = np.array([0, 0])  # Set agent position
    frame = env.render_grid_frame()
    #assert isinstance(frame, pygame.Surface)  # Should return a Surface object
    assert frame.get_width() == env.columns * 50  # Assuming pix_square_width = 50 (default Pygame scaling)
    assert frame.get_height() == env.rows * 50  # Assuming pix_square_height = 50

def test_seeding():
    """Test seeding for reproducibility."""
    env1 = GridWorldMDPFromLayoutEnv(
        gamma=0.9,
        layout=sample_layout,
        color_to_feature_map=sample_color_to_feature_map,
        seed=42
    )
    env2 = GridWorldMDPFromLayoutEnv(
        gamma=0.9,
        layout=sample_layout,
        color_to_feature_map=sample_color_to_feature_map,
        seed=42
    )
    obs1 = env1.reset()
    obs2 = env2.reset()
    assert np.array_equal(env1.feature_weights, env2.feature_weights)
    assert np.array_equal(obs1["agent"], obs2["agent"])

if __name__ == "__main__":
    pytest.main([__file__])