import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gymnasium as gym
import numpy as np
import pytest
import pygame
from env.gridworld_env import NoisyLinearRewardFeaturizedGridWorldEnv



def test_environment_initialization():
    """Test environment initialization with different configurations."""
    # Test default configuration
    env = NoisyLinearRewardFeaturizedGridWorldEnv(gamma=0.9, size=5, noise_prob=0.1, num_features=4, seed=42)
    assert env.size == 5
    assert env.noise_prob == 0.1
    assert env.gamma == 0.9
    assert env.num_features == 4
    assert env.observation_space["agent"].shape == (2,)
    assert env.action_space.n == 4
    assert len(env.terminal_states) == 1  # Default terminal state
    assert env.terminal_states[0] == 24  # Last state in 5x5 grid

    # Test without terminal states
    env = NoisyLinearRewardFeaturizedGridWorldEnv(gamma=0.9, size=5, include_terminal=False, seed=42)
    assert env.terminal_states == []

    # Test with custom terminal states
    env = NoisyLinearRewardFeaturizedGridWorldEnv(gamma=0.9, size=5, terminal_states=[0, 5, 10], seed=42)
    assert env.terminal_states == [0, 5, 10]

    # Test goal-reaching mode
    env = NoisyLinearRewardFeaturizedGridWorldEnv(gamma=0.9, size=5, goal_reaching=True, seed=42)
    assert env.goal_reaching
    assert np.all(env.feature_weights == np.sort(env.feature_weights)[::-1])  # Check descending order

def test_reset():
    """Test the reset functionality."""
    env = NoisyLinearRewardFeaturizedGridWorldEnv(gamma=0.9, size=5, seed=42)
    
    # Test default reset (random start)
    obs = env.reset()
    assert "agent" in obs
    assert obs["agent"].shape == (2,)
    assert np.all(obs["agent"] >= 0) and np.all(obs["agent"] < 5)

    # Test fixed start
    obs = env.reset(fixed_start=True)
    assert np.array_equal(obs["agent"], np.array([0, 0]))  # Should start at (0,0)

def test_step():
    """Test the step functionality."""
    env = NoisyLinearRewardFeaturizedGridWorldEnv(gamma=0.9, size=5, seed=42)
    env.reset(fixed_start=True)  # Start at (0,0)
    
    # Test a step with action=1 (DOWN)
    obs, reward, terminated, truncated = env.step(1)
    assert "agent" in obs
    assert obs["agent"].shape == (2,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert not terminated  # Should not terminate unless in terminal state
    assert not truncated

    # Test step in terminal state
    env = NoisyLinearRewardFeaturizedGridWorldEnv(gamma=0.9, size=5, terminal_states=[0], seed=42)
    env.reset(fixed_start=True)  # Start at (0,0), which is terminal
    obs, reward, terminated, truncated = env.step(0)
    assert terminated  # Should terminate in terminal state

def test_compute_reward():
    """Test reward computation."""
    env = NoisyLinearRewardFeaturizedGridWorldEnv(gamma=0.9, size=5, seed=42)
    state = 10  # Arbitrary state
    reward = env.compute_reward(state)
    assert isinstance(reward, float)
    row, col = divmod(state, env.size)
    features = env.get_cell_features([row, col])
    expected_reward = np.dot(features, env.feature_weights)
    assert np.isclose(reward, expected_reward)

def test_transition_matrix():
    """Test the transition matrix initialization."""
    env = NoisyLinearRewardFeaturizedGridWorldEnv(gamma=0.9, size=5, noise_prob=0.1, seed=42)
    assert env.transitions.shape == (25, 4, 25)  # 5x5 grid, 4 actions, 25 states
    
    # Check probabilities sum to 1 for a non-terminal state
    state = 0  # Top-left corner
    action = 0  # UP
    assert np.isclose(np.sum(env.transitions[state, action, :]), 1.0)
    
    # Check terminal state transitions (self-loop)
    env = NoisyLinearRewardFeaturizedGridWorldEnv(gamma=0.9, size=5, terminal_states=[24], seed=42)
    assert np.all(env.transitions[24, :, 24] == 1.0)  # Self-loop for terminal state

def test_visualize_policy():
    """Test policy visualization."""
    env = NoisyLinearRewardFeaturizedGridWorldEnv(gamma=0.9, size=5, seed=42)
    
    # Create a dummy policy: move right for all states
    policy = [(s, 3) for s in range(env.get_num_states())]
    env._visualize_policy(policy, save_path="test_policy_visualization.png", title="Test Policy")
    
    # Verify that the file was created (basic check)
    import os
    assert os.path.exists("test_policy_visualization.png")
    os.remove("test_policy_visualization.png")  # Clean up

def test_render():
    """Test rendering functionality."""
    env = NoisyLinearRewardFeaturizedGridWorldEnv(gamma=0.9, size=5, render_mode="rgb_array", seed=42)
    env.reset()
    frame = env.render()
    assert isinstance(frame, np.ndarray)  # Should return an RGB array
    assert frame.shape[0] == env.window_size and frame.shape[1] == env.window_size

def test_seed_reproducibility():
    """Test seed reproducibility."""
    env1 = NoisyLinearRewardFeaturizedGridWorldEnv(gamma=0.9, size=5, seed=42)
    env2 = NoisyLinearRewardFeaturizedGridWorldEnv(gamma=0.9, size=5, seed=42)
    
    env1.reset()
    env2.reset()
    obs1 = env1.get_observation()
    obs2 = env2.get_observation()
    assert np.array_equal(obs1["agent"], obs2["agent"])
    
    # Test same sequence of actions
    env1.step(0)
    env2.step(0)
    obs1 = env1.get_observation()
    obs2 = env2.get_observation()
    assert np.array_equal(obs1["agent"], obs2["agent"])

if __name__ == "__main__":
    pytest.main([__file__])