
import numpy as np
from scipy.special import logsumexp
import random
## New way of generating improvement feedback where the improved and the original trajectory have same start state, while both of them are suboptimal.
## It differs from simulate_improvement_v2, where it has one correction point and it takes optimal actions from that point afterward

def generate_random_trajectory_same_start(env, start_state, trajectory_length):
    """
    Generate a random trajectory starting from the specified state and with the specified length.

    Args:
        env: The environment object.
        start_state: The state from which the trajectory should start.
        trajectory_length: The length of the trajectory to generate.

    Returns:
        list of (state_index, action) tuples representing the trajectory.
    """
    trajectory = []
    state = start_state
    terminal_states = env.terminal_states  # List of terminal states as indices

    for step in range(trajectory_length):
        if state in terminal_states:
            trajectory.append((state, None))  # Append terminal state with None action
            break  # Stop if terminal state is reached

        # Choose a random action uniformly
        action = np.random.choice(env.num_actions)

        # Sample the next state based on transition probabilities
        next_state = np.random.choice(env.num_states, p=env.transitions[state][action])

        trajectory.append((state, action))
        state = next_state

    return trajectory

# def generate_random_trajectory(env, max_horizon=25, fixed_start=False, row=3, col=2):
def generate_random_trajectory(env, max_horizon=25, fixed_start=False):
    """
    Generate a random trajectory of fixed length (max_horizon + 1) using random actions.
    The state is stored as an integer index (raw_index) instead of (row, col).

    Args:
        env: The GridWorld environment.
        max_horizon (int): Maximum length of the trajectory.

    Returns:
        list of (state_index, action) tuples.
    """

    trajectory = []
    obsv = env.reset(fixed_start=fixed_start)  # Reset environment and get initial observation.
    agent_position = obsv["agent"]  # [row, col]
    terminal_states = obsv["terminal states"]  # List of terminal states as indices

    # Compute the raw index (integer) for the initial state.
    try:
        state = agent_position[0] * env.columns + agent_position[1]
    except:
        state = agent_position[0] * env.size + agent_position[1]

    for step in range(max_horizon):
        # Append the current state and chosen action
        if state in terminal_states:
            # Append terminal state with None action to indicate stopping
            trajectory.append((state, None))
            break  # Stop generating the trajectory if a terminal state is reached.

        # Choose a random action uniformly.
        #np.random.seed(seed) => wrong place to fix seed, because it always pick the same action in the trajectory
        action = np.random.choice(env.num_actions)

        # Sample the next state based on transition probabilities.

        next_state = np.random.choice(env.num_states, p=env.transitions[state][action])

        # Append (current state, chosen action) to the trajectory.
        trajectory.append((state, action))

        # Update state (now directly using raw index).
        state = next_state

    return trajectory

def generate_suboptimal_demonstrations(demonstrations, percent):
    """
    Generate suboptimal demonstrations by changing the action in a specified percentage of demonstrations.

    Args:
        demonstrations: List of tuples, where each tuple is (state, action) and action is in {0, 1, 2, 3}.
        percent: Float between 0 and 1, indicating the fraction of demonstrations to modify.

    Returns:
        List of demonstrations with the specified percentage having a different action.
    """
    # Check if percent is between 0 and 1
    if not 0 <= percent <= 1:
        raise ValueError("Percent must be between 0 and 1")

    # Valid actions
    valid_actions = [0, 1, 2, 3]

    # Calculate number of demonstrations to modify
    num_to_modify = int(len(demonstrations) * percent)

    suboptimal_demonstrations = []

    # Process the first 'percent' of demonstrations
    for i, demo in enumerate(demonstrations):
        state, action = demo
        if i < num_to_modify:
            # Create list of possible actions excluding the current action
            possible_actions = [a for a in valid_actions if a != action]
            # Choose a random different action
            new_action = np.random.choice(possible_actions)
            # Append modified demonstration
            suboptimal_demonstrations.append((state, new_action))
        else:
            # Keep original demonstration
            suboptimal_demonstrations.append(demo)

    return suboptimal_demonstrations

def simulate_improvement_feedback_v4(env, pre_generated_trajectories, num_random_trajs=25):
    """
    For each pre-generated trajectory, generate a set of new random trajectories starting from the same state,
    and compare their rewards. Select the trajectory with the highest reward as the improved trajectory.

    Args:
        env: The environment object.
        pre_generated_trajectories: List of pre-generated trajectories.
        num_random_trajs: The number of random trajectories to generate for comparison.

    Returns:
        List of tuples of paired trajectories: (improved_trajectory, original_trajectory)
    """
    def evaluate_trajectory(traj):
        """Compute total reward of a trajectory."""
        return sum(env.compute_reward(s) for s, _ in traj)

    paired_trajectories = []

    for traj in pre_generated_trajectories:
        # Extract the start state of the trajectory (start state is the first state of the trajectory)
        start_state = traj[0][0]

        # Generate a set of random trajectories from the same start state and same length
        random_trajectories = [generate_random_trajectory_same_start(env, start_state, len(traj)) for _ in range(num_random_trajs)]

        # Compare the rewards of the original and all generated random trajectories
        original_return = evaluate_trajectory(traj)

        # Select the random trajectory with the highest reward
        max_random_return = float('-inf')
        best_random_traj = None

        for new_traj in random_trajectories:
            new_return = evaluate_trajectory(new_traj)
            if new_return > max_random_return:
                max_random_return = new_return
                best_random_traj = new_traj

        # Select the trajectory with the highest reward
        if max_random_return > original_return:
            paired_trajectories.append((best_random_traj, traj))  # The best random trajectory is the improvement
        else:
            paired_trajectories.append((traj, best_random_traj))  # The original trajectory is better or equal

    return paired_trajectories

def simulate_human_estop_v2(env, full_trajectory, beta=2.0, gamma=1.0):
    """
    Simulates E-stop data based on the provided likelihood model.

    Args:
        env (NoisyLinearRewardFeaturizedGridWorldEnv): The environment instance.
        full_trajectory (list): A full-length trajectory as [(state, action), ...].
        beta (float): Sensitivity parameter for Boltzmann distribution.
        gamma (float): Discount factor for cumulative rewards.

    Returns:
        tuple: (trajectory, stopping_time)
    """
    traj_len = len(full_trajectory)

    # Compute cumulative reward for the entire trajectory
    traj_reward = sum(env.compute_reward(s) for s, _ in full_trajectory)

    # Initialize variables
    cumulative_rewards = []
    probabilities = []

    # Compute cumulative rewards up to each time step and probabilities
    for t in range(traj_len):
        # Reward up to time t
        reward_up_to_t = sum(env.compute_reward(s) for s, _ in full_trajectory[:t+1])

        # Add repeated reward for the last step
        #reward_up_to_t += (traj_len - t - 1) * env.compute_reward(full_trajectory[t][0])

        # Numerator and denominator for the stopping probability
        numerator = beta * reward_up_to_t
        denominator = logsumexp([beta * traj_reward, numerator])

        # Compute the probability of stopping at time t
        stop_probability = numerator - denominator
        probabilities.append(stop_probability)

    # Normalize probabilities (to ensure numerical stability)
    probabilities = np.array(probabilities)
    #probabilities /= probabilities.sum()

    # Sample stopping point t_stop from the computed probabilities
    #t_stop = np.random.choice(len(probabilities), p=probabilities)
    t_stop = np.argmax(probabilities)

    # Return the trajectory and the stopping point
    return (full_trajectory, t_stop)

def generate_pairwise_comparisons(env, trajectories, num_comparisons=10):
    """
    Generates a fixed number of pairwise comparisons between pre-generated trajectories.

    Args:
        env: The GridWorld environment.
        trajectories (list): List of pre-generated trajectories (each a list of (state, action) tuples).
        num_comparisons (int): Number of comparisons to return.

    Returns:
        List of tuples containing an ordered pair of trajectories (with the higher reward trajectory first).
    """
    pairwise_comparisons = []

    # Evaluate and attach rewards to each trajectory
    rewarded_trajectories = []
    for traj in trajectories:
        total_reward = sum(env.compute_reward(state) for state, action in traj)
        rewarded_trajectories.append((traj, total_reward))

    # Compare all unique trajectory pairs in fixed order
    for i in range(len(rewarded_trajectories)):
        for j in range(i + 1, len(rewarded_trajectories)):
            traj_1, reward_1 = rewarded_trajectories[i]
            traj_2, reward_2 = rewarded_trajectories[j]

            if reward_1 > reward_2:
                pairwise_comparisons.append((traj_1, traj_2))
            elif reward_2 > reward_1:
                pairwise_comparisons.append((traj_2, traj_1))
            # Ignore equal rewards if not meaningful

    # Return the first `num_comparisons` comparisons !! BUG BUG BUG
    #return pairwise_comparisons[:min(num_comparisons, len(pairwise_comparisons))] # This line lead to a bug where it returns a fixed traj in compare to others


    # Randomly pick pairwise preferences and not in order to make sure about the diversity
    random_picked_pairwise_comparisons = random.sample(pairwise_comparisons, num_comparisons)

    return random_picked_pairwise_comparisons

def generate_valid_trajectories(env, num_demonstration, min_length, max_horizon, fixed_start=None):
    """
    Generates a specified number of random trajectories with at least `min_length`.

    Args:
        env: The environment to generate trajectories from.
        num_demonstration (int): Number of valid trajectories to generate.
        min_length (int): Minimum trajectory length to accept.
        max_horizon (int): Max horizon for trajectory generation.

    Returns:
        List of valid trajectories.
    """
    random_trajs = []
    while len(random_trajs) < num_demonstration:
        traj = generate_random_trajectory(env, max_horizon=max_horizon, fixed_start=fixed_start)
        if len(traj) >= min_length:
            random_trajs.append(traj)
    return random_trajs

# def generate_noisy_comparisons(comparisons, percent):
#     """
#     Generate suboptimal pairwise comparisons by flipping the order of a specified percentage of comparisons.

#     Args:
#         comparisons: List of tuples, where each tuple represents a pairwise comparison (e.g., (item1, item2)).
#         percent: Float between 0 and 1, indicating the fraction of comparisons to flip.

#     Returns:
#         List of comparisons with the specified percentage flipped to suboptimal order.
#     """
#     # Check if percent is between 0 and 1
#     if not 0 <= percent <= 1:
#         raise ValueError("Percent must be between 0 and 1")

#     # Calculate number of comparisons to flip
#     num_to_flip = int(len(comparisons) * percent)

#     suboptimal_comparisons = []

#     # Process the first 'percent' of comparisons
#     for i, comparison in enumerate(comparisons):
#         if i < num_to_flip:
#             # Flip the order to create suboptimal comparison
#             suboptimal_comparisons.append((comparison[1], comparison[0]))
#         else:
#             # Keep original comparison
#             suboptimal_comparisons.append(comparison)

#     return suboptimal_comparisons

# def generate_noisy_estop(estops, percent):
#     # Check if percent is between 0 and 1
#     if not 0 <= percent <= 1:
#         raise ValueError("Percent must be between 0 and 1")

#     # Calculate number of estops to process (first 'percent' of estops)
#     num_to_process = int(len(estops) * percent)

#     noisy_estops = []

#     # Process only the first 'percent' of estops
#     for estop in estops[:num_to_process]:
#         # Get all possible stop times except the optimal one
#         non_optimal_stop_times = list(range(len(estop[0])))
#         non_optimal_stop_times.remove(int(estop[1]))

#         # Create noisy estop by choosing a random non-optimal stop time
#         noisy_estop = (estop[0], np.random.choice(non_optimal_stop_times))
#         noisy_estops.append(noisy_estop)

#     # Add remaining original estops if percent < 1
#     noisy_estops.extend(estops[num_to_process:])

#     return noisy_estops

# def generate_suboptimal_comparisons(comparisons, percent):
#     """
#     Generate suboptimal pairwise comparisons by flipping the order of a specified percentage of comparisons.

#     Args:
#         comparisons: List of tuples, where each tuple represents a pairwise comparison (e.g., (item1, item2)).
#         percent: Float between 0 and 1, indicating the fraction of comparisons to flip.

#     Returns:
#         List of comparisons with the specified percentage flipped to suboptimal order.
#     """
#     # Check if percent is between 0 and 1
#     if not 0 <= percent <= 1:
#         raise ValueError("Percent must be between 0 and 1")

#     # Calculate number of comparisons to flip
#     num_to_flip = int(len(comparisons) * percent)

#     suboptimal_comparisons = []

#     # Process the first 'percent' of comparisons
#     for i, comparison in enumerate(comparisons):
#         if i < num_to_flip:
#             # Flip the order to create suboptimal comparison
#             suboptimal_comparisons.append((comparison[1], comparison[0]))
#         else:
#             # Keep original comparison
#             suboptimal_comparisons.append(comparison)

#     return suboptimal_comparisons


# def generate_pairwise_comparisons(env, num_trajs=10, max_horizon=25, num_comparisons=10,  fixed_start=False):
#     """
#     Generates a fixed number of pairwise comparisons between randomly generated trajectories.

#     Args:
#         env: The GridWorld environment.
#         num_trajs (int): Number of trajectories to generate.
#         max_horizon (int): Maximum length of each trajectory.
#         num_comparisons (int): Number of comparisons to return.

#     Returns:
#         List of tuples containing a pair of trajectories for comparison.
#     """
#     pairwise_comparisons = []
#     trajectories = []

#     # Generate random trajectories
#     for _ in range(num_trajs):
#         traj = generate_random_trajectory(env, max_horizon, fixed_start=fixed_start)
#         total_reward = sum(env.compute_reward(state) for state, action in traj)  # Ignore terminal state
#         trajectories.append((traj, total_reward))

#     # Compare all unique trajectory pairs
#     for i in range(len(trajectories)):
#         for j in range(i + 1, len(trajectories)):  # Avoid duplicate comparisons
#             traj_1, reward_1 = trajectories[i]
#             traj_2, reward_2 = trajectories[j]

#             if reward_1 > reward_2:
#                 pairwise_comparisons.append((traj_1, traj_2))
#             elif reward_2 > reward_1:
#                 pairwise_comparisons.append((traj_2, traj_1))

#     # Ensure we return exactly `num_comparisons` comparisons
#     return random.sample(pairwise_comparisons, min(num_comparisons, len(pairwise_comparisons)))

# def simulate_improvement_feedback_DEPRICATED(env, trajectory, optimal_policy):
#     """
#     This kind of improvement where it improves the trajectory to reach the goal was so informative than the pairwise comparison

#     Simulates improvement feedback by modifying a randomly chosen suboptimal step in the trajectory.

#     Args:
#         env: The GridWorld environment.
#         trajectory (list): A list of (state, action) tuples representing the original trajectory.
#         optimal_policy (list of tuples): A list of (state, optimal_action) pairs.

#     Returns:
#         tuple: (improved_trajectory, original_trajectory)
#             - improved_trajectory: The modified trajectory with an improved action sequence.
#             - original_trajectory: The input trajectory (unchanged).
#             - If the given trajectory was already optimal, improved_trajectory is an empty list.
#     """
#     # Convert optimal_policy from list of tuples to dictionary for fast lookup
#     optimal_policy_dict = dict(optimal_policy)

#     if len(trajectory) < 2:
#         return ([], trajectory)  # Too short to improve

#     # Find all suboptimal action indices
#     suboptimal_indices = [
#         i for i in range(len(trajectory) - 1)  # Exclude the last state
#         if trajectory[i][1] != optimal_policy_dict.get(trajectory[i][0], trajectory[i][1])
#     ]

#     if not suboptimal_indices:
#         return ([], trajectory)  # No suboptimal actions found, return empty improved trajectory

#     # Randomly select one of the suboptimal indices
#     suboptimal_index = random.choice(suboptimal_indices)
#     state, _ = trajectory[suboptimal_index]  # Start from the randomly chosen suboptimal state
#     optimal_action = optimal_policy_dict[state]  # Get the optimal action

#     # Create the improved trajectory
#     improved_trajectory = trajectory[:suboptimal_index]  # Keep trajectory up to this state

#     while state not in env.terminal_states:
#         improved_trajectory.append((state, optimal_action))

#         # Get next state probabilities based on transition model
#         next_state_probs = env.transitions[state][optimal_action]
#         state = np.random.choice(env.get_num_states(), p=next_state_probs)  # Sample next state

#         if state in env.terminal_states:
#             improved_trajectory.append((state, None))  # Append terminal state
#             break

#         # Update action based on the optimal policy
#         optimal_action = optimal_policy_dict.get(state, optimal_action)

#     return (improved_trajectory, trajectory)



# def evaluate_trajectory(traj):
#     """Compute total reward of a trajectory."""
#     return sum(env.compute_reward(s) for s, _ in traj)

#     # Convert optimal_policy to dict for lookup
#     optimal_policy_dict = dict(optimal_policy)

#     if len(trajectory) < 2:
#         return ([], trajectory)

#     suboptimal_indices = [
#         i for i in range(len(trajectory) - 1)
#         if trajectory[i][1] != optimal_policy_dict.get(trajectory[i][0], trajectory[i][1])
#     ]

#     if not suboptimal_indices:
#         return ([], trajectory)

#     suboptimal_index = random.choice(suboptimal_indices)

#     original_return = evaluate_trajectory(trajectory)
#     improved_trajectory = []

#     while True:
#         # Start reconstruction of a corrected trajectory
#         improved_trajectory = trajectory[:suboptimal_index]
#         state, _ = trajectory[suboptimal_index]

#         for _ in range(suboptimal_index, len(trajectory)):
#             # With 50% probability use optimal action, otherwise random
#             if random.random() < 0.5:
#                 action = optimal_policy_dict.get(state, env.sample_action())
#             else:
#                 action = env.sample_action()

#             improved_trajectory.append((state, action))

#             # Sample next state
#             next_state_probs = env.transitions[state][action]
#             next_state = np.random.choice(env.get_num_states(), p=next_state_probs)

#             if next_state in env.terminal_states:
#                 improved_trajectory.append((next_state, None))
#                 break

#             state = next_state

#         # Evaluate and break if improved
#         improved_return = evaluate_trajectory(improved_trajectory)
#         if improved_return > original_return:
#             break

#     return (improved_trajectory, trajectory)



# def simulate_human_estop(env, full_trajectory, beta=2.0, gamma=1.0, fixed_length=None):
#     """
#     Simulates human E-stop (early stopping) behavior in a GridWorld environment and ensures all output trajectories have the same length.

#     Args:
#         env (NoisyLinearRewardFeaturizedGridWorldEnv): The environment instance.
#         full_trajectory (list): A full-length trajectory as [(state, action), ...].
#         beta (float): Sensitivity parameter for Boltzmann distribution.
#         gamma (float): Discount factor for cumulative rewards.
#         fixed_length (int, optional): Desired fixed length for the output trajectory. If the trajectory is shorter, the last step is repeated.

#     Returns:
#         tuple: (trajectory, stopping_time)
#     """
#     cumulative_rewards = []
#     current_reward = 0

#     for k, (state, _) in enumerate(full_trajectory):
#         if state is None:  # Handle padding
#             break

#         # Compute reward for the current state using the environment function
#         reward = env.compute_reward(state)  # Now using the built-in reward function

#         # Discounted cumulative reward up to step k
#         current_reward += (gamma**k) * reward
#         cumulative_rewards.append(current_reward)

#     # Convert to numpy array for stable computation
#     cumulative_rewards = np.array(cumulative_rewards)

#     # Use `logsumexp` for numerical stability
#     log_denominator = logsumexp(beta * cumulative_rewards)
#     log_numerator = beta * cumulative_rewards  # Log of exp(beta * cumulative_rewards)

#     # Compute stopping probabilities in log-space
#     log_probabilities = log_numerator - log_denominator
#     probabilities = np.exp(log_probabilities)  # Convert back to normal probability values

#     # Select stopping time using highest probability
#     t_stop = np.argmax(probabilities)

#     # Pad the trajectory to ensure it matches the fixed length
#     if fixed_length is not None:
#         last_step = full_trajectory[-1]
#         while len(full_trajectory) < fixed_length:
#             full_trajectory.append(last_step)

#     return (full_trajectory[:fixed_length] if fixed_length else full_trajectory, t_stop)


# """
# Some ideas for future debug

# In generate_optimal_demo, do I need to multiply discount factor when I summing rewards?

# """
# import sys
# import os
# import time
# import yaml
# import numpy as np
# import random
# from scipy.special import logsumexp
# # Get current and parent directory to handle import paths
# current = os.path.dirname(os.path.realpath(__file__))
# parent = os.path.dirname(current)
# sys.path.append(parent)

# from agent.q_learning_agent import ValueIteration

# class GridWorldMDPDataGenerator:
#     def __init__(self, env, q_values=None, seed=None):
#         """
#         Initializes the generator with the environment.
#         :param env: Markov decision process environment.
#         """
#         self.env = env
#         self.q_values = q_values

#         if seed is not None:
#             self._set_random_seed(seed=seed)

#     def generate_optimal_demo(self, num_trajs, start_states=None):
#         """
#         Generates multiple optimal demonstrations consisting of state-action pairs (s, a),
#         and computes the cumulative reward for each trajectory.

#         :param num_trajs: Number of trajectories to generate.
#         :param start_states: Optional list of starting states. If not provided, random non-terminal states will be used.
#         :return: A list of tuples where each tuple contains an optimal trajectory and its associated cumulative reward.
#         """

#         trajectories_with_rewards = []

#         # Get all non-terminal states
#         non_terminal_states = [s for s in range(self.env.get_num_states())
#                             if s not in self.env.terminal_states]

#         # Handle start states: If not provided, randomly select unique non-terminal starting states
#         if start_states is None:
#             if num_trajs > len(non_terminal_states):
#                 raise ValueError("Number of trajectories exceeds the number of available non-terminal states.")
#             start_states = np.random.choice(non_terminal_states, size=num_trajs, replace=False)

#         for current_state in start_states:
#             max_traj_length = self.env.get_num_states()
#             optimal_trajectory = []
#             cumulative_reward = 0  # Initialize cumulative reward for the trajectory

#             # Generate the trajectory until a terminal state is reached or max length is reached
#             while current_state not in self.env.terminal_states and len(optimal_trajectory) < max_traj_length:
#                 # Generate an optimal action, breaking ties uniformly at random
#                 act = np.random.choice(self.arg_max_set(self.q_values[current_state]))
#                 optimal_trajectory.append((current_state, act))

#                 # Compute reward for the current state
#                 reward = self.env.compute_reward(current_state)
#                 cumulative_reward += reward

#                 # Sample the next state based on transition probabilities
#                 probs = self.env.transitions[current_state][act]
#                 next_state = np.random.choice(self.env.num_states, p=probs)
#                 current_state = next_state

#             # Handle the last state if it's terminal
#             if current_state in self.env.terminal_states:
#                 reward = self.env.compute_reward(current_state)
#                 cumulative_reward += reward
#                 # Append the terminal state with a dummy action (-1 or None) if needed
#                 optimal_trajectory.append((current_state, None))  # Terminal state, no action


#             # Store the trajectory and its cumulative reward
#             trajectories_with_rewards.append((optimal_trajectory, cumulative_reward))

#         return trajectories_with_rewards

#     def generate_random_demo(self, num_trajs, start_states=None):
#         """
#         Generates multiple random trajectories consisting of state-action pairs (s, a),
#         and computes the cumulative reward for each trajectory.

#         :param num_trajs: Number of trajectories to generate.
#         :param start_states: Optional list of starting states. If not provided, random non-terminal states will be used.
#         :return: A list of tuples where each tuple contains a random trajectory and its associated cumulative reward.
#         """
#         trajectories_with_rewards = []

#         # Get all non-terminal states
#         non_terminal_states = [
#             s for s in range(self.env.get_num_states())
#             if s not in self.env.terminal_states
#         ]
#         # Handle start states: If not provided, randomly select unique non-terminal starting states
#         start_states = np.random.choice(non_terminal_states, size=num_trajs, replace=True)

#         for current_state in start_states:
#             max_traj_length = self.env.get_num_states()
#             random_trajectory = []
#             cumulative_reward = 0  # Initialize cumulative reward for the trajectory

#             # Generate the trajectory until a terminal state is reached or max length is reached
#             while current_state not in self.env.terminal_states and len(random_trajectory) < max_traj_length:
#                 # Select a random action
#                 act = np.random.choice(range(self.env.num_actions))
#                 random_trajectory.append((current_state, act))

#                 # Compute reward for the current state
#                 reward = self.env.compute_reward(current_state)
#                 cumulative_reward += reward

#                 # Sample the next state based on transition probabilities
#                 probs = self.env.transitions[current_state][act]
#                 next_state = np.random.choice(self.env.num_states, p=probs)
#                 current_state = next_state

#             # Handle the last state if it's terminal
#             if (current_state in self.env.terminal_states or
#                     len(random_trajectory) < max_traj_length):

#                 reward = self.env.compute_reward(current_state)
#                 cumulative_reward += reward
#                 # Append the terminal state with a dummy action (-1 or None) if needed
#                 random_trajectory.append((current_state, None))  # Terminal state, no action

#             # Store the trajectory and its cumulative reward
#             trajectories_with_rewards.append((random_trajectory, cumulative_reward))

#         return trajectories_with_rewards

#     def generate_pairwise_comparisons(self, strategy="random_vs_random", num_trajs=10):
#         """
#         Generates pairwise comparisons between trajectories based on rewards.

#         Strategies:
#         1. 'random_vs_random' - Generate multiple random trajectories and compare all pairs.
#         2. 'same_start_state' - Generate two trajectories from the same start state and compare them.

#         :param strategy: The strategy to use for generating pairwise comparisons.
#         :param num_trajs: Number of trajectories to generate.
#         :return: List of tuples where each tuple contains a pair of trajectories for comparison.
#         """
#         pairwise_comparisons = []

#         if strategy == "random_vs_random":
#             # Generate multiple random trajectories and compare all unique pairs
#             trajectories = self.generate_random_demo(num_trajs)
#             for i in range(len(trajectories)):
#                 for j in range(len(trajectories)):
#                     if i != j:
#                         traj_1, reward_1 = trajectories[i]
#                         traj_2, reward_2 = trajectories[j]
#                         if reward_1 > reward_2:
#                             pairwise_comparisons.append((traj_1, traj_2))

#         elif strategy == "same_start_state":
#             # Generate two random trajectories from the same start state and compare them
#             non_terminal_states = [
#                 s for s in range(self.env.get_num_states()) if s not in self.env.terminal_states
#             ]

#             start_states = np.random.choice(non_terminal_states, size=num_trajs, replace=True)

#             for start_state in start_states:
#                 traj_1, reward_1 = self.generate_random_demo(1, [start_state])[0]
#                 traj_2, reward_2 = self.generate_random_demo(1, [start_state])[0]

#                 if reward_1 > reward_2:
#                     pairwise_comparisons.append((traj_1, traj_2))
#                 elif reward_2 > reward_1:
#                     pairwise_comparisons.append((traj_2, traj_1))

#         else:
#             raise ValueError(f"Invalid strategy: {strategy}")

#         return pairwise_comparisons

#     def generate_estop(self, beta, num_trajs, start_states=None):
#         """
#         Generates E-stops for random trajectories using the human likelihood model.

#         :param beta: Sensitivity parameter for human decision-making.
#         :param num_trajs: Number of trajectories to generate.
#         :return: List of E-stop events (trajectories with stopping point t).
#         """
#         # Generate random trajectories first
#         trajectories_with_rewards = self.generate_random_demo(num_trajs, start_states)

#         estop_trajectories = []

#         # Iterate over each trajectory
#         for trajectory, cumulative_reward in trajectories_with_rewards:
#             T = len(trajectory)  # Length of the trajectory
#             stop_probs = []

#             # Calculate stop probabilities for each possible stopping point t
#             for t in range(T):
#                 # Sub-trajectory ξ_0:t (use cumulative rewards up to point t)
#                 reward_up_to_t = sum(self.env.compute_reward(s) for s, _ in trajectory[:t+1])
#                 stop_prob_numerator = np.exp(beta * reward_up_to_t)

#                 # Compute denominator (normalization factor for the entire trajectory)
#                 stop_prob_denominator = sum(np.exp(beta * sum(self.env.compute_reward(s) for s, _ in trajectory[:k+1])) for k in range(T))

#                 # Calculate stop probability for stopping at time t
#                 stop_prob = stop_prob_numerator / stop_prob_denominator
#                 stop_probs.append(stop_prob)

#             # Select stopping point t based on the calculated probabilities
#             stop_point = np.random.choice(range(T), p=stop_probs)

#             # Append the trajectory with its stopping point to the result list
#             estop_trajectories.append((trajectory, stop_point))

#         return estop_trajectories

#     @staticmethod
#     def arg_max_set(q_values):
#         """
#         Returns the set of actions corresponding to the maximum Q-value for a given state.
#         :param q_values: Q-values for the current state.
#         :return: List of actions with the maximum Q-value.
#         """
#         max_q = np.max(q_values)
#         return np.flatnonzero(q_values == max_q)

#     def _set_random_seed(self, seed):
#         """
#         Set the random seed for numpy, random, and any other libraries that require seeding.
#         """
#         np.random.seed(seed)
#         random.seed(seed)



# def simulate_improvement_feedback_v2(env, trajectory, optimal_policy):
#     """
#     Simulates improvement feedback by modifying a randomly chosen suboptimal step in the trajectory.

#     Args:
#         env: The GridWorld environment.
#         trajectory (list): A list of (state, action) tuples representing the original trajectory.
#         optimal_policy (list of tuples): A list of (state, optimal_action) pairs.

#     Returns:
#         tuple: (improved_trajectory, original_trajectory)
#             - improved_trajectory: The modified trajectory with an improved action sequence.
#             - original_trajectory: The input trajectory (unchanged).
#             - If the given trajectory was already optimal, improved_trajectory is an empty list.
#     """
#     # Convert optimal_policy from list of tuples to dictionary for fast lookup
#     optimal_policy_dict = dict(optimal_policy)

#     if len(trajectory) < 2:
#         return ([], trajectory)  # Too short to improve

#     # Find all suboptimal action indices
#     suboptimal_indices = [
#         i for i in range(len(trajectory) - 1)  # Exclude the last state
#         if trajectory[i][1] != optimal_policy_dict.get(trajectory[i][0], trajectory[i][1])
#     ]

#     if not suboptimal_indices:
#         return ([], trajectory)  # No suboptimal actions found, return empty improved trajectory

#     # Randomly select one of the suboptimal indices
#     suboptimal_index = random.choice(suboptimal_indices)
#     state, _ = trajectory[suboptimal_index]  # Start from the randomly chosen suboptimal state
#     optimal_action = optimal_policy_dict[state]  # Get the optimal action

#     # Create the improved trajectory
#     improved_trajectory = trajectory[:suboptimal_index]  # Keep trajectory up to this state

#     # Continue improving for the same length as the original trajectory
#     for _ in range(suboptimal_index, len(trajectory)):
#         improved_trajectory.append((state, optimal_action))

#         # Get next state probabilities based on transition model
#         next_state_probs = env.transitions[state][optimal_action]
#         state = np.random.choice(env.get_num_states(), p=next_state_probs)  # Sample next state
#         if state in env.terminal_states:
#             improved_trajectory.append((state, None))  # Append terminal state
#             break

#         # Update action based on the optimal policy
#         optimal_action = optimal_policy_dict.get(state, optimal_action)

#     return (improved_trajectory, trajectory)

# def simulate_improvement_feedback_v3(env, trajectory, optimal_policy):
#     def evaluate_trajectory(traj):
#         """Compute total reward of a trajectory."""
#         return sum(env.compute_reward(s) for s, _ in traj)

#     optimal_policy_dict = dict(optimal_policy)

#     if len(trajectory) < 2:
#         return ([], trajectory)

#     suboptimal_indices = [
#         i for i in range(len(trajectory) - 1)
#         if trajectory[i][1] != optimal_policy_dict.get(trajectory[i][0], trajectory[i][1])
#     ]

#     if not suboptimal_indices:
#         return ([], trajectory)

#     suboptimal_index = random.choice(suboptimal_indices)
#     improved_trajectory = trajectory[:suboptimal_index]
#     state, _ = trajectory[suboptimal_index]

#     for _ in range(suboptimal_index, len(trajectory)):
#         optimal_action = optimal_policy_dict[state]

#         # Choose optimal or non-optimal action
#         if random.random() < 0.5:
#             action = optimal_action
#         else:
#             non_optimal_actions = [a for a in range(env.num_actions) if a != optimal_action]
#             action = np.random.choice(non_optimal_actions)

#         improved_trajectory.append((state, action))

#         next_state_probs = env.transitions[state][action]
#         next_state = np.random.choice(env.get_num_states(), p=next_state_probs)

#         if next_state in env.terminal_states:
#             improved_trajectory.append((next_state, None))
#             break

#         state = next_state

#     # Evaluate and compare rewards
#     original_return = evaluate_trajectory(trajectory)
#     improved_return = evaluate_trajectory(improved_trajectory)

#     if improved_return > original_return:
#         return (improved_trajectory, trajectory)
#     else:
#         return (trajectory, improved_trajectory)