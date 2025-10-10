import numpy as np

class ValueIteration:
    """
    Implements the Value Iteration algorithm for solving a Markov Decision Process (MDP).

    Attributes:
        mdp: An MDP object containing transition probabilities, rewards, and discount factor.
    """
    def __init__(self, mdp):
        """
        Initializes the ValueIteration class with the MDP.

        Args:
            mdp: The Markov Decision Process (MDP) that the value iteration will solve.
        """
        self.mdp = mdp
        self.state_values = np.zeros(self.mdp.get_num_states(), dtype=float)

    def run_value_iteration(self, epsilon=1e-10):
        """
        Performs the Value Iteration algorithm to compute the optimal value function.

        Args:
            epsilon: The convergence threshold for stopping the iteration (default is 0.0001).

        Returns:
            state_values: A numpy array containing the optimal value for each state.
        """
        # Initialize delta to a large number and state_values to zero for all states
        delta = np.inf
        # Initialize all state values to zero

        # Continue iterating until the maximum value change (delta) is smaller than the threshold
        discount_factor = self.mdp.get_discount_factor()
        
        threshold = epsilon * (1 - discount_factor) / discount_factor

        while delta > threshold:
        #for _ in range(80000):
            # Copy current state values to use in the next iteration
            previous_state_values = self.state_values.copy()
            delta = 0

            # Iterate over all states in the MDP
            for state in range(self.mdp.get_num_states()):
                #if state in self.mdp.terminal_states:
                #    continue
                max_action_value = -np.inf  # Start with the lowest possible value for action evaluation
                
                # Iterate over all possible actions in the current state
                for action in range(self.mdp.get_num_actions()):
                    # Calculate the expected value of taking this action in the current state
                    expected_action_value = np.dot(self.mdp.transitions[state][action], previous_state_values)
                    # Keep track of the highest action value (optimal action)
                    max_action_value = max(expected_action_value, max_action_value)

                # Update the value of the current state using the Bellman optimality equation
                self.state_values[state] = self.mdp.compute_reward(state) + discount_factor * max_action_value

                # Calculate the difference between the new and old value for this state (for convergence check)
                delta = max(delta, np.abs(self.state_values[state] - previous_state_values[state]))

        # Return the optimal state values once convergence is reached
        return self.state_values
    
    def get_optimal_policy(self):
        """
        Extracts the optimal policy based on the computed state values.

        Returns:
            policy: A list of tuples (state, optimal_action) representing the optimal policy.
        """
        # Check if state values have been computed; if not, run value iteration
        if np.all(self.state_values == 0):
            print("State values are all zero. Running value iteration...")
            self.run_value_iteration()

        optimal_policy = []  # List to store (state, optimal_action) pairs

        # Compute the optimal action for each state
        for state in range(self.mdp.get_num_states()):
            max_action_value = -np.inf
            best_action = None
            # Iterate over all actions to find the optimal one
            for action in range(self.mdp.get_num_actions()):
                expected_action_value = np.dot(
                    self.mdp.transitions[state][action], self.state_values
                )
                if expected_action_value > max_action_value:
                    max_action_value = expected_action_value
                    best_action = action

            # Store the (state, optimal_action) pair
            optimal_policy.append((state, best_action))

        return optimal_policy

    def get_q_values(self, state_values=None):
        """
        Computes the Q-values for all state-action pairs based on the provided or computed state values.

        Args:
            state_values: A numpy array containing the value of each state. If not provided, the function
                        will run value iteration to compute the optimal state values.

        Returns:
            qvalues: A numpy array of shape (num_states, num_actions) containing the Q-value for each state-action pair.
        """
        # Run value iteration if state_values are not provided
        if state_values is None:
            state_values = self.run_value_iteration()

        # Initialize Q-values array with shape (num_states, num_actions)
        num_states = self.mdp.get_num_states()
        num_actions = self.mdp.get_num_actions()
        qvalues = np.zeros((num_states, num_actions), dtype=float)

        discount_factor = self.mdp.get_discount_factor()

        # Compute Q-values for each state-action pair
        for state in range(self.mdp.get_num_states()):
            #if state in self.mdp.terminal_states:
            #    qvalues[state,:] = 0
            #else:
            for action in range(self.mdp.get_num_actions()):
                # Q(s, a) = R(s, a) + γ * Σ_s' P(s' | s, a) * V(s')
                expected_value_of_next_state = np.dot(self.mdp.transitions[state][action], state_values)
                reward = self.mdp.compute_reward(state)
                qvalues[state][action] = reward + discount_factor * expected_value_of_next_state

        return qvalues

class PolicyEvaluation:
    """
    Implements policy evaluation for a given policy in a Markov Decision Process (MDP).

    Attributes:
        mdp: The MDP object that contains transitions, rewards, and discount factor.
        policy: The policy being evaluated, which can be deterministic.
        uniform_random: A boolean flag indicating if the policy is a uniform random policy.
    """

    def __init__(self, mdp, policy=None, uniform_random=False):
        """
        Initializes the PolicyEvaluation class with the MDP and policy.

        Args:
            mdp: The Markov Decision Process (MDP) object.
            policy: A policy mapping states to actions (deterministic).
            uniform_random: Boolean indicating if the policy is a uniform random policy.
        """
        self.mdp = mdp
        self.policy = policy
        self.uniform_random = uniform_random

    def run_policy_evaluation(self, epsilon):
        """
        Runs the policy evaluation algorithm to compute the state value function for the given policy.

        Args:
            epsilon: The convergence threshold for stopping the iteration.

        Returns:
            state_values: A numpy array representing the value of each state under the given policy.
        """
        
        if self.uniform_random:
            return self.run_uniform_policy_evaluation(epsilon)
        else:
            return self.run_deterministic_policy_evaluation(epsilon)

    def run_deterministic_policy_evaluation(self, epsilon):
        """
        Runs policy evaluation for deterministic policies.
        
        Args:
            epsilon: The convergence threshold for stopping the iteration.
        
        Returns:
            state_values: A numpy array representing the value of each state under the given policy.
        """
        # Initialize delta to a large number and state_values to zero for all states
        self.policy = [x[1] for x in self.policy]
        delta = np.inf
        state_values = np.zeros(self.mdp.get_num_states(), dtype=float)  # Initialize all state values to zero

        # Continue iterating until the maximum value change (delta) is smaller than the threshold
        discount_factor = self.mdp.get_discount_factor()
        threshold = epsilon * (1 - discount_factor) / discount_factor

        while delta > threshold:
            # Copy current state values to use in the next iteration
            previous_state_values = state_values.copy()
            delta = 0

            # Iterate over all states in the MDP
            for state in range(self.mdp.num_states):
                #if state in self.mdp.terminal_states:
                #    continue
                # Deterministic policy: a single action for the state
                action = self.policy[state]
                policy_action_value = np.dot(self.mdp.transitions[state][action], previous_state_values)

                # Update the state value using the reward and the expected future discounted value
                state_values[state] = self.mdp.compute_reward(state) + discount_factor * policy_action_value

                # Update delta to track the maximum change in state values
                delta = max(delta, abs(state_values[state] - previous_state_values[state]))

        return state_values

    def run_uniform_policy_evaluation(self, epsilon):
        """
        Runs policy evaluation assuming a uniform random policy where all actions are equally likely.

        Args:
            epsilon: The convergence threshold for stopping the iteration.

        Returns:
            state_values: A numpy array representing the value of each state under the uniform random policy.
        """
        # Number of states and actions in the environment
        num_states = self.mdp.num_states
        num_actions = self.mdp.num_actions
        
        # Initialize state values to zero
        state_values = np.zeros(num_states)
        
        # Initialize delta for convergence checking
        delta = np.inf
        discount_factor = self.mdp.get_discount_factor()
        threshold = epsilon * (1 - discount_factor) / discount_factor

        # Iterative process for policy evaluation
        while delta > threshold:
            # Copy the current state values to use in the next iteration
            previous_state_values = state_values.copy()
            delta = 0
            
            # Iterate over each state
            for state in range(num_states):
                #if state in self.mdp.terminal_states:
                #    continue
                # Calculate the expected value by averaging over all actions (uniform policy)
                policy_action_value = sum(
                    np.dot(self.mdp.transitions[state][action], previous_state_values) for action in range(num_actions)
                )
                
                # Update the value for the current state
                state_values[state] = self.mdp.compute_reward(state) + discount_factor * (1 / num_actions) * policy_action_value
                
                # Calculate the change in value (delta) to check convergence
                delta = max(delta, abs(state_values[state] - previous_state_values[state]))

        return state_values


class QIteration:
    pass
class Qlearning:
    pass
