
def print_policy_2(policy, size):
    '''
    Print the policy in a human-readable format.
    
    Args:
        policy: A list of (state, action) tuples representing the policy.
        size: Size of the grid (number of rows/columns).
    '''
    # Action mappings to arrow symbols
    action_arrows = {0: "^", 1: "v", 2: "<", 3: ">"}
    
    # Initialize an empty grid to store the policy
    grid_policy = [[" " for _ in range(size)] for _ in range(size)]
    
    # Populate the grid with arrows corresponding to actions
    for state, action in policy:
        if action is not None:  # Ensure the action is valid
            row, col = divmod(state, size)
            grid_policy[row][col] = action_arrows.get(action, "?")  # Use "?" for unknown actions

    # Print the grid
    for row in grid_policy:
        print(" | ".join(row))
    print()  # Add an empty line for better formatting

def print_policy(policy, rows, cols):
    """
    Print the policy in a human-readable format.
    
    Args:
        policy: A list of ((row, col), action) tuples representing the policy.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
    """
    # Action mappings to arrow symbols
    action_arrows = {0: "^", 1: "v", 2: "<", 3: ">"}
    
    # Initialize an empty grid to store the policy
    grid_policy = [[" " for _ in range(cols)] for _ in range(rows)]
    
    # Populate the grid with arrows corresponding to actions
    for state, action in policy:
        if action is not None:  # Ensure the action is valid
            row, col = divmod(state, cols)
            grid_policy[row][col] = action_arrows.get(action, "?")  # Use "?" for unknown actions

    # Print the grid
    for row in grid_policy:
        print(" | ".join(row))
    print()  # Add an empty line for better formatting