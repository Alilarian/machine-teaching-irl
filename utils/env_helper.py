import numpy as np

# Edit if your action ordering differs
ACTION_ARROWS = {0: "^", 1: "v", 2: "<", 3: ">"}

def optimal_actions_from_q(Q, tie_eps=1e-8, terminal_states=None):
    """
    Q: (S, A) Q-values
    Returns: list of lists; out[s] = list of optimal action indices at state s.
    """
    S, A = Q.shape
    terminals = set(terminal_states or [])
    out = [[] for _ in range(S)]
    for s in range(S):
        if s in terminals:
            continue
        row = Q[s]
        m = np.max(row)
        out[s] = [a for a in range(A) if (m - row[a]) <= tie_eps]
    return out

def policy_grid_from_q(Q, rows, cols, action_arrows=ACTION_ARROWS,
                       tie_eps=1e-8, terminal_states=None,
                       empty_char="·", terminal_char="T"):
    """
    Build a rectangular grid (rows x cols) of strings showing all optimal actions per cell.
    """
    assert Q.ndim == 2, "Q must be (S, A)"
    S = rows * cols
    assert Q.shape[0] == S, f"Q has {Q.shape[0]} states but rows*cols={S}"
    opt = optimal_actions_from_q(Q, tie_eps=tie_eps, terminal_states=terminal_states)
    terminals = set(terminal_states or [])

    grid = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            s = r * cols + c
            if s in terminals:
                row_cells.append(terminal_char)
            else:
                acts = opt[s]
                if not acts:
                    row_cells.append(empty_char)
                else:
                    row_cells.append("".join(action_arrows.get(a, "?") for a in acts))
        grid.append(row_cells)
    return grid

def print_policy_from_q(Q, rows, cols, action_arrows=ACTION_ARROWS,
                        tie_eps=1e-8, terminal_states=None,
                        empty_char="·", terminal_char="T"):
    """
    Pretty-print the rectangular policy grid with equal-width cells.
    """
    grid = policy_grid_from_q(Q, rows, cols, action_arrows, tie_eps,
                              terminal_states, empty_char, terminal_char)
    cell_w = max(max(len(cell) for cell in row) for row in grid) if grid else 1
    for row in grid:
        print(" | ".join(cell.ljust(cell_w) for cell in row))
    print()



# def print_policy_2(policy, size):
#     '''
#     Print the policy in a human-readable format.
    
#     Args:
#         policy: A list of (state, action) tuples representing the policy.
#         size: Size of the grid (number of rows/columns).
#     '''
#     # Action mappings to arrow symbols
#     action_arrows = {0: "^", 1: "v", 2: "<", 3: ">"}
    
#     # Initialize an empty grid to store the policy
#     grid_policy = [[" " for _ in range(size)] for _ in range(size)]
    
#     # Populate the grid with arrows corresponding to actions
#     for state, action in policy:
#         if action is not None:  # Ensure the action is valid
#             row, col = divmod(state, size)
#             grid_policy[row][col] = action_arrows.get(action, "?")  # Use "?" for unknown actions

#     # Print the grid
#     for row in grid_policy:
#         print(" | ".join(row))
#     print()  # Add an empty line for better formatting

# def print_policy(policy, rows, cols):
#     """
#     Print the policy in a human-readable format.
    
#     Args:
#         policy: A list of ((row, col), action) tuples representing the policy.
#         rows: Number of rows in the grid.
#         cols: Number of columns in the grid.
#     """
#     # Action mappings to arrow symbols
#     action_arrows = {0: "^", 1: "v", 2: "<", 3: ">"}
    
#     # Initialize an empty grid to store the policy
#     grid_policy = [[" " for _ in range(cols)] for _ in range(rows)]
    
#     # Populate the grid with arrows corresponding to actions
#     for state, action in policy:
#         if action is not None:  # Ensure the action is valid
#             row, col = divmod(state, cols)
#             grid_policy[row][col] = action_arrows.get(action, "?")  # Use "?" for unknown actions

#     # Print the grid
#     for row in grid_policy:
#         print(" | ".join(row))
#     print()  # Add an empty line for better formatting