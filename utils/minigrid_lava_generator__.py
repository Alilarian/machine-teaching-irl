################################################################################################  Env without rotation - five features

import numpy as np
from typing import List, Dict, Optional, Tuple

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Lava, Goal
from minigrid.core.mission import MissionSpace


# ======================================================
# Feature extraction (5 features)
# ======================================================

FEATURE_SET = "L2.5"

# W_MAP = {
#     # [dist_goal, on_lava, adj_lava, free_deg, step]
#     "L2.5": np.array([-1.0, -8.0, -2.0, 0.5, -0.05]) /
#             np.linalg.norm([-1.0, -8.0, -2.0, 0.5, -0.05]),
# }


W_MAP = {
    # [dist_goal, on_lava, adj_lava, step]
    "L2.5": np.array([-1.0, -8.0, -2.0, -0.05]) /
            np.linalg.norm([-1.0, -8.0, -2.0, -0.05]),
}

def manhattan(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def on_lava_state(lava_mask, y, x):
    return int(lava_mask[y, x])


def lava_adjacent_count(lava_mask, y, x):
    H, W = lava_mask.shape
    c = 0
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W:
            c += int(lava_mask[ny, nx])
    return c


def free_neighbor_count(wall_mask, y, x):
    H, W = wall_mask.shape
    c = 0
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W and not wall_mask[ny, nx]:
            c += 1
    return c


def phi_from_state(state, goal_yx, lava_mask, wall_mask, size):
    y, x = state
    gy, gx = goal_yx

    dist_goal = manhattan((y, x), (gy, gx))
    on_lava = on_lava_state(lava_mask, y, x)
    adj_lava = lava_adjacent_count(lava_mask, y, x)
    #free_deg = free_neighbor_count(wall_mask, y, x)
    step = 1.0

    return np.array([
        dist_goal,
        on_lava,
        adj_lava,
        #free_deg,
        step,
    ], dtype=float)


# ======================================================
# Actions (No rotation)
# ======================================================

ACT_UP = 0
ACT_DOWN = 1
ACT_LEFT = 2
ACT_RIGHT = 3

ACTIONS = [ACT_UP, ACT_DOWN, ACT_LEFT, ACT_RIGHT]

ACTION_TO_DELTA = {
    ACT_UP: (-1, 0),
    ACT_DOWN: (1, 0),
    ACT_LEFT: (0, -1),
    ACT_RIGHT: (0, 1),
}


# ======================================================
# Environment
# ======================================================

mission_space = MissionSpace(mission_func=lambda: "reach the goal")
class LavaWorldEnv(MiniGridEnv):

    def __init__(
        self,
        size: int,
        lava_mask: np.ndarray,
        goal_yx: Tuple[int, int],
        agent_start_pos=(1, 1),
        max_steps=None,
        **kwargs,
    ):
        self.size = size
        self.lava_mask = lava_mask
        self.goal_yx = goal_yx
        self.agent_start_pos = agent_start_pos

        if max_steps is None:
            max_steps = 4 * size * size

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=max_steps,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        for y in range(height):
            for x in range(width):
                if self.lava_mask[y, x]:
                    self.put_obj(Lava(), x, y)

        gy, gx = self.goal_yx
        self.put_obj(Goal(), gx, gy)

        self.agent_pos = self.agent_start_pos
        self.mission = "reach the goal"


# ======================================================
# Static Map Extraction
# ======================================================

def build_static_maps(env: LavaWorldEnv):
    size = env.width
    wall_mask = np.zeros((size, size), dtype=bool)
    lava_mask = env.lava_mask.copy()
    goal_yx = env.goal_yx

    for y in range(size):
        for x in range(size):
            obj = env.grid.get(x, y)
            if isinstance(obj, Wall):
                wall_mask[y, x] = True

    lava_cells = np.argwhere(lava_mask)
    return size, wall_mask, lava_mask, lava_cells, goal_yx

# ======================================================
# MDP Construction
# ======================================================

def is_terminal_state(state, goal_yx, lava_mask):
    y, x = state
    return (y, x) == goal_yx


def step_model(state, action, wall_mask, goal_yx, lava_mask):
    y, x = state
    size = wall_mask.shape[0]

    if is_terminal_state(state, goal_yx, lava_mask):
        return state, True

    dy, dx = ACTION_TO_DELTA[action]
    ny, nx = y + dy, x + dx

    if (
        ny < 0 or ny >= size or
        nx < 0 or nx >= size or
        wall_mask[ny, nx]
    ):
        ns = (y, x)
    else:
        ns = (ny, nx)

    return ns, is_terminal_state(ns, goal_yx, lava_mask)


def enumerate_states(size, wall_mask):
    states = []
    for y in range(size):
        for x in range(size):
            if not wall_mask[y, x]:
                states.append((y, x))
    return states


def build_tabular_mdp(
    states,
    wall_mask,
    goal_yx,
    lava_mask,
    lava_cells,
    size,
    gamma=0.99,
):
    S = len(states)
    A = len(ACTIONS)
    D = len(W_MAP[FEATURE_SET])

    idx_of = {s: i for i, s in enumerate(states)}

    T = np.zeros((S, A, S), dtype=float)
    terminal = np.zeros(S, dtype=bool)
    Phi = np.zeros((S, D), dtype=float)

    for i, s in enumerate(states):
        terminal[i] = is_terminal_state(s, goal_yx, lava_mask)
        Phi[i] = phi_from_state(s, goal_yx, lava_mask, wall_mask, size)

        for a_idx, a in enumerate(ACTIONS):
            sp, _ = step_model(s, a, wall_mask, goal_yx, lava_mask)
            T[i, a_idx, idx_of[sp]] = 1.0

    return {
        "states": states,
        "idx_of": idx_of,
        "true_w": W_MAP[FEATURE_SET],
        "T": T,
        "Phi": Phi,
        "terminal": terminal,
        "gamma": gamma,
        "goal_yx": goal_yx,
        "lava_mask": lava_mask,
        "lava_cells": lava_cells,
        "wall_mask": wall_mask,
        "size": size,
    }


# ======================================================
# Layout Generators
# ======================================================

def generate_lava_wall_layout(size, rng):
    lava_mask = np.zeros((size, size), dtype=bool)

    vertical = rng.random() < 0.5

    if vertical:
        col = rng.integers(1, size - 1)
        wall = [(y, col) for y in range(1, size - 1)]
    else:
        row = rng.integers(1, size - 1)
        wall = [(row, x) for x in range(1, size - 1)]

    n_holes = rng.integers(1, 3)
    holes = set(rng.choice(len(wall), size=n_holes, replace=False))

    for i, (y, x) in enumerate(wall):
        if i not in holes:
            lava_mask[y, x] = True

    goal_rows = [y for y in range(1, size - 1) if not lava_mask[y, size - 2]]
    goal_y = int(rng.choice(goal_rows))
    goal_yx = (goal_y, size - 2)

    return lava_mask, goal_yx


def generate_random_lava_layout(size, rng, lava_frac=1/3):
    lava_mask = np.zeros((size, size), dtype=bool)

    candidates = [
        (y, x)
        for y in range(1, size - 1)
        for x in range(1, size - 1)
        if (y, x) != (1, 1)
    ]

    n_lava = int(lava_frac * len(candidates))
    lava_indices = rng.choice(len(candidates), size=n_lava, replace=False)

    for idx in lava_indices:
        y, x = candidates[idx]
        lava_mask[y, x] = True

    free_cells = [(y, x) for (y, x) in candidates if not lava_mask[y, x]]
    goal_yx = free_cells[rng.integers(len(free_cells))]

    return lava_mask, goal_yx


# ======================================================
# Main Entry
# ======================================================

# def generate_lavaworld(
#     n_envs: int,
#     size: int,
#     seed: Optional[int] = None,
#     gamma: float = 0.99,
# ):
#     rng = np.random.default_rng(seed)

#     envs = []
#     mdps = []
#     meta = {
#         "lava_masks": [],
#         "goals": [],
#         "layout_type": [],
#         "seed": seed,
#     }

#     for i in range(n_envs):
#         if i < n_envs // 2:
#             lava_mask, goal_yx = generate_lava_wall_layout(size, rng)
#             layout_type = "wall"
#         else:
#             lava_mask, goal_yx = generate_random_lava_layout(size, rng, lava_frac=1/3)
#             layout_type = "random"

#         env = LavaWorldEnv(
#             size=size,
#             lava_mask=lava_mask,
#             goal_yx=goal_yx,
#             render_mode="human",
#         )

#         size_, wall_mask, lava_mask, lava_cells, goal_yx = build_static_maps(env)
#         states = enumerate_states(size_, wall_mask)

#         mdp = build_tabular_mdp(
#             states,
#             wall_mask,
#             goal_yx,
#             lava_mask,
#             lava_cells,
#             size_,
#             gamma,
#         )

#         envs.append(env)
#         mdps.append(mdp)
#         meta["lava_masks"].append(lava_mask)
#         meta["goals"].append(goal_yx)
#         meta["layout_type"].append(layout_type)

#     return envs, mdps, meta

def generate_lavaworld(
    n_envs: int,
    size: int,
    seed: Optional[int] = None,
    gamma: float = 0.99,
    no_lava_fraction: float = 0.2,   # NEW
):
    rng = np.random.default_rng(seed)

    envs = []
    mdps = []
    meta = {
        "lava_masks": [],
        "goals": [],
        "layout_type": [],
        "no_lava": [],
        "seed": seed,
    }

    if not (0.0 <= no_lava_fraction <= 1.0):
        raise ValueError("no_lava_fraction must be between 0 and 1.")

    for i in range(n_envs):

        # --------------------------------------------------
        # Decide whether this MDP has lava
        # --------------------------------------------------
        remove_lava = rng.random() < no_lava_fraction

        if remove_lava:
            # ----------------------------------------------
            # No-lava layout
            # ----------------------------------------------
            lava_mask = np.zeros((size, size), dtype=bool)

            # Choose goal randomly inside grid (not boundary)
            candidates = [
                (y, x)
                for y in range(1, size - 1)
                for x in range(1, size - 1)
                if (y, x) != (1, 1)
            ]

            goal_yx = candidates[rng.integers(len(candidates))]
            layout_type = "no_lava"

        else:
            # ----------------------------------------------
            # Original layouts
            # ----------------------------------------------
            if i < n_envs // 2:
                lava_mask, goal_yx = generate_lava_wall_layout(size, rng)
                layout_type = "wall"
            else:
                lava_mask, goal_yx = generate_random_lava_layout(
                    size,
                    rng,
                    lava_frac=1/3,
                )
                layout_type = "random"

        # --------------------------------------------------
        # Build environment
        # --------------------------------------------------
        env = LavaWorldEnv(
            size=size,
            lava_mask=lava_mask,
            goal_yx=goal_yx,
            render_mode="human",
        )

        size_, wall_mask, lava_mask, lava_cells, goal_yx = build_static_maps(env)
        states = enumerate_states(size_, wall_mask)

        mdp = build_tabular_mdp(
            states,
            wall_mask,
            goal_yx,
            lava_mask,
            lava_cells,
            size_,
            gamma,
        )

        # --------------------------------------------------
        # Store
        # --------------------------------------------------
        envs.append(env)
        mdps.append(mdp)

        meta["lava_masks"].append(lava_mask)
        meta["goals"].append(goal_yx)
        meta["layout_type"].append(layout_type)
        meta["no_lava"].append(remove_lava)

    return envs, mdps, meta

# ======================================================
# Random Trajectory Rollout
# ======================================================

def rollout_random_trajectory(
    start_state,
    wall_mask,
    goal_yx,
    lava_mask,
    max_horizon=150,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    traj = []
    s = start_state

    for _ in range(max_horizon):

        if is_terminal_state(s, goal_yx, lava_mask):
            break

        a = rng.choice(ACTIONS)
        sp, done = step_model(s, a, wall_mask, goal_yx, lava_mask)

        traj.append((s, a, sp))
        s = sp

        if done:
            break

    return traj