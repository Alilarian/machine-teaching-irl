## TODO: 1. make half of the env radnom placement of lava - one third of cells

# lavaworld_generator.py
import numpy as np
from typing import List, Dict, Optional, Tuple

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Lava, Goal
from minigrid.core.mission import MissionSpace

# ======================================================
# Feature extraction (LINEAR reward)
# ======================================================
FEATURE_SET = "L1.3"   # or pass as argument later

W_MAP = {
    "L1.2": np.array([-0.05, -2.0, -0.01])/np.linalg.norm([-0.05, -2.0, -0.01]),            # [dist, on_lava, step]
    "L1.3": np.array([-0.8, -0.5, -5.0, -0.05])/np.linalg.norm([-0.8, -0.5, -9.0, -0.1]),       # [dist, lava_ahead, on_lava, step]
}

def manhattan(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])

def lava_ahead_state(lava_mask, y, x, direction):
    dx, dy = DIR_TO_VEC[direction]
    ny, nx = y + dy, x + dx
    if 0 <= ny < lava_mask.shape[0] and 0 <= nx < lava_mask.shape[1]:
        return int(lava_mask[ny, nx])
    return 0

def on_lava_state(lava_mask, y, x):
    return int(lava_mask[y, x])

def phi_from_state(state, goal_yx, lava_mask, size):
    y, x, direction = state
    gy, gx = goal_yx

    dist = manhattan((y, x), (gy, gx))
    step = 1.0

    if FEATURE_SET == "L1.2":
        return np.array([
            dist,
            on_lava_state(lava_mask, y, x),
            step,
        ], dtype=float)

    if FEATURE_SET == "L1.3":
        return np.array([
            dist,
            lava_ahead_state(lava_mask, y, x, direction),
            on_lava_state(lava_mask, y, x),
            step,
        ], dtype=float)

    raise ValueError(f"Unknown FEATURE_SET {FEATURE_SET}")

# ======================================================
# Utilities
# ======================================================

def l2_normalize(w, eps=1e-8):
    n = np.linalg.norm(w)
    return w if n < eps else w / n


# ======================================================
# Directions & Actions
# ======================================================

DIR_TO_VEC = {
    0: (1, 0),   # right
    1: (0, 1),   # down
    2: (-1, 0),  # left
    3: (0, -1),  # up
}

ACT_LEFT = 0
ACT_RIGHT = 1
ACT_FORWARD = 2
ACTIONS = [ACT_LEFT, ACT_RIGHT, ACT_FORWARD]


# ======================================================
# Simple LavaWorld Environment
# ======================================================

mission_space = MissionSpace(mission_func=lambda: "reach the goal")


class LavaWorldEnv(MiniGridEnv):
    """
    MiniGrid env with externally provided lava_mask and goal_yx.
    """

    def __init__(
        self,
        size: int,
        lava_mask: np.ndarray,
        goal_yx: Tuple[int, int],
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps=None,
        **kwargs,
    ):
        self.size = size
        self.lava_mask = lava_mask
        self.goal_yx = goal_yx
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

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
        self.agent_dir = self.agent_start_dir
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
    #print(state)
    if not isinstance(state, (tuple, list, np.ndarray)):
        raise TypeError(
            f"is_terminal_state expected (y,x,dir), got {state} of type {type(state)}"
        )

    if len(state) != 3:
        raise ValueError(f"State must be length 3, got {state}")
    y, x, _ = state
    return (y, x) == goal_yx or lava_mask[y, x]


def step_model(state, action, wall_mask, goal_yx, lava_mask):
    y, x, direction = state
    size = wall_mask.shape[0]

    if is_terminal_state(state, goal_yx, lava_mask):
        return state, True

    if action == ACT_LEFT:
        ns = (y, x, (direction - 1) % 4)
        return ns, is_terminal_state(ns, goal_yx, lava_mask)

    if action == ACT_RIGHT:
        ns = (y, x, (direction + 1) % 4)
        return ns, is_terminal_state(ns, goal_yx, lava_mask)

    if action == ACT_FORWARD:
        dx, dy = DIR_TO_VEC[direction]
        ny, nx = y + dy, x + dx

        if (
            ny < 0 or ny >= size or
            nx < 0 or nx >= size or
            wall_mask[ny, nx]
        ):
            ns = (y, x, direction)
        else:
            ns = (ny, nx, direction)

        return ns, is_terminal_state(ns, goal_yx, lava_mask)

    raise ValueError("Unknown action")


def enumerate_states(size, wall_mask):
    states = []
    for y in range(size):
        for x in range(size):
            if wall_mask[y, x]:
                continue
            for d in range(4):
                states.append((y, x, d))
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
        Phi[i] = phi_from_state(s, goal_yx, lava_mask, size)

        for a_idx, a in enumerate(ACTIONS):
            sp, _ = step_model(s, a, wall_mask, goal_yx, lava_mask)
            T[i, a_idx, idx_of[sp]] = 1.0

    return {
        "states": states,
        "idx_of": idx_of,
        "true_w":W_MAP["L1.3"], 
        "T": T,
        "Phi": Phi,              # ← linear features
        "terminal": terminal,
        "gamma": gamma,
        "goal_yx": goal_yx,
        "lava_mask": lava_mask,
        "lava_cells": lava_cells,
        "wall_mask": wall_mask,
        "size": size,
    }


# ======================================================
# Layout Generator
# ======================================================

def generate_lava_layout(size, rng):
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
# ======================================================
# MAIN ENTRY POINT
# ======================================================

def generate_lavaworld(
    n_envs: int,
    size: int,
    seed: Optional[int] = None,
    gamma: float = 0.99,
):
    """
    MAIN FUNCTION TO IMPORT.

    Returns:
        envs: list[LavaWorldEnv]
        mdps: list[dict]
        meta: dict
    """
    rng = np.random.default_rng(seed)

    envs = []
    mdps = []
    meta = {"lava_masks": [], "goals": [], "seed": seed}

    for _ in range(n_envs):
        lava_mask, goal_yx = generate_lava_layout(size, rng)

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

        envs.append(env)
        mdps.append(mdp)
        meta["lava_masks"].append(lava_mask)
        meta["goals"].append(goal_yx)

    return envs, mdps, meta

def rollout_random_trajectory(
    start_state,
    wall_mask,
    goal_yx,
    lava_mask,
    max_horizon=30,
    rng=None,
):
    """
    Roll out a random trajectory from a fixed start state.
    Returns list of (s, a, s_next).
    """
    if rng is None:
        rng = np.random.default_rng()

    traj = []
    s = start_state

    for _ in range(max_horizon):
        #print(type(s))
        if is_terminal_state(s, goal_yx, lava_mask):
            break

        a = rng.choice(ACTIONS)
        sp, done = step_model(s, a, wall_mask, goal_yx, lava_mask)

        traj.append((s, a, sp))
        s = sp

        if done:
            break

    return traj