# =============================================================================
# Two-Stage SCOT vs Random (GLOBAL POOL) — FULL EXPERIMENT
# =============================================================================

import argparse
import json
import os
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)


from minigrid.core.world_object import Goal, Wall, Lava
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace

# test_lavaworld_manual.py
from utils import (generate_lavaworld,
                   policy_evaluation_next_state_multi, 
                   value_iteration_next_state_multi,
                   compute_successor_features_multi)





# =============================================================================
# Ground-truth reward generator
# =============================================================================



# =============================================================================
# Regret across envs
# =============================================================================


# =============================================================================
# RANDOM BASELINE — GLOBAL ATOM POOL
# =============================================================================
