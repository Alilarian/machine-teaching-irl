import os
import sys

# ----------------------------
# Project imports
# ----------------------------
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)
"""
CLEAR TESTS for generate_feedback_parallel.py

Covers:
  - Parallel environment execution
  - Pairwise preference reward ordering
  - E-stop prefix reward logic
  - Improvement SAME-START constraint
  - Index-based atom payload correctness
  - Non-empty / deterministic structure

Run:
    pytest -s test_generate_feedback_parallel.py
or:
    python test_generate_feedback_parallel.py
"""

import numpy as np

from utils.generate_feedback_parallel import (
    generate_candidate_atoms_for_scot_parallel,
    evaluate_trajectory,
)

from utils import parallel_value_iteration

from mdp.gridworld_env_layout import GridWorldMDPFromLayoutEnv
from utils import generate_random_gridworld_envs

# ============================================================
# Helpers
# ============================================================

def ok(msg):
    print(f"✔ {msg}")

def fail(msg):
    raise AssertionError(f"❌ {msg}")


# ============================================================
# Test
# ============================================================

def test_parallel_feedback_generation():
    print("\n--- Running PARALLEL feedback generation tests ---")

    # --------------------------------------------------------
    # 1. Build environments ONCE (for Q computation)
    # --------------------------------------------------------
    n_envs = 4
    mdp_size = 4
    feature_dim = 2
    seed = 0

    color_to_feature_map = {
        "f0": [1, 0],
        "f1": [0, 1],
    }

    envs, _ = generate_random_gridworld_envs(
        n_envs=n_envs,
        rows=mdp_size,
        cols=mdp_size,
        color_to_feature_map=color_to_feature_map,
        palette=list(color_to_feature_map.keys()),
        p_color_range={"f0": (0.5, 0.5), "f1": (0.5, 0.5)},
        terminal_policy=dict(kind="random_k", k_min=1, k_max=1, p_no_terminal=0.0),
        gamma_range=(0.99, 0.99),
        noise_prob_range=(0.0, 0.0),
        w_mode="fixed",
        W_fixed=np.array([1.0, -1.0]),
        seed=seed,
        GridEnvClass=GridWorldMDPFromLayoutEnv,
    )

    # --------------------------------------------------------
    # 2. Value Iteration
    # --------------------------------------------------------
    Q_list = parallel_value_iteration(
        envs,
        epsilon=1e-10,
        n_jobs=1,
        log=lambda _: None,
    )

    ok("Value iteration completed")

    # --------------------------------------------------------
    # 3. env_builder for subprocesses
    # --------------------------------------------------------
    def env_builder(env_idx):
        # recreate env deterministically
        return envs[env_idx]

    # --------------------------------------------------------
    # 4. Parallel feedback generation
    # --------------------------------------------------------
    candidates = generate_candidate_atoms_for_scot_parallel(
        env_builder,
        Q_list,
        max_workers=2,
        seed=123,
        use_q_demos=True,
        use_pairwise=True,
        use_estop=True,
        use_improvement=True,
        n_pairwise=50,
        n_estops=50,
        n_improvements=20,
        base_pool_size=100,
    )

    if len(candidates) != n_envs:
        fail("Wrong number of env outputs")

    if any(len(a) == 0 for a in candidates):
        fail("Some envs produced zero atoms")

    ok("Parallel atom generation")

    # --------------------------------------------------------
    # 5. Semantic checks (INDEX-BASED)
    # --------------------------------------------------------
    for env_idx, atoms in enumerate(candidates):
        env = envs[env_idx]

        # reconstruct base pool for checking
        base_trajs = []
        for atom in atoms:
            if atom.feedback_type == "demo":
                base_trajs.append(atom.data)

        # Use all base trajs seen in pairwise / estop / improvement
        all_trajs = []
        for atom in atoms:
            if atom.feedback_type in {"pairwise", "estop"}:
                i = atom.data[0]
                if i < len(base_trajs):
                    all_trajs.append(base_trajs[i])

        for atom in atoms:

            # -------------------------
            # Pairwise preference
            # -------------------------
            if atom.feedback_type == "pairwise":
                i, j = atom.data
                ti = base_trajs[i]
                tj = base_trajs[j]

                ri = evaluate_trajectory(env, ti)
                rj = evaluate_trajectory(env, tj)

                if ri <= rj:
                    fail(
                        f"Pairwise violated in env {env_idx}: "
                        f"ri={ri}, rj={rj}"
                    )

            # -------------------------
            # E-stop semantics
            # -------------------------
            if atom.feedback_type == "estop":
                i, t_stop = atom.data
                traj = base_trajs[i]

                if not (0 <= t_stop < len(traj)):
                    fail("E-stop index out of bounds")

                prefix_r = sum(env.compute_reward(s) for s, _ in traj[: t_stop + 1])
                full_r = evaluate_trajectory(env, traj)

                if prefix_r > full_r + 1e-8:
                    fail("E-stop prefix reward exceeds full reward")

            # -------------------------
            # Improvement semantics
            # -------------------------
            if atom.feedback_type == "improvement":
                better, base = atom.data

                if better[0][0] != base[0][0]:
                    fail("Improvement violates SAME START STATE")

                if evaluate_trajectory(env, better) < evaluate_trajectory(env, base):
                    fail("Improvement trajectory is worse than base")

    ok("Pairwise / E-stop / Improvement semantics")

    # --------------------------------------------------------
    # Final PASS banner
    # --------------------------------------------------------
    print("\n==============================================")
    print("✅ ALL PARALLEL FEEDBACK GENERATION TESTS PASSED")
    print("==============================================\n")


# ------------------------------------------------------------
# Allow running as script
# ------------------------------------------------------------
if __name__ == "__main__":
    test_parallel_feedback_generation()
