import numpy as np
import os
import sys

# ----------------------------
# Project imports
# ----------------------------
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from agent.q_learning_agent_ import ValueIteration
from mdp.gridworld_env_layout import GridWorldMDPFromLayoutEnv
from utils import generate_random_gridworld_envs

# Import Atom definition
from utils import Atom

# BIRL module
from reward_learning.multi_env_atomic_birl import MultiEnvAtomicBIRL


# =====================================================
# FIXED: Construct demo atoms using 1-step trajectories
# =====================================================

def make_demo_atoms(env, Q):
    """
    Produce Atom(env_idx, 'demo', traj_array)
    where traj_array is shape (1, 2):
        [[s, a_opt]]
    This satisfies Numba's requirement in demo_ll_numba().
    """
    atoms = []
    env_idx = 0
    terminals = set(env.terminal_states or [])

    for s in range(env.get_num_states()):
        if s in terminals:
            continue

        a_opt = int(np.argmax(Q[s]))

        # IMPORTANT FIX:
        # Create a 1-step trajectory array (T=1, features=2)
        traj = np.array([[s, a_opt]], dtype=np.int64)

        atom = Atom(env_idx, "demo", traj)
        atoms.append((env_idx, atom))

    return atoms


# =====================================================
# REWARD RECOVERY EXPERIMENT
# =====================================================

def test_reward_recovery(
    mdp_size=8,
    feature_dim=5,
    seed=10200,
    mcmc_samples=4000,
    stepsize=0.5,
    beta=10.0,
):

    print("\n========================================================")
    print("TEST: Reward Recovery on Single Environment (Full Demos)")
    print("========================================================\n")

    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # 1. Sample true reward vector
    # --------------------------------------------------
    W_true = rng.normal(size=feature_dim)
    W_true /= np.linalg.norm(W_true)

    print("[INFO] W_true =", W_true, "\n")

    # --------------------------------------------------
    # 2. Create environment
    # --------------------------------------------------
    color_to_feature_map = {
        f"f{i}": [1 if j == i else 0 for j in range(feature_dim)]
        for i in range(feature_dim)
    }
    palette = list(color_to_feature_map.keys())
    p_color_range = {c: (0.3, 0.8) for c in palette}

    envs, _ = generate_random_gridworld_envs(
        n_envs=1,
        rows=mdp_size,
        cols=mdp_size,
        color_to_feature_map=color_to_feature_map,
        palette=palette,
        p_color_range=p_color_range,
        w_mode="fixed",
        W_fixed=W_true,
        gamma_range=(0.95, 0.99),
        noise_prob_range=(0.0, 0.0),
        terminal_policy=dict(kind="random_k", k_min=0, k_max=1, p_no_terminal=0.1),
        seed=seed,
        GridEnvClass=GridWorldMDPFromLayoutEnv,
    )

    env = envs[0]
    print("[INFO] Environment loaded with", env.get_num_states(), "states.\n")

    # --------------------------------------------------
    # 3. Compute optimal policies & Q*
    # --------------------------------------------------
    VI = ValueIteration(env, reward_convention="on")
    V_opt = VI.run_value_iteration(epsilon=1e-12)
    Q_opt = VI.get_q_values(V_opt)

    # --------------------------------------------------
    # 4. Build demonstration atoms (FIXED FORMAT)
    # --------------------------------------------------
    atoms_flat = make_demo_atoms(env, Q_opt)
    print(f"[INFO] Generated {len(atoms_flat)} demo atoms.\n")

    # --------------------------------------------------
    # 5. Run Bayesian IRL
    # --------------------------------------------------
    birl = MultiEnvAtomicBIRL(
        envs=envs,
        atoms_flat=atoms_flat,
        beta_demo=beta,
        beta_pairwise=beta,
        beta_estop=beta,
        beta_improvement=beta,
        epsilon=1e-4
    )

    print("[INFO] Running MCMC... (this may take a moment)\n")

    birl.run_mcmc(
        samples=mcmc_samples,
        stepsize=stepsize,
        normalize=True,
        adaptive=True
    )

    # --------------------------------------------------
    # 6. Retrieve results
    # --------------------------------------------------
    w_map = birl.get_map_solution()
    w_mean = birl.get_mean_solution(burn_frac=0.2, skip_rate=5)

    print("\n=========== RESULTS ===========\n")
    print("W_true =", W_true)
    print("\nw_map =", w_map)
    print("\nw_mean =", w_mean)

    print("\nL2 error (MAP)  =", np.linalg.norm(w_map - W_true))
    print("L2 error (mean) =", np.linalg.norm(w_mean - W_true))
    print("\nAcceptance rate =", birl.accept_rate)
    print("\n================================\n")

    return W_true, w_map, w_mean


# =====================================================
# RUN TEST
# =====================================================

if __name__ == "__main__":
    test_reward_recovery()
