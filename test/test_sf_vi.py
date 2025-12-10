import numpy as np
import os
import sys

# --------------------------------------------------------
# Load your project modules
# --------------------------------------------------------

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from agent.q_learning_agent_ import ValueIteration
from utils.successor_features import (
    compute_successor_features_iterative_from_q,
    build_Pi_from_q,
)
from utils import generate_random_gridworld_envs
from mdp.gridworld_env_layout import GridWorldMDPFromLayoutEnv


# -------------------------------------------------------------------
# Helper: compute Q from successor features
# -------------------------------------------------------------------

def q_from_sf(mu_sa, w):
    """
    μ_sa: (S, A, d)
    w: (d,)
    Returns Q_sf(s,a) = μ_sa[s,a] @ w
    """
    return np.tensordot(mu_sa, w, axes=(2, 0))


# -------------------------------------------------------------------
# Test function
# -------------------------------------------------------------------

def test_vi_vs_sf(
    n_envs=3,
    mdp_size=6,
    feature_dim=4,
    seed=10,
    convention="entering",
    zero_terminal_features=True,
):
    print("\n======================================================")
    print(" TEST: ValueIteration Q vs Successor-Feature Q")
    print("======================================================\n")

    # --------------------------------------------------------
    # 1. Generate random environments
    # --------------------------------------------------------
    print("Generating environments...")

    color_to_feature_map = {
        f"f{i}": [1 if j == i else 0 for j in range(feature_dim)]
        for i in range(feature_dim)
    }
    palette = list(color_to_feature_map.keys())
    p_color_range = {c: (0.3, 0.8) for c in palette}

    # Create environments matching your main script
    W_true = np.random.randn(feature_dim)
    W_true /= np.linalg.norm(W_true)

    envs, _ = generate_random_gridworld_envs(
        n_envs=n_envs,
        rows=mdp_size,
        cols=mdp_size,
        color_to_feature_map=color_to_feature_map,
        palette=palette,
        p_color_range=p_color_range,
        gamma_range=(0.98, 0.995),
        noise_prob_range=(0.0, 0.0),
        w_mode="fixed",
        W_fixed=W_true,
        terminal_policy=dict(kind="random_k", k_min=0, k_max=1, p_no_terminal=0.1),
        seed=seed,
        GridEnvClass=GridWorldMDPFromLayoutEnv,
    )

    # --------------------------------------------------------
    # 2. For each environment run VI and SF and compare Q-values
    # --------------------------------------------------------

    for idx, env in enumerate(envs):
        print(f"\n--- Environment {idx} ---")
        print(f"   States = {env.get_num_states()}, Actions = {env.get_num_actions()}")
        print(f"   Convention = {convention}, zero_terminal_features={zero_terminal_features}")

        # ----------------------------------------------------
        # A) Compute Q-values using Value Iteration
        # ----------------------------------------------------
        vi = ValueIteration(env, reward_convention=convention)
        V_star = vi.run_value_iteration(epsilon=1e-12)
        Q_vi = vi.get_q_values(V_star)

        # ----------------------------------------------------
        # B) Compute successor features
        # ----------------------------------------------------
        mu_sa, mu_s, Phi, P_pi = compute_successor_features_iterative_from_q(
            env,
            Q_vi,
            convention=convention,
            zero_terminal_features=zero_terminal_features,
            tol=1e-12,
            max_iters=20000,
        )

        # ----------------------------------------------------
        # C) Compute Q using successor features
        # ----------------------------------------------------
        Q_sf = q_from_sf(mu_sa, env.feature_weights)

        # ----------------------------------------------------
        # D) Compare
        # ----------------------------------------------------
        diff = np.max(np.abs(Q_vi - Q_sf))

        print(f"   Max |Q_vi - Q_sf| = {diff:.12f}")

        if diff < 1e-6:
            print("   ✔ PASS: Successor-feature Q matches ValueIteration Q")
        else:
            print("   ❌ FAIL: MISMATCH DETECTED")
            print(f"   (Investigate transitions, terminal feature handling, or tie-breaking)")

        # Print some details
        print("   Example Q_vi[0] =", Q_vi[0])
        print("   Example Q_sf[0] =", Q_sf[0])

    print("\n======================================================")
    print(" TEST COMPLETE")
    print("======================================================\n")


# -------------------------------------------------------------------
# Run the test
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Test both reward conventions
    test_vi_vs_sf(convention="entering", zero_terminal_features=True)
    test_vi_vs_sf(convention="on", zero_terminal_features=True)

    # Also test without zero-terminal-feature
    test_vi_vs_sf(convention="entering", zero_terminal_features=False)
