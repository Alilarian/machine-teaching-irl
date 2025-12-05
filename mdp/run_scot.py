import argparse
import json
import os
import time
import numpy as np

import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

# ----------------------------
# Import your project's modules
# ----------------------------
from agent.q_learning_agent import ValueIteration
from utils.common_helper import calculate_expected_value_difference
from utils.successor_features import build_Pi_from_q
from utils import (
    generate_random_gridworld_envs,
    compute_successor_features_family,
    derive_constraints_from_q_family,
    derive_constraints_from_atoms,
    generate_candidate_atoms_for_scot,
    sample_random_atoms_like_scot,
    compute_Q_from_weights_with_VI,
    remove_redundant_constraints,
)
from teaching import scot_greedy_family_atoms_tracked
from reward_learning.multi_env_atomic_birl import MultiEnvAtomicBIRL

from gridworld_env_layout import GridWorldMDPFromLayoutEnv
from gridworld_env import NoisyLinearRewardFeaturizedGridWorldEnv
import numpy as np
from agent.q_learning_agent import ValueIteration, PolicyEvaluation
from scipy.optimize import linprog
from utils import generate_random_gridworld_envs

from utils import simulate_all_feedback
from utils import (
    compute_successor_features_family,
    derive_constraints_from_q_family,
    derive_constraints_from_atoms,
    generate_candidate_atoms_for_scot,
    sample_random_atoms_like_scot,
    compute_Q_from_weights_with_VI,
    regrets_from_Q,
    atom_to_constraints,
)

from teaching import scot_greedy_family_atoms_tracked

from reward_learning.multi_env_atomic_birl import MultiEnvAtomicBIRL


# =============================================================================
# 1. Ground-truth reward generator
# =============================================================================

def generate_w_true(d, mode="random_signed", seed=None):
    rng = np.random.default_rng(seed)

    if mode == "random_signed":
        w = rng.normal(size=d)
        return w / np.linalg.norm(w)

    elif mode == "one_hot":
        w = np.zeros(d)
        idx_pos = rng.integers(0, d)
        idx_neg = (idx_pos + 1) % d
        w[idx_pos] = 1
        w[idx_neg] = -1
        return w / np.linalg.norm(w)

    elif mode == "biased":
        w = rng.normal(size=d)
        w[0] += 4.0
        return w / np.linalg.norm(w)

    else:
        raise ValueError(f"Unknown W_TRUE generation mode: {mode}")


# =============================================================================
# 2. BIRL Wrapper
# =============================================================================

def birl_atomic_to_Q_lists(envs, atoms_flat, *, 
                           beta=5.0, epsilon=1e-4,
                           samples=2000, stepsize=0.1,
                           normalize=True, adaptive=True,
                           burn_frac=0.2, skip_rate=10,
                           vi_epsilon=1e-6):

    birl = MultiEnvAtomicBIRL(
        envs,
        atoms_flat,
        beta_demo=beta,
        beta_pairwise=beta,
        beta_estop=beta,
        beta_improvement=beta,
        epsilon=epsilon,
    )

    birl.run_mcmc(
        samples=samples,
        stepsize=stepsize,
        normalize=normalize,
        adaptive=adaptive,
    )

    w_map = birl.get_map_solution()
    w_mean = birl.get_mean_solution(
        burn_frac=burn_frac,
        skip_rate=skip_rate
    )


    ## we can parallelize
    Q_map_list = [
        compute_Q_from_weights_with_VI(env, w_map, vi_epsilon=vi_epsilon)
        for env in envs
    ]

    Q_mean_list = [
        compute_Q_from_weights_with_VI(env, w_mean, vi_epsilon=vi_epsilon)
        for env in envs
    ]

    return w_map, w_mean, Q_map_list, Q_mean_list, birl


# =============================================================================
# 3. Regret Utilities
# =============================================================================

def regrets_from_Q(envs, Q_list, *, tie_eps=1e-10,
                   epsilon=1e-4, normalize_with_random_policy=False):
    regrets = []
    for env, Q in zip(envs, Q_list):
        pi = build_Pi_from_q(env, Q, tie_eps=tie_eps)
        r = calculate_expected_value_difference(
            env=env,
            eval_policy=pi,
            epsilon=epsilon,
            normalize_with_random_policy=normalize_with_random_policy,
        )
        regrets.append(float(r))
    return np.array(regrets)


# =============================================================================
# 4. Combined SCOT vs Random regret analyzer
# =============================================================================

def run_scot_vs_random_Q_regret_atomic(
    envs,
    chosen_scot,
    make_random_chosen,
    *,
    n_random_trials=10,
    birl_kwargs=None,
    vi_epsilon=1e-6,
    regret_epsilon=1e-4,
):
    birl_kwargs = birl_kwargs or {}

    # ---------------- SCOT ----------------
    w_scot_map, w_scot_mean, Q_scot_map, Q_scot_mean, birl_scot = \
        birl_atomic_to_Q_lists(
            envs,
            chosen_scot,
            vi_epsilon=vi_epsilon,
            **birl_kwargs,
        )

    reg_scot_map = regrets_from_Q(envs, Q_scot_map, epsilon=regret_epsilon)
    reg_scot_mean = regrets_from_Q(envs, Q_scot_mean, epsilon=regret_epsilon)

    # ---------------- RANDOM ----------------
    rand_map_regs = []
    rand_mean_regs = []

    for sd in range(n_random_trials):
        chosen_rand = make_random_chosen(sd)

        _, _, Q_rand_map, Q_rand_mean, _ = birl_atomic_to_Q_lists(
            envs,
            chosen_rand,
            vi_epsilon=vi_epsilon,
            **birl_kwargs
        )

        rand_map_regs.append(
            regrets_from_Q(envs, Q_rand_map, epsilon=regret_epsilon)
        )
        rand_mean_regs.append(
            regrets_from_Q(envs, Q_rand_mean, epsilon=regret_epsilon)
        )

    return {
        "SCOT": {
            "regret_map": reg_scot_map.tolist(),
            "regret_mean": reg_scot_mean.tolist(),
        },
        "RANDOM": {
            "regret_map": np.vstack(rand_map_regs).tolist(),
            "regret_mean": np.vstack(rand_mean_regs).tolist(),
        },
        "BIRL": {
            "SCOT_accept_rate": float(birl_scot.accept_rate),
        },
    }



def run_universal_experiment(
    *,
    n_envs,
    mdp_size,
    feature_dim,
    w_true_mode,
    feedback_demos,
    feedback_pairwise,
    feedback_estop,
    feedback_improvement,
    feedback_count,
    random_trials,
    seed,
    result_dir,
    **birl_kwargs
):
    """
    Runs the full pipeline and prints detailed logs of every stage.
    """

    print("\n======================================================")
    print(" UNIVERSAL ATOMIC SCOT EXPERIMENT — STARTING")
    print("======================================================\n")

    start_all = time.time()

    # ---------------------------------------------------
    # Create unique results directory
    # ---------------------------------------------------
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_name = (
        f"exp_env{n_envs}_size{mdp_size}_fd{feature_dim}_"
        f"fb{feedback_count}_{timestamp}"
    )
    out_dir = os.path.join(result_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INIT] Results will be stored in:\n       {out_dir}\n")

    # ---------------------------------------------------
    # 1. Generate W_TRUE
    # ---------------------------------------------------
    print("[1/12] Generating ground-truth reward W_TRUE...")
    W_TRUE = generate_w_true(feature_dim, mode=w_true_mode, seed=seed)
    print(f"       W_TRUE = {W_TRUE}\n")

    # ---------------------------------------------------
    # 2. Generate random MDPs
    # ---------------------------------------------------
    print("[2/12] Generating random GridWorld environments...")
    t0 = time.time()

    # Build feature map
    color_to_feature_map = {
        f"f{i}": [1 if j == i else 0 for j in range(feature_dim)]
        for i in range(feature_dim)
    }
    palette = list(color_to_feature_map.keys())
    p_color_range = {c: (0.3, 0.8) for c in palette}

    print(f"       → MDP size: {mdp_size}x{mdp_size}")
    print(f"       → Feature dim: {feature_dim}")
    print(f"       → Colors/features: {palette}")
    print(f"       → True reward W = {W_TRUE}")

    envs, _ = generate_random_gridworld_envs(
        n_envs=n_envs,
        rows=mdp_size,
        cols=mdp_size,
        color_to_feature_map=color_to_feature_map,
        palette=palette,
        p_color_range=p_color_range,
        terminal_policy=dict(kind="random_k", k_min=0, k_max=1, p_no_terminal=0.1),
        gamma_range=(0.98, 0.995),
        noise_prob_range=(0.0, 0.0),
        w_mode="fixed",
        W_fixed=W_TRUE,
        seed=seed,
        GridEnvClass=GridWorldMDPFromLayoutEnv  # ← REQUIRED FIX
    )

    print(f"       ✔ Generated {n_envs} environments in {time.time() - t0:.2f}s\n")

    # ---------------------------------------------------
    # 3. Value Iteration
    # ---------------------------------------------------
    print("[3/12] Running Value Iteration on all MDPs...")
    t0 = time.time()
    vis = [ValueIteration(e) for e in envs]
    for i, v in enumerate(vis):
        v.run_value_iteration(epsilon=1e-10)
        if (i+1) % max(1, n_envs//5) == 0:
            print(f"       VI progress: {i+1}/{n_envs} MDPs solved...")
    Q_list = [v.get_q_values() for v in vis]
    print(f"       ✔ VI completed in {time.time() - t0:.2f}s\n")

    # ---------------------------------------------------
    # 4. Successor features
    # ---------------------------------------------------
    print("[4/12] Computing successor features for each MDP...")
    t0 = time.time()
    SFs = compute_successor_features_family(
        envs, Q_list,
        convention="entering",
        zero_terminal_features=True,
        tol=1e-10, max_iters=10000
    )
    print(f"       ✔ Successor features computed "
          f"in {time.time() - t0:.2f}s\n")

    # ---------------------------------------------------
    # 5. Q-based constraints
    # ---------------------------------------------------
    print("[5/12] Deriving Q-based constraints...")
    t0 = time.time()
    _, U_q_global = derive_constraints_from_q_family(
        SFs, Q_list, envs,
        skip_terminals=True,
        normalize=True
    )
    print(f"       ✔ Q-only global constraints: {len(U_q_global)} "
          f"in {time.time() - t0:.2f}s\n")

    # ---------------------------------------------------
    # 6. Generate feedback atoms
    # ---------------------------------------------------
    print("[6/12] Generating candidate atoms for SCOT...")
    t0 = time.time()
    candidates_per_env = generate_candidate_atoms_for_scot(
        envs,
        Q_list,
        use_q_demos=feedback_demos,
        use_pairwise=feedback_pairwise,
        use_estop=feedback_estop,
        use_improvement=feedback_improvement,
        n_pairwise=feedback_count,
        n_estops=feedback_count,
        n_improvements=feedback_count,
    )
    atom_counts = [len(a) for a in candidates_per_env]
    print(f"       ✔ Atoms per env (min/mean/max): "
          f"{min(atom_counts)}/{np.mean(atom_counts):.1f}/{max(atom_counts)} "
          f"in {time.time() - t0:.2f}s\n")

    # Flatten
    atoms_flat = [(i, atom)
                  for i, atoms in enumerate(candidates_per_env)
                  for atom in atoms]

    # ---------------------------------------------------
    # 7. Atom constraints
    # ---------------------------------------------------
    print("[7/12] Deriving constraint vectors from atoms...")
    t0 = time.time()
    _, U_atoms_global = derive_constraints_from_atoms(
        candidates_per_env, SFs, envs
    )
    print(f"       ✔ Atom-derived constraints: {len(U_atoms_global)} "
          f"in {time.time() - t0:.2f}s\n")

    # ---------------------------------------------------
    # 8. Universal constraint merging
    # ---------------------------------------------------
    print("[8/12] Merging Q-based + Atom constraints into Universal set...")
    t0 = time.time()

    compounded = []
    if len(U_q_global) > 0:
        compounded.append(U_q_global)
    if len(U_atoms_global) > 0:
        compounded.append(U_atoms_global)

    if compounded:
        U_universal = remove_redundant_constraints(
            np.vstack(compounded),
            epsilon=1e-4
        )
    else:
        U_universal = np.zeros((0, feature_dim))

    print(f"       ✔ Universal constraint set size: "
          f"{len(U_universal)} "
          f"(reduced from {sum(len(x) for x in compounded)}) "
          f"in {time.time() - t0:.2f}s\n")

    # ---------------------------------------------------
    # 9. SCOT greedy selection
    # ---------------------------------------------------
    print("[9/12] Running SCOT greedy selection...")
    t0 = time.time()
    chosen_scot, scot_stats = scot_greedy_family_atoms_tracked(
        U_universal,
        candidates_per_env,
        SFs,
        envs,
    )
    print(f"       ✔ SCOT selected {len(chosen_scot)} atoms "
          f"in {time.time() - t0:.2f}s")
    print(f"       SCOT env coverage summary (per-env #selected):")
    for env_idx, v in scot_stats.items():
        print(f"         env {env_idx:02d}: {len(v['atoms'])} atoms")

    chosen_scot_flat = [(i, atom) for (i, atom) in chosen_scot]
    print()

    # ---------------------------------------------------
    # 10. Regret evaluation
    # ---------------------------------------------------
    print("[10/12] Computing regret (SCOT vs Random)...")
    t0 = time.time()
    make_random = lambda sd: sample_random_atoms_like_scot(
        candidates_per_env, chosen_scot, seed=sd)

    results = run_scot_vs_random_Q_regret_atomic(
        envs,
        chosen_scot_flat,
        make_random,
        n_random_trials=random_trials,
        birl_kwargs=birl_kwargs,
    )

    print(f"       ✔ Regret computed in {time.time() - t0:.2f}s")
    print(f"       SCOT mean regret: {np.mean(results['SCOT']['regret_map']):.4f}")
    print(f"       Random mean regret: "
          f"{np.mean(results['RANDOM']['regret_map']):.4f}\n")

    # ---------------------------------------------------
    # Save result file
    # ---------------------------------------------------
    print("[11/12] Saving results to JSON...")
    res ults["metadata"] = {
        "W_TRUE": W_TRUE.tolist(),
        "experiment_dir": out_dir,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"       ✔ Saved as {out_dir}/results.json")

    # ---------------------------------------------------
    # Final Stats
    # ---------------------------------------------------
    print("\n[12/12] EXPERIMENT COMPLETED")
    print("------------------------------------------------------")
    print(f"Total runtime: {time.time() - start_all:.2f} seconds")
    print("Results directory:", out_dir)
    print("======================================================\n")

    return results, out_dir

# ============================================================
# CLI ENTRY POINT (argparse)
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run universal SCOT vs Random atomic IRL experiment."
    )

    # -----------------------------------------------------------
    # Environment / MDP parameters
    # -----------------------------------------------------------
    parser.add_argument("--n_envs", type=int, default=30,
                        help="Number of MDP environments to generate.")
    parser.add_argument("--mdp_size", type=int, default=5,
                        help="Gridworld size (rows = cols = mdp_size).")
    parser.add_argument("--feature_dim", type=int, default=2,
                        help="Number of reward features.")

    parser.add_argument("--w_true_mode", type=str, default="random_signed",
                        choices=["random_signed", "one_hot", "biased"],
                        help="How to generate ground-truth reward weights.")

    # -----------------------------------------------------------
    # Feedback settings
    # -----------------------------------------------------------
    parser.add_argument(
        "--feedback",
        nargs="+",
        default=["demo", "pairwise", "estop", "improvement"],
        help="Feedback types to enable. Options: demo pairwise estop improvement"
    )

    parser.add_argument("--feedback_count", type=int, default=50,
                        help="Atoms per feedback type per environment.")

    # -----------------------------------------------------------
    # Random baseline (for regret comparison)
    # -----------------------------------------------------------
    parser.add_argument("--random_trials", type=int, default=10,
                        help="How many random baseline seeds to try.")

    # -----------------------------------------------------------
    # BIRL (MCMC) settings
    # -----------------------------------------------------------
    parser.add_argument("--samples", type=int, default=5000,
                        help="Number of MCMC samples.")
    parser.add_argument("--stepsize", type=float, default=0.1,
                        help="Proposal stepsize in MCMC.")
    parser.add_argument("--beta", type=float, default=10.0,
                        help="Inverse temperature for likelihood.")

    # -----------------------------------------------------------
    # General settings
    # -----------------------------------------------------------
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility.")
    parser.add_argument("--result_dir", type=str, default="results_universal",
                        help="Parent directory to store experiment results.")

    args = parser.parse_args()

    # -----------------------------------------
    # Convert feedback list → booleans
    # -----------------------------------------
    fb_list = args.feedback
    fb_demo        = ("demo" in fb_list)
    fb_pairwise    = ("pairwise" in fb_list)
    fb_estop       = ("estop" in fb_list)
    fb_improvement = ("improvement" in fb_list)

    # -----------------------------------------
    # BIRL kwargs
    # -----------------------------------------
    birl_kwargs = dict(
        beta=args.beta,
        samples=args.samples,
        stepsize=args.stepsize,
        normalize=True,
        adaptive=True,
        burn_frac=0.2,
        skip_rate=10,
    )

    # -----------------------------------------
    # Run the experiment
    # -----------------------------------------
    run_universal_experiment(
        n_envs=args.n_envs,
        mdp_size=args.mdp_size,
        feature_dim=args.feature_dim,
        w_true_mode=args.w_true_mode,
        feedback_demos=fb_demo,
        feedback_pairwise=fb_pairwise,
        feedback_estop=fb_estop,
        feedback_improvement=fb_improvement,
        feedback_count=args.feedback_count,
        random_trials=args.random_trials,
        seed=args.seed,
        result_dir=args.result_dir,
        **birl_kwargs
    )