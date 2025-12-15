import argparse
import json
import os
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor


import numpy as np

def extract_optimal_actions_from_Q(Q, tie_eps=1e-12):
    """
    For each state s, return a list of all optimal actions.

    Args:
        Q: np.ndarray of shape (S, A)
        tie_eps: numerical tolerance for ties

    Returns:
        optimal_actions: list of lists, optimal_actions[s] = [a1, a2, ...]
    """
    optimal_actions = []
    for s in range(Q.shape[0]):
        row = Q[s]
        max_q = np.max(row)
        acts = np.where(np.abs(row - max_q) <= tie_eps)[0].tolist()
        optimal_actions.append(acts)
    return optimal_actions

def print_policy_from_Q(env, Q, tie_eps=1e-12):
    """
    Print optimal policy for a single GridWorld env from Q-values.
    Shows ALL optimal actions per state.
    """
    arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    optimal_actions = extract_optimal_actions_from_Q(Q, tie_eps)

    print("\n========== Optimal Policy ==========")
    for r in range(env.rows):
        row_syms = []
        for c in range(env.columns):
            s = r * env.columns + c

            if s in env.terminal_states:
                row_syms.append("  T  ")
                continue

            acts = optimal_actions[s]
            sym = "".join(arrows[a] for a in acts)
            row_syms.append(f"{sym:^5}")
        print("".join(row_syms))
    print("===================================\n")

    return optimal_actions

def print_all_env_policies(envs, Q_list, tie_eps=1e-12):
    """
    Print optimal policy for each env using precomputed Q-values.
    """
    for i, (env, Q) in enumerate(zip(envs, Q_list)):
        print(f"\n########## ENV {i} ##########")
        print_policy_from_Q(env, Q, tie_eps)

# ---------------------------------------------------------------------
# 0. Path & imports
# ---------------------------------------------------------------------
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from agent.q_learning_agent_ import ValueIteration
from utils.common_helper import calculate_expected_value_difference
from utils.successor_features import build_Pi_from_q, max_q_sa_pairs
from utils import (
    generate_random_gridworld_envs,
    compute_successor_features_family,
    derive_constraints_from_q_family,
    derive_constraints_from_atoms,
    generate_candidate_atoms_for_scot,
    sample_random_atoms_like_scot,
    compute_Q_from_weights_with_VI,
    remove_redundant_constraints,
    parallel_value_iteration,
)
from teaching import scot_greedy_family_atoms_tracked
from reward_learning.multi_env_atomic_birl import MultiEnvAtomicBIRL
from gridworld_env_layout import GridWorldMDPFromLayoutEnv


# =============================================================================
# 1. Ground-truth reward generator
# =============================================================================
def generate_w_true(d, mode="random_signed", seed=None):
    rng = np.random.default_rng(seed)

    if mode == "random_signed":
        w = rng.normal(size=d)
        return w / np.linalg.norm(w)

    if mode == "one_hot":
        w = np.zeros(d)
        idx_pos = rng.integers(0, d)
        idx_neg = (idx_pos + 1) % d
        w[idx_pos] = 1
        w[idx_neg] = -1
        return w / np.linalg.norm(w)

    if mode == "biased":
        w = rng.normal(size=d)
        w[0] += 4.0
        return w / np.linalg.norm(w)

    raise ValueError(f"Unknown W_TRUE generation mode: {mode}")


# =============================================================================
# 2. BIRL Wrapper
# =============================================================================
def _compute_Q_wrapper(args):
    """Helper for parallel execution: (env, w, vi_epsilon)."""
    env, w, vi_eps = args
    return compute_Q_from_weights_with_VI(env, w, vi_epsilon=vi_eps)


def birl_atomic_to_Q_lists(
    envs,
    atoms_flat,
    *,
    beta=5.0,
    epsilon=1e-4,
    samples=2000,
    stepsize=0.1,
    normalize=True,
    adaptive=False,
    burn_frac=0.2,
    skip_rate=10,
    vi_epsilon=1e-6,
    n_jobs=None,  # number of parallel workers for Q computation
):
    """
    Run MultiEnvAtomicBIRL → return MAP/mean Q-functions.
    Parallel Q-computation for each environment.
    """

    # ---------------------------
    # Run BIRL (sequential)
    # ---------------------------
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
        adaptive=False,
    )

    # Extract MAP and mean weights
    w_map = birl.get_map_solution()
    w_mean = birl.get_mean_solution(
        burn_frac=burn_frac,
        skip_rate=skip_rate,
    )

    # ---------------------------
    # Parallel Q computation
    # ---------------------------
    worker_args_map = [(env, w_map, vi_epsilon) for env in envs]
    worker_args_mean = [(env, w_mean, vi_epsilon) for env in envs]

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        Q_map_list = list(executor.map(_compute_Q_wrapper, worker_args_map))
        Q_mean_list = list(executor.map(_compute_Q_wrapper, worker_args_mean))

    return w_map, w_mean, Q_map_list, Q_mean_list, birl


# =============================================================================
# 3. Regret Utilities
# =============================================================================
def _compute_regret_wrapper(args):
    env, Q, tie_eps, epsilon, normalize = args
    #pi = build_Pi_from_q(env, Q, tie_eps=tie_eps) ## This one returns probabilsitic things
    pi = max_q_sa_pairs(env, Q,)
    print("Pi insided compute regret wrapper")
    print(pi)
    r = calculate_expected_value_difference(
        env=env,
        eval_policy=pi,
        epsilon=epsilon,
        normalize_with_random_policy=normalize,
    )
    return float(r)


def regrets_from_Q(
    envs,
    Q_list,
    *,
    tie_eps=1e-10,
    epsilon=1e-4,
    normalize_with_random_policy=False,
    n_jobs=None,
):
    """Parallel regret computation for each env."""
    worker_args = [
        (env, Q, tie_eps, epsilon, normalize_with_random_policy)
        for env, Q in zip(envs, Q_list)
    ]

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        regrets = list(executor.map(_compute_regret_wrapper, worker_args))

    return np.array(regrets)


# =============================================================================
# 4. Random baseline helpers (picklable)
# =============================================================================
def make_random_chosen(sd, candidates_per_env, chosen_scot):
    """Picklable version of random atom generator."""
    return sample_random_atoms_like_scot(
        candidates_per_env=candidates_per_env,
        chosen_scot=chosen_scot,
        seed=sd,
    )


def _random_trial_worker(args):
    (
        sd,
        envs,
        make_random_args,  # (candidates_per_env, chosen_scot_nonflat)
        birl_kwargs,
        vi_epsilon,
        regret_epsilon,
    ) = args

    candidates_per_env, chosen_scot_nonflat = make_random_args

    # Generate random atoms inside worker
    chosen_rand = make_random_chosen(
        sd,
        candidates_per_env=candidates_per_env,
        chosen_scot=chosen_scot_nonflat,
    )

    # BIRL → Qs
    _, _, Q_rand_map, Q_rand_mean, _ = birl_atomic_to_Q_lists(
        envs,
        chosen_rand,
        vi_epsilon=vi_epsilon,
        **birl_kwargs,
    )

    # Regrets
    reg_map = regrets_from_Q(envs, Q_rand_map, epsilon=regret_epsilon)
    reg_mean = regrets_from_Q(envs, Q_rand_mean, epsilon=regret_epsilon)

    return reg_map, reg_mean


# =============================================================================
# 5. Option B: Split evaluation into SCOT-only and Random-only
# =============================================================================
def eval_scot_regret_atomic(
    envs,
    chosen_scot_flat,
    *,
    birl_kwargs=None,
    vi_epsilon=1e-6,
    regret_epsilon=1e-4,
    n_jobs=None,
):
    """
    Evaluate SCOT only:
      - run BIRL on chosen_scot_flat
      - compute regrets for MAP/mean policies per env
    """
    birl_kwargs = birl_kwargs or {}

    w_scot_map, w_scot_mean, Q_scot_map, Q_scot_mean, birl_scot = birl_atomic_to_Q_lists(
        envs,
        chosen_scot_flat,
        vi_epsilon=vi_epsilon,
        **birl_kwargs,
    )

    print_all_env_policies(envs, Q_scot_map, tie_eps=1e-12)

    reg_scot_map = regrets_from_Q(envs, Q_scot_map, epsilon=regret_epsilon, n_jobs=n_jobs)
    reg_scot_mean = regrets_from_Q(envs, Q_scot_mean, epsilon=regret_epsilon, n_jobs=n_jobs)

    return {
        "SCOT": {
            "regret_map": reg_scot_map.tolist(),
            "regret_mean": reg_scot_mean.tolist(),
        },
        "BIRL": {
            "SCOT_accept_rate": float(birl_scot.accept_rate),
            # (Optional: if you ever want to log weights)
            # "w_scot_map": w_scot_map.tolist(),
            # "w_scot_mean": w_scot_mean.tolist(),
        },
    }


def eval_random_regret_atomic(
    envs,
    make_random_args,  # (candidates_per_env, chosen_scot_nonflat)
    *,
    n_random_trials=10,
    birl_kwargs=None,
    vi_epsilon=1e-6,
    regret_epsilon=1e-4,
    n_jobs=None,
):
    """
    Evaluate Random baseline only:
      - runs n_random_trials trials in parallel (each trial: sample random atoms, run BIRL, compute regrets)
      - returns stacked regrets: shape (n_trials, n_envs)
    """
    birl_kwargs = birl_kwargs or {}

    if n_random_trials <= 0:
        raise ValueError("eval_random_regret_atomic called with n_random_trials <= 0")

    worker_args = [
        (
            sd,
            envs,
            make_random_args,
            birl_kwargs,
            vi_epsilon,
            regret_epsilon,
        )
        for sd in range(n_random_trials)
    ]

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(_random_trial_worker, worker_args))

    rand_map_regs = [res[0] for res in results]   # each: (n_envs,)
    rand_mean_regs = [res[1] for res in results]  # each: (n_envs,)

    return {
        "RANDOM": {
            "regret_map": np.vstack(rand_map_regs).tolist(),
            "regret_mean": np.vstack(rand_mean_regs).tolist(),
        }
    }


def run_regret_evaluation_atomic(
    envs,
    chosen_scot_flat,
    *,
    make_random_args=None,
    random_trials=10,
    birl_kwargs=None,
    vi_epsilon=1e-6,
    regret_epsilon=1e-4,
    n_jobs=None,
):
    """
    Orchestrator:
      - always computes SCOT
      - optionally computes RANDOM if random_trials > 0
    """
    results = eval_scot_regret_atomic(
        envs,
        chosen_scot_flat,
        birl_kwargs=birl_kwargs,
        vi_epsilon=vi_epsilon,
        regret_epsilon=regret_epsilon,
        n_jobs=n_jobs,
    )

    if random_trials > 0:
        if make_random_args is None:
            raise ValueError("make_random_args must be provided when random_trials > 0")

        results.update(
            eval_random_regret_atomic(
                envs,
                make_random_args,
                n_random_trials=random_trials,
                birl_kwargs=birl_kwargs,
                vi_epsilon=vi_epsilon,
                regret_epsilon=regret_epsilon,
                n_jobs=n_jobs,
            )
        )

    return results


# =============================================================================
# 6. Main experiment
# =============================================================================
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
    **birl_kwargs,
):
    """
    Run the full universal SCOT experiment.
    If random_trials > 0, also runs Random benchmark; otherwise runs SCOT-only.
    Logs progress and saves all results to JSON / NPY.
    """

    def log(msg):
        print(msg, flush=True)

    log("\n======================================================")
    log(" UNIVERSAL ATOMIC SCOT EXPERIMENT — STARTING")
    log("======================================================\n")

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

    log(f"[INIT] Results will be stored in:\n       {out_dir}\n")

    # ---------------------------------------------------
    # 1. Generate W_TRUE
    # ---------------------------------------------------
    log("[1/12] Generating ground-truth reward W_TRUE...")
    W_TRUE = generate_w_true(feature_dim, mode=w_true_mode, seed=seed)
    log(f"       W_TRUE = {W_TRUE}\n")

    # ---------------------------------------------------
    # 2. Generate random MDPs
    # ---------------------------------------------------
    log("[2/12] Generating random GridWorld environments...")
    t0 = time.time()

    color_to_feature_map = {
        f"f{i}": [1 if j == i else 0 for j in range(feature_dim)]
        for i in range(feature_dim)
    }
    palette = list(color_to_feature_map.keys())
    p_color_range = {c: (0.3, 0.8) for c in palette}

    log(f"       → MDP size: {mdp_size}x{mdp_size}")
    log(f"       → Feature dim: {feature_dim}")
    log(f"       → Colors/features: {palette}")
    log(f"       → True reward W = {W_TRUE}")

    envs, _ = generate_random_gridworld_envs(
        n_envs=n_envs,
        rows=mdp_size,
        cols=mdp_size,
        color_to_feature_map=color_to_feature_map,
        palette=palette,
        p_color_range=p_color_range,
        terminal_policy=dict(kind="random_k", k_min=1, k_max=1, p_no_terminal=0.0),
        gamma_range=(0.99, 0.99),
        noise_prob_range=(0.0, 0.0),
        w_mode="fixed",
        W_fixed=W_TRUE,
        seed=seed,
        GridEnvClass=GridWorldMDPFromLayoutEnv,
    )

    log(f"       ✔ Generated {n_envs} environments in {time.time() - t0:.2f}s\n")

    #### To debug #####
    for env in envs:
        env.print_mdp_info()
    #### To debug #####

    # ---------------------------------------------------
    # 3. Value Iteration
    # ---------------------------------------------------
    Q_list = parallel_value_iteration(
        envs,
        epsilon=1e-10,
        n_jobs=None,
        log=log,
    )

    ### Print optimal policy ##### Debug
    print_all_env_policies(envs, Q_list, tie_eps=1e-12)

    # ---------------------------------------------------
    # 4. Successor features
    # ---------------------------------------------------
    log("[4/12] Computing successor features for each MDP...")
    t0 = time.time()
    SFs = compute_successor_features_family(
        envs,
        Q_list,
        convention="entering",
        zero_terminal_features=True,
        tol=1e-10,
        max_iters=10000,
    )
    log(f"       ✔ Successor features computed in {time.time() - t0:.2f}s\n")

    # ---------------------------------------------------
    # 5. Q-based constraints
    # ---------------------------------------------------
    log("[5/12] Deriving Q-based constraints...")
    t0 = time.time()
    _, U_q_global = derive_constraints_from_q_family(
        SFs,
        Q_list,
        envs,
        skip_terminals=True,
        normalize=True,
    )
    log(
        f"       ✔ Q-only global constraints: {len(U_q_global)} "
        f"in {time.time() - t0:.2f}s\n"
    )

    # ---------------------------------------------------
    # 6. Generate feedback atoms
    # ---------------------------------------------------
    log("[6/12] Generating candidate atoms for SCOT...")
    t0 = time.time()
    candidates_per_env = generate_candidate_atoms_for_scot(
        envs,
        Q_list,
        use_q_demos=feedback_demos,
        num_q_rollouts_per_state=1,
        q_demo_max_steps=1,
        use_pairwise=feedback_pairwise,
        use_estop=feedback_estop,
        use_improvement=feedback_improvement,
        n_pairwise=feedback_count,
        n_estops=feedback_count,
        n_improvements=feedback_count,
    )
    atom_counts = [len(a) for a in candidates_per_env]
    log(
        "       ✔ Atoms per env (min/mean/max): "
        f"{min(atom_counts)}/{np.mean(atom_counts):.1f}/{max(atom_counts)} "
        f"in {time.time() - t0:.2f}s\n"
    )

    # Flatten atoms for BIRL (not strictly required later, but kept)
    atoms_flat = [
        (i, atom)
        for i, atoms in enumerate(candidates_per_env)
        for atom in atoms
    ]

    # ---------------------------------------------------
    # 7. Atom constraints
    # ---------------------------------------------------
    log("[7/12] Deriving constraint vectors from atoms...")
    t0 = time.time()
    _, U_atoms_global = derive_constraints_from_atoms(
        candidates_per_env,
        SFs,
        envs,
    )
    log(
        f"       ✔ Atom-derived constraints: {len(U_atoms_global)} "
        f"in {time.time() - t0:.2f}s\n"
    )

    # ---------------------------------------------------
    # 8. Universal constraint merging
    # ---------------------------------------------------
    log("[8/12] Merging Q-based + Atom constraints into Universal set...")
    t0 = time.time()

    compounded = []
    if len(U_q_global) > 0:
        compounded.append(U_q_global)
    if len(U_atoms_global) > 0:
        compounded.append(U_atoms_global)

    if compounded:
        U_universal = remove_redundant_constraints(
            np.vstack(compounded),
            epsilon=1e-4,
        )
    else:
        U_universal = np.zeros((0, feature_dim))

    log(
        f"       ✔ Universal constraint set size: {len(U_universal)} "
        f"(reduced from {sum(len(x) for x in compounded)}) "
        f"in {time.time() - t0:.2f}s\n"
    )

    # ---------------------------------------------------
    # 9. SCOT greedy selection
    # ---------------------------------------------------
    log("[9/12] Running SCOT greedy selection...")
    t0 = time.time()
    chosen_scot, scot_stats, scot_constraint_sets = scot_greedy_family_atoms_tracked(
        U_universal,
        candidates_per_env,
        SFs,
        envs,
    )
    log(
        f"       ✔ SCOT selected {len(chosen_scot)} atoms "
        f"in {time.time() - t0:.2f}s"
    )


    for i in chosen_scot:
        print(i)
        print(i[1].data)


    # Env activation summary
    log("       SCOT env coverage summary (per-env #selected):")
    num_envs_activated = 0
    env_stats_summary = {}
    for env_idx, stats in scot_stats.items():
        num_atoms = len(stats["atoms"])
        total_cov = stats["total_coverage"]
        if num_atoms > 0:
            num_envs_activated += 1
        env_stats_summary[env_idx] = {
            "num_atoms": num_atoms,
            "total_coverage": total_cov,
            "indices": stats["indices"],
            "coverage_counts": stats["coverage_counts"],
        }
        log(f"         env {env_idx:02d}: {num_atoms} atoms, coverage={total_cov}")
    log(f"       #activated envs: {num_envs_activated}/{n_envs}\n")

        
    # BIRL expects flattened: (env_idx, atom)
    chosen_scot_flat = [(i, atom) for (i, atom) in chosen_scot]

    print("Universal Const")
    for i in U_atoms_global:
        print(i)

    print("SCOT Const")
    for i in scot_constraint_sets:
        print(i)

    log("[9.5/12] Checking constraint coverage...")

    def key_for(v, decimals=12):
        n = np.linalg.norm(v)
        if n == 0 or not np.isfinite(n):
            return ("ZERO",)
        return tuple(np.round(v / n, decimals))

    def fmt_vec(v, precision=6):
        return np.array2string(
            np.asarray(v),
            precision=precision,
            floatmode="fixed",
            separator=", ",
        )

    U_universal_arr = np.asarray(U_universal, dtype=float)

    if len(scot_constraint_sets) > 0:
        scot_constraints_flat = np.vstack(scot_constraint_sets).astype(float)
    else:
        scot_constraints_flat = np.zeros((0, feature_dim), dtype=float)

    # Key sets
    U_keys = {key_for(v) for v in U_universal_arr}
    SCOT_keys = {key_for(v) for v in scot_constraints_flat}

    covered_keys = U_keys & SCOT_keys
    missed_keys  = U_keys - SCOT_keys

    num_universal = len(U_keys)
    num_scot = len(SCOT_keys)
    num_covered = len(covered_keys)
    num_missed  = len(missed_keys)
    coverage_pct = 100 * (num_covered / max(1, num_universal))

    log(f"       Universal constraints count   : {num_universal}")
    log(f"       SCOT constraints count        : {num_scot}")
    log(f"       SCOT-covered constraints      : {num_covered}")
    log(f"       Missed universal constraints  : {num_missed}")
    log(f"       Coverage percentage           : {coverage_pct:.2f}%\n")

    # -------------------------
    # Print constraints (bounded)
    # -------------------------
    MAX_SHOW = 30

    log(f"       Universal constraints (raw) — showing {min(MAX_SHOW, len(U_universal_arr))}/{len(U_universal_arr)}")
    for i, v in enumerate(U_universal_arr[:MAX_SHOW]):
        log(f"         U[{i:04d}] {fmt_vec(v)}")

    log(f"\n       SCOT constraints (raw) — showing {min(MAX_SHOW, len(scot_constraints_flat))}/{len(scot_constraints_flat)}")
    for i, v in enumerate(scot_constraints_flat[:MAX_SHOW]):
        log(f"         S[{i:04d}] {fmt_vec(v)}")

    # -------------------------
    # Optional: print missed directions (unit vectors)
    # -------------------------
    if num_missed > 0:
        # Build representative unit vectors for missed keys
        missed_units = []
        for v in U_universal_arr:
            k = key_for(v)
            if k in missed_keys and k != ("ZERO",):
                n = np.linalg.norm(v)
                if n > 0 and np.isfinite(n):
                    missed_units.append(v / n)

        missed_units = np.asarray(missed_units, dtype=float)

        log(f"\n       Missed universal directions (unit) — showing {min(MAX_SHOW, len(missed_units))}/{len(missed_units)}")
        for i, v in enumerate(missed_units[:MAX_SHOW]):
            log(f"         M[{i:04d}] {fmt_vec(v)}")

    log("")  # final newline

    coverage_stats = {
        "num_universal": num_universal,
        "num_scot_constraints": int(scot_constraints_flat.shape[0]),
        "num_covered": num_covered,
        "num_missed": num_missed,
        "coverage_pct": coverage_pct,
    }


    # ---------------------------------------------------
    # 10. Regret evaluation (SCOT-only or SCOT+Random)
    # ---------------------------------------------------
    if random_trials > 0:
        log("[10/12] Computing regret (SCOT vs Random)...")
    else:
        log("[10/12] Computing regret (SCOT only; random_trials=0)...")

    t0 = time.time()

    # Data tuple only (picklable)
    make_random_args = (candidates_per_env, chosen_scot)
    print(chosen_scot_flat)
    results = run_regret_evaluation_atomic(
        envs,
        chosen_scot_flat,
        make_random_args=make_random_args if random_trials > 0 else None,
        random_trials=random_trials,
        birl_kwargs=birl_kwargs,
        vi_epsilon=1e-6,
        regret_epsilon=1e-4,
        n_jobs=None,
    )

    log(f"       ✔ Regret computed in {time.time() - t0:.2f}s")
    log(f"       SCOT mean regret: {np.mean(results['SCOT']['regret_map']):.4f}")

    if "RANDOM" in results:
        log(f"       Random mean regret: {np.mean(results['RANDOM']['regret_map']):.4f}\n")
    else:
        log("       Random baseline skipped.\n")

    # ---------------------------------------------------
    # 11. Save raw constraint logs
    # ---------------------------------------------------
    log("[11/12] Saving SCOT and Universal constraint arrays...")

    np.save(os.path.join(out_dir, "U_universal.npy"), U_universal)
    np.save(os.path.join(out_dir, "SCOT_constraints.npy"), scot_constraints_flat)

    log("       ✔ Saved U_universal.npy and SCOT_constraints.npy\n")

    # ---------------------------------------------------
    # 12. Save results JSON (with coverage + env stats)
    # ---------------------------------------------------
    log("[12/12] Saving results to JSON...")

    results["metadata"] = {
        "W_TRUE": W_TRUE.tolist(),
        "experiment_dir": out_dir,
        "n_envs": n_envs,
        "mdp_size": mdp_size,
        "feature_dim": feature_dim,
        "w_true_mode": w_true_mode,
        "feedback": {
            "demos": bool(feedback_demos),
            "pairwise": bool(feedback_pairwise),
            "estop": bool(feedback_estop),
            "improvement": bool(feedback_improvement),
            "feedback_count": feedback_count,
        },
        "random_trials": int(random_trials),
        "seed": seed,
        "coverage": coverage_stats,
        "scot_env_stats": env_stats_summary,
        "num_envs_activated": num_envs_activated,
    }

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    log(f"       ✔ Saved results as {os.path.join(out_dir, 'results.json')}\n")

    # ---------------------------------------------------
    # Final Stats
    # ---------------------------------------------------
    log("EXPERIMENT COMPLETED")
    log("------------------------------------------------------")
    log(f"Total runtime: {time.time() - start_all:.2f} seconds")
    log(f"Results directory: {out_dir}")
    log("======================================================\n")

    return results, out_dir


# ============================================================
# CLI ENTRY POINT (argparse)
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run universal SCOT vs Random atomic IRL experiment (supports SCOT-only when --random_trials 0)."
    )

    # Environment / MDP parameters
    parser.add_argument("--n_envs", type=int, default=30, help="Number of MDP environments to generate.")
    parser.add_argument("--mdp_size", type=int, default=10, help="Gridworld size (rows = cols = mdp_size).")
    parser.add_argument("--feature_dim", type=int, default=2, help="Number of reward features.")
    parser.add_argument(
        "--w_true_mode",
        type=str,
        default="random_signed",
        choices=["random_signed", "one_hot", "biased"],
        help="How to generate ground-truth reward weights.",
    )

    # Feedback settings
    parser.add_argument(
        "--feedback",
        nargs="+",
        default=["demo", "pairwise", "estop", "improvement"],
        help="Feedback types to enable. Options: demo pairwise estop improvement",
    )
    parser.add_argument("--feedback_count", type=int, default=50, help="Atoms per feedback type per environment.")

    # Random baseline (for regret comparison)
    parser.add_argument("--random_trials", type=int, default=10, help="How many random baseline seeds to try. Use 0 for SCOT-only.")

    # BIRL (MCMC) settings
    parser.add_argument("--samples", type=int, default=5000, help="Number of MCMC samples.")
    parser.add_argument("--stepsize", type=float, default=0.1, help="Proposal stepsize in MCMC.")
    parser.add_argument("--beta", type=float, default=10.0, help="Inverse temperature for likelihood.")

    # General settings
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--result_dir", type=str, default="results_universal", help="Parent directory to store experiment results.")

    args = parser.parse_args()

    fb_list = args.feedback
    fb_demo = "demo" in fb_list
    fb_pairwise = "pairwise" in fb_list
    fb_estop = "estop" in fb_list
    fb_improvement = "improvement" in fb_list

    birl_kwargs = dict(
        beta=args.beta,
        samples=args.samples,
        stepsize=args.stepsize,
        normalize=True,
        adaptive=False,
        burn_frac=0.2,
        skip_rate=10,
    )

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
        **birl_kwargs,
    )
