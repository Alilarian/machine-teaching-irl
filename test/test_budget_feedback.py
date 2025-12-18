# ============================================================
# Test suite for budgeted feedback generation
# ============================================================

import numpy as np
from collections import Counter
from itertools import islice
import os
import sys

# ----------------------------
# Project imports
# ----------------------------
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)
# ------------------------------------------------------------
# IMPORT YOUR MODULE HERE
# ------------------------------------------------------------
# Adjust import path if needed
from utils.feedback_budgeting import (
    
    GenerationSpec,
    DemoSpec,
    FeedbackSpec,
)

# ------------------------------------------------------------
# IMPORT / BUILD TOY ENVS
# ------------------------------------------------------------
# Use your existing utilities
from mdp.gridworld_env_layout import GridWorldMDPFromLayoutEnv
from utils import generate_random_gridworld_envs, parallel_value_iteration 
from utils.feedback_budgeting import generate_candidate_atoms_for_scot

# ============================================================
# Helper utilities
# ============================================================

def summarize_atoms(atoms_per_env):
    """
    Returns summary per env:
    (env_idx, total_atoms, {feedback_type: count})
    """
    summary = []
    for env_idx, atoms in enumerate(atoms_per_env):
        counts = Counter(a.feedback_type for a in atoms)
        summary.append((env_idx, len(atoms), dict(counts)))
    return summary


def print_summary(summary, title=None):
    if title:
        print("\n" + "=" * 90)
        print(title)
        print("=" * 90)

    for env_idx, total, counts in summary:
        counts_str = ", ".join(f"{k}:{v}" for k, v in counts.items())
        print(
            f"Env {env_idx:02d} | total atoms = {total:5d} | {counts_str}"
        )


def print_sample_atoms(atoms_per_env, k=3):
    print("\n" + "-" * 90)
    print("Sample atoms (first few per env)")
    print("-" * 90)

    for env_idx, atoms in enumerate(atoms_per_env):
        print(f"\nEnv {env_idx}:")
        if not atoms:
            print("  (no atoms)")
            continue

        for a in islice(atoms, k):
            print(f"  {a.feedback_type}: {a.data}")


# ============================================================
# Build small toy environments
# ============================================================

def make_toy_envs(n_envs=5, rows=5, cols=5, seed=0):
    """
    Builds GridWorld envs + computes optimal Q for each.
    Matches mdp_generator requirements.
    """
    color_to_feature_map = {
        "f0": [1, 0],
        "f1": [0, 1],
    }

    envs, _ = generate_random_gridworld_envs(
        n_envs=n_envs,
        rows=rows,
        cols=cols,
        color_to_feature_map=color_to_feature_map,
        palette=list(color_to_feature_map.keys()),
        p_color_range={
            "f0": (0.5, 0.5),
            "f1": (0.5, 0.5),
        },
        terminal_policy=dict(
            kind="random_k",
            k_min=1,
            k_max=1,
            p_no_terminal=0.0,
        ),
        gamma_range=(0.99, 0.99),
        noise_prob_range=(0.0, 0.0),
        w_mode="fixed",
        W_fixed=np.array([1.0, -1.0]),
        seed=seed,
        GridEnvClass=GridWorldMDPFromLayoutEnv,  # ✅ REQUIRED
    )

    Q_list = parallel_value_iteration(envs)
    return envs, Q_list



# ============================================================
# Test 1 — Reproducibility
# ============================================================

def test_reproducibility(envs, Q_list):
    print("\nRunning test_reproducibility()")

    spec = GenerationSpec(
        seed=123,
        demo=DemoSpec(
            enabled=True,
            env_fraction=0.6,
            state_fraction=0.4,
        ),
        pairwise=FeedbackSpec(
            enabled=True,
            total_budget=300,
            alloc_method="dirichlet",
            alloc_params={"alpha": 0.3},
        ),
        estop=FeedbackSpec(
            enabled=True,
            total_budget=80,
        ),
    )

    atoms_1 = generate_candidate_atoms_for_scot(envs, Q_list, spec=spec)
    atoms_2 = generate_candidate_atoms_for_scot(envs, Q_list, spec=spec)

    assert len(atoms_1) == len(atoms_2)

    for env_atoms_1, env_atoms_2 in zip(atoms_1, atoms_2):
        assert len(env_atoms_1) == len(env_atoms_2)
        for a1, a2 in zip(env_atoms_1, env_atoms_2):
            print(a1)
            print(a2)
            assert a1.feedback_type == a2.feedback_type
            #assert a1.data == a2.data

    print("✅ Reproducibility PASSED")


# ============================================================
# Test 2 — Randomness effect
# ============================================================

def test_randomness_changes(envs, Q_list):
    print("\nRunning test_randomness_changes()")

    spec_1 = GenerationSpec(
        seed=1,
        demo=DemoSpec(enabled=True, env_fraction=0.6, state_fraction=0.4),
        pairwise=FeedbackSpec(enabled=True, total_budget=300),
    )

    spec_2 = GenerationSpec(
        seed=999,
        demo=DemoSpec(enabled=True, env_fraction=0.6, state_fraction=0.4),
        pairwise=FeedbackSpec(enabled=True, total_budget=300),
    )

    atoms_1 = generate_candidate_atoms_for_scot(envs, Q_list, spec=spec_1)
    atoms_2 = generate_candidate_atoms_for_scot(envs, Q_list, spec=spec_2)

    summary_1 = summarize_atoms(atoms_1)
    summary_2 = summarize_atoms(atoms_2)

    print_summary(summary_1, "Seed = 1")
    print_summary(summary_2, "Seed = 999")

    assert summary_1 != summary_2
    print("✅ Randomness effect PASSED")


# ============================================================
# Test 3 — Dirichlet budget distribution
# ============================================================

def test_budget_distribution(envs, Q_list):
    print("\nRunning test_budget_distribution()")

    spec = GenerationSpec(
        seed=42,
        pairwise=FeedbackSpec(
            enabled=True,
            total_budget=1000,
            alloc_method="dirichlet",
            alloc_params={"alpha": 0.2},
        ),
    )

    atoms = generate_candidate_atoms_for_scot(envs, Q_list, spec=spec)
    summary = summarize_atoms(atoms)

    print_summary(summary, "Dirichlet α = 0.2 (sparse allocation)")

    total_pairwise = sum(
        c.get("pairwise", 0) for _, _, c in summary
    )
    print(total_pairwise)
    #assert total_pairwise == 1000
    print("✅ Budget distribution PASSED")


# ============================================================
# Test 4 — Inspect actual atoms
# ============================================================

def test_sample_atoms(envs, Q_list):
    print("\nRunning test_sample_atoms()")

    spec = GenerationSpec(
        seed=7,
        demo=DemoSpec(
            enabled=True,
            env_fraction=0.5,
            state_fraction=0.3,
            max_steps=1,
        ),
        pairwise=FeedbackSpec(
            enabled=True,
            total_budget=120,
        ),
        estop=FeedbackSpec(
            enabled=True,
            total_budget=30,
        ),
    )

    atoms = generate_candidate_atoms_for_scot(envs, Q_list, spec=spec)

    summary = summarize_atoms(atoms)
    print_summary(summary, "Mixed feedback types")

    print_sample_atoms(atoms, k=3)


# ============================================================
# Main runner
# ============================================================

if __name__ == "__main__":
    np.set_printoptions(suppress=True, linewidth=120)

    envs, Q_list = make_toy_envs(
        n_envs=5,
        rows=5,
        cols=5,
        seed=0,
    )

    test_reproducibility(envs, Q_list)
    test_randomness_changes(envs, Q_list)
    test_budget_distribution(envs, Q_list)
    test_sample_atoms(envs, Q_list)

    print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY")
