import json
import numpy as np
import matplotlib.pyplot as plt


def plot_grouped_regret(
    feedback_to_file,
    *,
    sort_by_random=False,
    show_values=True,
    save_path=None,
):
    """
    Parameters
    ----------
    feedback_to_file : dict
        {"feedback_name": "path/to/result.json"}

    sort_by_random : bool
        If True, sort feedback types by random mean regret (ascending).

    show_values : bool
        If True, print numeric values on top of bars.

    save_path : str or None
        If provided, save figure to this path. Otherwise show plot.
    """

    feedback_names = []
    two_stage_means = []
    random_means = []

    # Metadata (assumed identical across experiments)
    n_envs = None
    mdp_size = None
    random_trials = None

    # ---------------- Load results ----------------
    for feedback, file_path in feedback_to_file.items():
        with open(file_path, "r") as f:
            data = json.load(f)

        feedback_names.append(feedback)
        two_stage_means.append(float(data["two_stage_mean"]))
        random_means.append(float(data["random_mean"]))

        cfg = data.get("config", {})
        n_envs = n_envs or cfg.get("n_envs")
        mdp_size = mdp_size or cfg.get("mdp_size")
        random_trials = random_trials or cfg.get("random_trials")

    feedback_names = np.array(feedback_names)
    two_stage_means = np.array(two_stage_means)
    random_means = np.array(random_means)

    # ---------------- Optional sorting ----------------
    if sort_by_random:
        order = np.argsort(random_means)
        feedback_names = feedback_names[order]
        two_stage_means = two_stage_means[order]
        random_means = random_means[order]

    # ---------------- Plot ----------------
    x = np.arange(len(feedback_names))
    width = 0.35

    plt.figure(figsize=(10, 6))

    bars_scot = plt.bar(
        x - width / 2,
        two_stage_means,
        width,
        label="Two-Stage SCOT",
    )

    bars_random = plt.bar(
        x + width / 2,
        random_means,
        width,
        label="Random Selection",
    )

    plt.xticks(x, feedback_names)
    plt.xlabel("Feedback Type")
    plt.ylabel("Mean Regret")

    plt.title(
        f"Mean Regret Comparison | "
        f"{n_envs} MDPs, Grid {mdp_size}×{mdp_size}, "
        f"Random Trials = {random_trials}"
    )

    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # ---------------- Value labels ----------------
    if show_values:
        def annotate(bars):
            for bar in bars:
                h = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    h,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        annotate(bars_scot)
        annotate(bars_random)

    # ---------------- Save or show ----------------
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    feedback_to_file = {
        "demo": "two_stage_vs_random_env20_size3_fd3_budget0_seed11000_20260107-194859.json",
        "pairwise": "two_stage_vs_random_env20_size3_fd3_budget100000_seed100_20260107-183005.json",
        "correction": "two_stage_vs_random_env20_size3_fd3_budget100000_seed100_20260107-195729.json",
    }

    plot_grouped_regret(
        feedback_to_file,
        sort_by_random=True,
        show_values=True,
        save_path="mean_regret_comparison.png",
    )
