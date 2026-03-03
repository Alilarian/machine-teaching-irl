import numpy as np
import random
from typing import List, Tuple, Dict, Optional

def generate_random_gridworld_envs(
    n_envs: int,
    rows: int = 2,
    cols: int = 3,
    color_to_feature_map: Dict[str, List[float]] = None,
    palette: Tuple[str, ...] = ("red", "blue"),
    p_color_range: Dict[str, Tuple[float, float]] = None,
    # terminal policy
    terminal_policy: Dict = None,
    # dynamics / discount
    gamma_range: Tuple[float, float] = (0.95, 0.995),
    noise_prob_range: Tuple[float, float] = (0.0, 0.2),
    # reward weights
    w_mode: str = "fixed",   # "fixed" | "rand_unit" | "jitter"
    W_fixed: Optional[np.ndarray] = None,
    w_jitter_sigma: float = 0.10,
    # reproducibility/uniqueness
    seed: Optional[int] = None,
    ensure_unique_layouts: bool = True,
    ensure_unique_layout_terminals: bool = True,
    max_resample_tries: int = 200,
    # pass-through ctor args
    render_mode: Optional[str] = None,
    GridEnvClass=None,  # pass GridWorldMDPFromLayoutEnv
):
    """
    Returns:
        envs: List[GridEnvClass]
        meta: dict with layouts, terminals, weights, noise, gamma for inspection
    """
    assert GridEnvClass is not None, "Pass GridWorldMDPFromLayoutEnv via GridEnvClass"

    rng = np.random.default_rng(seed)
    random.seed(seed)

    if color_to_feature_map is None:
        color_to_feature_map = {"red": [1.0, 0.0], "blue": [0.0, 1.0]}
    feat_dim = len(next(iter(color_to_feature_map.values())))

    if p_color_range is None:
        # Example: red between 0.2–0.6; rest mass goes to other colors uniformly
        p_color_range = {c: (0.2, 0.6) if c == "red" else (0.4, 0.8) for c in palette}

    if terminal_policy is None:
        # “random_k” with chance of no terminal
        terminal_policy = dict(kind="random_k", k_min=0, k_max=1, p_no_terminal=0.3)

    def sample_W():
        if w_mode == "fixed":
            assert W_fixed is not None, "Provide W_fixed for w_mode='fixed'"
            w = np.array(W_fixed, dtype=float)
        elif w_mode == "rand_unit":
            w = rng.normal(size=feat_dim)
        elif w_mode == "jitter":
            assert W_fixed is not None, "Provide W_fixed for w_mode='jitter'"
            w = np.array(W_fixed, dtype=float) + rng.normal(scale=w_jitter_sigma, size=feat_dim)
        else:
            raise ValueError(f"Unknown w_mode {w_mode}")
        # normalize to unit length to keep scale comparable across envs
        norm = np.linalg.norm(w)
        return w / (norm + 1e-12)

    def sample_palette_probs():
        # independently sample each color’s target share within its range,
        # then renormalize to sum=1
        raw = []
        for c in palette:
            lo, hi = p_color_range[c]
            raw.append(rng.uniform(lo, hi))
        raw = np.maximum(1e-6, np.array(raw))
        return {c: float(x / np.sum(raw)) for c, x in zip(palette, raw)}

    def sample_layout(rows, cols):
        p = sample_palette_probs()
        choices = list(p.keys())
        probs = [p[c] for c in choices]
        colors = rng.choice(choices, size=(rows, cols), p=probs)
        return [[str(colors[r, c]) for c in range(cols)] for r in range(rows)]

    def flat_index(r, c): return r * cols + c

    def sample_terminals(rows, cols):
        kind = terminal_policy.get("kind", "random_k")
        if kind == "none":
            return []
        if kind == "random_k":
            if rng.random() < terminal_policy.get("p_no_terminal", 0.0):
                return []
            k_min = terminal_policy.get("k_min", 1)
            k_max = terminal_policy.get("k_max", 1)
            k = int(rng.integers(k_min, k_max + 1))
            k = min(k, rows * cols)
            all_idxs = list(range(rows * cols))
            rng.shuffle(all_idxs)
            return sorted(all_idxs[:k])
        if kind == "corners":
            corners = sorted({flat_index(0, 0), flat_index(0, cols - 1),
                              flat_index(rows - 1, 0), flat_index(rows - 1, cols - 1)})
            # optionally choose a random subset of corners
            k = terminal_policy.get("k", len(corners))
            k = min(k, len(corners))
            rng.shuffle(corners)
            return sorted(corners[:k])
        if kind == "single_random":
            return [int(rng.integers(0, rows * cols))]
        raise ValueError(f"Unknown terminal kind: {kind}")

    # Uniqueness guards
    seen_layouts = set()
    seen_layout_term = set()

    envs = []
    meta = dict(layouts=[], terminals=[], weights=[], gammas=[], noise_probs=[])

    tries = 0
    while len(envs) < n_envs and tries < max_resample_tries * n_envs:
        tries += 1
        layout = sample_layout(rows, cols)
        layout_flat = tuple(sum(layout, []))  # flatten to 1D tuple
        terminals = sample_terminals(rows, cols)

        # uniqueness checks
        if ensure_unique_layouts and layout_flat in seen_layouts:
            continue
        if ensure_unique_layout_terminals and (layout_flat, tuple(terminals)) in seen_layout_term:
            continue

        gamma = float(rng.uniform(*gamma_range))
        noise_prob = float(rng.uniform(*noise_prob_range))
        W = sample_W()

        env = GridEnvClass(
            gamma=gamma,
            layout=layout,
            color_to_feature_map=color_to_feature_map,
            noise_prob=noise_prob,
            terminal_states=terminals if len(terminals) > 0 else [],  # allow “no terminal”
            custom_feature_weights=W,
            render_mode=render_mode,
        )

        envs.append(env)
        meta["layouts"].append(layout)
        meta["terminals"].append(terminals)
        meta["weights"].append(W)
        meta["gammas"].append(gamma)
        meta["noise_probs"].append(noise_prob)

        seen_layouts.add(layout_flat)
        seen_layout_term.add((layout_flat, tuple(terminals)))

    if len(envs) < n_envs:
        print(f"[warn] Only generated {len(envs)} unique envs after {tries} attempts.")

    return envs, meta
