import numpy as np
from scipy.optimize import linprog

def _normalize_dir(v, tol=1e-12):
    v = np.asarray(v, float)
    nrm = np.linalg.norm(v)
    if nrm < tol:
        return None
    v = v / nrm
    for x in v:
        if abs(x) > tol:
            if x < 0:
                v = -v
            break
    return tuple(np.round(v, 12))


def is_redundant_constraint(h, H, epsilon=1e-4):
    h = np.asarray(h, float)
    H = np.asarray(H, float)

    if H.size == 0:
        return False
    if H.ndim == 1:
        H = H.reshape(1, -1)

    m, n = H.shape
    assert h.shape == (n,)

    b = np.zeros(m)
    res = linprog(h, A_ub=-H, b_ub=b, bounds=[(-1, 1)]*n, method='highs')
    if res.status != 0:
        return False

    return res.fun >= -epsilon


def remove_redundant_constraints(halfspaces, epsilon=1e-4):
    halfspaces = [np.asarray(h, float) for h in halfspaces]
    seen = set()
    unique = []

    for h in halfspaces:
        key = _normalize_dir(h)
        if key is None:
            continue
        if key not in seen:
            seen.add(key)
            unique.append(h)

    kept = []
    for h in unique:
        H_keep = np.vstack(kept) if kept else np.empty((0, h.size))
        if not is_redundant_constraint(h, H_keep, epsilon):
            kept.append(h)

    final = []
    for i, h in enumerate(kept):
        others = [kept[j] for j in range(len(kept)) if j != i]
        H_others = np.vstack(others) if others else np.empty((0, h.size))
        if not is_redundant_constraint(h, H_others, epsilon):
            final.append(h)

    return final
