import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def _intersection_polygon_2d(V, box=1.0, tol=1e-12):
    V = np.asarray(V, float).reshape(-1, 2)
    half = [(-vx, -vy, 0.0) for (vx, vy) in V]
    half += [(1,0,-box), (-1,0,-box), (0,1,-box), (0,-1,-box)]

    pts = []
    m = len(half)
    for i in range(m):
        a1, b1, c1 = half[i]
        for j in range(i+1, m):
            a2, b2, c2 = half[j]
            D = a1*b2 - a2*b1
            if abs(D) < tol:
                continue
            x = (b1*c2 - b2*c1) / D
            y = (c1*a2 - c2*a1) / D
            if all(a*x + b*y + c <= tol for (a,b,c) in half):
                pts.append((x,y))

    if not pts:
        return np.empty((0,2))

    pts = np.unique(np.round(pts, 12), axis=0)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:,1] - c[1], pts[:,0] - c[0])
    order = np.argsort(ang)
    return pts[order]


def plot_halfspace_intersection_2d(
    V,
    *,
    box=1.0,
    colors=None,
    labels=None,
    w_true=None,
    scot_sol=None,
    title="Intersection of half-spaces",
):
    V = np.asarray(V, float).reshape(-1,2)
    m = len(V)
    xs = np.linspace(-box, box, 400)

    if colors is None:
        colors = ["#d81b60", "#008080", "#1f77b4", "#ff7f0e"]
    if labels is None:
        labels = [f"Constraint {i+1}" for i in range(m)]

    fig, ax = plt.subplots(figsize=(6,6))
    handles = []

    for i, (vx, vy) in enumerate(V):
        col = colors[i % len(colors)]
        if abs(vy) < 1e-12:
            h = ax.axvline(0, color=col, lw=4, label=labels[i])
        else:
            y_line = -(vx/vy) * xs
            h, = ax.plot(xs, y_line, color=col, lw=1, label=labels[i])
        handles.append(h)

    poly = _intersection_polygon_2d(V, box=box)
    if poly.shape[0] > 0:
        patch = Polygon(poly, closed=True, facecolor="#f5bd23", alpha=0.9,
                        edgecolor="none", hatch="///")
        ax.add_patch(patch)

    ax.axhline(0, color="k", lw=1)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlim(-box, box)
    ax.set_ylim(-box, box)
    ax.set_aspect("equal", "box")

    if w_true is not None:
        ax.plot(w_true[0], w_true[1], "k*", ms=10)
    if scot_sol is not None:
        ax.plot(scot_sol[0], scot_sol[1], "*", color="#23f523", ms=10)

    ax.legend(loc="upper right")
    ax.set_title(title)
    plt.show()
