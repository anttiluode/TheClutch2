"""viz.py — matplotlib rendering for the Clutch demo (headless / Agg backend)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import imageio.v2 as imageio
import tempfile, os

from nav import W, H, A, B

HABIT = "#2e9e5b"   # green  = cheap cached step
COG = "#d64545"     # red    = expensive replan
INK = "#1b1b1b"
GRID_BG = "#f4f1ea"


def _render_nav_frame(walls, pos, mode, cum_expanded, target, step, plan_calls):
    fig = Figure(figsize=(4.2, 4.2), dpi=96)
    ax = fig.add_subplot(111)
    ax.set_facecolor(GRID_BG)
    img = np.ones((H, W, 3))
    img[walls] = [0.12, 0.12, 0.14]        # walls dark
    ax.imshow(img, origin="lower", interpolation="nearest")
    ax.scatter([A[0]], [A[1]], s=60, marker="s", c="#3a6ea5", zorder=3)
    ax.scatter([B[0]], [B[1]], s=60, marker="s", c="#3a6ea5", zorder=3)
    tcol = "#e8a33d"
    ax.scatter([target[0]], [target[1]], s=180, marker="*", c=tcol, zorder=4,
               edgecolors=INK, linewidths=0.5)
    col = HABIT if mode == "HABITUAL" else COG
    ax.scatter([pos[0]], [pos[1]], s=90, c=col, zorder=5,
               edgecolors=INK, linewidths=0.6)
    ax.set_xlim(-1, W); ax.set_ylim(-1, H)
    ax.set_xticks([]); ax.set_yticks([])
    label = "REPLAN (expensive)" if mode == "COGNITIVE" else "cached step (cheap)"
    ax.set_title(f"step {step}   {label}", fontsize=10, color=col, fontweight="bold")
    ax.text(0.5, -0.06, f"BFS cells expanded: {cum_expanded:,}   plan calls: {plan_calls}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=INK)
    fig.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def make_nav_gif(frames, max_frames=90, fps=12):
    """frames: list of (walls, pos, mode, cum_expanded, target). Returns gif path."""
    if not frames:
        return None
    n = len(frames)
    idxs = np.linspace(0, n - 1, min(max_frames, n)).astype(int)
    plan_calls = 0
    prev_mode = None
    imgs = []
    # precompute cumulative plan calls along the full trace
    calls_at = []
    pc = 0
    for (_, _, mode, _, _) in frames:
        if mode == "COGNITIVE":
            pc += 1
        calls_at.append(pc)
    for i in idxs:
        walls, pos, mode, cum, target = frames[i]
        img = _render_nav_frame(walls, pos, mode, cum, target, i, calls_at[i])
        imgs.append(img)
    # hold last frame a beat
    imgs += [imgs[-1]] * 6
    path = os.path.join(tempfile.gettempdir(), f"nav_{np.random.randint(1e9)}.gif")
    imageio.mimsave(path, imgs, fps=fps, loop=0)
    return path


def make_compute_plot(clutch_frames, cog_frames, clutch_label="CLUTCH"):
    fig = Figure(figsize=(5.2, 3.4), dpi=96)
    ax = fig.add_subplot(111)
    cc = [f[3] for f in clutch_frames]
    gc = [f[3] for f in cog_frames]
    ax.plot(range(len(gc)), gc, color=COG, lw=2, label="ALWAYS_COGNITIVE (replan every step)")
    ax.plot(range(len(cc)), cc, color=HABIT, lw=2, label=clutch_label)
    ax.set_xlabel("step"); ax.set_ylabel("cumulative BFS cells expanded")
    ax.set_title("Compute spent over time", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    if gc and cc:
        ratio = cc[-1] / gc[-1] * 100 if gc[-1] else 0
        ax.text(0.98, 0.05, f"clutch = {ratio:.1f}% of always-replan",
                transform=ax.transAxes, ha="right", fontsize=9,
                color=HABIT, fontweight="bold")
    fig.tight_layout()
    return fig


def make_drift_plot(y, change_pts, result, window):
    fig = Figure(figsize=(6.4, 3.6), dpi=96)
    ax = fig.add_subplot(111)
    t = np.arange(len(y))
    ax.plot(t, y, color="#9aa0a6", lw=1.0, label="true signal", zorder=1)
    ax.plot(t, result["pred"], color=HABIT, lw=1.6, label="clutch prediction", zorder=2)
    for i, cp in enumerate(sorted(change_pts)):
        ax.axvline(cp, color="#c9b358", ls=":", lw=1.0,
                   label="regime change" if i == 0 else None)
    for i, tt in enumerate(result["trip_times"]):
        ax.axvline(tt, color=COG, ls="-", lw=0.9, alpha=0.7,
                   label="gate trip -> REFIT" if i == 0 else None)
    ax.set_xlabel("time step"); ax.set_ylabel("value")
    ax.set_title(f"Drift-gated retraining — {result['refits']} refits, "
                 f"MAE {result['mae']:.2f}", fontsize=11)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def make_pareto_plot(sw, best, fallback, front):
    """Accuracy-vs-compute scatter of every swept config + Pareto frontier + winner."""
    fig = Figure(figsize=(6.2, 3.8), dpi=96)
    ax = fig.add_subplot(111)
    ref = sw["ref"]
    mags = [r for r in sw["rows"] if r["gate"] == "MagnitudeGate"]
    accs = [r for r in sw["rows"] if r["gate"] == "AcceleratorGate"]
    ax.scatter([r["samples"] for r in mags], [r["mae"] for r in mags],
               s=22, c=HABIT, alpha=0.55, label="MagnitudeGate configs")
    ax.scatter([r["samples"] for r in accs], [r["mae"] for r in accs],
               s=22, c="#7a5fb5", alpha=0.55, label="AcceleratorGate configs")
    fx = [r["samples"] for r in front]; fy = [r["mae"] for r in front]
    ax.plot(fx, fy, color=INK, lw=1.2, ls="--", alpha=0.7, label="Pareto frontier")
    ax.axhline(ref["mae"], color=COG, lw=1.2, ls=":",
               label=f"refit-every-step MAE ({ref['mae']:.3g})")
    ax.axvline(ref["refit_samples"], color=COG, lw=1.0, ls=":", alpha=0.5)
    win = best or fallback
    if win:
        ax.scatter([win["samples"]], [win["mae"]], s=170, marker="*",
                   c="#e8a33d", edgecolors=INK, linewidths=0.8, zorder=5,
                   label="chosen config")
    ax.set_xlabel("training samples spent (compute)")
    ax.set_ylabel("prediction MAE")
    ax.set_title("Every gate config on YOUR data — down-left is better", fontsize=11)
    ax.legend(fontsize=7.5, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig
