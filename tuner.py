"""
tuner.py — turn the Clutch demo into a TOOL: tune the gate on the USER's own data.

The visitor pastes/uploads any 1-D time series (latency metric, sensor stream, price
feed, error signal...). We run the real closed-loop drift substrate on it:
  cheap  = extrapolate the cached linear model        (O(1))
  costly = least-squares refit on the last `window`   (O(window))
  error  = normalized residual of the last prediction
Then we SWEEP both gate families over a parameter grid, plot the honest
accuracy-vs-compute Pareto frontier, and pick the cheapest config whose MAE is within
`tol` of refit-every-step. Output: a copy-paste Clutch(...) snippet + $ savings.

All counterfactuals are valid because the loop is closed (refitting changes future
errors) — this is the same honest accounting as the benchmark, on the user's data.
"""

import io
import re
import numpy as np
from drift import run_drift

MAX_POINTS = 20_000  # keep server-side sweeps snappy


# ------------------------------------------------------------------ parsing
def parse_series(text=None, file_obj=None):
    """Extract a 1-D float series from pasted text or an uploaded CSV/TXT.
    Multi-column CSV: uses the LAST numeric column (usually the value column).
    Returns (y, message)."""
    raw = ""
    if file_obj is not None:
        path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
        if path:
            with open(path, "r", errors="ignore") as f:
                raw = f.read()
    elif text:
        raw = text
    if not raw.strip():
        return None, "No data provided."

    rows = []
    for line in raw.strip().splitlines():
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
        if nums:
            rows.append([float(x) for x in nums])
    if not rows:
        return None, "Could not find any numbers in the input."

    ncol = max(len(r) for r in rows)
    if ncol == 1:
        y = np.array([r[0] for r in rows if len(r) == 1], dtype=float)
        note = ""
    else:
        # take the last column present in full-width rows (skips header remnants)
        full = [r for r in rows if len(r) == ncol]
        y = np.array([r[-1] for r in full], dtype=float)
        note = f" (took last of {ncol} numeric columns)"
    y = y[np.isfinite(y)]
    if len(y) < 60:
        return None, f"Found only {len(y)} points — need at least 60 for a meaningful sweep."
    if len(y) > MAX_POINTS:
        y = y[:MAX_POINTS]
        note += f"; truncated to first {MAX_POINTS:,} points"
    return y, f"Loaded {len(y):,} points{note}."


# ------------------------------------------------------------------ sweep
MAG_GRID = [dict(gain=g, leak=l, trip_mag=t)
            for g in (2.0, 4.0, 6.0, 8.0)
            for l in (0.2, 0.5)
            for t in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)]
ACC_GRID = [dict(trip_acc=t, refractory=r)
            for t in (0.2, 0.4, 0.8, 1.2, 2.0)
            for r in (0, 3, 6)]


def sweep(y, window=25):
    """Run baselines + full grid. Returns dict with baselines and per-config rows."""
    cps = set()
    ref = run_drift(y, cps, "ALWAYS_REFIT", window=window)
    never = run_drift(y, cps, "NEVER_REFIT", window=window)
    rows = []
    for gp in MAG_GRID:
        r = run_drift(y, cps, "CLUTCH_MAG", window=window, gate_params=gp)
        rows.append(dict(gate="MagnitudeGate", params=gp, mae=r["mae"],
                         refits=r["refits"], samples=r["refit_samples"]))
    for gp in ACC_GRID:
        r = run_drift(y, cps, "CLUTCH_ACC", window=window, gate_params=gp)
        rows.append(dict(gate="AcceleratorGate", params=gp, mae=r["mae"],
                         refits=r["refits"], samples=r["refit_samples"]))
    return dict(ref=ref, never=never, rows=rows, window=window, n=len(y))


def pick_best(sw, tol=0.10):
    """Cheapest config with MAE <= (1+tol) * refit-every-step MAE.
    Returns (best_or_None, fallback, limit). fallback = Pareto point with the
    smallest MAE increase that still saves >= 30% compute (honest 'closest option')."""
    limit = sw["ref"]["mae"] * (1.0 + tol)
    ok = [r for r in sw["rows"] if r["mae"] <= limit]
    best = min(ok, key=lambda r: (r["samples"], r["mae"])) if ok else None
    base = sw["ref"]["refit_samples"] or 1
    savers = [r for r in pareto_front(sw["rows"]) if r["samples"] <= 0.7 * base]
    fallback = min(savers, key=lambda r: r["mae"]) if savers else None
    return best, fallback, limit


def pareto_front(rows):
    """Non-dominated set in (samples, mae), sorted by samples."""
    pts = sorted(rows, key=lambda r: (r["samples"], r["mae"]))
    front, best_mae = [], float("inf")
    for r in pts:
        if r["mae"] < best_mae - 1e-12:
            front.append(r)
            best_mae = r["mae"]
    return front


# ------------------------------------------------------------------ output
def code_snippet(best, window):
    p = best["params"]
    if best["gate"] == "MagnitudeGate":
        gate = (f"MagnitudeGate(gain={p['gain']}, leak={p['leak']}, "
                f"trip={p['trip_mag']})")
    else:
        gate = f"AcceleratorGate(trip={p['trip_acc']}, refractory={p['refractory']})"
    return f"""# tuned on YOUR data — cheapest gate within tolerance of refit-every-step
from clutch import Clutch, MagnitudeGate, AcceleratorGate

clutch = Clutch({gate})
# refit window used during tuning: {window}
# supply your three callbacks:
#   cheap_step(state)     -> action from the cached model/plan (O(1))
#   expensive_plan(state) -> (action, calm_bool)  # your costly call
#   error_signal(state)   -> scalar >= 0, e.g. |prediction - truth| / scale
action, mode = clutch.step(state, cheap_step, expensive_plan, error_signal)"""


def report(sw, best, fallback, limit, tol, cost_per_call):
    ref, never = sw["ref"], sw["never"]
    win = best or fallback
    lines = [f"#### Result on your {sw['n']:,}-point series (window {sw['window']})\n"]
    lines.append("| strategy | MAE | expensive calls | training samples |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| refit every step | {ref['mae']:.4g} | {ref['refits']:,} | {ref['refit_samples']:,} |")
    if win:
        tag = "tuned clutch" if best else "closest clutch (outside tolerance)"
        lines.append(f"| **{tag}** | **{win['mae']:.4g}** | **{win['refits']:,}** | **{win['samples']:,}** |")
    lines.append(f"| never refit | {never['mae']:.4g} | {never['refits']} | {never['refit_samples']:,} |")
    lines.append("")
    if best is None:
        lines.append(f"⚠️ **No gate config stayed within {tol*100:.0f}% of the refit-every-step MAE "
                     f"(limit {limit:.4g}).** Honest verdict: on this series the cheap extrapolation "
                     "itself loses accuracy between refits, so gating is not free here.")
        if fallback:
            inc = (fallback["mae"] / ref["mae"] - 1) * 100
            frac = fallback["samples"] / (ref["refit_samples"] or 1) * 100
            lines.append(f"\nClosest trade-off on the Pareto frontier: `{fallback['gate']}` "
                         f"{fallback['params']} — **{frac:.1f}% of the compute for +{inc:.1f}% MAE**. "
                         "Take it only if that accuracy hit is acceptable; otherwise refit every step.")
        return "\n".join(lines), win
    frac = best["samples"] / ref["refit_samples"] if ref["refit_samples"] else 1.0
    saved_calls = ref["refits"] - best["refits"]
    lines.append(f"**Winner: `{best['gate']}` {best['params']}** — matches refit-every-step accuracy "
                 f"(within {tol*100:.0f}%) using **{frac*100:.1f}% of the training compute** "
                 f"and **{best['refits']:,} expensive calls instead of {ref['refits']:,}**.")
    if cost_per_call and cost_per_call > 0:
        lines.append(f"\n💰 At **${cost_per_call:.4g} per expensive call** (e.g. an LLM re-plan), "
                     f"this trace costs **${ref['refits']*cost_per_call:,.2f}** always-refit vs "
                     f"**${best['refits']*cost_per_call:,.2f}** with the clutch — "
                     f"**${saved_calls*cost_per_call:,.2f} saved ({(1-best['refits']/ref['refits'])*100:.1f}%)** "
                     f"on this trace alone.")
    return "\n".join(lines), win


# ------------------------------------------------------------------ demo presets
def preset_series(name, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(900, dtype=float)
    if name.startswith("Server latency"):
        y = 120 + 8 * np.sin(t / 40) + rng.normal(0, 6, len(t))
        for s in (200, 480, 700):                      # incidents
            y[s:s + 60] += np.linspace(0, 140, 60) * rng.uniform(0.6, 1.2)
        return y
    if name.startswith("Sensor with"):
        y = np.cumsum(rng.normal(0, 0.05, len(t)))
        for s in (150, 400, 650, 800):
            y[s:] += rng.uniform(-3, 3)                # calibration jumps
        return y + rng.normal(0, 0.15, len(t))
    # "Price-like random walk with regime shifts"
    vol = np.where((t > 300) & (t < 500), 0.9, 0.25)
    y = 100 + np.cumsum(rng.normal(0.02, 1, len(t)) * vol)
    return y
