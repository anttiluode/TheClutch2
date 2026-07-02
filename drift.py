"""
drift.py — a SECOND substrate for the same Clutch, to show it is not grid-specific.

Problem: an online predictor must forecast a streaming signal y_t. The signal is
piecewise-linear with occasional REGIME CHANGES (the slope/offset jumps) — the
time-series analogue of a wall dropping. Refitting a model every step is accurate but
expensive; never refitting drifts badly after a regime change. The Clutch refits only
when prediction error trips the gate.

Substrate mapping (identical Clutch API, different callbacks):
  cheap_step(state)      -> extrapolate the CACHED linear model            (O(1))
  expensive_plan(state)  -> REFIT a linear model on the last `window` obs  (O(window))
  error_signal(state)    -> normalized residual of the last prediction

This is exactly "concept-drift-gated retraining": spend training compute only when the
world has actually drifted. Compute is counted honestly as samples fit (refits x window).
"""

import numpy as np
from clutch import Clutch, MagnitudeGate, AcceleratorGate


def make_signal(rng, T=600, n_regimes=5, noise=0.4):
    """Piecewise-linear signal with abrupt regime changes."""
    change_pts = sorted(rng.choice(range(40, T - 20), size=n_regimes - 1, replace=False))
    bounds = [0] + list(change_pts) + [T]
    y = np.zeros(T)
    val = float(rng.uniform(-2, 2))
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        slope = float(rng.uniform(-0.15, 0.15))
        for t in range(lo, hi):
            val += slope
            y[t] = val
    y = y + rng.normal(0, noise, size=T)
    return y, set(change_pts)


class DriftPredictor:
    """Holds streaming state; exposes the three Clutch callbacks."""
    def __init__(self, y, window=25):
        self.y = y
        self.window = window
        self.t = 0
        self.a = 0.0          # cached slope
        self.b = float(y[0])  # cached offset
        self.origin = 0       # x-origin the cached model was fit at
        self.last_resid = 1.0 # normalized; start "surprised" so we plan first
        self.scale = np.std(y) + 1e-6
        self.pred_log = np.full(len(y), np.nan)
        self.refit_samples = 0

    def _predict_at(self, t):
        return self.a * (t - self.origin) + self.b

    # ---- Clutch callbacks -------------------------------------------------
    def cheap_step(self, _):
        return self._predict_at(self.t)          # extrapolate cache; never None

    def expensive_plan(self, _):
        lo = max(0, self.t - self.window)
        xs = np.arange(lo, self.t + 1)
        ys = self.y[lo:self.t + 1]
        self.refit_samples += len(xs)
        if len(xs) >= 2:
            a, b = np.polyfit(xs - lo, ys, 1)
            self.a, self.b, self.origin = float(a), float(b), lo
        pred = self._predict_at(self.t)
        insample = np.mean(np.abs(np.polyval([self.a, self.b], xs - lo) - ys)) if len(xs) else 0.0
        calm = (insample / self.scale) < 0.6     # clean fit -> safe to latch to habit
        return pred, calm

    def error_signal(self, _):
        return self.last_resid


def run_drift(y, change_pts, strategy, window=25, gate_params=None):
    """strategy in {ALWAYS_REFIT, NEVER_REFIT, CLUTCH_MAG, CLUTCH_ACC}."""
    gp = gate_params or {}
    pred = DriftPredictor(y, window=window)
    clutch = None
    if strategy == "CLUTCH_MAG":
        clutch = Clutch(MagnitudeGate(gain=gp.get("gain", 4.0),
                                      leak=gp.get("leak", 0.5),
                                      trip=gp.get("trip_mag", 3.0)))
    elif strategy == "CLUTCH_ACC":
        clutch = Clutch(AcceleratorGate(trip=gp.get("trip_acc", 0.8),
                                        refractory=gp.get("refractory", 3)))

    refits = 0
    trip_times = []
    for t in range(len(y)):
        pred.t = t
        if strategy == "ALWAYS_REFIT":
            p, _ = pred.expensive_plan(None); refits += 1; mode = "COGNITIVE"
        elif strategy == "NEVER_REFIT":
            if t == 0:
                p, _ = pred.expensive_plan(None); refits += 1
            else:
                p = pred.cheap_step(None)
            mode = "HABITUAL"
        else:
            before = clutch.stats.expensive_calls
            p, mode = clutch.step(None, pred.cheap_step,
                                  pred.expensive_plan, pred.error_signal)
            if clutch.stats.expensive_calls > before:
                refits += 1
                if clutch.stats.trips and mode == "COGNITIVE":
                    trip_times.append(t)
        pred.pred_log[t] = p
        # observe truth, update normalized residual for next step's gate
        pred.last_resid = abs(p - y[t]) / pred.scale

    mae = float(np.nanmean(np.abs(pred.pred_log - y)))
    return dict(strategy=strategy, mae=mae, refits=refits,
                refit_samples=pred.refit_samples, pred=pred.pred_log,
                trip_times=trip_times)


def benchmark_drift(seeds, window=25, gate_params=None, T=600):
    strategies = ["ALWAYS_REFIT", "NEVER_REFIT", "CLUTCH_MAG", "CLUTCH_ACC"]
    agg = {s: {"mae": [], "refits": [], "samples": []} for s in strategies}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        y, cps = make_signal(rng, T=T)
        for s in strategies:
            r = run_drift(y, cps, s, window=window, gate_params=gate_params)
            agg[s]["mae"].append(r["mae"])
            agg[s]["refits"].append(r["refits"])
            agg[s]["samples"].append(r["refit_samples"])
    out = []
    base = np.mean(agg["ALWAYS_REFIT"]["samples"])
    for s in strategies:
        out.append(dict(strategy=s,
                        mae=float(np.mean(agg[s]["mae"])),
                        refits=float(np.mean(agg[s]["refits"])),
                        samples=float(np.mean(agg[s]["samples"])),
                        vs_ceiling=float(np.mean(agg[s]["samples"]) / base * 100.0)))
    return out
