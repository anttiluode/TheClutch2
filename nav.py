"""
nav.py — the grid-navigation substrate for the Clutch demo.

Refactored from Antti Luode's benchmark.py so the app can:
  (a) run a single seed and capture per-step frames (grid, position, mode) for animation,
  (b) run the full multi-seed benchmark table (the honest measurement),
with configurable gate parameters exposed to the UI.

Compute is counted honestly as BFS cells expanded — the real O(N^2) cost of replanning.
"""

import numpy as np
from collections import deque
from clutch import Clutch, MagnitudeGate, AcceleratorGate, FilteredAcceleratorGate

W = H = 60
A = (8, 30)
B = (52, 30)


def in_bounds(x, y):
    return 0 <= x < W and 0 <= y < H


def bfs(grid, start, goal):
    if grid[start[1], start[0]] or grid[goal[1], goal[0]]:
        return None, 0
    q = deque([start]); came = {start: None}; expanded = 0
    while q:
        cur = q.popleft(); expanded += 1
        if cur == goal:
            path = []; c = cur
            while c is not None:
                path.append(c); c = came[c]
            return path[::-1], expanded
        cx, cy = cur
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if in_bounds(nx, ny) and not grid[ny, nx] and (nx, ny) not in came:
                came[(nx, ny)] = cur; q.append((nx, ny))
    return None, expanded


def connected(grid):
    p, _ = bfs(grid, A, B)
    return p is not None


def make_wall_schedule(rng):
    """Barriers at distinct x, distinct times; reject any barrier that disconnects A-B."""
    grid = np.zeros((H, W), dtype=bool)
    schedule = {}
    times = sorted(rng.choice(range(15, 250), size=5, replace=False))
    xs = list(rng.choice(range(18, 46), size=5, replace=False))
    for t, bx in zip(times, xs):
        placed = None
        for _ in range(12):
            gap = int(rng.integers(8, H - 8))
            cells = [(bx, y) for y in range(H) if abs(y - gap) > 5]
            trial = grid.copy()
            for (x, y) in cells:
                if (x, y) != A and (x, y) != B:
                    trial[y, x] = True
            if connected(trial):
                placed = cells; grid = trial; break
        if placed:
            schedule[int(t)] = placed
    return schedule


def make_gate(strategy, gate_params):
    gp = gate_params or {}
    if strategy == "CLUTCH_MAG":
        return MagnitudeGate(gain=gp.get("gain", 5.0),
                             leak=gp.get("leak", 0.5),
                             trip=gp.get("trip_mag", 10.0))
    if strategy == "CLUTCH_ACC":
        return AcceleratorGate(trip=gp.get("trip_acc", 0.9),
                               refractory=gp.get("refractory", 0))
    if strategy == "CLUTCH_ACC_REF":
        return AcceleratorGate(trip=gp.get("trip_acc", 0.9),
                               refractory=gp.get("refractory", 4))
    if strategy == "CLUTCH_ACC_FILT":
        return FilteredAcceleratorGate(trip=gp.get("trip_acc", 0.9),
                                       refractory=gp.get("refractory", 4),
                                       alpha=gp.get("alpha", 0.4))
    return None


class Runner:
    def __init__(self, strategy, seed, p_noise=0.0, max_steps=1500,
                 gate_params=None, capture=False):
        self.rng = np.random.default_rng(seed)
        self.strategy = strategy
        self.p_noise = p_noise
        self.max_steps = max_steps
        self.gate_params = gate_params or {}
        self.capture = capture
        self.grid = np.zeros((H, W), dtype=bool)
        self.schedule = make_wall_schedule(self.rng)
        self.pos = A
        self.target = B
        self.path = []
        self.idx = 0
        self.expanded_total = 0
        self.patrols_done = 0
        self.target_patrols = 3
        self.frames = []  # (walls_copy, pos, mode, cum_expanded, target)

    def perceived_blocked(self, cell):
        real = self.grid[cell[1], cell[0]]
        if not real and self.p_noise > 0 and self.rng.random() < self.p_noise:
            return True
        return real

    def cheap_step(self, _):
        if self.idx + 1 >= len(self.path):
            return None
        nxt = self.path[self.idx + 1]
        if self.perceived_blocked(nxt):
            return None
        return nxt

    def expensive_plan(self, _):
        path, expanded = bfs(self.grid, self.pos, self.target)
        self.expanded_total += expanded
        if path is None or len(path) < 2:
            return None, False
        self.path = path
        self.idx = 1
        return path[1], True

    def error_signal(self, _):
        if self.idx + 1 >= len(self.path):
            return 1.0
        return 1.0 if self.perceived_blocked(self.path[self.idx + 1]) else 0.0

    def _snap(self, mode):
        if self.capture:
            self.frames.append((self.grid.copy(), self.pos, mode,
                                self.expanded_total, self.target))

    def run(self):
        clutch = None
        if self.strategy.startswith("CLUTCH"):
            clutch = Clutch(make_gate(self.strategy, self.gate_params))

        if self.strategy == "ALWAYS_HABITUAL":
            path, e = bfs(self.grid, self.pos, self.target)
            self.expanded_total += e
            if path:
                self.path, self.idx = path, 0

        for t in range(self.max_steps):
            if t in self.schedule:
                for (x, y) in self.schedule[t]:
                    if (x, y) != A and (x, y) != B:
                        self.grid[y, x] = True

            mode = "HABITUAL"
            if self.strategy == "ALWAYS_COGNITIVE":
                path, e = bfs(self.grid, self.pos, self.target)
                self.expanded_total += e
                mode = "COGNITIVE"
                if path and len(path) >= 2:
                    self.pos = path[1]

            elif self.strategy == "ALWAYS_HABITUAL":
                nxt = self.cheap_step(None)
                if nxt is not None:
                    self.pos = nxt; self.idx += 1

            else:
                action, mode = clutch.step(None, self.cheap_step,
                                           self.expensive_plan, self.error_signal)
                if action is not None:
                    self.pos = action
                    if mode == "HABITUAL":
                        self.idx += 1

            self._snap(mode)

            if self.pos == self.target:
                self.patrols_done += 1
                if self.patrols_done >= self.target_patrols:
                    return self._result(True, t + 1, clutch)
                self.target = A if self.target == B else B
                if clutch:
                    clutch.mode = "COGNITIVE"
                self.path, self.idx = [], 0

        return self._result(False, self.max_steps, clutch)

    def _result(self, success, steps, clutch):
        if clutch:
            expensive = clutch.stats.expensive_calls
            trips = clutch.stats.trips
        elif self.strategy == "ALWAYS_COGNITIVE":
            expensive, trips = steps, 0
        else:
            expensive, trips = 1, 0
        return dict(success=success, steps=steps, expanded=self.expanded_total,
                    expensive=expensive, trips=trips, frames=self.frames)


def benchmark_table(strategies, seeds, p_noise, gate_params=None):
    """Full honest measurement across seeds. Returns list of per-strategy dicts."""
    out = []
    base = None
    for strat in strategies:
        rows = [Runner(strat, s, p_noise=p_noise, gate_params=gate_params).run()
                for s in seeds]
        ok = [r for r in rows if r["success"]]

        def m(k, src):
            return float(np.mean([r[k] for r in src])) if src else float("nan")

        rec = dict(strategy=strat, success=len(ok) / len(rows),
                   steps=m("steps", ok), expanded=m("expanded", rows),
                   expensive=m("expensive", rows), trips=m("trips", rows))
        if strat == "ALWAYS_COGNITIVE":
            base = rec["expanded"]
        out.append(rec)
    for rec in out:
        rec["vs_ceiling"] = (rec["expanded"] / base * 100.0) if base else float("nan")
    return out
