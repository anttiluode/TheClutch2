"""
benchmark.py — measure the clutch on a dynamic-environment navigation task.

Agent patrols A<->B on a grid. Barrier walls (with a gap) drop at scripted times, so a
cached route periodically breaks. Wall schedules are rejected if they would disconnect
A from B, so every environment stays solvable — success-rate gaps then reflect the GATE,
not luck. Compute is measured as BFS cells expanded (the honest O(N^2) cost).

Strategies:
  ALWAYS_COGNITIVE : replan every step.                 quality ceiling / most expensive
  ALWAYS_HABITUAL  : plan once, never replan.           cheap / brittle
  CLUTCH_MAG       : replan when leaky-integrator trips. (the Loom's gate)
  CLUTCH_ACC       : replan when 2nd-derivative trips.   (accelerometer / jerk)
  CLUTCH_ACC_REF   : accelerometer + refractory period.  (noise-tolerant)
"""

import numpy as np
from collections import deque
from clutch import Clutch, MagnitudeGate, AcceleratorGate

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
        for nx, ny in ((cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)):
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
        for _ in range(12):  # try gaps until one keeps A-B connected
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


class Runner:
    def __init__(self, strategy, seed, p_noise=0.0, max_steps=1500):
        self.rng = np.random.default_rng(seed)
        self.strategy = strategy
        self.p_noise = p_noise
        self.max_steps = max_steps
        self.grid = np.zeros((H, W), dtype=bool)
        self.schedule = make_wall_schedule(self.rng)
        self.pos = A
        self.target = B
        self.path = []
        self.idx = 0
        self.expanded_total = 0
        self.patrols_done = 0
        self.target_patrols = 3

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
        self.idx = 1          # we are about to move onto path[1]
        return path[1], True

    def error_signal(self, _):
        if self.idx + 1 >= len(self.path):
            return 1.0
        return 1.0 if self.perceived_blocked(self.path[self.idx + 1]) else 0.0

    def run(self):
        clutch = None
        if self.strategy == "CLUTCH_MAG":
            clutch = Clutch(MagnitudeGate(gain=5, leak=0.5, trip=10))
        elif self.strategy == "CLUTCH_ACC":
            clutch = Clutch(AcceleratorGate(trip=0.9, refractory=0))
        elif self.strategy == "CLUTCH_ACC_REF":
            clutch = Clutch(AcceleratorGate(trip=0.9, refractory=4))

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

            if self.strategy == "ALWAYS_COGNITIVE":
                path, e = bfs(self.grid, self.pos, self.target)
                self.expanded_total += e
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
                    expensive=expensive, trips=trips)


def summarize(strategy, p_noise, seeds):
    rows = [Runner(strategy, s, p_noise=p_noise).run() for s in seeds]
    ok = [r for r in rows if r["success"]]
    def m(k, src): return np.mean([r[k] for r in src]) if src else float("nan")
    return dict(strategy=strategy, success=len(ok)/len(rows),
                steps=m("steps", ok), expanded=m("expanded", rows),
                expensive=m("expensive", rows), trips=m("trips", rows))


if __name__ == "__main__":
    seeds = list(range(16))
    strategies = ["ALWAYS_COGNITIVE", "ALWAYS_HABITUAL",
                  "CLUTCH_MAG", "CLUTCH_ACC", "CLUTCH_ACC_REF"]
    for p_noise in (0.0, 0.03):
        print(f"\n=== sensor noise p={p_noise}  ({len(seeds)} seeds, patrol x3, {W}x{H}) ===")
        print(f"{'strategy':<18}{'success':>8}{'steps':>8}{'expanded':>11}"
              f"{'planCalls':>10}{'gateTrips':>10}")
        base = None; results = []
        for s in strategies:
            r = summarize(s, p_noise, seeds); results.append(r)
            if s == "ALWAYS_COGNITIVE":
                base = r["expanded"]
            steps = "-" if np.isnan(r["steps"]) else f"{r['steps']:.0f}"
            print(f"{r['strategy']:<18}{r['success']*100:>7.0f}%{steps:>8}"
                  f"{r['expanded']:>11.0f}{r['expensive']:>10.1f}{r['trips']:>10.1f}")
        print("  compute vs ALWAYS_COGNITIVE (successful strategies):")
        for r in results:
            if r["success"] > 0.99:
                print(f"    {r['strategy']:<18} {r['expanded']/base*100:>5.1f}%")
