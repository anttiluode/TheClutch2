"""
clutch.py — a substrate-agnostic dual-process controller.

Distilled from Antti Luode's Loom Navigator. The one reusable idea in that demo:
run a CHEAP cached policy by default, and only pay for an EXPENSIVE planner when a
"surprise" signal trips a gate. When calm, latch the expensive result back into the
cheap cache.

This module makes no assumptions about *what* the substrates are. You supply:
  - cheap_step(state)      -> next action, from the cached plan (O(1)-ish)
  - expensive_plan(state)  -> a fresh plan (may be O(N^2) or worse)
  - error_signal(state)    -> scalar in [0, inf): "how wrong was my last prediction?"

Two gate strategies are provided, corresponding to two readings of "surprise":
  - MagnitudeGate:  leaky integrator of error, trips over a threshold. (the Loom's gate)
  - AcceleratorGate: triggers on the *second difference* of error — the "accelerometer"
                     / jerk reading (Park & Cohen 2025 framing). Faster, noise-sensitive.

Nothing here is hyped. It's a clutch: it decides when to spend compute.
"""

from dataclasses import dataclass, field


class MagnitudeGate:
    """Leaky integrator of error. Trips when accumulated surprise crosses `trip`."""
    def __init__(self, gain=5.0, leak=0.5, trip=10.0, reset=0.0):
        self.gain, self.leak, self.trip, self.reset = gain, leak, trip, reset
        self.surprise = 0.0

    def update(self, err):
        self.surprise = max(0.0, self.surprise + self.gain * err - self.leak)
        return self.surprise > self.trip

    def relax(self, amount=0.5):
        self.surprise = max(0.0, self.surprise - amount)

    def clear(self):
        self.surprise = self.reset


class AcceleratorGate:
    """Second-difference ('jerk') detector. Trips on a sudden change in error.

    Optional refractory period suppresses re-triggering for `refractory` steps after
    a fire — the biological low-pass that makes a derivative signal usable in noise.
    """
    def __init__(self, trip=1.5, refractory=0):
        self.trip, self.refractory = trip, refractory
        self.e1 = 0.0   # err at t-1
        self.e2 = 0.0   # err at t-2
        self.cool = 0
        self.surprise = 0.0  # exposed for logging/UI parity with MagnitudeGate

    def update(self, err):
        accel = err - 2.0 * self.e1 + self.e2   # discrete 2nd derivative
        self.e2, self.e1 = self.e1, err
        self.surprise = abs(accel)
        if self.cool > 0:
            self.cool -= 1
            return False
        if abs(accel) > self.trip:
            self.cool = self.refractory
            return True
        return False

    def relax(self, amount=0.5):
        pass  # derivative gate has no accumulator to bleed

    def clear(self):
        self.e1 = self.e2 = 0.0
        self.cool = 0


@dataclass
class ClutchStats:
    steps: int = 0
    expensive_calls: int = 0     # how many times the planner ran
    habitual_steps: int = 0
    cognitive_steps: int = 0
    trips: int = 0               # gate fired
    history: list = field(default_factory=list)  # ('H'|'C') per step


class Clutch:
    """The controller. Owns the mode and the gate; delegates the substrates to you."""
    def __init__(self, gate):
        self.gate = gate
        self.mode = "COGNITIVE"       # start uncached: must plan first
        self.stats = ClutchStats()

    def step(self, state, cheap_step, expensive_plan, error_signal,
             latch_when_calm=True):
        """Advance one tick. Returns (action, mode).

        cheap_step(state)     -> action or None if the cache is exhausted/invalid
        expensive_plan(state) -> (action, calm_bool). calm_bool=True means "I found a
                                 clean plan, safe to latch back to habit."
        error_signal(state)   -> scalar >= 0
        """
        s = self.stats
        s.steps += 1
        err = error_signal(state)
        tripped = self.gate.update(err)
        if tripped:
            s.trips += 1

        if self.mode == "HABITUAL":
            action = cheap_step(state)
            if tripped or action is None:
                self.mode = "COGNITIVE"          # shed the habit
            else:
                self.gate.relax()
                s.habitual_steps += 1
                s.history.append("H")
                return action, "HABITUAL"

        # COGNITIVE
        action, calm = expensive_plan(state)
        s.expensive_calls += 1
        s.cognitive_steps += 1
        s.history.append("C")
        if latch_when_calm and calm:
            self.gate.clear()
            self.mode = "HABITUAL"               # latch the fresh plan
        return action, "COGNITIVE"


class FilteredAcceleratorGate(AcceleratorGate):
    """Accelerometer gate with an EMA low-pass on the error before differentiating.
    The biological analogue: dendritic integration time-constant smoothing the jerk signal.
    """
    def __init__(self, trip=1.5, refractory=0, alpha=0.4):
        super().__init__(trip=trip, refractory=refractory)
        self.alpha = alpha
        self.filt = 0.0

    def update(self, err):
        self.filt = self.alpha * err + (1 - self.alpha) * self.filt
        return super().update(self.filt)

    def clear(self):
        super().clear(); self.filt = 0.0
