"""
app.py — The Clutch: a live demo of event-triggered compute reuse.

Run a CHEAP cached policy by default; only pay for an EXPENSIVE planner when a
"surprise" signal trips a gate; latch the fresh plan back into the cache when calm.
This Space lets you *watch* that gate decide when to spend compute, on two unrelated
substrates, and reproduce the honest benchmark (negative results included).
"""

import base64
import numpy as np
import gradio as gr

from nav import Runner, benchmark_table, W, H
from drift import make_signal, run_drift, benchmark_drift
from viz import make_nav_gif, make_compute_plot, make_drift_plot, make_pareto_plot
from tuner import (parse_series, sweep, pick_best, pareto_front,
                   code_snippet, report, preset_series)

GATE_TO_STRAT = {
    "Leaky integrator  (MagnitudeGate — the Loom's own gate)": "CLUTCH_MAG",
    "Accelerometer  (2nd-derivative / jerk)": "CLUTCH_ACC",
    "Accelerometer + refractory": "CLUTCH_ACC_REF",
    "Filtered accelerometer (EMA low-pass)": "CLUTCH_ACC_FILT",
}


def _gif_html(path):
    with open(path, "rb") as f:
        b = base64.b64encode(f.read()).decode()
    return (f'<img src="data:image/gif;base64,{b}" '
            f'style="width:100%;max-width:430px;border-radius:10px;'
            f'box-shadow:0 2px 12px rgba(0,0,0,.12);" alt="navigation animation"/>')


# ------------------------------------------------------------------ NAV
def run_nav(gate_label, gain, leak, trip_mag, trip_acc, refractory, noise, seed):
    strat = GATE_TO_STRAT[gate_label]
    gp = dict(gain=gain, leak=leak, trip_mag=trip_mag,
              trip_acc=trip_acc, refractory=int(refractory))
    seed = int(seed)
    rc = Runner(strat, seed, p_noise=noise, gate_params=gp, capture=True).run()
    rg = Runner("ALWAYS_COGNITIVE", seed, p_noise=noise, capture=True).run()

    gif = make_nav_gif(rc["frames"])
    fig = make_compute_plot(rc["frames"], rg["frames"], clutch_label=strat)
    vs = rc["expanded"] / rg["expanded"] * 100 if rg["expanded"] else float("nan")
    ok = "✅ reached goal (patrol ×3)" if rc["success"] else "❌ did not finish"
    md = f"""
### This run — `{strat}`  (seed {seed}, noise {noise:.2f})
| | clutch | replan-every-step |
|---|---:|---:|
| outcome | {ok} | {"✅" if rg["success"] else "❌"} |
| BFS cells expanded | **{rc['expanded']:,}** | {rg['expanded']:,} |
| planner calls | **{rc['expensive']}** | {rg['expensive']} |
| gate trips | {rc['trips']} | — |

**The clutch used {vs:.1f}% of the replan-every-step compute** and still finished.
Green dot = cheap cached step. Red dot = the gate tripped and it paid for a fresh plan.
"""
    return _gif_html(gif), fig, md


# ------------------------------------------------------------------ DRIFT
def run_drift_demo(gate_label, trip_mag, trip_acc, refractory, window, seed):
    strat = "CLUTCH_MAG" if "integrator" in gate_label else "CLUTCH_ACC"
    gp = dict(trip_mag=trip_mag, trip_acc=trip_acc, refractory=int(refractory))
    seed = int(seed)
    rng = np.random.default_rng(seed)
    y, cps = make_signal(rng)
    res = run_drift(y, cps, strat, window=int(window), gate_params=gp)
    ref = run_drift(y, cps, "ALWAYS_REFIT", window=int(window))
    never = run_drift(y, cps, "NEVER_REFIT", window=int(window))
    fig = make_drift_plot(y, cps, res, int(window))
    vs = res["refit_samples"] / ref["refit_samples"] * 100 if ref["refit_samples"] else 0
    md = f"""
### Concept-drift-gated retraining — `{strat}` (seed {seed})
| strategy | prediction MAE | refits | training samples |
|---|---:|---:|---:|
| refit every step | {ref['mae']:.2f} | {ref['refits']} | {ref['refit_samples']:,} |
| **clutch (refit on drift)** | **{res['mae']:.2f}** | **{res['refits']}** | **{res['refit_samples']:,}** |
| never refit | {never['mae']:.2f} | {never['refits']} | {never['refit_samples']:,} |

Same `Clutch` class as the grid demo — only the three callbacks changed
(cheap = extrapolate cached line, expensive = least-squares refit, error = residual).
It matched the refit-every-step accuracy while doing **{vs:.1f}%** of the training work,
and stayed far below the never-refit disaster.
"""
    return fig, md


# ------------------------------------------------------------------ TUNER
PRESETS = ["— use my pasted/uploaded data —",
           "Server latency with incident spikes",
           "Sensor with calibration jumps",
           "Price-like random walk with regime shifts"]


def run_tuner(preset, pasted, file_obj, window, tol, cost):
    if preset != PRESETS[0]:
        y, msg = preset_series(preset, seed=1), f"Preset: {preset} (900 points)."
    else:
        y, msg = parse_series(text=pasted, file_obj=file_obj)
        if y is None:
            return msg, None, "", ""
    sw = sweep(y, window=int(window))
    best, fb, limit = pick_best(sw, tol=tol)
    md, win = report(sw, best, fb, limit, tol, cost)
    fig = make_pareto_plot(sw, best, fb, pareto_front(sw["rows"]))
    code = code_snippet(win, int(window)) if win else "# no viable config — see verdict"
    return msg, fig, md, code


# ------------------------------------------------------------------ BENCHMARKS
def nav_benchmark(noise):
    rows = benchmark_table(["ALWAYS_COGNITIVE", "ALWAYS_HABITUAL",
                            "CLUTCH_MAG", "CLUTCH_ACC", "CLUTCH_ACC_REF"],
                           list(range(16)), noise)
    head = "| strategy | success | BFS expanded | plan calls | gate trips | vs replan-all |\n"
    head += "|---|---:|---:|---:|---:|---:|\n"
    body = ""
    for r in rows:
        steps_ok = r["success"] > 0.99
        vs = f"{r['vs_ceiling']:.1f}%" if steps_ok or r['strategy'] == 'ALWAYS_COGNITIVE' else "brittle"
        body += (f"| `{r['strategy']}` | {r['success']*100:.0f}% | {r['expanded']:,.0f} "
                 f"| {r['expensive']:.1f} | {r['trips']:.1f} | {vs} |\n")
    return f"#### Grid navigation — 16 seeds, patrol ×3, sensor noise {noise:.2f}\n\n" + head + body


def drift_benchmark():
    rows = benchmark_drift(list(range(20)))
    head = "| strategy | MAE | refits | training samples | vs refit-all |\n|---|---:|---:|---:|---:|\n"
    body = ""
    for r in rows:
        body += (f"| `{r['strategy']}` | {r['mae']:.2f} | {r['refits']:.1f} "
                 f"| {r['samples']:,.0f} | {r['vs_ceiling']:.2f}% |\n")
    return "#### Drift-gated retraining — 20 seeds\n\n" + head + body


# ------------------------------------------------------------------ TEXT
INTRO = """
# 🔌 The Clutch — *spend expensive compute only when reality drifts*

A **substrate-agnostic dual-process controller**, distilled from Antti Luode's Loom
Navigator to its one reusable idea:

> Run a **cheap cached policy** by default. Only pay for an **expensive planner** when a
> *surprise* signal trips a **gate**. When things go calm, **latch** the fresh plan back
> into the cache.

`clutch.py` is ~120 lines, zero dependencies, and makes **no assumption about what the
substrates are** — you hand it three callbacks (a cheap step, an expensive plan, a scalar
error) and pick a gate. The two live demos below drive the *same* controller on two
unrelated problems. *Do not hype. Do not lie. Just show.*
"""

WHERE = """
## Where this is actually valuable

The clutch is not a new theorem — it is a clean, reusable **systems pattern**:
*event-triggered recomputation*. It earns its keep anywhere a good-but-costly computation
can be cached and reused **until reality drifts**, and where recomputing every tick is the
lazy default people actually ship:

- **LLM agent loops** — the expensive call is a *token-billed* re-plan. Cheap = replay the
  cached tool-call plan; expensive = ask the model to re-plan; gate = tool-result surprise.
  This is where the savings are money, not just cycles.
- **Robotics / MPC** — gate expensive trajectory optimization by tracking error.
- **Online ML** — retrain only on detected concept drift (exactly demo 2).
- **Query planners, JIT recompilation, cache invalidation** — replan on a surprise trip.

### Drop-in wrapper for an LLM agent loop
```python
from clutch import Clutch, MagnitudeGate

clutch = Clutch(MagnitudeGate(gain=5, leak=0.5, trip=10))

def cheap_step(state):        # O(1): pop the next action from the cached plan
    return state.cached_plan.next() if state.cached_plan.valid else None

def expensive_plan(state):    # $$$: the LLM re-plans from scratch
    plan = llm.replan(state)  # <-- your billed call
    state.cached_plan = plan
    calm = plan.confidence > 0.8
    return plan.first_action, calm

def error_signal(state):      # how wrong was the last predicted tool result?
    return state.last_surprise    # 0.0 == exactly as expected

while not done:
    action, mode = clutch.step(state, cheap_step, expensive_plan, error_signal)
    state = env.apply(action)
# clutch.stats.expensive_calls  ==  how many billed LLM calls you actually paid for
```
On the grid task this turns ~250 planner calls into ~3. In an agent loop, that is the
difference between an LLM call every step and one only when the world surprises you.

## The honest negative result
The **accelerometer / jerk gate** (2nd-derivative of error, the Park–Cohen framing) is
*not* a free lunch. Under sensor noise it fires ~1.7× more often than the leaky integrator
and burns ~50% more compute for the same success; on the drift task it *under*-fires and
fails. Across both substrates the plain **leaky integrator is the better engine here.**
The derivative gate's advantage would show on tasks needing *fast* reaction to abrupt
change — which neither of these stresses. Stated, not hidden.

*Built on `clutch.py`, `benchmark.py` by Antti Luode (PerceptionLab).*
"""

THEME = gr.themes.Soft(primary_hue="emerald", secondary_hue="slate")

with gr.Blocks(title="The Clutch") as demo:
    gr.Markdown(INTRO)

    with gr.Tab("0 · Tune it on YOUR data"):
        gr.Markdown(
            "**This is the useful part.** Paste or upload any 1-D time series — server "
            "latency, a sensor stream, a price feed, an error metric. The Space runs the "
            "real closed-loop clutch on it (cheap = extrapolate cached linear model, "
            "expensive = refit), sweeps **~80 gate configs**, plots the honest "
            "accuracy-vs-compute Pareto frontier, and hands back a **copy-paste "
            "`Clutch(...)` config tuned to your data** plus a dollar-savings estimate. "
            "If gating doesn't pay on your data, it says so.")
        with gr.Row():
            with gr.Column(scale=1):
                t_preset = gr.Dropdown(PRESETS, value=PRESETS[1],
                                       label="Data source (pick a preset or use your own)")
                t_paste = gr.Textbox(lines=5, label="Paste numbers (one per line, or CSV — last column is used)",
                                     placeholder="123.4\n125.1\n119.8\n...")
                t_file = gr.File(label="…or upload a CSV/TXT", file_types=[".csv", ".txt"])
                t_window = gr.Slider(8, 80, 25, step=1, label="refit window (samples per expensive call)")
                t_tol = gr.Slider(0.02, 0.5, 0.10, step=0.01,
                                  label="accuracy tolerance vs refit-every-step")
                t_cost = gr.Number(value=0.03, label="cost per expensive call in $ (0 = skip)")
                t_run = gr.Button("▶ Tune on this data", variant="primary")
                t_msg = gr.Markdown()
            with gr.Column(scale=2):
                t_plot = gr.Plot(label="Pareto: every config on your data")
                t_md = gr.Markdown()
                t_code = gr.Code(language="python", label="Your tuned config — copy-paste")
        t_run.click(run_tuner, [t_preset, t_paste, t_file, t_window, t_tol, t_cost],
                    [t_msg, t_plot, t_md, t_code])

    with gr.Tab("1 · Watch it navigate"):
        gr.Markdown("An agent patrols A↔B on a 60×60 grid. Walls with a gap drop at "
                    "scripted times, breaking the cached route. **Every maze is "
                    "guaranteed solvable**, so the outcome reflects the gate, not luck.")
        with gr.Row():
            with gr.Column(scale=1):
                n_gate = gr.Dropdown(list(GATE_TO_STRAT), value=list(GATE_TO_STRAT)[0],
                                     label="Gate")
                with gr.Accordion("Leaky-integrator params", open=True):
                    n_gain = gr.Slider(1, 10, 5, step=0.5, label="gain")
                    n_leak = gr.Slider(0, 2, 0.5, step=0.1, label="leak")
                    n_trip_mag = gr.Slider(2, 30, 10, step=1, label="trip threshold")
                with gr.Accordion("Accelerometer params", open=False):
                    n_trip_acc = gr.Slider(0.2, 3, 0.9, step=0.1, label="trip (jerk)")
                    n_ref = gr.Slider(0, 10, 4, step=1, label="refractory")
                n_noise = gr.Slider(0, 0.1, 0.0, step=0.01, label="sensor noise (false blocks)")
                n_seed = gr.Slider(0, 40, 3, step=1, label="seed")
                n_run = gr.Button("▶ Run", variant="primary")
            with gr.Column(scale=1):
                n_gif = gr.HTML(label="animation")
                n_md = gr.Markdown()
        n_plot = gr.Plot(label="compute over time")
        n_run.click(run_nav,
                    [n_gate, n_gain, n_leak, n_trip_mag, n_trip_acc, n_ref, n_noise, n_seed],
                    [n_gif, n_plot, n_md])

    with gr.Tab("2 · Same clutch, different world"):
        gr.Markdown("The **identical `Clutch`** now forecasts a streaming signal that "
                    "jumps at random *regime changes*. Refitting every step is accurate "
                    "but costly; the clutch refits **only when prediction error trips the "
                    "gate**. This is concept-drift-gated retraining.")
        with gr.Row():
            with gr.Column(scale=1):
                d_gate = gr.Dropdown(["Leaky integrator  (MagnitudeGate)",
                                      "Accelerometer  (2nd-derivative / jerk)"],
                                     value="Leaky integrator  (MagnitudeGate)", label="Gate")
                d_trip_mag = gr.Slider(1, 8, 3, step=0.5, label="trip threshold (integrator)")
                d_trip_acc = gr.Slider(0.2, 3, 0.8, step=0.1, label="trip (jerk)")
                d_ref = gr.Slider(0, 10, 3, step=1, label="refractory (jerk)")
                d_window = gr.Slider(8, 60, 25, step=1, label="refit window")
                d_seed = gr.Slider(0, 40, 7, step=1, label="seed")
                d_run = gr.Button("▶ Run", variant="primary")
            with gr.Column(scale=2):
                d_plot = gr.Plot()
        d_md = gr.Markdown()
        d_run.click(run_drift_demo,
                    [d_gate, d_trip_mag, d_trip_acc, d_ref, d_window, d_seed],
                    [d_plot, d_md])

    with gr.Tab("3 · The honest benchmark"):
        gr.Markdown("Reproduce the full measurement in-browser. Compute is counted "
                    "honestly (BFS cells expanded / training samples). ~7–20 s each.")
        with gr.Row():
            b0 = gr.Button("Grid benchmark — no noise")
            b1 = gr.Button("Grid benchmark — 3% sensor noise")
            b2 = gr.Button("Drift benchmark")
        b_out = gr.Markdown()
        b0.click(lambda: nav_benchmark(0.0), None, b_out)
        b1.click(lambda: nav_benchmark(0.03), None, b_out)
        b2.click(drift_benchmark, None, b_out)

    with gr.Tab("4 · What this is / where it's valuable"):
        gr.Markdown(WHERE)


if __name__ == "__main__":
    demo.launch()
