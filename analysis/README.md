# analysis/ — experiment logging & plotting

Two phases: **collect** telemetry while running the sim, then **plot** it.

## 1. Collect (built into `scripts/sim.py`)

Pass `--log_tag` to label the run; telemetry is written automatically:

```bash
python scripts/sim.py --config level2.toml --log_tag pmm
python scripts/sim.py --config level2.toml --log_tag cubic   # baseline, same MPCC
```

Each episode writes `results/<tag>_lvl<L>_seed<S>_run<i>/` containing
`trajectory.csv`, `tracking.csv`, `control.csv`, `mpc_solve.csv`,
`planner_stats.csv`, `events.csv`, `optimization.csv`, plus a shared
`results/runs_summary.csv`. Without `--log_tag`, nothing is written and the
sim behaves exactly as before.

For statistics, run multiple seeds (edit `seed` in the config or loop over
configs) — aim for >=10 per level/planner.

### What is captured automatically
- `tracking`/`control`: from the controller's existing `self._log_*` lists.
- `mpc_solve`: wall-clock time of each `compute_control` call (proxy for MPC
  solve time).
- `trajectory`: drone position and velocity from `obs["pos"]` / `obs["vel"]`.
- `events`: gate passes (detected from `obs["target_gate"]` advancing).
- `track.csv`: gate and obstacle positions (so the overlay plot can draw the
  track without a separate config file).
- `runs_summary`: lap time, gates passed, success, mean/max solve time, etc.

### Not captured yet (plots skip gracefully)
- Per-plan planner statistics (`planner_stats`) and per-iteration solver loss
  (`optimization`) — add calls to `RunLogger.record_plan` /
  `record_opt_iteration` from inside the planner if you want those plots.

## 2. Plot

```bash
cd analysis
python make_all_plots.py --results ../results --out ../figures \
    --pmm-run pmm_lvl2_seed3_run0 --baseline-run cubic_lvl2_seed3_run0
```

Writes publication-quality PDF+PNG figures into `../figures/`. Aggregate plots
(lap time, success rate, solve-time histogram, planner runtime, clearance) use
every run under `--results`; single-run plots use the named runs.

For a single run's **speed/position track overlay** (path coloured by speed,
gates and obstacles drawn from that run's `track.csv`):

```bash
python plot_track_overlay.py --run ../results/pmm_lvl2_seed3_run0 --out ../figures
```

Run it once per run (e.g. the PMM run and the baseline run) to compare lines.
This shows the *flown* path and *flown* speed, not the planner's internal
speed profile (the reference path is not logged yet).

Dependencies: matplotlib, pandas, numpy (no seaborn). `plotting.py` and
`plot_style.py` are imported by `make_all_plots.py` — run that, not them.
