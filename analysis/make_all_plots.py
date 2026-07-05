"""Generate every report figure from a results directory in one command.

Usage::

    python make_all_plots.py --results results --out figures \
        --pmm-run lvl2_pmm_seed03 --baseline-run lvl2_cubic_seed03

Aggregate plots (solve-time histogram, lap time, success rate, planner
runtime, obstacle clearance) are built from every run under ``--results``.
Per-run plots (trajectory, tracking, control, gate timeline, optimization)
use the runs named by ``--pmm-run`` / ``--baseline-run``; if omitted, the
first run directory found is used for the single-run plots and the
trajectory comparison is skipped.

Track geometry for the trajectory overlay is optional. To draw gates and
obstacles, edit ``load_track`` below to return them from your config, or
pass ``--track-npz`` pointing at a file with ``obstacles`` (N,3) and
``gates`` (M,2) arrays.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np

import plotting as P


def load_track(track_npz: Optional[str]
               ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (obstacles[N,3]=x,y,r, gates[M,2]=x,y) or (None, None).

    [INSERT] Point this at your track config, or leave --track-npz unset to
    plot trajectories without the environment drawn.
    """
    if track_npz and os.path.exists(track_npz):
        data = np.load(track_npz)
        return data.get("obstacles"), data.get("gates")
    return None, None


def _first_run(results_dir: str) -> Optional[str]:
    for entry in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, entry)
        if os.path.isdir(path):
            return path
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results",
                    help="directory containing runs_summary.csv and run dirs")
    ap.add_argument("--out", default="figures", help="output directory")
    ap.add_argument("--pmm-run", default=None, help="run_id for PMM plots")
    ap.add_argument("--baseline-run", default=None,
                    help="run_id for the baseline trajectory")
    ap.add_argument("--track-npz", default=None,
                    help="optional .npz with 'obstacles' and 'gates' arrays")
    ap.add_argument("--budget-ms", type=float, default=20.0,
                    help="MPC solve-time budget line (50 Hz -> 20 ms)")
    args = ap.parse_args()

    results, out = args.results, args.out
    obstacles, gates = load_track(args.track_npz)

    # --- aggregate plots (all runs) ---
    P.plot_solve_time_hist(results, out, budget_ms=args.budget_ms)
    P.plot_lap_time_comparison(results, out)
    P.plot_success_rate(results, out)
    P.plot_planner_runtime(results, out)
    P.plot_obstacle_clearance(results, out)

    # --- single-run plots ---
    single = (os.path.join(results, args.pmm_run) if args.pmm_run
              else _first_run(results))
    if single:
        P.plot_tracking_error(single, out)
        P.plot_velocity_tracking(single, out)
        P.plot_control_inputs(single, out)
        P.plot_gate_timestamps(single, out)
        P.plot_optimization_loss(single, out)

    # --- trajectory comparison (needs both runs) ---
    if args.pmm_run and args.baseline_run:
        P.plot_trajectory_comparison(
            os.path.join(results, args.pmm_run),
            os.path.join(results, args.baseline_run),
            out, obstacles=obstacles, gates=gates)

    print(f"figures written to: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
