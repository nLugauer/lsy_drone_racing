"""Standalone: speed/position overlay of the track for path-planner analysis.

Draws the flown path of a single run, colour-coded by speed, over the gates
and obstacles (read automatically from the run's track.csv). Useful for
inspecting where the planner's line is fast vs. slow, and for eyeballing the
PMM line against the baseline.

Usage::

    cd analysis
    python plot_track_overlay.py --run ../results/pmm_lvl2_seed3_run0 \
        --out ../figures

Run it once per run you want to inspect (e.g. the PMM run and the baseline
run) to get one figure each.
"""

from __future__ import annotations

import argparse
import os

import plotting as P


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True,
                    help="path to a single results/<run_id>/ directory")
    ap.add_argument("--out", default="figures", help="output directory")
    ap.add_argument("--obstacle-radius", type=float, default=0.05,
                    help="radius used to draw obstacles (m); track.csv stores "
                         "only positions")
    ap.add_argument("--name", default=None,
                    help="figure filename (default: track_speed_<run_id>)")
    args = ap.parse_args()

    run_id = os.path.basename(os.path.normpath(args.run))
    name = args.name or f"track_speed_{run_id}"
    P.plot_track_speed_overlay(args.run, args.out,
                               obstacle_radius=args.obstacle_radius, name=name)
    print(f"figure written to: {os.path.abspath(args.out)}/{name}.pdf")


if __name__ == "__main__":
    main()
