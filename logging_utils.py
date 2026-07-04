"""CSV logging utilities for drone-racing experiments.

This module is meant to be imported by the simulation / controller entry
point (e.g. ``scripts/sim.py`` or the controller's ``compute_control``). It
has *no third-party dependencies* so it can run inside the flight loop
without adding import weight or measurable overhead: rows are buffered in
memory and written once at the end of a run with the standard-library
``csv`` module.

Usage
-----
One :class:`RunLogger` per episode (one lap attempt)::

    from analysis.logging_utils import RunLogger

    log = RunLogger(output_dir="results", run_id="lvl2_pmm_seed03",
                    level=2, planner="pmm", seed=3)

    # inside the control loop:
    log.record_state(t, pos, vel, ref_pos=ref, theta=theta, s=s)
    log.record_solve(t, solve_time_ms, sqp_iters, qp_status, cost)
    log.record_tracking(t, contour_err, lag_err, speed, v_theta, v_theta_ref)
    log.record_control(t, u)                      # u = [roll,pitch,yaw,thrust,a_theta]

    # when a plan is (re)computed:
    log.record_plan(t_trigger, plan_time_ms, n_vel_samples, horizon,
                    path_length, cost, min_clearance, is_replan=True)

    # discrete events:
    log.record_event(t, "gate_passed", gate_idx=1)
    log.record_event(t, "collision", detail="obstacle_2")

    # at the end of the episode:
    log.finish(lap_time=lap_time, gates_passed=4, success=True, collisions=0)

``finish`` writes one CSV per stream into ``results/<run_id>/`` and appends a
single summary row to ``results/runs_summary.csv`` (shared across all runs).

The column names below are the contract with ``plotting.py`` — keep them in
sync if you extend the schema.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Sequence


# --- CSV schemas (column order is preserved on write) --------------------
_FIELDS: Dict[str, List[str]] = {
    "trajectory": ["t", "x", "y", "z", "vx", "vy", "vz",
                   "ref_x", "ref_y", "ref_z", "theta", "s"],
    "mpc_solve": ["t", "solve_time_ms", "sqp_iters", "qp_status", "cost"],
    "tracking": ["t", "contour_err", "lag_err", "speed",
                 "v_theta", "v_theta_ref"],
    "control": ["t", "roll_cmd", "pitch_cmd", "yaw_cmd",
                "thrust_cmd", "a_theta"],
    "planner_stats": ["plan_id", "t_trigger", "is_replan", "plan_time_ms",
                      "n_vel_samples", "horizon", "path_length", "cost",
                      "min_clearance", "n_cubic", "fallback"],
    # Only used if your planner exposes an iterative solver loss.
    "optimization": ["plan_id", "iteration", "loss"],
    "events": ["t", "event_type", "gate_idx", "detail"],
}

_SUMMARY_FIELDS = ["run_id", "level", "planner", "seed", "lap_time",
                   "gates_passed", "success", "collisions",
                   "n_replans", "mean_solve_ms", "max_solve_ms",
                   "mean_contour_err", "min_clearance"]


class RunLogger:
    """Buffers per-run telemetry and writes it to CSV on ``finish``."""

    def __init__(self, output_dir: str, run_id: str,
                 level: int, planner: str, seed: int = 0) -> None:
        self.output_dir = output_dir
        self.run_id = run_id
        self.level = level
        self.planner = planner
        self.seed = seed
        self._buffers: Dict[str, List[Dict[str, Any]]] = {
            name: [] for name in _FIELDS
        }
        self._plan_counter = 0

    # -- recording ------------------------------------------------------
    def record_state(self, t: float, pos: Sequence[float],
                     vel: Sequence[float],
                     ref_pos: Optional[Sequence[float]] = None,
                     theta: float = float("nan"),
                     s: float = float("nan")) -> None:
        """Vehicle state and the reference it is tracking, per control tick."""
        rx, ry, rz = (ref_pos if ref_pos is not None
                      else (float("nan"),) * 3)
        self._buffers["trajectory"].append(dict(
            t=t, x=pos[0], y=pos[1], z=pos[2],
            vx=vel[0], vy=vel[1], vz=vel[2],
            ref_x=rx, ref_y=ry, ref_z=rz, theta=theta, s=s))

    def record_solve(self, t: float, solve_time_ms: float,
                     sqp_iters: int = -1, qp_status: int = 0,
                     cost: float = float("nan")) -> None:
        """MPC solver diagnostics, per control tick."""
        self._buffers["mpc_solve"].append(dict(
            t=t, solve_time_ms=solve_time_ms, sqp_iters=sqp_iters,
            qp_status=qp_status, cost=cost))

    def record_tracking(self, t: float, contour_err: float, lag_err: float,
                        speed: float, v_theta: float,
                        v_theta_ref: float = float("nan")) -> None:
        """Tracking errors and progress speed, per control tick."""
        self._buffers["tracking"].append(dict(
            t=t, contour_err=contour_err, lag_err=lag_err, speed=speed,
            v_theta=v_theta, v_theta_ref=v_theta_ref))

    def record_control(self, t: float, u: Sequence[float]) -> None:
        """Control input u = [roll, pitch, yaw, thrust, a_theta]."""
        self._buffers["control"].append(dict(
            t=t, roll_cmd=u[0], pitch_cmd=u[1], yaw_cmd=u[2],
            thrust_cmd=u[3], a_theta=u[4] if len(u) > 4 else float("nan")))

    def record_plan(self, t_trigger: float, plan_time_ms: float,
                    n_vel_samples: int, horizon: int, path_length: float,
                    cost: float, min_clearance: float = float("nan"),
                    n_cubic: int = -1, fallback: bool = False,
                    is_replan: bool = False) -> int:
        """One (re)planning event. Returns the plan_id (for optimization log)."""
        plan_id = self._plan_counter
        self._plan_counter += 1
        self._buffers["planner_stats"].append(dict(
            plan_id=plan_id, t_trigger=t_trigger, is_replan=int(is_replan),
            plan_time_ms=plan_time_ms, n_vel_samples=n_vel_samples,
            horizon=horizon, path_length=path_length, cost=cost,
            min_clearance=min_clearance, n_cubic=n_cubic,
            fallback=int(fallback)))
        return plan_id

    def record_opt_iteration(self, plan_id: int, iteration: int,
                             loss: float) -> None:
        """Per-iteration solver loss for a given plan.

        Only call this if your JAX planner runs an iterative optimizer whose
        objective is meaningful to plot. For a pure graph-search planner this
        stream can be left empty.
        """
        self._buffers["optimization"].append(dict(
            plan_id=plan_id, iteration=iteration, loss=loss))

    def record_event(self, t: float, event_type: str,
                     gate_idx: int = -1, detail: str = "") -> None:
        """Discrete event: 'gate_passed', 'collision', 'crash', ..."""
        self._buffers["events"].append(dict(
            t=t, event_type=event_type, gate_idx=gate_idx, detail=detail))

    # -- writing --------------------------------------------------------
    def finish(self, lap_time: float, gates_passed: int, success: bool,
               collisions: int = 0) -> str:
        """Write all per-stream CSVs and append the summary row.

        Returns the run directory path.
        """
        run_dir = os.path.join(self.output_dir, self.run_id)
        os.makedirs(run_dir, exist_ok=True)

        for name, rows in self._buffers.items():
            path = os.path.join(run_dir, f"{name}.csv")
            with open(path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=_FIELDS[name])
                writer.writeheader()
                writer.writerows(rows)

        self._append_summary(lap_time, gates_passed, success, collisions)
        return run_dir

    def _append_summary(self, lap_time: float, gates_passed: int,
                        success: bool, collisions: int) -> None:
        solves = self._buffers["mpc_solve"]
        tracks = self._buffers["tracking"]
        plans = self._buffers["planner_stats"]
        solve_ms = [r["solve_time_ms"] for r in solves]
        contour = [abs(r["contour_err"]) for r in tracks]
        clearances = [r["min_clearance"] for r in plans
                      if r["min_clearance"] == r["min_clearance"]]  # drop NaN

        row = dict(
            run_id=self.run_id, level=self.level, planner=self.planner,
            seed=self.seed, lap_time=lap_time, gates_passed=gates_passed,
            success=int(success), collisions=collisions,
            n_replans=sum(r["is_replan"] for r in plans),
            mean_solve_ms=_mean(solve_ms), max_solve_ms=_max(solve_ms),
            mean_contour_err=_mean(contour), min_clearance=_min(clearances))

        summary_path = os.path.join(self.output_dir, "runs_summary.csv")
        os.makedirs(self.output_dir, exist_ok=True)
        write_header = not os.path.exists(summary_path)
        with open(summary_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# --- tiny NaN-safe aggregates (avoid importing numpy in the flight loop) --
def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _max(xs: Sequence[float]) -> float:
    return max(xs) if xs else float("nan")


def _min(xs: Sequence[float]) -> float:
    return min(xs) if xs else float("nan")
