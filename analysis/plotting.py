"""Plotting functions for the drone-racing results section.

Each function loads the CSV streams written by :mod:`logging_utils`, produces
one publication-quality figure, and saves it via :func:`plot_style.save`.
Functions degrade gracefully: if a required CSV is missing or empty they emit
a warning and return without raising, so partial data still yields the plots
it can support.

Dependencies: matplotlib, pandas, numpy (no seaborn).

Directory convention (see logging_utils)::

    results/
        runs_summary.csv
        <run_id>/
            trajectory.csv  mpc_solve.csv  tracking.csv  control.csv
            planner_stats.csv  optimization.csv  events.csv

Most per-run plots take a ``run_dir``; aggregate plots take ``results_dir``.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_style import apply_style, save, COL_WIDTH, DBL_WIDTH, COLORS


# --- loading helpers -----------------------------------------------------
def _load(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        warnings.warn(f"missing CSV: {path}")
        return None
    df = pd.read_csv(path)
    if df.empty:
        warnings.warn(f"empty CSV: {path}")
        return None
    return df


def _stream(run_dir: str, name: str) -> Optional[pd.DataFrame]:
    return _load(os.path.join(run_dir, f"{name}.csv"))


# --- A. Planning ---------------------------------------------------------
def plot_trajectory_comparison(pmm_run_dir: str, baseline_run_dir: str,
                               out_dir: str,
                               obstacles: Optional[np.ndarray] = None,
                               gates: Optional[np.ndarray] = None,
                               name: str = "trajectory_comparison") -> None:
    """Top-down (x-y) overlay of the PMM vs. baseline flown paths.

    ``obstacles``: (N,3) array of [x, y, radius]; ``gates``: (M,2) [x, y].
    Pass these from your track config to draw the environment.
    """
    apply_style()
    pmm = _stream(pmm_run_dir, "trajectory")
    base = _stream(baseline_run_dir, "trajectory")
    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH))

    if obstacles is not None:
        for ox, oy, r in obstacles:
            ax.add_patch(plt.Circle((ox, oy), r, color=COLORS["grey"],
                                    alpha=0.3, zorder=0))
    if gates is not None:
        ax.scatter(gates[:, 0], gates[:, 1], marker="s", s=40,
                   facecolors="none", edgecolors="k", label="gates", zorder=3)

    if pmm is not None:
        ax.plot(pmm.x, pmm.y, color=COLORS["pmm"], label="PMM")
    if base is not None:
        ax.plot(base.x, base.y, color=COLORS["baseline"], ls="--",
                label="cubic spline")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="best")
    save(fig, out_dir, name)


def plot_planner_runtime(results_dir: str, out_dir: str,
                         name: str = "planner_runtime") -> None:
    """Distribution of planner runtimes, split into initial vs. replan.

    Aggregates planner_stats.csv across every run under ``results_dir``.
    """
    apply_style()
    frames = []
    for run in _iter_runs(results_dir):
        df = _stream(run, "planner_stats")
        if df is not None:
            frames.append(df)
    if not frames:
        warnings.warn("no planner_stats found")
        return
    stats = pd.concat(frames, ignore_index=True)
    initial = stats.loc[stats.is_replan == 0, "plan_time_ms"].dropna()
    replan = stats.loc[stats.is_replan == 1, "plan_time_ms"].dropna()

    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 0.7))
    data = [d.values for d in (initial, replan) if len(d)]
    labels = [lbl for lbl, d in zip(("initial", "replan"), (initial, replan))
              if len(d)]
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("plan time [ms]")
    ax.set_title("Planner runtime")
    save(fig, out_dir, name)


def plot_obstacle_clearance(results_dir: str, out_dir: str,
                            name: str = "obstacle_clearance") -> None:
    """Minimum planned clearance per run, grouped by planner."""
    apply_style()
    summary = _load(os.path.join(results_dir, "runs_summary.csv"))
    if summary is None or "min_clearance" not in summary:
        return
    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 0.7))
    groups = [g["min_clearance"].dropna().values
              for _, g in summary.groupby("planner")]
    labels = [k for k, _ in summary.groupby("planner")]
    ax.boxplot(groups, tick_labels=labels, showfliers=True)
    ax.set_ylabel("min clearance [m]")
    ax.set_title("Planned obstacle clearance")
    save(fig, out_dir, name)


def plot_optimization_loss(run_dir: str, out_dir: str,
                           name: str = "optimization_loss") -> None:
    """Solver loss vs. iteration, one line per plan_id.

    Only meaningful if your JAX planner logs an iterative loss via
    ``record_opt_iteration``. If optimization.csv is empty this is skipped.
    """
    apply_style()
    df = _stream(run_dir, "optimization")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 0.7))
    for plan_id, g in df.groupby("plan_id"):
        ax.plot(g.iteration, g.loss, label=f"plan {plan_id}", alpha=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    if df.plan_id.nunique() <= 6:
        ax.legend(loc="best")
    ax.set_title("Optimization convergence")
    save(fig, out_dir, name)


# --- B. Controller -------------------------------------------------------
def plot_tracking_error(run_dir: str, out_dir: str,
                        name: str = "tracking_error") -> None:
    """Contour and lag error over time, with RMS annotations."""
    apply_style()
    df = _stream(run_dir, "tracking")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(DBL_WIDTH, COL_WIDTH * 0.7))
    ax.plot(df.t, df.contour_err, color=COLORS["pmm"], label="contour error")
    ax.plot(df.t, df.lag_err, color=COLORS["baseline"], label="lag error")
    rms_c = float(np.sqrt(np.mean(df.contour_err ** 2)))
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error [m]")
    ax.set_title(f"Tracking error (contour RMS = {rms_c:.3f} m)")
    ax.legend(loc="best")
    save(fig, out_dir, name)


def plot_velocity_tracking(run_dir: str, out_dir: str,
                           name: str = "velocity_tracking") -> None:
    """Achieved speed vs. reference progress speed v_theta."""
    apply_style()
    df = _stream(run_dir, "tracking")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(DBL_WIDTH, COL_WIDTH * 0.7))
    ax.plot(df.t, df.speed, color=COLORS["pmm"], label="speed")
    if df.v_theta_ref.notna().any():
        ax.plot(df.t, df.v_theta_ref, color=COLORS["grey"], ls=":",
                label=r"$v_\theta$ ref")
    ax.plot(df.t, df.v_theta, color=COLORS["baseline"], ls="--",
            label=r"$v_\theta$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("speed [m/s]")
    ax.legend(loc="best")
    ax.set_title("Velocity tracking")
    save(fig, out_dir, name)


def plot_control_inputs(run_dir: str, out_dir: str,
                        name: str = "control_inputs") -> None:
    """Commanded inputs over time (roll/pitch/yaw, thrust, a_theta)."""
    apply_style()
    df = _stream(run_dir, "control")
    if df is None:
        return
    fig, axes = plt.subplots(3, 1, figsize=(DBL_WIDTH, DBL_WIDTH * 0.6),
                             sharex=True)
    axes[0].plot(df.t, df.roll_cmd, label="roll")
    axes[0].plot(df.t, df.pitch_cmd, label="pitch")
    axes[0].plot(df.t, df.yaw_cmd, label="yaw")
    axes[0].set_ylabel("attitude cmd")
    axes[0].legend(loc="upper right", ncol=3)
    axes[1].plot(df.t, df.thrust_cmd, color=COLORS["accent"])
    axes[1].set_ylabel("thrust cmd")
    axes[2].plot(df.t, df.a_theta, color=COLORS["warn"])
    axes[2].set_ylabel(r"$a_\theta$")
    axes[2].set_xlabel("time [s]")
    fig.align_ylabels(axes)
    save(fig, out_dir, name)


def plot_solve_time_hist(results_dir: str, out_dir: str,
                         budget_ms: float = 20.0,
                         name: str = "solve_time_hist") -> None:
    """Histogram of MPC solve times across all runs, with the 50 Hz budget."""
    apply_style()
    frames = [_stream(r, "mpc_solve") for r in _iter_runs(results_dir)]
    frames = [f for f in frames if f is not None]
    if not frames:
        return
    solve = pd.concat(frames, ignore_index=True).solve_time_ms.dropna()
    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 0.7))
    ax.hist(solve.values, bins=40, color=COLORS["pmm"], alpha=0.8)
    ax.axvline(budget_ms, color=COLORS["baseline"], ls="--",
               label=f"{budget_ms:.0f} ms budget")
    ax.axvline(solve.mean(), color="k", ls=":",
               label=f"mean {solve.mean():.1f} ms")
    ax.set_xlabel("solve time [ms]")
    ax.set_ylabel("count")
    ax.legend(loc="best")
    ax.set_title(f"MPC solve time (max {solve.max():.1f} ms)")
    save(fig, out_dir, name)


# --- C. Overall performance ---------------------------------------------
def plot_lap_time_comparison(results_dir: str, out_dir: str,
                             name: str = "lap_time") -> None:
    """Grouped bar chart of lap time by level and planner (mean +/- std)."""
    apply_style()
    summary = _load(os.path.join(results_dir, "runs_summary.csv"))
    if summary is None:
        return
    # Only count successful runs toward lap-time statistics.
    ok = summary[summary.success == 1]
    _grouped_bar(ok, value="lap_time", ylabel="lap time [s]",
                 title="Lap time", out_dir=out_dir, name=name)


def plot_success_rate(results_dir: str, out_dir: str,
                      name: str = "success_rate") -> None:
    """Grouped bar chart of success rate [%] by level and planner."""
    apply_style()
    summary = _load(os.path.join(results_dir, "runs_summary.csv"))
    if summary is None:
        return
    rate = (summary.groupby(["level", "planner"]).success.mean() * 100
            ).reset_index()
    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 0.7))
    _bars_from_pivot(ax, rate.pivot(index="level", columns="planner",
                                    values="success"))
    ax.set_ylabel("success rate [%]")
    ax.set_xlabel("level")
    ax.set_ylim(0, 100)
    ax.legend(loc="best")
    ax.set_title("Success rate")
    save(fig, out_dir, name)


def plot_gate_timestamps(run_dir: str, out_dir: str,
                         name: str = "gate_timestamps") -> None:
    """Timeline of gate-pass and collision events for a single run."""
    apply_style()
    df = _stream(run_dir, "events")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(DBL_WIDTH, COL_WIDTH * 0.5))
    passes = df[df.event_type == "gate_passed"]
    coll = df[df.event_type == "collision"]
    ax.scatter(passes.t, passes.gate_idx, color=COLORS["accent"],
               marker="o", label="gate passed", zorder=3)
    for _, row in passes.iterrows():
        ax.annotate(f"{row.t:.2f}s", (row.t, row.gate_idx),
                    textcoords="offset points", xytext=(0, 5), fontsize=6)
    if not coll.empty:
        ax.scatter(coll.t, [-1] * len(coll), color=COLORS["baseline"],
                   marker="x", label="collision", zorder=3)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("gate index")
    ax.legend(loc="best")
    ax.set_title("Gate completion timeline")
    save(fig, out_dir, name)


# --- shared bar-chart machinery -----------------------------------------
def _grouped_bar(summary: pd.DataFrame, value: str, ylabel: str,
                 title: str, out_dir: str, name: str) -> None:
    agg = summary.groupby(["level", "planner"])[value].agg(["mean", "std"])
    means = agg["mean"].unstack("planner")
    stds = agg["std"].unstack("planner")
    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 0.7))
    _bars_from_pivot(ax, means, yerr=stds)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("level")
    ax.legend(loc="best")
    ax.set_title(title)
    save(fig, out_dir, name)


def _bars_from_pivot(ax, pivot: pd.DataFrame,
                     yerr: Optional[pd.DataFrame] = None) -> None:
    """Draw grouped bars from a (level x planner) pivot table."""
    levels = list(pivot.index)
    planners = list(pivot.columns)
    x = np.arange(len(levels))
    width = 0.8 / max(len(planners), 1)
    for i, planner in enumerate(planners):
        err = yerr[planner].values if yerr is not None else None
        ax.bar(x + i * width, pivot[planner].values, width,
               yerr=err, capsize=2, label=str(planner),
               color=COLORS.get(str(planner), None))
    ax.set_xticks(x + width * (len(planners) - 1) / 2)
    ax.set_xticklabels([str(l) for l in levels])


def _iter_runs(results_dir: str):
    """Yield each per-run subdirectory under ``results_dir``."""
    for entry in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, entry)
        if os.path.isdir(path):
            yield path
