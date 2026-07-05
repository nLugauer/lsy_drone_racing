"""Glue between ``scripts/sim.py`` and :class:`analysis.logging_utils.RunLogger`.

This module collects the data that is only visible from the simulation loop
(drone position, per-tick control-step time, gate-pass events, episode
outcome) and merges it with the telemetry the controller already accumulates
in its ``self._log_*`` lists. It is deliberately dependency-free so it can be
imported and used inside the flight loop without side effects.

Typical use in ``sim.py`` (see that file for the exact hooks)::

    rec = SimRecorder(config, config_name, run_idx=r, tag="pmm")
    # in the loop, after each compute_control:
    rec.record_tick(curr_time, obs, control_ms)
    # after the episode:
    rec.finish(controller, obs, curr_time)

If anything goes wrong (missing keys, unusual controller) the recorder fails
soft: it prints a warning and never raises, so logging can never break a run.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple


def _level_from_config_name(name: str) -> int:
    """Parse the difficulty level from a config filename like 'level2.toml'."""
    m = re.search(r"level(\d+)", str(name))
    return int(m.group(1)) if m else -1


class SimRecorder:
    """Accumulates per-run telemetry from the sim loop and writes it on finish."""

    def __init__(self, config: Any, config_name: str, run_idx: int = 0,
                 tag: str = "run", results_dir: Optional[str] = None) -> None:
        # env step frequency -> control period (curr_time = tick / freq in sim.py)
        try:
            self.dt = 1.0 / float(config.env.freq)
        except Exception:
            self.dt = 0.02
        try:
            self.n_gates = len(config.env.track.gates)
        except Exception:
            self.n_gates = -1
        self.level = _level_from_config_name(config_name)
        try:
            self.seed = int(getattr(config.env, "seed", 0) or 0)
        except Exception:
            self.seed = 0
        self.tag = tag
        self.run_id = f"{tag}_lvl{self.level}_seed{self.seed}_run{run_idx}"
        self.results_dir = results_dir or os.path.abspath("results")

        self._traj: List[Tuple[float, float, float, float]] = []
        self._control_ms: List[float] = []
        self._events: List[Tuple[float, str, int]] = []
        self._prev_target: Optional[int] = None

    # -- per-tick ------------------------------------------------------
    def record_tick(self, t: float, obs: Dict[str, Any],
                    control_ms: float) -> None:
        """Call once per control tick with the wall-clock cost of compute_control."""
        try:
            pos = obs.get("pos") if hasattr(obs, "get") else None
            vel = obs.get("vel") if hasattr(obs, "get") else None
            if pos is not None and len(pos) >= 3:
                if vel is not None and len(vel) >= 3:
                    vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
                else:
                    vx = vy = vz = float("nan")
                self._traj.append((t, float(pos[0]), float(pos[1]),
                                   float(pos[2]), vx, vy, vz))
            self._control_ms.append(float(control_ms))
            # Detect gate passes via the target-gate index advancing.
            tg = int(obs["target_gate"])
            if self._prev_target is not None and tg != self._prev_target:
                if self._prev_target >= 0:
                    self._events.append((t, "gate_passed", self._prev_target))
            self._prev_target = tg
        except Exception as exc:  # never let logging break the sim
            print(f"[analysis] record_tick skipped: {exc}")

    # -- end of episode ------------------------------------------------
    def finish(self, controller: Any, obs: Dict[str, Any],
               curr_time: float) -> Optional[str]:
        """Write all CSV streams for this run. Returns the run directory."""
        try:
            from analysis.logging_utils import RunLogger
        except Exception as exc:
            print(f"[analysis] could not import RunLogger, no logs written: {exc}")
            return None

        try:
            log = RunLogger(self.results_dir, self.run_id, self.level,
                            self.tag, self.seed)

            # 1) controller telemetry (tracking + control), if exposed.
            contour = list(getattr(controller, "_log_contour", []) or [])
            n = len(contour)
            if n:
                lag = _col(controller, "_log_lag", n)
                roll = _col(controller, "_log_roll", n)
                pitch = _col(controller, "_log_pitch", n)
                thrust = _col(controller, "_log_thrust", n)
                v_theta = _col(controller, "_log_v_theta", n)
                a_theta = _col(controller, "_log_a_theta", n)
                for i in range(n):
                    t = i * self.dt
                    log.record_tracking(t, contour[i], lag[i],
                                        float("nan"), v_theta[i])
                    log.record_control(t, [roll[i], pitch[i], 0.0,
                                           thrust[i], a_theta[i]])
            else:
                print("[analysis] controller exposes no _log_* telemetry; "
                      "writing trajectory/solve/events only.")

            # 2) control-step time (proxy for MPC solve time), from the loop.
            for i, ms in enumerate(self._control_ms):
                log.record_solve(i * self.dt, ms)

            # 3) trajectory (drone position + velocity over time).
            for (t, x, y, z, vx, vy, vz) in self._traj:
                log.record_state(t, [x, y, z], [vx, vy, vz], s=float("nan"))

            # 4) gate-pass events.
            for (t, kind, gate_idx) in self._events:
                log.record_event(t, kind, gate_idx=gate_idx)

            # 5) outcome.
            gates_passed = int(obs.get("target_gate", -1))
            if gates_passed == -1 and self.n_gates > 0:
                gates_passed = self.n_gates  # -1 == passed the final gate
            success = self.n_gates > 0 and gates_passed == self.n_gates
            run_dir = log.finish(lap_time=curr_time,
                                 gates_passed=gates_passed, success=success)

            # 6) track geometry, so the overlay plot can draw gates/obstacles
            #    without a separate track file.
            self._write_track(run_dir, obs)

            print(f"[analysis] wrote {run_dir}")
            return run_dir
        except Exception as exc:
            print(f"[analysis] finish() failed, no logs written: {exc}")
            return None

    @staticmethod
    def _write_track(run_dir: Optional[str], obs: Dict[str, Any]) -> None:
        """Write gate/obstacle positions to <run_dir>/track.csv (kind,x,y,z).

        Uses the final observation, so gate positions reflect the revealed
        (true) track. Fails soft if the keys are absent.
        """
        if not run_dir:
            return
        import csv
        rows = []
        for key, kind in (("gates_pos", "gate"), ("obstacles_pos", "obstacle")):
            arr = obs.get(key) if hasattr(obs, "get") else None
            if arr is None:
                continue
            for p in arr:
                try:
                    rows.append((kind, float(p[0]), float(p[1]),
                                 float(p[2]) if len(p) > 2 else float("nan")))
                except Exception:
                    continue
        if not rows:
            return
        try:
            with open(os.path.join(run_dir, "track.csv"), "w",
                      newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["kind", "x", "y", "z"])
                w.writerows(rows)
        except Exception as exc:
            print(f"[analysis] could not write track.csv: {exc}")


def _col(obj: Any, name: str, n: int) -> List[float]:
    """Return controller list ``name`` if it has length ``n``, else NaNs."""
    v = list(getattr(obj, name, []) or [])
    return v if len(v) == n else [float("nan")] * n
