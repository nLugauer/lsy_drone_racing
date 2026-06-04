"""Trajectory planning helper for MPC spline generation.

This module encapsulates spline creation, arc-length parameterization, and
segment coefficient extraction for the attitude MPC controller.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


class TrajectoryPlanner:
    """Encapsulates path generation and spline evaluation for MPC."""

    def __init__(
        self,
        start_pos: np.ndarray | None = None,
        gates_pos: np.ndarray | None = None,
        n_eval_points: int = 500,
        min_z: float = 0.15,
    ) -> None:
        """Build the spline from start pos + gates, ensuring ground clearance."""
        # 1. Handle fallback to hardcoded waypoints if gates_pos is None
        if gates_pos is None:
            waypoints = np.array(
                [
                    [-1.5, 0.75, 0.05],
                    [-1.0, 0.55, 0.4],
                    [0.3, 0.35, 0.7],
                    [1.3, -0.15, 0.9],
                    [0.85, 0.85, 1.2],
                    [-0.5, -0.05, 0.7],
                    [-1.2, -0.2, 0.8],
                    [-1.2, -0.2, 1.2],
                    [-0.0, -0.7, 1.2],
                    [0.5, -0.75, 1.2],
                ]
            )
            if start_pos is not None:
                waypoints = np.vstack((start_pos, waypoints))
        else:
            waypoints = np.vstack((start_pos, gates_pos))

        # 2. Build the spline with ground-clearance checks
        self._s, self._des_pos_spline = self._build_safe_spline(waypoints, min_z)
        self._s_total = float(self._s[-1])

        # 3. Precompute derivatives and fine evaluation points
        self._des_vel_spline = self._des_pos_spline.derivative()
        self._waypoints_pos = self._des_pos_spline(np.linspace(0, self._s_total, n_eval_points))

    def _build_safe_spline(
        self, waypoints: np.ndarray, min_z: float
    ) -> tuple[np.ndarray, CubicSpline]:
        """Pre-emptively injects safe midpoints between all waypoints to prevent floor dips."""
        safe_waypoints = [waypoints[0]]

        for i in range(len(waypoints) - 1):
            p1 = waypoints[i]
            p2 = waypoints[i + 1]

            # Calculate exact geometric midpoint
            midpoint = (p1 + p2) / 2.0

            # Clamp the Z coordinate to ensure it never drops below the safety margin
            midpoint[2] = max(midpoint[2], min_z)

            safe_waypoints.append(midpoint)
            safe_waypoints.append(p2)

        safe_wpts_arr = np.array(safe_waypoints)

        # Calculate standard chord lengths for the newly augmented waypoints
        distances = np.linalg.norm(np.diff(safe_wpts_arr, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(distances)))
        spline = CubicSpline(s, safe_wpts_arr)

        return s, spline

    @property
    def total_length(self) -> float:
        """Return the total length of the planned trajectory."""
        return self._s_total

    @property
    def waypoints_pos(self) -> np.ndarray:
        """Return fine-grained sampled positions along the path."""
        return self._waypoints_pos

    @property
    def knot_points(self) -> np.ndarray:
        """Return the spline knot points used for segment indexing."""
        return self._s

    def final_waypoint(self) -> np.ndarray:
        """Return the final waypoint at the end of the spline."""
        return self._des_pos_spline(self._s_total)

    def evaluate(self, s: float) -> np.ndarray:
        """Evaluate the desired path position at a given parameter s."""
        return self._des_pos_spline(s)

    def evaluate_velocity(self, s: float) -> np.ndarray:
        """Evaluate the desired path velocity at a given parameter s."""
        return self._des_vel_spline(s)

    def get_nearest_waypoint_index(self, pos: np.ndarray) -> int:
        """Return the index of the sampled path point nearest to a world-space position."""
        return int(np.argmin(np.linalg.norm(self._waypoints_pos - pos, axis=1)))

    def get_segment_index(self, nearest_idx: int) -> int:
        """Clamp a nearest waypoint index to a valid spline segment index."""
        n_segments = len(self._s) - 1
        return min(max(nearest_idx, 0), max(0, n_segments - 1))

    def get_segment_coeffs(self, seg_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the polynomial coefficients for a spline segment."""
        cs_c = self._des_pos_spline.c
        try:
            c_seg = cs_c[:, seg_idx, :]
        except Exception:
            c_seg = cs_c[:, :, seg_idx]
        return c_seg[:, 0], c_seg[:, 1], c_seg[:, 2]

    def get_polynomial_coeffs_at(
        self, theta_pred: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Return polynomial coefficients and local theta offset for a predicted path point."""
        theta_pred = float(np.clip(theta_pred, self._s[0], self._s[-1]))
        seg_idx = int(np.searchsorted(self._s[1:], theta_pred, side="right"))
        n_segments = len(self._s) - 1
        seg_idx = min(max(seg_idx, 0), n_segments - 1)

        cs_c = self._des_pos_spline.c
        try:
            c_seg = cs_c[:, seg_idx, :]
        except Exception:
            c_seg = cs_c[:, :, seg_idx]

        return c_seg[:, 0], c_seg[:, 1], c_seg[:, 2], float(self._s[seg_idx])
