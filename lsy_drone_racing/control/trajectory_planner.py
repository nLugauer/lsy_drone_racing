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
        gate_rpys: np.ndarray | None = None,
        n_eval_points: int = 500,
        min_z: float = 0.15,
        gate_approach_dist: float = 0.3,
    ) -> None:
        """Build the spline from start pos + gates, ensuring ground clearance.

        Args:
            start_pos: Drone start position [x, y, z].
            gates_pos: (N, 3) gate center positions. If None, uses hardcoded waypoints.
            gate_rpys: (N, 3) gate orientations [roll, pitch, yaw]. Used to inject
                pre/post-gate approach waypoints for perpendicular gate traversal.
            n_eval_points: Resolution of the precomputed path sample array.
            min_z: Minimum z height for ground clearance injection.
            gate_approach_dist: Distance in front of / behind each gate for approach waypoints.
        """
        self._n_eval_points = n_eval_points
        self._min_z = min_z
        self._gate_approach_dist = gate_approach_dist

        waypoints = self._build_waypoints(start_pos, gates_pos, gate_rpys)

        # Build the spline with ground-clearance checks
        self._s, self._des_pos_spline = self._build_safe_spline(waypoints, min_z)
        self._s_total = float(self._s[-1])

        # Precompute derivatives and fine evaluation points
        self._des_vel_spline = self._des_pos_spline.derivative()
        self._waypoints_pos = self._des_pos_spline(np.linspace(0, self._s_total, n_eval_points))

    def _build_waypoints(
        self,
        start_pos: np.ndarray | None,
        gates_pos: np.ndarray | None,
        gate_rpys: np.ndarray | None,
    ) -> np.ndarray:
        """Construct the ordered waypoint list, optionally with gate approach vectors."""
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
            return waypoints

        # Build waypoints from gate centers, optionally with approach vectors
        pts = [np.array(start_pos, dtype=np.float64)]

        for i, gate_pos in enumerate(gates_pos):
            gate_pos = np.array(gate_pos, dtype=np.float64)

            if gate_rpys is not None:
                yaw = float(gate_rpys[i, 2])
                # Gate normal points along the gate's local X axis after yaw rotation
                normal = np.array([np.cos(yaw), np.sin(yaw), 0.0])

                # Determine approach side: which side of the gate are we coming from?
                prev_pt = pts[-1]
                to_gate = gate_pos - prev_pt
                if np.dot(to_gate, normal) >= 0:
                    # Approaching from the negative-normal side
                    pre_gate = gate_pos - self._gate_approach_dist * normal
                    post_gate = gate_pos + self._gate_approach_dist * normal
                else:
                    # Approaching from the positive-normal side
                    pre_gate = gate_pos + self._gate_approach_dist * normal
                    post_gate = gate_pos - self._gate_approach_dist * normal

                pts.append(pre_gate)
                pts.append(gate_pos)
                pts.append(post_gate)
            else:
                pts.append(gate_pos)

        # Add a continuation waypoint 0.5m past the last gate to prevent the optimizer
        # from stalling at the spline endpoint during the MPC horizon.
        if len(gates_pos) >= 2:
            direction = np.array(gates_pos[-1]) - np.array(gates_pos[-2])
            direction = direction / (np.linalg.norm(direction) + 1e-6)
        else:
            direction = np.array([1.0, 0.0, 0.0])
        pts.append(np.array(gates_pos[-1]) + 0.5 * direction)

        return np.array(pts, dtype=np.float64)

    def rebuild(
        self, start_pos: np.ndarray, gates_pos: np.ndarray, gate_rpys: np.ndarray | None = None
    ) -> None:
        """Rebuild the spline in-place with updated gate positions.

        Call this when a gate's true position becomes known. The trajectory object
        is updated and all subsequent evaluate() / get_polynomial_coeffs_at() calls
        will use the new spline. Theta values from the previous spline are invalid
        after this call — use nearest_theta() to re-initialize.

        Args:
            start_pos: Current drone position used as spline anchor.
            gates_pos: (N, 3) updated gate positions (mix of nominal and true).
            gate_rpys: (N, 3) gate orientations. Pass None to skip approach waypoints.
        """
        waypoints = self._build_waypoints(start_pos, gates_pos, gate_rpys)
        self._s, self._des_pos_spline = self._build_safe_spline(waypoints, self._min_z)
        self._s_total = float(self._s[-1])
        self._des_vel_spline = self._des_pos_spline.derivative()
        self._waypoints_pos = self._des_pos_spline(
            np.linspace(0, self._s_total, self._n_eval_points)
        )

    def nearest_theta(self, pos: np.ndarray) -> float:
        """Return the arc-length parameter of the path point nearest to pos.

        Use this to re-initialize _current_theta after a trajectory rebuild.

        Args:
            pos: World-space position [x, y, z].

        Returns:
            Arc-length parameter in [0, total_length].
        """
        idx = self.get_nearest_waypoint_index(pos)
        return float(self._s_total * idx / max(self._n_eval_points - 1, 1))

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
