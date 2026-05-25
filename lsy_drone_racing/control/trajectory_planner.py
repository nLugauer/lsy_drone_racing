"""Trajectory planner for drone racing.

Basic Spline trajectory planner that generates
 position and velocity references for the MPC controller.
"""

import numpy as np
from scipy.interpolate import CubicSpline


class TrajectoryPlanner:
    """Trajectory planner for drone racing."""

    def __init__(self, waypoints: np.ndarray, t_total: float, freq: int):
        """Initialize the trajectory planner."""
        t = np.linspace(0, t_total, len(waypoints))
        self._des_pos_spline = CubicSpline(t, waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()

        self.waypoints_pos = self._des_pos_spline(np.linspace(0, t_total, int(freq * t_total)))
        self.waypoints_vel = self._des_vel_spline(np.linspace(0, t_total, int(freq * t_total)))
        self.waypoints_yaw = self.waypoints_pos[:, 0] * 0  # Zero yaw for now

        self.max_ticks = len(self.waypoints_pos) - 1

    def get_references(
        self, current_tick: int, horizon: int
    ) -> tuple[
        np.ndarray,  # pos_ref (N, 3)
        np.ndarray,  # vel_ref (N, 3)
        np.ndarray,  # yaw_ref (N,)
        np.ndarray,  # pos_e (3,)
        np.ndarray,  # vel_e (3,)
        float,  # yaw_e scalar
    ]:
        """Returns the reference slice for the prediction horizon."""
        i = min(current_tick, max(0, self.max_ticks - horizon))

        pos_ref = self.waypoints_pos[i : i + horizon]
        vel_ref = self.waypoints_vel[i : i + horizon]
        yaw_ref = self.waypoints_yaw[i : i + horizon]

        # Terminal state
        end_idx = min(i + horizon, self.max_ticks)
        pos_e = self.waypoints_pos[end_idx]
        vel_e = self.waypoints_vel[end_idx]
        yaw_e = self.waypoints_yaw[end_idx]

        return pos_ref, vel_ref, yaw_ref, pos_e, vel_e, yaw_e
