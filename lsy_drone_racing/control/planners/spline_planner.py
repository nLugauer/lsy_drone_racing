"""Basic Spline trajectory planner."""

import numpy as np

from .base_planner import BasePlanner


class SplinePlanner(BasePlanner):
    """Basic spline trajectory planner for drone racing."""

    def __init__(self, waypoints: np.ndarray, t_total: float, freq: int) -> None:
        """Initialize the spline planner and generate the trajectory.

        Args:
            waypoints: Sequence of track waypoints to fit.
            t_total: Total trajectory duration in seconds.
            freq: Reference generation frequency in Hz.
        """
        super().__init__(freq)
        self.waypoints = waypoints
        self.t_total = t_total
        # You can call the planning logic directly in init for simple planners
        self.plan_trajectory()

    def plan_trajectory(self) -> None:
        """Generate the spline trajectory references from the provided waypoints."""
        from scipy.interpolate import CubicSpline

        t_nodes = np.linspace(0.0, 1.0, len(self.waypoints))
        spline = CubicSpline(t_nodes, self.waypoints, axis=0, bc_type="clamped")
        t_fixed = np.linspace(0.0, 1.0, int(np.ceil(self.t_total * self.freq)) + 1)

        self.traj_pos = spline(t_fixed)
        self.traj_vel = spline.derivative(1)(t_fixed)
        self.traj_yaw = np.zeros(len(t_fixed))
        self.t_fixed = t_fixed
        self.max_ticks = len(t_fixed) - 1

    def get_references(
        self, current_tick: int, horizon: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Return horizon reference slices and terminal values from the planned spline.

        Args:
            current_tick: Current planner tick.
            horizon: Number of future reference steps.

        Returns:
            Position, velocity, yaw, terminal position, terminal velocity, and terminal yaw.
        """
        i = min(current_tick, max(0, self.max_ticks - horizon))
        pos_ref = self.traj_pos[i : i + horizon]
        vel_ref = self.traj_vel[i : i + horizon]
        yaw_ref = self.traj_yaw[i : i + horizon]
        end_idx = min(i + horizon, self.max_ticks)
        return (
            pos_ref,
            vel_ref,
            yaw_ref,
            self.traj_pos[end_idx],
            self.traj_vel[end_idx],
            self.traj_yaw[end_idx],
        )
