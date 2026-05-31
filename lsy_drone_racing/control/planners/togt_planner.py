"""Time-Optimal Gate-Traversing (TOGT) Planner for Drone Racing.

Implements the unconstrained optimization framework from arXiv:2309.06837v3.
Uses differential flatness to map flat outputs to states/controls, penalizes
actuator limits, and uses spatial mappings for gate boundaries.
"""

import casadi as ca
import numpy as np
from scipy.interpolate import CubicSpline

from .base_planner import BasePlanner


class TOGTPlanner(BasePlanner):
    """TOGT Planner implementing the optimization framework from arXiv:2309.06837v3."""

    def __init__(
        self,
        gates_data: list[dict],
        parameters: dict,
        freq: int,
        start_pos: np.ndarray | list[float] | None = None,
    ) -> None:
        """Initialize the TOGT planner with gate data and drone parameters.

        Args:
            gates_data: Gate polygon definitions used by the spatial mapping.
            parameters: Drone parameters including mass and thrust limits.
            freq: Reference generation frequency in Hz.
            start_pos: Initial drone position in world coordinates.
        """
        super().__init__(freq)
        self.params = parameters
        self.gates = gates_data
        self.L = len(gates_data)
        self.start_pos = (
            np.asarray(start_pos, dtype=np.float64) if start_pos is not None else np.zeros(3)
        )

        # Polynomial order and resolution for constraint checking (Eq. 12)
        self.s = 5
        self.kappa = 10
        self.max_tilt = 0.5

        self._setup_unconstrained_nlp()

    def _flat_to_state_control(
        self, p: ca.MX, v: ca.MX, a: ca.MX, j: ca.MX, s: ca.MX, yaw: ca.MX, yaw_dot: ca.MX
    ) -> ca.MX:
        """Map flat outputs and derivatives to the required collective thrust.

        The mapping is simplified to focus on thrust constraints by using the flat
        output acceleration and gravity vector.

        Args:
            p: Position flat output.
            v: Velocity flat output.
            a: Acceleration flat output.
            j: Jerk flat output.
            s: Snap flat output.
            yaw: Yaw flat output.
            yaw_dot: Yaw rate flat output.

        Returns:
            A CasADi MX expression for the collective thrust.
        """
        g_vec = np.array([0, 0, 9.81])
        thrust_vec = a + g_vec

        # Collective thrust f = m * ||a + g||
        thrust = self.params["mass"] * ca.norm_2(thrust_vec)
        return thrust

    def _setup_unconstrained_nlp(self) -> None:
        """Build the unconstrained objective (Eq. 15) using CasADi."""
        # 1. Unconstrained Variables
        # K: Time allocation variables (L segments between start and each gate)
        self.K = ca.MX.sym("K", self.L)
        # T: Actual time durations (strictly positive mapping: T = K^2 + 1e-3)
        self.T = self.K**2 + 1e-3

        # D: Gate spatial variables. Assuming 4 corners per gate (v=4), D is length 4*L
        self.D = ca.MX.sym("D", 4 * self.L)

        # 2. Gate Transformations (Eq. 13 & 14)
        P_list = []  # Waypoints constrained inside gates
        for i, gate in enumerate(self.gates):
            if gate["type"] == "polygon":
                d_i = self.D[i * 4 : (i + 1) * 4]
                o = gate["origin"]
                V = gate["V"]

                # Eq 14: g_P(d) = o + V * ( [d]^2 / (d^T d)^2 )
                d_sq = d_i**2
                d_norm_sq_sq = (ca.dot(d_i, d_i) + 1e-6) ** 2
                p_i = o + ca.mtimes(V, d_sq / d_norm_sq_sq)
                P_list.append(p_i)

        # Start and end positions (hovering)
        P_full = [ca.MX(self.start_pos)] + P_list

        # 3. Objective Formulation (Eq. 15)
        # T_Sigma: Total flight time
        T_Sigma = ca.sum1(self.T)

        # I_T: Penalty for dynamic constraint violations (Eq. 12)
        I_T = 0

        # For a full MINCO functional, we would invert a matrix here to get piece-wise
        # polynomial coefficients from P and T. To keep this tractable in standard CasADi,
        # we approximate the MINCO mapping by evaluating finite-difference derivatives
        # across the segments to check maximum thrust limits.

        for i in range(self.L):
            dt = self.T[i]
            p0 = P_full[i]
            p1 = P_full[i + 1]

            # Simple average velocity and acceleration for the segment
            v_avg = (p1 - p0) / dt
            # Assuming start/end velocities are near zero for conservative accel check
            a_avg = (v_avg - 0) / dt

            # Thrust required for this segment
            thrust = self._flat_to_state_control(p0, v_avg, a_avg, 0, 0, 0, 0)

            # Tilt constraint penalty: controller input bounds are ±0.5 rad for roll/pitch.
            # Use the same simplified flat-output mapping to estimate the thrust vector tilt.
            g_vec = ca.DM([0.0, 0.0, 9.81])
            thrust_vec = a_avg + g_vec
            cos_phi = (a_avg[2] + 9.81) / ca.norm_2(thrust_vec)
            cos_phi_min = np.cos(self.max_tilt)
            tilt_violation = ca.fmax(cos_phi_min - cos_phi, 0)

            # Eq 12 Penalty: max(h(x,u), 0)^3
            # Limit: f_i <= f_max -> thrust <= 4 * f_max
            max_thrust = self.params["thrust_max"] * 4
            thrust_violation = ca.fmax(thrust - max_thrust, 0)

            I_T += (thrust_violation**3 + 1e3 * tilt_violation**3) * dt

        # Total unconstrained objective
        self.objective = T_Sigma + 100.0 * I_T

        # Create a CasADi function and its gradient for the SciPy solver
        vars_cat = ca.vcat([self.K, self.D])
        self.f_obj = ca.Function("f_obj", [vars_cat], [self.objective])
        self.f_grad = ca.Function("f_grad", [vars_cat], [ca.gradient(self.objective, vars_cat)])

    def _build_basic_spline_waypoints(self, num_samples_per_segment: int = 2) -> np.ndarray:
        """Build a compact waypoint sequence from a spline through start and gate centers.

        We intentionally keep the waypoint count low here so the planner does not
        create an overly dense reference trajectory. The cubic spline is still
        evaluated at a small number of points for smoothness, but the path is
        defined mainly by the gate centers.
        """
        waypoints = self._get_augmented_waypoints(approach_dist=0.05)
        t_nodes = np.linspace(0.0, 1.0, len(waypoints))
        spline = CubicSpline(t_nodes, waypoints, axis=0, bc_type="clamped")
        t_dense = np.linspace(0.0, 1.0, (len(waypoints) - 1) * num_samples_per_segment + 1)
        return spline(t_dense)

    def _get_augmented_waypoints(self, approach_dist: float = 0.15) -> np.ndarray:
        """Add waypoints including pre- and post-gate approach points to enforce direction."""
        waypoints = [self.start_pos.reshape(1, 3)]

        for gate in self.gates:
            origin = np.asarray(gate["origin"])
            yaw = gate.get("yaw", 0.0)

            # The normal vector pointing strictly forward through the gate
            normal = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)

            # Add a point before the gate, the gate itself, and a point after
            waypoints.append((origin - approach_dist * normal).reshape(1, 3))
            waypoints.append(origin.reshape(1, 3))
            waypoints.append((origin + approach_dist * normal).reshape(1, 3))

        return np.vstack(waypoints)

    def _build_basic_spline_path(self) -> np.ndarray:
        """Build a high-resolution reference spline for plotting."""
        waypoints = self._get_augmented_waypoints(approach_dist=0.05)
        t_nodes = np.linspace(0.0, 1.0, len(waypoints))
        spline = CubicSpline(t_nodes, waypoints, axis=0, bc_type="clamped")
        t_plot = np.linspace(0.0, 1.0, max(200, (len(waypoints) - 1) * 20 + 1))
        return spline(t_plot)

    def _max_horizontal_acceleration(self) -> float:
        """Estimate the maximum available horizontal acceleration from thrust."""
        thrust_total = float(self.params["thrust_max"] * 4)
        mass = float(self.params["mass"])
        gravity = abs(float(self.params["gravity_vec"][-1]))
        specific_thrust = thrust_total / mass
        if specific_thrust <= gravity:
            return 0.0
        return float(np.sqrt(max(specific_thrust**2 - gravity**2, 0.0)))

    def _segment_times_from_dynamics(self, distances: np.ndarray) -> np.ndarray:
        """Compute segment timing using available horizontal acceleration."""
        a_horiz = self._max_horizontal_acceleration()
        max_speed = 3.0
        if a_horiz <= 0.0:
            return distances / max_speed
        times = np.sqrt(4.0 * distances / a_horiz)
        return np.maximum(times, distances / max_speed)

    def plan_trajectory(self) -> None:
        """Plan the trajectory by solving the unconstrained optimization problem."""
        self.basic_spline_pos = self._build_basic_spline_path()

        # self.waypoints_pos = self._build_basic_spline_waypoints(num_samples_per_segment=0)
        self.waypoints_pos = self._get_augmented_waypoints(approach_dist=0.05)

        distances = np.linalg.norm(np.diff(self.waypoints_pos, axis=0), axis=1)
        segment_times = self._segment_times_from_dynamics(distances)

        t_nodes = np.concatenate([[0.0], np.cumsum(segment_times)])
        t_total = t_nodes[-1]
        self.t_total = float(t_total)
        print(f"TOGT planned trajectory duration: {self.t_total:.2f} seconds")

        self.t_fixed = np.linspace(0.0, t_total, int(np.ceil(t_total * self.freq)) + 1)
        self.max_ticks = len(self.t_fixed) - 1

        bc_type = ((1, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 0.0]))
        spline = CubicSpline(t_nodes, self.waypoints_pos, axis=0, bc_type=bc_type)
        self.traj_pos = spline(self.t_fixed)
        self.traj_pos[:, 2] = np.maximum(self.traj_pos[:, 2], 0.0)
        self.traj_vel = spline.derivative(1)(self.t_fixed)
        self.traj_yaw = np.zeros(len(self.t_fixed))

    def _process_solution(self, opt_x: np.ndarray) -> None:
        """Extract optimized variables and build the interpolated reference trajectory."""
        opt_K = opt_x[: self.L]

        T_opt = opt_K**2 + 1e-3

        waypoints = [self.start_pos.copy()]
        for gate in self.gates:
            waypoints.append(np.asarray(gate["origin"], dtype=np.float64))

        self.waypoints_pos = np.array(waypoints)

        # Generate time vector for the waypoints
        t_nodes = [0.0]
        for t in T_opt:
            t_nodes.append(t_nodes[-1] + t)

        t_total = t_nodes[-1]
        self.t_total = float(t_total)
        print(f"TOGT planned trajectory duration after optimization: {self.t_total:.2f} seconds")
        self.t_fixed = np.arange(0, t_total, 1.0 / self.freq)
        self.max_ticks = len(self.t_fixed) - 1

        # Cubic spline interpolation for high-frequency tracking in the MPC
        from scipy.interpolate import CubicSpline

        bc_type = ((1, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 0.0]))
        spline = CubicSpline(t_nodes, self.waypoints_pos, bc_type=bc_type)
        self.traj_pos = spline(self.t_fixed)
        self.traj_pos[:, 2] = np.maximum(self.traj_pos[:, 2], 0.0)
        self.traj_vel = spline.derivative(1)(self.t_fixed)
        self.traj_yaw = np.zeros(len(self.t_fixed))

    def get_references(
        self, current_tick: int, horizon: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Return reference trajectories and terminal state for the MPC horizon.

        Args:
            current_tick: Current planner tick index.
            horizon: Number of future steps to return.

        Returns:
            Tuple of horizon position refs, velocity refs, yaw refs, terminal position,
            terminal velocity, and terminal yaw.
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
