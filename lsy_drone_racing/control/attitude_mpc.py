"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from a symbolic drone_model."""
    # Build base symbolic variables (outside the read-only drone-models library)
    x_base = ca.MX.sym("x_base", 12, 1)
    u_base = ca.MX.sym("u_base", 4, 1)

    # Call the library helper to get the base dynamics expression
    X_dot_lib, X_lib, U_lib, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )

    # Substitute library-internal state/input symbols with our x_base / u_base symbols
    X_dot_base = ca.substitute(X_dot_lib, X_lib, x_base)
    X_dot_base = ca.substitute(X_dot_base, U_lib, u_base)

    # MPCC augmentation: add virtual progress states theta and v_theta, and virtual input a_theta
    theta = ca.MX.sym("theta")
    v_theta = ca.MX.sym("v_theta")
    a_theta = ca.MX.sym("a_theta")

    # Derivatives for augmented states
    theta_dot = v_theta
    v_theta_dot = a_theta

    # Concatenate augmented state, input and derivative vectors
    x_aug = ca.vertcat(x_base, theta, v_theta)
    u_aug = ca.vertcat(u_base, a_theta)
    x_dot_aug = ca.vertcat(X_dot_base, theta_dot, v_theta_dot)

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = "mpcc_attitude_mpc"
    model.x = x_aug
    model.u = u_aug
    model.f_expl_expr = x_dot_aug
    model.f_impl_expr = None

    # model parameter vector p (12 spline coeffs + 1 contouring weight q_c + 1 local offset + 8 obstacle XY coords)
    model.p = ca.MX.sym("p", 22)
    px = model.p[0:4]
    py = model.p[4:8]
    pz = model.p[8:12]
    q_c = model.p[12]
    theta_offset = model.p[13]
    obs_xy_flat = model.p[14:22]

    theta_sym = x_aug[12]
    v_theta_sym = x_aug[13]

    # Local parameter within the current segment to avoid large absolute powers
    d_theta = theta_sym - theta_offset

    # ----------------------------------------------------------------------
    # EXACT MPCC MATH (Section III-D: Derivation of Contour and Lag Errors)
    # ----------------------------------------------------------------------

    # Equation 8: Evaluate nominal path p^d(d_theta)
    theta_pows = ca.vertcat(d_theta**3, d_theta**2, d_theta, ca.MX(1.0))
    pos_ref = ca.vertcat(ca.dot(px, theta_pows), ca.dot(py, theta_pows), ca.dot(pz, theta_pows))

    # Equation 9: Calculate tangent t(d_theta) = dp^d(d_theta)/dd_theta
    theta_dot_pows = ca.vertcat(3 * d_theta**2, 2 * d_theta, ca.MX(1.0), ca.MX(0.0))
    t_vec = ca.vertcat(
        ca.dot(px, theta_dot_pows), ca.dot(py, theta_dot_pows), ca.dot(pz, theta_dot_pows)
    )

    # The paper assumes perfect arc-length parameterization (||t|| = 1).
    # Since cubic splines fluctuate slightly, we explicitly normalize to prevent math breakdown.
    t_norm = t_vec / ca.sqrt(ca.sumsqr(t_vec) + 1e-4)

    # Position Error: e(theta_k) = p_k - p^d(theta_k)
    e_pos = x_aug[0:3] - pos_ref

    # Equation 10: Lag Error e^l(theta_k)
    e_lag_scalar = ca.dot(e_pos, t_norm)
    e_lag_vec = e_lag_scalar * t_norm

    # Equation 11: Contour Error e^c(theta_k)
    e_cont_vec = e_pos - e_lag_vec

    # Dynamic Weighting Integration:
    # To apply the dynamic contour weight q_c inside a Least Squares formulation,
    # we multiply the vector by sqrt(q_c) so that squaring it yields q_c * ||e^c||^2
    weighted_e_cont = ca.sqrt(q_c) * e_cont_vec

    # Obstacle repulsion penalty from cylindrical obstacle XY positions.
    pos_xy = x_aug[0:2]
    obs_penalty = ca.MX(0.0)
    for i in range(4):
        # Extract the 2x1 column vector for obstacle i directly from the flat array
        obs_i = obs_xy_flat[i * 2 : i * 2 + 2]
        dist_sq_obs = ca.sumsqr(pos_xy - obs_i)
        obs_penalty += 5.0 * ca.exp(-dist_sq_obs / 0.05)

    # Final cost_y_expr. Size: 3 (Contour) + 1 (Lag) + 4 (u_base) + 1 (v_theta) + 1 (obstacle) = 10
    model.cost_y_expr = ca.vertcat(
        weighted_e_cont, e_lag_scalar, u_aug[0:4], v_theta_sym, obs_penalty
    )

    # Terminal cost expression (Size 4: Contour + Lag + obstacle)
    model.cost_y_expr_e = ca.vertcat(weighted_e_cont, e_lag_scalar, obs_penalty)

    return model


def create_ocp_solver(
    Tf: float, N: int, parameters: dict, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # Set model (augmented with MPCC states/inputs)
    ocp.model = create_acados_model(parameters)

    # Get Dimensions
    nx = ocp.model.x.rows()
    ocp.model.u.rows()
    # For NONLINEAR_LS cost we rely on the model.cost_y_expr size
    ny = int(ocp.model.cost_y_expr.rows())
    # terminal residual
    ny_e = int(ocp.model.cost_y_expr_e.rows())

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf
    #

    # Cost Type: use NONLINEAR_LS for MPCC formulation
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    W = np.zeros((ny, ny))
    # Weights: [Contour(3), Lag(1), Controls(4), Progress(1), Obstacle(1)]
    W[0:3, 0:3] = np.diag(
        [50.0, 50.0, 400.0]
    )  # The actual weight is driven dynamically by q_c in the model
    W[3, 3] = 400.0  # High constant weight q_l to keep the virtual state tied to reality
    W[4:8, 4:8] = np.diag([1.0, 1.0, 1.0, 50.0])  # Control regularization R
    W[8, 8] = 5.0  # Progress weight mu
    W[9, 9] = 500.0  # Obstacle repulsion penalty
    ocp.cost.W = W

    # Terminal weights
    W_e = np.zeros((ny_e, ny_e))
    W_e[0:3, 0:3] = np.diag([50.0, 50.0, 400.0])  # Terminal contour
    W_e[3, 3] = 400.0  # Terminal lag
    W_e[4, 4] = 500.0  # Terminal obstacle repulsion
    ocp.cost.W_e = W_e

    # Set initial references.
    yref = np.zeros((ny,))
    # Here is the trick: set the reference for v_theta (index 8) to a high target speed.
    # The solver will minimize (v_theta - 15.0)^2, pushing the drone to go faster.
    yref[8] = 15.0
    # Prevent the optimizer from collapsing thrust to zero by targeting hover thrust.
    yref[7] = parameters["mass"] * np.linalg.norm(parameters["gravity_vec"])
    ocp.cost.yref = yref

    ocp.cost.yref_e = np.zeros((ny_e,))

    # Hook up model nonlinear residuals to ocp
    ocp.model.cost_y_expr = ocp.model.cost_y_expr
    ocp.model.cost_y_expr_e = ocp.model.cost_y_expr_e

    # Set State Constraints (roll/pitch/yaw and forward progress velocity)
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5, 0.0])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5, 10.0])
    ocp.constraints.idxbx = np.array([3, 4, 5, 13])

    # Set Input Constraints (roll/pitch/thrust and virtual acceleration a_theta)
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4, 0.0001])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4, 10.0])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options - use RTI for real-time MPCC execution
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # switch to RTI
    ocp.solver_options.tol = 1e-6

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    ocp.parameter_values = np.zeros((22,))

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/lsy_example_mpc.json",
        verbose=verbose,
        build=True,
        generate=True,
    )

    return acados_ocp_solver, ocp


class AttitudeMPC(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._N = 25
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        self._current_theta = 0.0
        self._current_v_theta = 0.0
        self._predicted_trajectory = np.zeros((self._N, 3))
        self._log_thrust = []
        self._log_roll = []
        self._log_pitch = []
        self._log_contour = []
        self._log_lag = []
        self._log_v_theta = []
        self._gate_positions = np.array([g["pos"] for g in config.env.track.gates])

        # Same waypoints as in the trajectory controller. Determined by trial and error.
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

        # 1. Calculate the Euclidean distance between consecutive waypoints
        distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)

        # 2. Create the cumulative chord length array (starts at 0)
        # s will look something like: [0.0, 0.6, 2.1, 3.4, ...]
        s = np.concatenate(([0.0], np.cumsum(distances)))
        self._s_total = s[-1]  # Total physical length of the track

        # 3. Create the arc-length parameterized spline
        self._des_pos_spline = CubicSpline(s, waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()

        # 4. Generate fine evaluation points for the nearest-neighbor search
        # We now evaluate over the spatial parameter 's' instead of time 't'
        n_eval_points = 500
        self._waypoints_pos = self._des_pos_spline(np.linspace(0, self._s_total, n_eval_points))
        self._waypoints_yaw = self._waypoints_pos[:, 0] * 0

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        # For NONLINEAR_LS MPCC we read the residual sizes from the model
        try:
            self._ny = int(self._ocp.model.cost_y_expr.rows())
        except Exception:
            self._ny = int(self._nx + self._nu)
        try:
            self._ny_e = int(self._ocp.model.cost_y_expr_e.rows())
        except Exception:
            self._ny_e = int(3)

        self._tick = 0
        self._tick_max = len(self._waypoints_pos) - 1 - self._N
        self._config = config
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        min(self._tick, self._tick_max)
        if self._tick >= self._tick_max:
            self._finished = True

        # Setting initial state
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))

        # Estimate current progress along the spline by nearest waypoint index
        nearest_idx = int(np.argmin(np.linalg.norm(self._waypoints_pos - obs["pos"], axis=1)))
        # Choose segment where the drone currently is (clamp)
        n_segments = len(self._des_pos_spline.x) - 1
        seg_idx = min(max(nearest_idx, 0), max(0, n_segments - 1))

        # Use persistent virtual progress states instead of resetting them every tick.
        theta0 = self._current_theta
        v_theta0 = self._current_v_theta

        # Augmented initial state
        x0_aug = np.concatenate((x0, np.array([theta0, v_theta0])))
        self._acados_ocp_solver.set(0, "lbx", x0_aug)
        self._acados_ocp_solver.set(0, "ubx", x0_aug)

        # For MPCC we provide the spline coefficients and a contouring weight in model.p.
        # Define the reference residual target with hover thrust and target speed
        yref_target = np.zeros((self._ny,))
        yref_target[7] = self.drone_params["mass"] * 9.81  # Hover thrust
        yref_target[8] = 15.0  # Target progress speed (v_theta)

        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref_target)

        # Terminal yref zero (size ny_e)
        yref_e_zero = np.zeros((self._ny_e,))
        self._acados_ocp_solver.set(self._N, "y_ref", yref_e_zero)

        # current reference at the current virtual progress.
        pos_ref_current = self._des_pos_spline(self._current_theta)

        # Prepare and set parameter vector p for each stage (polynomial coeffs + contour weight)
        # Extract cubic coefficients for the current segment from the CubicSpline object
        # SciPy CubicSpline stores coefficients in `.c` with shape (4, n_segments, dim)
        cs_c = self._des_pos_spline.c
        # Ensure indexing works for vector-valued spline
        # c_seg shape expected (4, dim)
        try:
            c_seg = cs_c[:, seg_idx, :]
        except Exception:
            # fallback if shape differs
            c_seg = cs_c[:, :, seg_idx]

        # c_seg: rows are powers [3,2,1,0], columns are dims (x,y,z)
        px = c_seg[:, 0]
        py = c_seg[:, 1]
        pz = c_seg[:, 2]

        # Track obstacle positions and dynamic contouring weight coefficients.
        obs_positions = np.array([o["pos"][:2] for o in self._config.env.track.obstacles])

        # Dynamic contouring weight (placeholder heuristic)
        dist_to_waypoint = np.linalg.norm(self._waypoints_pos[nearest_idx] - obs["pos"])
        q_c = 10.0 if dist_to_waypoint < 0.5 else 1.0

        params = np.zeros((22,))
        params[0:4] = px.flatten()
        params[4:8] = py.flatten()
        params[8:12] = pz.flatten()
        params[12] = q_c
        params[13] = self._des_pos_spline.x[seg_idx]

        # Set spline parameters stage-by-stage using predicted theta.
        for j in range(self._N):
            if self._tick == 0:
                # Provide a kinematic guess for the very first tick to prevent divergence
                theta_pred = (
                    self._current_theta + j * self._dt * 2.0
                )  # Assume 2.0 m/s initial progress
                xj_guess = x0_aug.copy()
                xj_guess[12] = theta_pred
                xj_guess[13] = 2.0
                hover_u = np.array([0.0, 0.0, 0.0, self.drone_params["mass"] * 9.81, 0.0])
                self._acados_ocp_solver.set(j, "x", xj_guess)
                self._acados_ocp_solver.set(j, "u", hover_u)
            else:
                # Use the solver's optimized trajectory from the previous tick as the segment guess
                xj_prev = self._acados_ocp_solver.get(j, "x")
                theta_pred = float(xj_prev[12])

            # Clip to valid spline bounds
            theta_pred = float(
                np.clip(theta_pred, self._des_pos_spline.x[0], self._des_pos_spline.x[-1])
            )

            seg_idx_j = int(np.searchsorted(self._des_pos_spline.x[1:], theta_pred, side="right"))
            seg_idx_j = min(max(seg_idx_j, 0), n_segments - 1)
            try:
                c_seg_j = cs_c[:, seg_idx_j, :]
            except Exception:
                c_seg_j = cs_c[:, :, seg_idx_j]

            px_j = c_seg_j[:, 0]
            py_j = c_seg_j[:, 1]
            pz_j = c_seg_j[:, 2]

            pos_pred = self._des_pos_spline(theta_pred)

            # Gaussian Dynamic Contouring Weight
            q_nom = 1.0
            q_wp = 400.0  # Peak weight at the gate
            sigma_sq = 0.5**2  # Variance

            q_c_j = q_nom
            for gate_pos in self._gate_positions:
                dist_sq = np.sum((pos_pred - gate_pos) ** 2)
                q_c_j += q_wp * np.exp(-0.5 * dist_sq / sigma_sq)

            theta_offset_j = self._des_pos_spline.x[seg_idx_j]

            params_j = np.zeros((22,))
            params_j[0:4] = px_j.flatten()
            params_j[4:8] = py_j.flatten()
            params_j[8:12] = pz_j.flatten()
            params_j[12] = q_c_j
            params_j[13] = theta_offset_j
            params_j[14:22] = obs_positions.flatten()

            self._acados_ocp_solver.set(j, "p", params_j)

        # Solve and extract first control. We run the RTI solver to meet 50 Hz real-time.
        self._acados_ocp_solver.solve()

        for j in range(self._N):
            xj = self._acados_ocp_solver.get(j, "x")
            self._predicted_trajectory[j] = xj[0:3]

        u0_aug = self._acados_ocp_solver.get(0, "u")
        x1_opt = self._acados_ocp_solver.get(1, "x")
        self._current_theta = float(x1_opt[12])
        self._current_v_theta = float(x1_opt[13])

        u0 = u0_aug[0:4]

        # Calculate Contour and Lag Error for Telemetry
        p_curr = obs["pos"]
        p_ref = self._des_pos_spline(self._current_theta)
        t_ref = self._des_vel_spline(self._current_theta)
        t_norm = t_ref / (np.linalg.norm(t_ref) + 1e-6)

        e_pos = p_curr - p_ref
        e_lag_val = np.dot(e_pos, t_norm)
        e_cont_vec = e_pos - e_lag_val * t_norm
        e_cont_val = np.linalg.norm(e_cont_vec)

        # Append to logs
        self._log_thrust.append(float(u0[3]))
        self._log_roll.append(float(u0[0]))
        self._log_pitch.append(float(u0[1]))
        self._log_contour.append(float(e_cont_val))
        self._log_lag.append(float(e_lag_val))
        self._log_v_theta.append(self._current_v_theta)

        return u0

    def render_callback(self, sim: Sim):
        """Visualize the spatial reference path, the current MPCC target, and the predicted horizon."""
        trajectory = self._des_pos_spline(np.linspace(0.0, self._s_total, 150))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))

        target_pos = self._des_pos_spline(self._current_theta)
        draw_points(sim, target_pos.reshape(1, -1), rgba=(1.0, 0.0, 0.0, 1.0), size=0.04)

        if self._predicted_trajectory.shape[0] > 0:
            draw_line(sim, self._predicted_trajectory, rgba=(1.0, 0.5, 0.0, 1.0))

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        self._tick += 1

        return self._finished

    def episode_callback(self):
        """Plot MPCC telemetry metrics and reset the integral error."""
        hover_thrust = self.drone_params["mass"] * 9.81
        max_thrust = self.drone_params["thrust_max"] * 4
        min_thrust = self.drone_params["thrust_min"] * 4

        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        # 1. Tracking Errors
        axs[0].plot(self._log_contour, label="Contour Error (e_c)")
        axs[0].plot(self._log_lag, label="Lag Error (e_l)")
        axs[0].set_ylabel("Error [m]")
        axs[0].legend()
        axs[0].grid(True)

        # 2. Attitude Commands
        axs[1].plot(self._log_roll, label="Roll Command")
        axs[1].plot(self._log_pitch, label="Pitch Command")
        axs[1].axhline(0.5, color="r", linestyle="--", label="Upper Limit")
        axs[1].axhline(-0.5, color="r", linestyle="--", label="Lower Limit")
        axs[1].set_ylabel("Angle [rad]")
        axs[1].legend()
        axs[1].grid(True)

        # 3. Thrust Command
        axs[2].plot(self._log_thrust, label="Thrust Command")
        axs[2].axhline(hover_thrust, color="g", linestyle=":", label="Hover")
        axs[2].axhline(max_thrust, color="r", linestyle="--", label="Max Thrust")
        axs[2].axhline(min_thrust, color="r", linestyle="--", label="Min Thrust")
        axs[2].set_ylabel("Thrust [N]")
        axs[2].legend()
        axs[2].grid(True)

        # 4. Progress Speed
        axs[3].plot(self._log_v_theta, label="Virtual Speed (v_theta)")
        axs[3].axhline(15.0, color="g", linestyle="--", label="Target Speed")
        axs[3].set_ylabel("Speed [m/s]")
        axs[3].set_xlabel("Timestep")
        axs[3].legend()
        axs[3].grid(True)

        fig.tight_layout()
        fig.savefig("mpcc_standard_metrics.png")
        plt.show()
        plt.close(fig)

        self._log_thrust.clear()
        self._log_roll.clear()
        self._log_pitch.clear()
        self._log_contour.clear()
        self._log_lag.clear()
        self._log_v_theta.clear()
        self._tick = 0
