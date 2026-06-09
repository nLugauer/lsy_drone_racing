"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

import uuid
from typing import TYPE_CHECKING

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from drone_models.so_rpy_rotor_drag import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.obstacle_manager import ObstacleManager
from lsy_drone_racing.control.trajectory_planner import TrajectoryPlanner

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


def create_acados_model(
    parameters: dict, obs_manager: ObstacleManager, unique_id: str
) -> AcadosModel:
    """Creates an acados model from a symbolic drone_model."""
    # Build base symbolic variables (outside the read-only drone-models library)
    x_base = ca.MX.sym("x_base", 13, 1)
    u_base = ca.MX.sym("u_base", 4, 1)

    # Call the library helper to get the base dynamics expression
    X_dot_lib, X_lib, U_lib, _ = symbolic_dynamics_euler(
        model_rotor_vel=True,
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        thrust_time_coef=parameters["thrust_time_coef"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
        drag_matrix=parameters["drag_matrix"],
    )

    # Substitute library-internal state/input symbols with our x_base / u_base symbols.
    # The library defines rotor_vel as a 4-vector, making X_lib size 16.
    # We pad our 13-element x_base with 3 zeros to match sizes during substitution.
    X_sub = ca.vertcat(x_base, ca.MX.zeros(3, 1))
    X_dot_base = ca.substitute(X_dot_lib, X_lib, X_sub)

    # Slice the first 13 elements to extract our 12 base states + the scalar thrust derivative
    X_dot_base = X_dot_base[0:13]
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
    model.name = f"mpcc_attitude_mpc_{unique_id}"
    model.x = x_aug
    model.u = u_aug
    model.f_expl_expr = x_dot_aug
    model.f_impl_expr = None

    # model parameter vector p (12 spline coeffs + 1 contouring weight q_c + 1 local offset)
    num_obs = len(obs_manager.obstacles)
    num_params = 14 + (6 * num_obs)  # 14 for MPCC, 6 per obstacle (p1, p2)

    model.p = ca.MX.sym("p", num_params)
    px = model.p[0:4]
    py = model.p[4:8]
    pz = model.p[8:12]
    q_c = model.p[12]
    theta_offset = model.p[13]
    p_obs = model.p[14:]  # Extract dynamic obstacle parameters

    # Extract the virtual states from the augmented state vector
    theta_sym = x_aug[13]
    v_theta_sym = x_aug[14]

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

    # Final cost_y_expr. Size: 3 (Contour) + 1 (Lag) + 4 (u_base) + 1 (v_theta) = 9
    model.cost_y_expr = ca.vertcat(weighted_e_cont, e_lag_scalar, u_aug[0:4], v_theta_sym)
    model.cost_y_expr_e = ca.vertcat(weighted_e_cont, e_lag_scalar)

    if num_obs > 0:
        model.con_h_expr = obs_manager.get_collision_expressions(x_aug, p_obs)
        model.con_h_expr_e = obs_manager.get_collision_expressions(x_aug, p_obs)

    return model


def create_ocp_solver(
    Tf: float, N: int, parameters: dict, obs_manager: ObstacleManager, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()
    unique_id = uuid.uuid4().hex[:8]

    # Set model (augmented with MPCC states/inputs)
    ocp.model = create_acados_model(parameters, obs_manager, unique_id)

    # Get Dimensions
    nx = ocp.model.x.rows()
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

    W_contour = parameters.get("Q_c", 20.0)
    W_lag = parameters.get("Q_l", 250.0)
    W_controls = parameters.get("R_u", 50.0)
    W_progress = parameters.get("mu", 0.1)
    W_thrust = parameters.get("R_T", 250.0)

    W = np.zeros((ny, ny))
    # Weights: [Contour(3), Lag(1), Controls(4), Progress(1)]
    W[0:3, 0:3] = np.diag([W_contour, W_contour, W_contour])
    W[3, 3] = W_lag  # High constant weight q_l to keep the virtual state tied to reality
    W[4:8, 4:8] = np.diag(
        [W_controls, W_controls, W_controls, W_thrust]
    )  # Control regularization R
    W[8, 8] = W_progress  # Progress weight mu
    ocp.cost.W = W

    # Terminal weights
    W_e = np.zeros((ny_e, ny_e))
    W_e[0:3, 0:3] = np.diag([40.0, 40.0, 40.0])  # Terminal contour
    W_e[3, 3] = 300.0  # Terminal lag
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
    # TODO: revert roll/pitch/yaw limits back to [-0.5, -0.5, -0.5] / [0.5, 0.5, 0.5]
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5, 0.0])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5, 10.0])
    ocp.constraints.idxbx = np.array([3, 4, 5, 14])

    # Set Input Constraints (roll/pitch/thrust and virtual acceleration a_theta)
    # TODO: revert roll/pitch limits back to [-0.5, -0.5, -0.5] / [0.5, 0.5, 0.5]
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4, 0.01])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4, 10.0])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    nh = len(obs_manager.obstacles)
    if nh > 0:
        # We want h(x) >= 0, so lower bound is 0, upper bound is infinity
        ocp.constraints.lh = np.zeros(nh)
        ocp.constraints.uh = 1e9 * np.ones(nh)
        ocp.constraints.lh_e = np.zeros(nh)
        ocp.constraints.uh_e = 1e9 * np.ones(nh)

        # Enable slacks on all nonlinear constraints
        ocp.constraints.idxsh = np.arange(nh)
        ocp.constraints.idxsh_e = np.arange(nh)

        # Linear-Quadratic Penalty Weights
        # Zl / Zu are L2 (Quadratic) weights
        # zl / zu are L1 (Linear) weights
        # We penalize violating the lower bound heavily
        Z_l_weight = parameters.get("Z_l", 4000.0)
        z_l_weight = parameters.get("z_l", 4000.0)

        Z_l = Z_l_weight * np.ones(nh)
        z_l = z_l_weight * np.ones(nh)

        ocp.cost.Zl = Z_l
        ocp.cost.Zu = np.zeros(nh)
        ocp.cost.zl = z_l
        ocp.cost.zu = np.zeros(nh)

        ocp.cost.Zl_e = Z_l
        ocp.cost.Zu_e = np.zeros(nh)
        ocp.cost.zl_e = z_l
        ocp.cost.zu_e = np.zeros(nh)

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

    num_params = 14 + (6 * nh)
    ocp.parameter_values = np.zeros((num_params,))

    # set prediction horizon
    ocp.solver_options.tf = Tf

    unique_id = uuid.uuid4().hex[:8]

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file=f"c_generated_code/lsy_example_mpc_{unique_id}.json",
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
        self._log_q_c = []
        self.last_solver_status = 0

        self._obstacle_manager = ObstacleManager(safety_margin=0.14)
        gate_positions = np.array([g["pos"] for g in config.env.track.gates])
        gate_rpys = np.array([g["rpy"] for g in config.env.track.gates])
        for gate_pos, gate_rpy in zip(gate_positions, gate_rpys):
            self._obstacle_manager.add_gate(gate_pos, gate_rpy)

        if hasattr(config.env.track, "obstacles") and config.env.track.obstacles:
            for pole_pos in config.env.track.obstacles:
                self._obstacle_manager.add_pole(pole_pos)

        # start_pos = obs["pos"]
        # None for hardcoded trajectory; gate_positions and start_pos for gates as waypoints
        self._trajectory = TrajectoryPlanner(start_pos=None, gates_pos=None)

        self.drone_params = load_params("so_rpy_rotor_drag", config.sim.drone_model)

        if hasattr(config, "mpcc_tune"):
            self.drone_params.update(config.mpcc_tune)

        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params, self._obstacle_manager
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
        self._config = config
        self._finished = False
        self._last_thrust = self.drone_params["mass"] * 9.81  # Track thrust state for next tick

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
        if info is not None:
            gates_pos = None
            gates_yaw = None
            if "gates_pos" in info:
                gates_pos = np.array(info["gates_pos"], dtype=np.float64)
            elif "gates_pos" in obs:
                gates_pos = np.array(obs["gates_pos"], dtype=np.float64)

            if "gates_yaw" in info:
                gates_yaw = np.array(info["gates_yaw"], dtype=np.float64)
            elif "gates_quat" in obs:
                gates_quat = np.array(obs["gates_quat"], dtype=np.float64)
                gates_yaw = R.from_quat(gates_quat).as_euler("xyz")[:, 2]

            if gates_pos is not None and gates_yaw is not None:
                gates_rpys = np.zeros((gates_pos.shape[0], 3), dtype=np.float64)
                gates_rpys[:, 2] = gates_yaw
                self._obstacle_manager.update_gate_positions(gates_pos, gates_rpys)

            obstacles_pos = None
            if "obstacles_pos" in info:
                obstacles_pos = np.array(info["obstacles_pos"], dtype=np.float64)
            elif "obstacles_pos" in obs:
                obstacles_pos = np.array(obs["obstacles_pos"], dtype=np.float64)

            if obstacles_pos is not None:
                self._obstacle_manager.update_pole_positions(obstacles_pos)

        # Define the terminal condition:
        # The environment sets target_gate to -1 exactly when the final gate plane is crossed.
        if "target_gate" in obs and int(obs["target_gate"]) == -1:
            self._finished = True

        # Setting initial state
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"], [self._last_thrust]))

        # Use persistent virtual progress states instead of resetting them every tick.
        theta0 = self._current_theta
        v_theta0 = self._current_v_theta

        # Augmented state has physical states, rotor velocity, and two virtual progress states.
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

        # Extract flattened obstacle parameters
        obs_params = self._obstacle_manager.get_obstacle_parameters()
        num_obs = len(self._obstacle_manager.obstacles)
        total_params = 14 + (6 * num_obs)

        # Set spline parameter guess stage-by-stage using predicted theta.
        for j in range(self._N):
            if self._tick == 0:
                # Provide a kinematic guess for the very first tick to prevent divergence
                theta_pred = self._current_theta + j * self._dt * 2.0
                xj_guess = x0_aug.copy()
                xj_guess[13] = theta_pred
                xj_guess[14] = 2.0
                hover_u = np.array([0.0, 0.0, 0.0, self.drone_params["mass"] * 9.81, 0.0])
                self._acados_ocp_solver.set(j, "x", xj_guess)
                self._acados_ocp_solver.set(j, "u", hover_u)
            else:
                # Use the solver's optimized trajectory from the previous tick as the segment guess
                xj_prev = self._acados_ocp_solver.get(j, "x")
                theta_pred = float(xj_prev[13])

            theta_pred = float(
                np.clip(
                    theta_pred, self._trajectory.knot_points[0], self._trajectory.knot_points[-1]
                )
            )

            px_j, py_j, pz_j, theta_offset_j = self._trajectory.get_polynomial_coeffs_at(theta_pred)
            pos_pred = self._trajectory.evaluate(theta_pred)
            q_c_j = self._obstacle_manager.dynamic_contour_weight(pos_pred)

            params_j = np.zeros((total_params,))
            params_j[0:4] = px_j.flatten()
            params_j[4:8] = py_j.flatten()
            params_j[8:12] = pz_j.flatten()
            params_j[12] = q_c_j
            params_j[13] = theta_offset_j
            params_j[14:] = obs_params  # Inject dynamic obstacles

            self._acados_ocp_solver.set(j, "p", params_j)

        # Set parameters for the terminal node (N)
        self._acados_ocp_solver.set(self._N, "p", params_j)

        # Solve and extract first control. We run the RTI solver to meet 50 Hz real-time.
        self.last_solver_status = self._acados_ocp_solver.solve()

        for j in range(self._N):
            xj = self._acados_ocp_solver.get(j, "x")
            self._predicted_trajectory[j] = xj[0:3]

        u0_aug = self._acados_ocp_solver.get(0, "u")
        x1_opt = self._acados_ocp_solver.get(1, "x")
        self._current_theta = float(x1_opt[13])
        self._current_v_theta = float(x1_opt[14])

        u0 = u0_aug[0:4]
        self._last_thrust = float(u0[3])  # Save thrust command for next tick

        # Calculate Contour and Lag Error for Telemetry
        p_curr = obs["pos"]
        p_ref = self._trajectory.evaluate(self._current_theta)
        t_ref = self._trajectory.evaluate_velocity(self._current_theta)
        t_norm = t_ref / (np.linalg.norm(t_ref) + 1e-6)

        e_pos = p_curr - p_ref
        e_lag_val = np.dot(e_pos, t_norm)
        e_cont_vec = e_pos - e_lag_val * t_norm
        e_cont_val = np.linalg.norm(e_cont_vec)

        q_c_current = self._obstacle_manager.dynamic_contour_weight(p_ref)

        # Append to logs
        self._log_thrust.append(float(u0[3]))
        self._log_roll.append(float(u0[0]))
        self._log_pitch.append(float(u0[1]))
        self._log_contour.append(float(e_cont_val))
        self._log_lag.append(float(e_lag_val))
        self._log_v_theta.append(self._current_v_theta)
        self._log_q_c.append(float(q_c_current))

        self._tick += 1
        return u0

    def render_callback(self, sim: Sim):
        """Visualize the reference path, current MPCC target, predicted horizon, and obstacles."""
        trajectory = self._trajectory.evaluate(np.linspace(0.0, self._trajectory.total_length, 150))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))

        target_pos = self._trajectory.evaluate(self._current_theta)
        draw_points(sim, target_pos.reshape(1, -1), rgba=(1.0, 0.0, 0.0, 1.0), size=0.04)

        if self._predicted_trajectory.shape[0] > 0:
            draw_line(sim, self._predicted_trajectory, rgba=(1.0, 0.5, 0.0, 1.0))

        self._obstacle_manager.render(sim, rgba=(1.0, 0.0, 0.0, 0.3))

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
        plotting = False
        if plotting is True:
            hover_thrust = self.drone_params["mass"] * 9.81
            max_thrust = self.drone_params["thrust_max"] * 4
            min_thrust = self.drone_params["thrust_min"] * 4

            fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

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
            axs[3].legend()
            axs[3].grid(True)

            # 5. Dynamic Contour Weight
            axs[4].plot(self._log_q_c, label="Contour Weight (q_c)", color="purple")
            axs[4].set_ylabel("Weight")
            axs[4].set_xlabel("Timestep")
            axs[4].legend()
            axs[4].grid(True)

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
            self._log_q_c.clear()
            self._tick = 0
