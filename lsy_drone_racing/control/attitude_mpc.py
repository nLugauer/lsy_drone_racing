"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from a symbolic drone_model."""
    # For more info on the models, check out https://github.com/learnsyslab/drone-models
    # Build base symbolic variables (outside the read-only drone-models library)
    # Base states: pos(3), rpy(3), vel(3), drpy(3) => 12
    x_base = ca.SX.sym("x_base", 12, 1)
    # Base inputs: rpy_cmd(3), thrust(1) => 4
    u_base = ca.SX.sym("u_base", 4, 1)

    # Call the library helper to get the base dynamics expression (it returns expressions
    # built with its own internal CasADi symbols). We'll substitute the internal symbols
    # with our externally-created symbols to keep the read-only library untouched.
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
    # NOTE: symbolic_dynamics_euler returns casadi SX/MX objects; ca.substitute handles SX
    X_dot_base = ca.substitute(X_dot_lib, X_lib, x_base)
    X_dot_base = ca.substitute(X_dot_base, U_lib, u_base)

    # MPCC augmentation: add virtual progress states theta and v_theta, and virtual input a_theta
    theta = ca.SX.sym("theta")
    v_theta = ca.SX.sym("v_theta")
    a_theta = ca.SX.sym("a_theta")

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

    # Define model parameter vector p to carry cubic polynomial coefficients for the
    # current spline segment (4 coeffs per x,y,z => 12) plus a contouring weight q_c => 13
    model.p = ca.SX.sym("p", 13)

    # Build a placeholder residual expression for nonlinear least squares cost.
    # p layout: [x_c3,x_c2,x_c1,x_c0, y_c3,..y_c0, z_c3..z_c0, q_c]
    px = model.p[0:4]
    py = model.p[4:8]
    pz = model.p[8:12]
    q_c = model.p[12]

    # theta is the last-but-one state in x_aug
    theta_sym = x_aug[12]
    v_theta_sym = x_aug[13]

    # Evaluate cubic polynomial at theta (here assumed to be local segment parameter)
    # polynomial is px[0]*theta^3 + px[1]*theta^2 + px[2]*theta + px[3]
    theta_pows = ca.vertcat(theta_sym**3, theta_sym**2, theta_sym, ca.SX(1.0))
    pos_ref = ca.vertcat(ca.dot(px, theta_pows), ca.dot(py, theta_pows), ca.dot(pz, theta_pows))

    # Residual: cartesian error (pos - pos_ref), appended with control inputs and -v_theta (we will
    # try to maximize progress by minimizing -v_theta, so include -v_theta in residual)
    residual_pos = x_aug[0:3] - pos_ref
    residual_u = u_aug[0:4]
    residual_vtheta = -v_theta_sym

    # final cost_y_expr. Size: 3 (pos) + 4 (u) + 1 (-v_theta) = 8
    model.cost_y_expr = ca.vertcat(residual_pos, residual_u, residual_vtheta)

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
    nu = ocp.model.u.rows()
    # For NONLINEAR_LS cost we rely on the model.cost_y_expr size
    ny = int(ocp.model.cost_y_expr.rows())
    # terminal residual -- for simplicity we use position-only terminal residual size 3
    ny_e = 3

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf
    #

    # Cost Type: use NONLINEAR_LS for MPCC formulation
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    # Weights
    # State weights
    Q = np.diag(
        [
            50.0,  # pos
            50.0,  # pos
            400.0,  # pos
            1.0,  # rpy
            1.0,  # rpy
            1.0,  # rpy
            10.0,  # vel
            10.0,  # vel
            10.0,  # vel
            5.0,  # drpy
            5.0,  # drpy
            5.0,  # drpy
        ]
    )
    # Input weights (reference is upright orientation and hover thrust)
    R = np.diag(
        [
            1.0,  # rpy
            1.0,  # rpy
            1.0,  # rpy
            50.0,  # thrust
        ]
    )

    # For nonlinear LS, define a simple weighting for the residual vector
    W = np.eye(ny)
    # Increase position residual weight (first 3 entries)
    W[0:3, 0:3] = np.eye(3) * 100.0
    ocp.cost.W = W

    # Terminal weight (only position in this simple skeleton)
    W_e = np.eye(ny_e) * 200.0
    ocp.cost.W_e = W_e

    # Set initial references (yref is zero since residual encodes errors)
    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # Hook up model nonlinear residuals to ocp (acados_template expects model.cost_y_expr)
    ocp.model.cost_y_expr = ocp.model.cost_y_expr
    # Terminal residual: position error between x[0:3] and the polynomial evaluated at theta
    # Reconstruct terminal pos_ref expression using model symbols
    p = ocp.model.p
    px = p[0:4]
    py = p[4:8]
    pz = p[8:12]
    theta_sym = ocp.model.x[12]
    theta_pows = ca.vertcat(theta_sym**3, theta_sym**2, theta_sym, ca.SX(1.0))
    pos_ref_e = ca.vertcat(ca.dot(px, theta_pows), ca.dot(py, theta_pows), ca.dot(pz, theta_pows))
    ocp.model.cost_y_expr_e = ocp.model.x[0:3] - pos_ref_e

    # Set State Constraints (rpy < 30°)
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    # Set Input Constraints (rpy < 30°)
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

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
        self._t_total = 15  # s
        t = np.linspace(0, self._t_total, len(waypoints))
        self._des_pos_spline = CubicSpline(t, waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()
        self._waypoints_pos = self._des_pos_spline(
            np.linspace(0, self._t_total, int(config.env.freq * self._t_total))
        )
        self._waypoints_vel = self._des_vel_spline(
            np.linspace(0, self._t_total, int(config.env.freq * self._t_total))
        )
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
        i = min(self._tick, self._tick_max)
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

        # Local theta initial guess: simple placeholder (0.0). TODO: replace with proper projection
        theta0 = 0.0
        v_theta0 = 0.0

        # Augmented initial state
        x0_aug = np.concatenate((x0, np.array([theta0, v_theta0])))
        self._acados_ocp_solver.set(0, "lbx", x0_aug)
        self._acados_ocp_solver.set(0, "ubx", x0_aug)

        # For MPCC we provide the spline coefficients and a contouring weight in model.p.
        # We will not use time-based yref trajectory anymore; set all yref entries to zero
        yref_zero = np.zeros((self._ny,))
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref_zero)

        # Terminal yref zero (size ny_e)
        yref_e_zero = np.zeros((self._ny_e,))
        self._acados_ocp_solver.set(self._N, "y_ref", yref_e_zero)

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

        # Dynamic contouring weight (placeholder heuristic)
        # Increase weight when close to a waypoint/gate (simple proximity test)
        dist_to_waypoint = np.linalg.norm(self._waypoints_pos[nearest_idx] - obs["pos"])
        q_c = 10.0 if dist_to_waypoint < 0.5 else 1.0

        params = np.zeros((13,))
        params[0:4] = px.flatten()
        params[4:8] = py.flatten()
        params[8:12] = pz.flatten()
        params[12] = q_c

        # Set the same (or a shifting) segment polynomial across the horizon.
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "p", params)

        # Solve and extract first control. We run the RTI solver to meet 50 Hz real-time.
        self._acados_ocp_solver.solve()
        u0_aug = self._acados_ocp_solver.get(0, "u")

        # Strip virtual input a_theta (last entry) and return base control [r,p,y,thrust]
        u0 = u0_aug[0:4]

        return u0

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
        """Reset the integral error."""
        self._tick = 0
