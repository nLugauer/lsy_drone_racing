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
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.obstacleManager import ObstacleManager
from lsy_drone_racing.control.planners import TOGTPlanner

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


def create_acados_model(parameters: dict, obs_manager: ObstacleManager) -> AcadosModel:
    """Creates an acados model from a symbolic drone_model."""
    # For more info on the models, check out https://github.com/learnsyslab/drone-models
    X_dot, X, U, _ = symbolic_dynamics_euler(
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

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = "basic_example_mpc"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U

    if len(obs_manager.obstacles) > 0:
        # Create a symbolic parameter vector: 6 values (p1_xyz, p2_xyz) per obstacle
        num_obs = len(obs_manager.obstacles)
        p_sym = ca.MX.sym("p_obs", num_obs * 6)
        model.p = p_sym  # Register parameter in the model

        # Pass the parameter vector to your expression builder
        model.con_h_expr = obs_manager.get_collision_expressions(X, p_sym)
        model.con_h_expr_e = obs_manager.get_collision_expressions(X, p_sym)

    return model


def create_ocp_solver(
    Tf: float, N: int, parameters: dict, obs_manager: ObstacleManager, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # Set model
    ocp.model = create_acados_model(parameters, obs_manager)

    # Get Dimensions
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf
    #

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    # State weights
    Q = np.diag(
        [
            200.0,  # pos
            200.0,  # pos
            200.0,  # pos
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

    Q_e = Q.copy()
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx_e = Vx_e

    # Set initial references. We will overwrite these later to track the trajectory
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))

    # Set State Constraints (rpy < 30°)
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    # Set Input Constraints (rpy < 30°)
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # Set hard constraints (obstacles)
    nh = len(obs_manager.obstacles)

    if nh > 0:
        # Path constraints
        ocp.constraints.lh = np.zeros(nh)
        ocp.constraints.uh = 1e9 * np.ones(nh)
        ocp.constraints.idxsh = np.arange(nh)  # Soften path constraints

        # Terminal constraints (at node N)
        # Without these, ns_e remains 0 and causes your error
        ocp.constraints.lh_e = np.zeros(nh)
        ocp.constraints.uh_e = 1e9 * np.ones(nh)
        ocp.constraints.idxsh_e = np.arange(nh)  # Soften terminal constraints

        # --- SLACK PENALTIES ---
        # Path penalties
        ocp.cost.Zl = 8e4 * np.ones(nh)
        ocp.cost.Zu = np.zeros(nh)
        ocp.cost.zl = 1e4 * np.ones(nh)
        ocp.cost.zu = np.zeros(nh)

        # Terminal penalties
        ocp.cost.Zl_e = 2e4 * np.ones(nh)
        ocp.cost.Zu_e = np.zeros(nh)
        ocp.cost.zl_e = 1e3 * np.ones(nh)
        ocp.cost.zu_e = np.zeros(nh)

        # Initialize the parameter values in the OCP
        initial_params = obs_manager.get_obstacle_parameters()
        ocp.parameter_values = initial_params

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_, PARTIAL_ ,_HPIPM, _QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP, SQP_RTI
    ocp.solver_options.tol = 1e-6

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 40
    ocp.solver_options.nlp_solver_max_iter = 100

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
        self._N = 30
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        self.obs_manager = ObstacleManager(safety_margin=0.15)
        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self.obs_manager.initialize_track(config.env.track)
        self._observed_gate_positions: np.ndarray | None = None
        self._observed_obstacle_positions: np.ndarray | None = None

        gate_polygons = self.obs_manager.get_togt_polygons(config.env.track)
        start_pos = np.asarray(obs["pos"], dtype=np.float64)

        self._trajectory_planner = TOGTPlanner(
            gate_polygons, parameters=self.drone_params, freq=config.env.freq, start_pos=start_pos
        )
        self._trajectory_planner.plan_trajectory()

        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params, self.obs_manager
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._tick = 0
        self._tick_max = max(0, self._trajectory_planner.max_ticks - self._N)
        self._config = config
        self._finished = False

        self._planned_trajectory = None
        self._path_history = []

    def _refresh_obstacle_gate_positions(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Update the obstacle manager when new gate or obstacle positions are observed."""
        if "gates_pos" in obs:
            gates_pos = np.asarray(obs["gates_pos"], dtype=np.float64)
            gates_rpy = None
            if "gates_rpy" in obs:
                gates_rpy = np.asarray(obs["gates_rpy"], dtype=np.float64)
            elif "gates_quat" in obs:
                gates_rpy = R.from_quat(np.asarray(obs["gates_quat"], dtype=np.float64)).as_euler(
                    "xyz"
                )

            if self._observed_gate_positions is None or not np.allclose(
                gates_pos, self._observed_gate_positions, atol=1e-8
            ):
                self.obs_manager.update_gate_positions(gates_pos, gates_rpy)
                self._observed_gate_positions = gates_pos.copy()

        if "obstacles_pos" in obs:
            obs_pos = np.asarray(obs["obstacles_pos"], dtype=np.float64)
            if self._observed_obstacle_positions is None or not np.allclose(
                obs_pos, self._observed_obstacle_positions, atol=1e-8
            ):
                self.obs_manager.update_obstacle_positions(obs_pos)
                self._observed_obstacle_positions = obs_pos.copy()

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
        if "pos" in obs:
            self._path_history.append(obs["pos"].copy())
            # Keep history from getting too long and lagging the sim
            if len(self._path_history) > 100:
                self._path_history.pop(0)

        self._refresh_obstacle_gate_positions(obs)

        if self._tick >= self._tick_max:
            self._finished = True

        # Setting initial state
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        pos_ref, vel_ref, yaw_ref, pos_e, vel_e, yaw_e = self._trajectory_planner.get_references(
            current_tick=self._tick, horizon=self._N
        )

        # Save the immediate tracking target for visualization
        self._current_tracking_point = pos_ref[0].copy()

        # Setting state reference
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = pos_ref
        # zero roll, pitch
        yref[:, 5] = yaw_ref
        yref[:, 6:9] = vel_ref
        # zero drpy

        # Setting input reference (index > self._nx)
        # zero rpy
        # hover thrust
        yref[:, 15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        # Setting final state reference
        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = pos_e
        # zero roll, pitch
        yref_e[5] = yaw_e
        yref_e[6:9] = vel_e
        # zero drpy
        # AcadosOcpSolver expects stage variable names, not the OCP cost attribute name.
        self._acados_ocp_solver.set(self._N, "yref", yref_e)

        current_obs_params = self.obs_manager.get_obstacle_parameters()
        for j in range(self._N + 1):
            self._acados_ocp_solver.set(j, "p", current_obs_params)

        # Solving problem and getting first input
        status = self._acados_ocp_solver.solve()  # check solver status and handle infeasibility
        if status != 0:
            print(f"Acados solver returned with status {status} at tick {self._tick}")
        u0 = self._acados_ocp_solver.get(0, "u")

        # visualization of the planned trajectory
        self._planned_trajectory = np.array(
            [self._acados_ocp_solver.get(j, "x")[0:3] for j in range(self._N + 1)]
        )

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

    def render_callback(self, sim: Sim) -> None:
        """Visualize the overall track, drone history, and MPC prediction."""
        # 1. Draw the reference waypoints (Green Line)
        # This shows the entire track the MPC is trying to follow

        # 4. Draw the current tracking target (Bright Green Dot)
        if hasattr(self, "_current_tracking_point"):
            draw_points(
                sim,
                self._current_tracking_point.reshape(1, 3),
                rgba=(0.0, 1.0, 0.0, 1.0),
                size=0.03,
            )

        if hasattr(self._trajectory_planner, "traj_pos"):
            traj_vis = self._trajectory_planner.traj_pos[
                :: max(1, len(self._trajectory_planner.traj_pos) // 80)
            ]
            draw_line(sim, traj_vis, rgba=(0.0, 0.8, 0.2, 0.7))
        if hasattr(self._trajectory_planner, "basic_spline_pos"):
            basic_vis = self._trajectory_planner.basic_spline_pos[
                :: max(1, len(self._trajectory_planner.basic_spline_pos) // 80)
            ]
            draw_line(sim, basic_vis, rgba=(1.0, 0.0, 0.0, 0.5))
        if hasattr(self._trajectory_planner, "waypoints_pos"):
            draw_points(
                sim, self._trajectory_planner.waypoints_pos, rgba=(1.0, 0.85, 0.0, 1.0), size=0.04
            )
        # Draw the obstacles (Red Transparent Capsules)
        self.obs_manager.render(sim)

        # 2. Draw actual flight path history (Blue Line)
        if len(self._path_history) > 1:
            # Downsample by 3 for performance
            path_array = np.array(self._path_history[::3])
            draw_line(sim, path_array, rgba=(0.0, 0.5, 1.0, 1.0))

        # 3. Draw the MPC Planned Horizon (Red Line & Dots)
        if self._planned_trajectory is not None:
            # Draw a line connecting the planned states
            draw_line(sim, self._planned_trajectory, rgba=(1.0, 0.0, 0.0, 1.0))

            # Draw dots at each prediction step to see the spacing (speed)
            # Red dots indicate horizon points inside obstacle safety margins.
            collision_mask = self.obs_manager.points_in_obstacles(
                self._planned_trajectory, margin=self.obs_manager.safety_margin
            )

            if np.any(collision_mask):
                draw_points(
                    sim,
                    self._planned_trajectory[collision_mask],
                    rgba=(1.0, 0.0, 0.0, 1.0),
                    size=0.01,
                )

            safe_mask = ~collision_mask
            if np.any(safe_mask):
                draw_points(
                    sim, self._planned_trajectory[safe_mask], rgba=(1.0, 0.5, 0.0, 1.0), size=0.01
                )
