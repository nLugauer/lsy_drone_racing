"""MPCC attitude controller for the drone-racing task.

Model Predictive Contouring Controller (Foehn et al. 2021, Sec. III) over the collective-thrust +
attitude interface: virtual progress states (theta, v_theta) advance along a reference path while
contour/lag errors, control effort and progress are penalised. The reference is a near time-optimal
PMM racing line (point_mass_planner.py); gates/poles enter the cost as a dynamic contour weight and
the solver as soft collision constraints (obstacle_manager.py). Built and solved with acados
(SQP-RTI) at 50 Hz.

based on OG attitude_mpc.py structure by LSY
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import TYPE_CHECKING

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import yaml
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from drone_models.so_rpy_rotor_drag import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.obstacle_manager import ObstacleManager
from lsy_drone_racing.control.point_mass_planner import AsyncPMMReplanner, PointMassPlanner
from lsy_drone_racing.control.trajectory_planner import TrajectoryPlanner

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

# Shared with the PMM planner so one setLevel controls all planner + replan logging.
logger = logging.getLogger("lsy_drone_racing.pmm")


def create_acados_model(
    parameters: dict, obs_manager: ObstacleManager, unique_id: str
) -> AcadosModel:
    """Build the MPCC acados model: base drone dynamics augmented with progress states."""
    x_base = ca.MX.sym("x_base", 13, 1)
    u_base = ca.MX.sym("u_base", 4, 1)
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
    # The library models rotor_vel as a 4-vector (X_lib size 16); pad our 13-state vector with 3
    # zeros for the substitution, then slice back to 13.
    X_dot_base = ca.substitute(X_dot_lib, X_lib, ca.vertcat(x_base, ca.MX.zeros(3, 1)))[0:13]
    X_dot_base = ca.substitute(X_dot_base, U_lib, u_base)

    # MPCC augmentation: virtual progress states theta, v_theta and virtual input a_theta.
    theta = ca.MX.sym("theta")
    v_theta = ca.MX.sym("v_theta")
    a_theta = ca.MX.sym("a_theta")
    x_aug = ca.vertcat(x_base, theta, v_theta)
    u_aug = ca.vertcat(u_base, a_theta)
    x_dot_aug = ca.vertcat(X_dot_base, v_theta, a_theta)

    model = AcadosModel()
    model.name = f"mpcc_attitude_mpc_{unique_id}"
    model.x = x_aug
    model.u = u_aug
    model.f_expl_expr = x_dot_aug

    # Parameters: 12 spline coeffs + q_c + theta_offset, then 6 (p1, p2) per obstacle.
    num_obs = len(obs_manager.obstacles)
    model.p = ca.MX.sym("p", 14 + 6 * num_obs)
    px, py, pz = model.p[0:4], model.p[4:8], model.p[8:12]
    q_c, theta_offset, p_obs = model.p[12], model.p[13], model.p[14:]

    # Exact MPCC contour/lag errors (Foehn et al. 2021, Sec. III-D, eqs. 8-11). d_theta is a local
    # segment parameter to avoid large absolute powers.
    d_theta = theta - theta_offset
    theta_pows = ca.vertcat(d_theta**3, d_theta**2, d_theta, ca.MX(1.0))  # eq. 8: path p^d
    pos_ref = ca.vertcat(ca.dot(px, theta_pows), ca.dot(py, theta_pows), ca.dot(pz, theta_pows))
    theta_dot_pows = ca.vertcat(
        3 * d_theta**2, 2 * d_theta, ca.MX(1.0), ca.MX(0.0)
    )  # eq. 9: tangent
    t_vec = ca.vertcat(
        ca.dot(px, theta_dot_pows), ca.dot(py, theta_dot_pows), ca.dot(pz, theta_dot_pows)
    )
    # Cubic splines are only approximately arc-length, so normalize the tangent explicitly.
    t_norm = t_vec / ca.sqrt(ca.sumsqr(t_vec) + 1e-4)
    e_pos = x_aug[0:3] - pos_ref
    e_lag = ca.dot(e_pos, t_norm)  # eq. 10: lag error
    e_cont = e_pos - e_lag * t_norm  # eq. 11: contour error
    # sqrt(q_c) so the least-squares residual squares to q_c * ||e_cont||^2 (contour weight).
    weighted_e_cont = ca.sqrt(q_c) * e_cont

    model.cost_y_expr = ca.vertcat(weighted_e_cont, e_lag, u_aug[0:4], v_theta)
    model.cost_y_expr_e = ca.vertcat(weighted_e_cont, e_lag)
    if num_obs > 0:
        model.con_h_expr = obs_manager.get_collision_expressions(x_aug, p_obs)
        model.con_h_expr_e = obs_manager.get_collision_expressions(x_aug, p_obs)
    return model


def create_ocp_solver(
    Tf: float, N: int, parameters: dict, obs_manager: ObstacleManager, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Build the acados OCP and RTI solver for the MPCC formulation."""
    ocp = AcadosOcp()
    unique_id = uuid.uuid4().hex[:8]
    ocp.model = create_acados_model(parameters, obs_manager, unique_id)

    nx = ocp.model.x.rows()
    ny = int(ocp.model.cost_y_expr.rows())
    ny_e = int(ocp.model.cost_y_expr_e.rows())
    ocp.solver_options.N_horizon = N

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    W_contour = parameters.get("Q_c", 20.0)
    W_lag = parameters.get("Q_l", 250.0)
    W_controls = parameters.get("R_u", 50.0)
    W_progress = parameters.get("mu", 0.3)
    W_thrust = parameters.get("R_T", 250.0)

    # Stage weights on [contour(3), lag(1), controls(3) + thrust(1), progress(1)].
    W = np.zeros((ny, ny))
    W[0:3, 0:3] = np.diag([W_contour, W_contour, W_contour])
    W[3, 3] = W_lag
    W[4:8, 4:8] = np.diag([W_controls, W_controls, W_controls, W_thrust])
    W[8, 8] = W_progress
    ocp.cost.W = W

    W_e = np.zeros((ny_e, ny_e))
    W_e[0:3, 0:3] = np.diag([40.0, 40.0, 40.0])  # terminal contour
    W_e[3, 3] = 300.0  # terminal lag
    ocp.cost.W_e = W_e

    # yref is overwritten every tick in compute_control; seed it with hover thrust and a v_theta
    # target so the progress incentive is active while the state bound caps the achievable speed.
    yref = np.zeros((ny,))
    yref[7] = parameters["mass"] * np.linalg.norm(parameters["gravity_vec"])
    yref[8] = 3.0
    ocp.cost.yref = yref
    ocp.cost.yref_e = np.zeros((ny_e,))

    # State bounds: roll/pitch/yaw and forward progress velocity v_theta (capped at 4 m/s).
    ocp.constraints.lbx = np.array([-2.0, -2.0, -2.0, 0.0])
    ocp.constraints.ubx = np.array([2.0, 2.0, 2.0, 4.0])
    ocp.constraints.idxbx = np.array([3, 4, 5, 14])
    # Input bounds: roll/pitch/yaw, collective thrust, virtual acceleration a_theta.
    ocp.constraints.lbu = np.array([-2.0, -2.0, -2.0, parameters["thrust_min"] * 4, 0.01])
    ocp.constraints.ubu = np.array([2.0, 2.0, 2.0, parameters["thrust_max"] * 4, 15.0])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])
    ocp.constraints.x0 = np.zeros((nx))  # overwritten each tick

    nh = len(obs_manager.obstacles)
    if nh > 0:
        # Soft constraints h(x) = signed distance in [0, inf), penalised with L1-L2 slacks.
        ocp.constraints.lh = np.zeros(nh)
        ocp.constraints.uh = 1e9 * np.ones(nh)
        ocp.constraints.lh_e = np.zeros(nh)
        ocp.constraints.uh_e = 1e9 * np.ones(nh)
        ocp.constraints.idxsh = np.arange(nh)
        ocp.constraints.idxsh_e = np.arange(nh)
        Z_l = parameters.get("Z_l", 4000.0) * np.ones(nh)
        z_l = parameters.get("z_l", 4000.0) * np.ones(nh)
        ocp.cost.Zl = Z_l
        ocp.cost.Zu = np.zeros(nh)
        ocp.cost.zl = z_l
        ocp.cost.zu = np.zeros(nh)
        ocp.cost.Zl_e = Z_l
        ocp.cost.Zu_e = np.zeros(nh)
        ocp.cost.zl_e = z_l
        ocp.cost.zu_e = np.zeros(nh)

    # RTI solver for 50 Hz execution.
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tol = 1e-6
    # Levenberg-Marquardt damping of the Gauss-Newton Hessian steadies the single RTI step near
    # obstacles and regularizes the weakly-weighted v_theta/a_theta directions.
    ocp.solver_options.levenberg_marquardt = 1e-3
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50
    ocp.parameter_values = np.zeros((14 + 6 * nh,))
    ocp.solver_options.tf = Tf

    solver = AcadosOcpSolver(
        ocp,
        json_file=f"c_generated_code/lsy_example_mpc_{unique_id}.json",
        verbose=verbose,
        build=True,
        generate=True,
    )
    return solver, ocp


class AttitudeMPC(Controller):
    """MPCC controller using the collective-thrust and attitude interface."""

    USE_PMM_PLANNER = True

    # Online replanning: True -> replan the reference as gates are revealed; False -> OG track
    PMM_REPLAN = True

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller.

        Args:
            obs: Initial observation of the environment state.
            info: Additional reset information from the environment.
            config: Environment configuration.
        """
        super().__init__(obs, info, config)
        self._N = 33
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        self._current_theta = 0.0
        self._current_v_theta = 0.0
        self._predicted_trajectory = np.zeros((self._N, 3))
        self._log_thrust, self._log_roll, self._log_pitch = [], [], []
        self._log_contour, self._log_lag = [], []
        self._log_v_theta, self._log_a_theta, self._log_q_c = [], [], []

        self._obstacle_manager = ObstacleManager(safety_margin=0.12)

        gate_positions = np.array(obs["gates_pos"], dtype=np.float64).reshape(-1, 3)
        gate_quats = np.array(obs["gates_quat"], dtype=np.float64).reshape(-1, 4)
        gate_rpys = R.from_quat(gate_quats).as_euler("xyz")
        poles = np.array(obs.get("obstacles_pos", np.empty((0, 3))), dtype=np.float64).reshape(
            -1, 3
        )
        for pole_pos in poles:
            self._obstacle_manager.add_pole(pole_pos)

        # Setting up gate geometry and adding to obstacle manager
        inner_w = 0.40
        outer_w = 0.72
        lower_r = 0.12
        for gate_pos, gate_rpy in zip(gate_positions, gate_rpys):
            self._obstacle_manager.add_gate(
                gate_pos,
                gate_rpy,
                inner_width=inner_w,
                outer_width=outer_w,
                lower_frame_radius=lower_r,
            )

        self._gates_visited_flags = np.zeros(len(gate_positions), dtype=bool)
        start_pos = np.array(obs["pos"], dtype=np.float64)

        if self.USE_PMM_PLANNER:
            # PMM racing line (Foehn et al. 2021, Sec. VI), refit as an arc-length cubic spline.
            # v_max matches the MPCC v_theta cap; the tail past the last gate must cover the MPCC
            # look-ahead so the reference does not pile up at the spline end. The snapshot gives the
            # planner a frozen, thread-safe obstacle copy.
            v_max = 4.0
            tail_extension = max(0.5, v_max * self._T_HORIZON + 0.5)
            self._trajectory = PointMassPlanner(
                start_pos=start_pos,
                gates_pos=gate_positions,
                gate_rpys=gate_rpys,
                start_vel=np.array(obs["vel"], dtype=np.float64),
                obstacle_manager=self._obstacle_manager.snapshot(),
                u_max=12.0,
                v_max=v_max,
                n_vel_samples=600,  # offline initial plan: more samples -> better global line
                tail_extension=tail_extension,
            )
        else:
            self._trajectory = TrajectoryPlanner(
                start_pos=start_pos, gates_pos=gate_positions, gate_rpys=gate_rpys
            )

        # Drone dynamics + MPCC cost weights. A local mpcc_config.yaml, an external file, or a
        # config.mpcc_tune override takes precedence over the drone defaults.
        self.drone_params = load_params("so_rpy_rotor_drag", config.sim.drone_model)
        mpcc_tune = getattr(config, "mpcc_tune", {})
        config_file = getattr(config, "mpcc_config_file", None)
        local_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpcc_config.yaml")
        mpcc_params: dict = {}
        if config_file and os.path.exists(config_file):
            with open(config_file) as f:
                if config_file.endswith((".yaml", ".yml")):
                    mpcc_params = yaml.safe_load(f) or {}
                elif config_file.endswith(".json"):
                    mpcc_params = json.load(f) or {}
        elif os.path.exists(local_yaml):
            with open(local_yaml) as f:
                mpcc_params = yaml.safe_load(f) or {}
        if hasattr(mpcc_tune, "to_container"):  # OmegaConf
            mpcc_params.update(mpcc_tune.to_container(structured=True))
        elif isinstance(mpcc_tune, dict):
            mpcc_params.update(mpcc_tune)
        else:
            keys = ("Q_c", "Q_l", "R_u", "mu", "R_T", "Z_l", "z_l")
            mpcc_params.update({k: getattr(mpcc_tune, k) for k in keys if hasattr(mpcc_tune, k)})
        for k, v in mpcc_params.items():
            numeric = isinstance(v, (int, float, str)) and not isinstance(v, bool)
            self.drone_params[k] = float(v) if numeric else v

        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params, self._obstacle_manager
        )
        self._ny = int(self._ocp.model.cost_y_expr.rows())

        self._tick = 0
        self._finished = False
        self._last_thrust = self.drone_params["mass"] * 9.81
        self._last_u0 = np.array([0.0, 0.0, 0.0, self._last_thrust])  # QP-failure fallback
        self._needs_warm_start_reset = False

        # Asynchronous PMM replanning so the 50 Hz loop never stalls; the obstacle
        # manager is updated every tick, so the MPCC constraints always use live positions.
        self._replanner = AsyncPMMReplanner() if self.USE_PMM_PLANNER else None
        # Offline backbone: the high-M global plan above
        self._backbone = self._trajectory if self.USE_PMM_PLANNER else None
        self._suffix_gap = 0.5  # [m] start the backbone suffix this far past the last window gate
        self._planned_gates_pos = gate_positions.copy()
        self._planned_target = 0
        self._replan_horizon = 3  # gates ahead of the target to replan through (paper Sec. VI-B)
        self._replan_vel_samples = 80
        self._replan_gate_move = 0.12  # [m] observed gate shift that triggers a replan
        self._commit_distance = (
            0.35  # [m] near-field kept fixed across a replan (no reference jump)
        )
        self._gate_approach_margin = 0.4  # [m] approach room left before the target gate on replan

    def _stage_params(
        self, theta: float, target_gate_idx: int, obs_params: NDArray[np.floating], n_params: int
    ) -> NDArray[np.floating]:
        """Assemble the acados parameter vector (spline coeffs, q_c, offset, obstacles) at theta."""
        px, py, pz, offset = self._trajectory.get_polynomial_coeffs_at(theta)
        pos = self._trajectory.evaluate(theta)
        params = np.zeros(n_params)
        params[0:4] = px.flatten()
        params[4:8] = py.flatten()
        params[8:12] = pz.flatten()
        params[12] = self._obstacle_manager.dynamic_contour_weight(pos, target_gate_idx)
        params[13] = offset
        params[14:] = obs_params
        return params

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next collective-thrust + roll/pitch/yaw command for one control tick."""
        gates_pos = np.array(obs["gates_pos"], dtype=np.float64)
        gate_quats = np.array(obs["gates_quat"], dtype=np.float64)
        gates_yaw = R.from_quat(gate_quats).as_euler("xyz")[:, 2]
        gates_rpys = np.zeros((gates_pos.shape[0], 3), dtype=np.float64)
        gates_rpys[:, 2] = gates_yaw
        self._obstacle_manager.update_gate_positions(gates_pos, gates_rpys)

        obstacles_pos = np.array(obs.get("obstacles_pos", np.empty((0, 3))), dtype=np.float64)
        self._obstacle_manager.update_pole_positions(obstacles_pos)

        # Replan when a gate's observed position changes
        if gates_pos is not None and self.PMM_REPLAN:
            if self.USE_PMM_PLANNER:
                self._maybe_replan_pmm(
                    obs, gates_pos, gates_rpys if gates_yaw is not None else None
                )
            elif "gates_visited" in obs:
                self._original_rebuild(obs, gates_pos)

        # The environment sets target_gate to -1 once the final gate plane is crossed.
        target_gate_idx = int(obs.get("target_gate", 0))
        if target_gate_idx == -1:
            self._finished = True

        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"], [self._last_thrust]))
        x0_aug = np.concatenate((x0, [self._current_theta, self._current_v_theta]))
        self._acados_ocp_solver.set(0, "lbx", x0_aug)
        self._acados_ocp_solver.set(0, "ubx", x0_aug)

        yref_target = np.zeros((self._ny,))
        yref_target[7] = self.drone_params["mass"] * 9.81  # hover thrust
        yref_target[8] = 5.0  # progress-speed target (clipped to the v_theta bound below)

        obs_params = self._obstacle_manager.get_obstacle_parameters()
        total_params = 14 + 6 * len(self._obstacle_manager.obstacles)
        knots = self._trajectory.knot_points
        use_kinematic_guess = self._tick == 0 or self._needs_warm_start_reset
        hover_thrust = self.drone_params["mass"] * 9.81

        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref_target)
            if use_kinematic_guess:
                # Kinematic warm start on the first tick and after any trajectory rebuild: advance
                # theta at the current progress speed and point the velocity along the path tangent.
                theta_pred = self._current_theta + j * self._dt * max(self._current_v_theta, 0.5)
                theta_pred = float(np.clip(theta_pred, knots[0], knots[-1]))
                xj = x0_aug.copy()
                xj[0:3] = self._trajectory.evaluate(theta_pred)
                t_ref = self._trajectory.evaluate_velocity(theta_pred)
                xj[6:9] = t_ref / (np.linalg.norm(t_ref) + 1e-6) * max(self._current_v_theta, 0.5)
                xj[13] = theta_pred
                xj[14] = max(self._current_v_theta, 0.5)
                self._acados_ocp_solver.set(j, "x", xj)
                self._acados_ocp_solver.set(j, "u", np.array([0.0, 0.0, 0.0, hover_thrust, 0.0]))
            else:
                theta_pred = float(self._acados_ocp_solver.get(j, "x")[13])

            theta_pred = float(np.clip(theta_pred, knots[0], knots[-1]))
            self._acados_ocp_solver.set(
                j, "p", self._stage_params(theta_pred, target_gate_idx, obs_params, total_params)
            )

            # Steer v_theta toward the PMM's time-optimal speed (fast on straights, slower into
            # turns), clipped to the v_theta bound. Legacy spline keeps the constant target.
            if self.USE_PMM_PLANNER:
                yref_j = yref_target.copy()
                yref_j[8] = float(
                    np.clip(self._trajectory.evaluate_speed(theta_pred), 0.5, yref_target[8])
                )
                self._acados_ocp_solver.set(j, "yref", yref_j)

        theta_N = float(np.clip(self._acados_ocp_solver.get(self._N, "x")[13], knots[0], knots[-1]))
        self._acados_ocp_solver.set(
            self._N, "p", self._stage_params(theta_N, target_gate_idx, obs_params, total_params)
        )

        status = self._acados_ocp_solver.solve()
        self._needs_warm_start_reset = False
        if status != 0:
            # QP failure (NaN / max iter): reuse the last command and force a fresh warm start.
            self._needs_warm_start_reset = True
            return self._last_u0.copy()

        for j in range(self._N):
            self._predicted_trajectory[j] = self._acados_ocp_solver.get(j, "x")[0:3]

        u0_aug = self._acados_ocp_solver.get(0, "u")
        x1_opt = self._acados_ocp_solver.get(1, "x")
        self._current_theta = float(x1_opt[13])
        self._current_v_theta = float(x1_opt[14])
        u0 = u0_aug[0:4]
        self._last_thrust = float(u0[3])
        self._last_u0 = u0.copy()

        # Telemetry: contour/lag errors and the contour weight at the adopted progress point.
        p_ref = self._trajectory.evaluate(self._current_theta)
        t_ref = self._trajectory.evaluate_velocity(self._current_theta)
        t_norm = t_ref / (np.linalg.norm(t_ref) + 1e-6)
        e_pos = obs["pos"] - p_ref
        e_lag = float(np.dot(e_pos, t_norm))
        self._log_thrust.append(float(u0[3]))
        self._log_roll.append(float(u0[0]))
        self._log_pitch.append(float(u0[1]))
        self._log_contour.append(float(np.linalg.norm(e_pos - e_lag * t_norm)))
        self._log_lag.append(e_lag)
        self._log_v_theta.append(self._current_v_theta)
        self._log_a_theta.append(float(u0_aug[4]))
        self._log_q_c.append(
            float(self._obstacle_manager.dynamic_contour_weight(p_ref, target_gate_idx))
        )
        return u0

    def _maybe_replan_pmm(
        self,
        obs: dict[str, NDArray[np.floating]],
        gates_pos: NDArray[np.floating],
        gates_rpys: NDArray[np.floating] | None,
    ) -> None:
        """Off-thread PMM replanning: adopt a finished background plan and/or start a new one.

        Each tick this swaps in a finished plan and, if a gate in the
        replan window moved, requests a fresh plan on a worker thread. The
        control loop keeps flying the current plan meanwhile.
        """
        target = int(obs.get("target_gate", 0))

        # Adopt a finished plan only if its starting gate has not been passed since the request
        new_planner = self._replanner.take()
        if new_planner is not None and target == self._planned_target:
            # Reject a reversing plan
            vel = np.array(obs["vel"], dtype=np.float64)
            speed = float(np.linalg.norm(vel))
            tang = np.asarray(
                new_planner.evaluate_velocity(new_planner.knot_points[0]), dtype=np.float64
            )
            tang_n = tang / (np.linalg.norm(tang) + 1e-9)
            reversing = speed > 0.3 and float(np.dot(tang_n, vel / speed)) < 0.0
            if not reversing:
                old_theta = self._current_theta
                self._trajectory = new_planner
                self._reanchor_progress(obs)
                # Keep the warm start and shift its theta
                self._shift_warmstart_theta(self._current_theta - old_theta)
                logger.info(
                    "REPLAN adopted: target=%d, len=%.2f m", target, self._trajectory.total_length
                )
            else:
                logger.warning("REPLAN discarded (reversing): target=%d", target)
        elif new_planner is not None:
            logger.warning(
                "REPLAN discarded (stale): built for %d, target %d", self._planned_target, target
            )

        if target < 0:
            return  # final gate passed; nothing left to plan

        # Trigger: a gate in the planning window moved beyond threshold.
        window = slice(target, min(target + self._replan_horizon, len(gates_pos)))
        moved = 0.0
        if window.stop > window.start:
            moved = float(
                np.max(np.linalg.norm(gates_pos[window] - self._planned_gates_pos[window], axis=1))
            )
        if moved <= self._replan_gate_move:
            return
        if self._replanner.busy():
            return  # a replan is already running; re-checked next tick

        reason = f"gate_moved={moved:.3f}m"
        logger.info(
            "REPLAN trigger @%d: %s, target=%d, window=[%d:%d]",
            self._tick,
            reason,
            target,
            window.start,
            window.stop,
        )

        # Start new plan a short look-ahead (commit_distance) ahead of
        # the drone and prepend the segment in between, so the immediate reference is unchanged
        # across adoption
        knots = self._trajectory.knot_points
        theta_now = float(np.clip(self._current_theta, knots[0], knots[-1]))
        theta_commit = min(theta_now + self._commit_distance, self._trajectory.total_length)
        theta_target = float(self._trajectory.nearest_theta(gates_pos[target]))
        if theta_target > theta_now:
            theta_commit = min(theta_commit, theta_target - self._gate_approach_margin)

        committed_pts = committed_speeds = None
        if theta_commit - theta_now > 0.05:
            n_pre = max(12, int((theta_commit - theta_now) / 0.025) + 1)
            s_pre = np.linspace(theta_now, theta_commit, n_pre)
            committed_pts = np.asarray(self._trajectory.evaluate(s_pre), dtype=np.float64)
            committed_speeds = np.array(
                [float(self._trajectory.evaluate_speed(float(s))) for s in s_pre], dtype=np.float64
            )
            start_pos = committed_pts[-1].copy()
            tang = np.asarray(self._trajectory.evaluate_velocity(theta_commit), dtype=np.float64)
            start_vel = (
                tang
                / (np.linalg.norm(tang) + 1e-9)
                * float(self._trajectory.evaluate_speed(theta_commit))
            )
        else:
            start_pos = np.array(obs["pos"], dtype=np.float64)
            start_vel = np.array(obs["vel"], dtype=np.float64)

        # Far-field suffix: reuse the offline backbone beyond the replan window
        # (if number of gates is smaller than total number of gates)
        committed_suffix_pts = committed_suffix_speeds = None
        if self._backbone is not None and window.stop < len(gates_pos):
            bb = self._backbone
            theta_suffix = min(
                float(bb.nearest_theta(gates_pos[window.stop - 1])) + self._suffix_gap,
                bb.total_length,
            )
            if bb.total_length - theta_suffix > 0.05:
                n_suf = int(np.clip((bb.total_length - theta_suffix) / 0.05, 10, 300))
                s_suf = np.linspace(theta_suffix, bb.total_length, n_suf)
                committed_suffix_pts = np.asarray(bb.evaluate(s_suf), dtype=np.float64)
                committed_suffix_speeds = np.asarray(bb.evaluate_speed(s_suf), dtype=np.float64)

        # Snapshots on this thread so the worker reads no shared mutable state. Exclude
        # only the gates this replan routes through, so it can fly through their openings.
        horizon_gates = np.array(gates_pos[window], dtype=np.float64)
        horizon_rpys = (
            np.array(gates_rpys[window], dtype=np.float64) if gates_rpys is not None else None
        )
        obs_snapshot = self._obstacle_manager.snapshot(exclude_gate_centers=horizon_gates)
        planner = self._trajectory  # captured by the closure; _spawn reuses its tuning
        self._replanner.request(
            lambda: planner._spawn(
                start_pos,
                horizon_gates,
                horizon_rpys,
                start_vel,
                obs_snapshot,
                committed_pts=committed_pts,
                committed_speeds=committed_speeds,
                committed_suffix_pts=committed_suffix_pts,
                committed_suffix_speeds=committed_suffix_speeds,
                n_vel_samples=self._replan_vel_samples,
            )
        )
        self._planned_gates_pos = np.array(gates_pos, dtype=np.float64)
        self._planned_target = target

    def _reanchor_progress(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Re-fit (theta, v_theta) to the current trajectory after a swap.

        Searches only the first 3 m of the new path so the reference cannot snap to a future
        segment where the path crosses over itself.
        """
        knot_start = self._trajectory.knot_points[0]
        knot_end = self._trajectory.knot_points[-1]
        search_thetas = np.linspace(knot_start, min(knot_start + 3.0, knot_end), 60)
        pts = self._trajectory.evaluate(search_thetas)
        self._current_theta = float(
            search_thetas[np.argmin(np.linalg.norm(pts - obs["pos"], axis=1))]
        )

        t_new = self._trajectory.evaluate_velocity(self._current_theta)
        t_norm = t_new / (np.linalg.norm(t_new) + 1e-6)
        v_proj = float(np.dot(np.array(obs["vel"], dtype=np.float64), t_norm))
        self._current_v_theta = max(0.01, v_proj)

    def _shift_warmstart_theta(self, delta: float) -> None:
        """Add a constant progress offset to every warm-start stage's theta after a replan swap.

        The near-field geometry is identical across the swap, so shifting only theta by
        the same constant keeps the stored solution valid. No-op on the first tick.
        """
        if self._tick == 0:
            return
        for j in range(self._N + 1):
            xj = self._acados_ocp_solver.get(j, "x")
            xj[13] += delta
            self._acados_ocp_solver.set(j, "x", xj)

    def _original_rebuild(
        self, obs: dict[str, NDArray[np.floating]], gates_pos: NDArray[np.floating]
    ) -> None:
        """Synchronous rebuild for the non-PMM spline planner (USE_PMM_PLANNER False).

        Rebuilds the reference through the revealed gate centers the first time any gate enters
        sensor range, then re-anchors the progress state.
        """
        gates_visited_now = np.array(obs["gates_visited"], dtype=bool)
        if not np.any(gates_visited_now & ~self._gates_visited_flags):
            return
        # Only gates from the current target onwards, to avoid routing back through a passed gate.
        target_gate_idx = int(obs.get("target_gate", 0))
        if 0 <= target_gate_idx < len(gates_pos):
            remaining_gates = gates_pos[target_gate_idx:]
        else:
            remaining_gates = gates_pos[-1:]
        self._trajectory.rebuild(obs["pos"], remaining_gates, gate_rpys=None)
        self._reanchor_progress(obs)
        self._gates_visited_flags = gates_visited_now.copy()
        self._needs_warm_start_reset = True

    def render_callback(self, sim: Sim):
        """Draw the reference path, current MPCC target, predicted horizon, and obstacles."""
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
        """Increment the tick counter and report whether the episode is finished."""
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Plot MPCC telemetry to mpcc_standard_metrics.png and clear the logs."""
        hover = self.drone_params["mass"] * 9.81
        max_thrust = self.drone_params["thrust_max"] * 4
        min_thrust = self.drone_params["thrust_min"] * 4

        if False:
            fig, axs = plt.subplots(6, 1, figsize=(10, 18), sharex=True)
            axs[0].plot(self._log_contour, label="Contour Error (e_c)")
            axs[0].plot(self._log_lag, label="Lag Error (e_l)")
            axs[0].set_ylabel("Error [m]")
            axs[1].plot(self._log_roll, label="Roll Command")
            axs[1].plot(self._log_pitch, label="Pitch Command")
            axs[1].axhline(0.5, color="r", linestyle="--", label="Upper Limit")
            axs[1].axhline(-0.5, color="r", linestyle="--", label="Lower Limit")
            axs[1].set_ylabel("Angle [rad]")
            axs[2].plot(self._log_thrust, label="Thrust Command")
            axs[2].axhline(hover, color="g", linestyle=":", label="Hover")
            axs[2].axhline(max_thrust, color="r", linestyle="--", label="Max Thrust")
            axs[2].axhline(min_thrust, color="r", linestyle="--", label="Min Thrust")
            axs[2].set_ylabel("Thrust [N]")
            axs[3].plot(self._log_v_theta, label="Virtual Speed (v_theta)")
            axs[3].axhline(7.5, color="g", linestyle="--", label="Target Speed")
            axs[3].set_ylabel("Speed [m/s]")
            axs[4].plot(self._log_q_c, label="Contour Weight (q_c)", color="purple")
            axs[4].set_ylabel("Weight")
            axs[4].set_xlabel("Timestep")
            axs[5].plot(self._log_a_theta, label="Virtual Accel (a_theta)", color="orange")
            axs[5].axhline(5.0, color="r", linestyle="--", label="Upper Limit")
            axs[5].axhline(0.01, color="r", linestyle="--", label="Lower Limit")
            axs[5].set_ylabel("Accel [m/s^2]")
            axs[5].set_xlabel("Timestep")
            for ax in axs:
                ax.legend()
                ax.grid(True)

            fig.tight_layout()
            fig.savefig("mpcc_standard_metrics.png")
            plt.show()
            plt.close(fig)

        for log in (
            self._log_thrust,
            self._log_roll,
            self._log_pitch,
            self._log_contour,
            self._log_lag,
            self._log_v_theta,
            self._log_a_theta,
            self._log_q_c,
        ):
            log.clear()
        self._tick = 0
