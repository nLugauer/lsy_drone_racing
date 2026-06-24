"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

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

# Shared with the PMM planner so one setLevel() controls all planner+replan logging:
#     logging.getLogger("lsy_drone_racing.pmm").setLevel(logging.DEBUG)
logger = logging.getLogger("lsy_drone_racing.pmm")


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
    # Cost Type: use NONLINEAR_LS for MPCC formulation
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    W_contour = parameters.get("Q_c", 20.0)
    W_lag = parameters.get("Q_l", 250.0)
    W_controls = parameters.get("R_u", 50.0)
    W_progress = parameters.get("mu", 0.3)
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
    # The solver minimizes (v_theta - target)^2, pushing the drone toward the target speed.
    # Set to match the v_θ state constraint upper bound so the incentive is always active
    # but the constraint prevents physically unreachable commands.
    yref[8] = 3.0
    # Prevent the optimizer from collapsing thrust to zero by targeting hover thrust.
    yref[7] = parameters["mass"] * np.linalg.norm(parameters["gravity_vec"])
    ocp.cost.yref = yref

    ocp.cost.yref_e = np.zeros((ny_e,))

    # Hook up model nonlinear residuals to ocp
    ocp.model.cost_y_expr = ocp.model.cost_y_expr
    ocp.model.cost_y_expr_e = ocp.model.cost_y_expr_e

    # Set State Constraints (roll/pitch/yaw and forward progress velocity)
    ocp.constraints.lbx = np.array([-2.0, -2.0, -2.0, 0.0])
    ocp.constraints.ubx = np.array([2.0, 2.0, 2.0, 4.0])  # v_θ capped at 4 m/s
    ocp.constraints.idxbx = np.array([3, 4, 5, 14])

    # Set Input Constraints (roll/pitch/thrust and virtual acceleration a_theta)
    ocp.constraints.lbu = np.array([-2.0, -2.0, -2.0, parameters["thrust_min"] * 4, 0.01])
    ocp.constraints.ubu = np.array([2.0, 2.0, 2.0, parameters["thrust_max"] * 4, 15.0])
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

    # Reference path source. True -> PMM sampling planner (point_mass_planner.py);
    # False -> the original chord-length cubic spline (trajectory_planner.py). Both expose
    # the same public API, so the rest of this controller is identical either way. Kept as a
    # toggle for A/B lap-time comparison.
    USE_PMM_PLANNER = True

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._N = 33
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
        self._log_a_theta = []
        self._log_q_c = []
        self.last_solver_status = 0

        self._obstacle_manager = ObstacleManager(safety_margin=0.11)

        # Resolve the initial track layout. Two separate cases, because level 3 hides the real
        # layout from the config file:
        #   * Levels 0-2 (config.env.track.randomize == False): the config holds the real nominal
        #     gate/obstacle positions, so read them from the config (original behavior).
        #   * Level 3 (randomize == True): the env fully regenerates the track per reset and the
        #     config positions are just origin placeholders. The actual randomized layout (already
        #     in visit order) is delivered through the reset observation, so read it from obs. The
        #     order is fixed (target_gate), so no reordering is needed. Without this, the first
        #     plan is built through gates stacked at the origin -> a degenerate path that ignores
        #     the true gate order.
        self._randomized_track = bool(getattr(config.env.track, "randomize", False))
        if not self._randomized_track:
            # ---- Levels 0-2: layout comes from the config (unchanged) ----
            gate_positions = np.array([g["pos"] for g in config.env.track.gates], dtype=np.float64)
            gate_rpys = np.array([g["rpy"] for g in config.env.track.gates], dtype=np.float64)

            if hasattr(config.env.track, "obstacles") and config.env.track.obstacles:
                for pole_pos in config.env.track.obstacles:
                    self._obstacle_manager.add_pole(pole_pos)
        else:
            # ---- Level 3: the full randomized layout is delivered through the reset observation
            # (the config holds only origin placeholders). Load ALL gate positions/orientations
            # from the obs/info vector here, before takeoff, so PMM can plan one path through every
            # gate in visit order. obs["gates_pos"] reports each gate's nominal position until the
            # drone senses it within range (then it switches to the measured position); at reset
            # that means the nominal randomized layout for every not-yet-seen gate. The per-gate
            # online refinement afterwards is handled exactly as in level 2 by _maybe_replan_pmm.
            # Check info first (matches compute_control pattern), then obs, then fall back to config
            if "gates_pos" in info:
                gate_positions = np.array(info["gates_pos"], dtype=np.float64).reshape(-1, 3)
            elif "gates_pos" in obs:
                gate_positions = np.array(obs["gates_pos"], dtype=np.float64).reshape(-1, 3)
            else:
                gate_positions = np.array(
                    [g["pos"] for g in config.env.track.gates], dtype=np.float64
                )

            if "gates_quat" in info:
                gate_quats = np.array(info["gates_quat"], dtype=np.float64).reshape(-1, 4)
            elif "gates_quat" in obs:
                gate_quats = np.array(obs["gates_quat"], dtype=np.float64).reshape(-1, 4)
            else:
                gate_quats = np.array(
                    [R.from_euler("xyz", g["rpy"]).as_quat() for g in config.env.track.gates],
                    dtype=np.float64,
                )

            gate_rpys = R.from_quat(gate_quats).as_euler("xyz")

            if "obstacles_pos" in info:
                for pole_pos in np.array(info["obstacles_pos"], dtype=np.float64).reshape(-1, 3):
                    self._obstacle_manager.add_pole(pole_pos)
            elif "obstacles_pos" in obs:
                for pole_pos in np.array(obs["obstacles_pos"], dtype=np.float64).reshape(-1, 3):
                    self._obstacle_manager.add_pole(pole_pos)

        # Register every gate as an obstacle/waypoint (same for all levels).
        for gate_pos, gate_rpy in zip(gate_positions, gate_rpys):
            self._obstacle_manager.add_gate(gate_pos, gate_rpy)

        # Persist the initial layout so the rest of the controller (PMM build, replanning
        # bookkeeping) works off a single, explicit source of truth.
        self._gate_positions = gate_positions.copy()
        self._gate_rpys = gate_rpys.copy()
        self._gates_visited_flags = np.zeros(len(gate_positions), dtype=bool)

        start_pos = np.array(obs["pos"], dtype=np.float64)

        # Sanity guard: in a real randomized track the gates are spread out (>= ~1 m apart). If two
        # or more loaded gate centers coincide in xy, the observation handed us origin placeholders
        # instead of the real layout (e.g. the running env predates the level-3 obs fix), and any
        # plan built now would squiggle at the origin.
        #  Surface that loudly instead of failing silently.
        if len(gate_positions) > 1:
            xy = gate_positions[:, :2]
            dmat = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
            np.fill_diagonal(dmat, np.inf)
            if float(dmat.min()) < 0.2:
                import inspect

                import lsy_drone_racing.envs.randomize as _rz

                has_fix = "nominal_gates_pos=gates_pos" in inspect.getsource(_rz)
                print(
                    "[WARN] AttitudeMPC: obs['gates_pos'] returned near-coincident gate centers "
                    f"at reset -> the observation is not exposing the real level-3 layout.\n"
                    f"       loaded gate xy =\n{np.array2string(gate_positions, precision=3)}\n"
                    f"       env randomize.py actually loaded from: {_rz.__file__}\n"
                    f"       that env file has the #91 nominal-layout fix: {has_fix}\n"
                    "      -> if False, your interpreter is importing an outdated lsy_drone_racing;"
                    "reinstall/point it at this repo so obs exposes all gate positions at reset."
                )

        if self.USE_PMM_PLANNER:
            # PMM planner: builds a near time-optimal racing line (Foehn et al. 2021, Sec. VI),
            # then refits it as an arc-length cubic spline with the same API as TrajectoryPlanner.
            # v_max is kept consistent with the MPCC's v_theta cap; the planner emits geometry
            # only (no timing) and reads obstacles for graph-edge collision pruning.
            # Tail past the last in-path gate must cover the MPCC look-ahead (~v_max * T_horizon),
            # otherwise the reference piles up at the spline end and the drone brakes there. With a
            # LOCAL replan horizon the last in-window gate is mid-track, so this matters on every
            # replan, not only at the finish. tail_extension is captured in the planner's tuning
            # snapshot, so async replans (_spawn) inherit the same value automatically.
            v_max = 4.0
            tail_extension = max(0.5, v_max * self._T_HORIZON + 0.5)
            # The initial plan routes through ALL gates, so exclude every gate's own frame from the
            # PMM's collision pruning (we MUST fly through those openings); poles still block. The
            # MPCC keeps the full obstacle set for its hard constraints, so clearance is unchanged.
            self._trajectory = PointMassPlanner(
                start_pos=start_pos,
                gates_pos=gate_positions,
                gate_rpys=gate_rpys,
                start_vel=np.array(obs["vel"], dtype=np.float64),
                obstacle_manager=self._obstacle_manager.snapshot(
                    exclude_gate_centers=gate_positions
                ),
                u_max=21.0,
                v_max=v_max,
                n_vel_samples=100,  # offline initial plan: more samples -> better global line
                tail_extension=tail_extension,
            )
        else:
            self._trajectory = TrajectoryPlanner(
                start_pos=start_pos, gates_pos=gate_positions, gate_rpys=gate_rpys
            )

        self.drone_params = load_params("so_rpy_rotor_drag", config.sim.drone_model)
        mpcc_params = {}

        # Safely detect if config is a dictionary namespace or dot-accessible object
        if isinstance(config, dict):
            mpcc_tune = config.get("mpcc_tune", {})
            config_file = config.get("mpcc_config_file", None)
        else:
            mpcc_tune = getattr(config, "mpcc_tune", {})
            config_file = getattr(config, "mpcc_config_file", None)

        # Stage 1: Check for an external config file path or fallback to local 'mpcc_config.yaml'
        current_module_dir = os.path.dirname(os.path.abspath(__file__))
        local_yaml_path = os.path.join(current_module_dir, "mpcc_config.yaml")

        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    if config_file.endswith((".yaml", ".yml")):
                        mpcc_params = yaml.safe_load(f) or {}
                    elif config_file.endswith(".json"):
                        mpcc_params = json.load(f) or {}
                print(f"Loaded MPCC configuration from file: {config_file}")
            except Exception as e:
                print(f"Failed to parse config file {config_file}: {e}")
        elif os.path.exists(local_yaml_path):
            try:
                with open(local_yaml_path, "r") as f:
                    mpcc_params = yaml.safe_load(f) or {}
                print(f"Loaded optimized parameters from local file: {local_yaml_path}")
            except Exception as e:
                print(f"Failed to parse local mpcc_config.yaml: {e}")

        # Stage 2: Merge or override from direct `config.mpcc_tune` if it exists
        if mpcc_tune:
            if hasattr(mpcc_tune, "to_container"):  # Support OmegaConf structures
                mpcc_params.update(mpcc_tune.to_container(structured=True))
            elif isinstance(mpcc_tune, dict):
                mpcc_params.update(mpcc_tune)
            else:
                for key in ["Q_c", "Q_l", "R_u", "mu", "R_T", "Z_l", "z_l"]:
                    if hasattr(mpcc_tune, key):
                        mpcc_params[key] = getattr(mpcc_tune, key)

        # Stage 3: Map values into drone parameter lookup dictionary
        if mpcc_params:
            print("Applying structural MPCC controller parameter updates:")
            for k, v in mpcc_params.items():
                print(f"  {k} -> {v}")
                self.drone_params[k] = (
                    float(v) if isinstance(v, (int, float, str)) and not isinstance(v, bool) else v
                )

        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params, self._obstacle_manager
        )

        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
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
        self._last_u0 = np.array([0.0, 0.0, 0.0, self._last_thrust])  # Fallback on QP failure
        self._needs_warm_start_reset = False

        # --- Phase 4: asynchronous PMM replanning state (PMM planner only) ---
        # In level 2 each gate's true position is revealed (obs["gates_pos"] switches from
        # nominal to real) once the drone comes within sensor range. We replan the PMM whenever
        # that observed position changes, but OFF the control thread so the 50 Hz loop never
        # stalls on the ~150 ms plan. We remember the gate positions baked into the current plan
        # to detect such changes.
        self._replanner = AsyncPMMReplanner() if self.USE_PMM_PLANNER else None
        # Offline backbone: the pristine high-M global plan built above. Online replans patch only
        # the local window and splice THIS backbone's far-field back in as a committed suffix, so
        # the global racing line is never discarded. It is never replaced (downstream gates stay
        # nominal until their own reveal patches them). None for the legacy spline planner.
        self._backbone = self._trajectory if self.USE_PMM_PLANNER else None
        self._suffix_gap = 0.5  # [m] start the backbone suffix this far past the last window gate
        self._planned_gates_pos = gate_positions.copy()  # gate positions used by the live plan
        self._planned_target = 0  # target-gate index at the last replan
        # Offline-global / online-local split:
        #   * Initial plan (constructor above) uses ALL gates -> globally optimal racing line.
        #   * Online REPLANS are local: they re-solve only the next few gates, where newly revealed
        #     gate positions actually change the path. Far gates are still nominal (no new info) and
        #     get re-timed by the MPCC anyway, so re-solving through them is wasted work.
        # Paper (Sec. VI-B): N>=3 gives near-identical flight times to full-track planning. We use
        # 2 here per the current experiment; raise toward 3 if boundary myopia shows up.
        self._replan_horizon = 3  # gates ahead of the current target to replan through
        # Online replans use fewer velocity samples than the offline initial plan (25): with the
        # ~M^2 graph cost, 12 keeps a 2-gate replan at ~150-200 ms so it lands before going stale,
        # while the initial plan can afford more samples for a better global line.
        self._replan_vel_samples = 150
        self._replan_gate_move = 0.06  # [m] observed gate shift that triggers a replan
        # Robustness: commit the near-field on replans. A replan only changes the path BEYOND this
        # look-ahead distance on the current trajectory; the segment in between is kept identical
        # so adoption never jumps the MPCC's immediate reference. Smaller = more reactive to a
        # newly revealed gate; larger = smoother. Keep it below the sensor range (0.7 m).
        self._commit_distance = 0.4
        # Minimum approach room [m] left in front of the target gate when committing the near-field
        # on a replan. Must be large enough for the PMM to swing its velocity onto the gate's +x
        # crossing direction after a position reveal. Too small (e.g. 0.1) forces the planner to
        # re-thread a freshly revealed, laterally shifted gate within a fraction of a metre while
        # still carrying near-full speed -> with bounded acceleration it overshoots and curls back,
        # producing the visible loop right before the gate.
        self._gate_approach_margin = 0.4

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone."""
        if info is not None:
            gates_pos = None
            gates_yaw = None
            gates_rpys = None
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

            # Replan when a gate's observed position changes — i.e. its true position was just
            # revealed (obs["gates_pos"] switches nominal->real once within the 0.7 m sensor
            # range) or refined. Without replanning, the nominal path can sit up to 0.15 m off
            # the true center in Level 2, leaving only 0.05 m clearance — a near-certain crash.
            # The PMM planner replans OFF the control thread (Phase 4) so the 50 Hz loop never
            # stalls; the legacy spline planner keeps its original synchronous rebuild.
            if gates_pos is not None:
                if self.USE_PMM_PLANNER:
                    replan_rpys = gates_rpys if gates_yaw is not None else None
                    self._maybe_replan_pmm(obs, gates_pos, replan_rpys)
                elif "gates_visited" in obs:
                    self._legacy_rebuild(obs, gates_pos)

        # Define the terminal condition:
        # The environment sets target_gate to -1 exactly when the final gate plane is crossed.
        # Define the terminal condition and get the current target gate:
        target_gate_idx = int(obs.get("target_gate", 0))
        if target_gate_idx == -1:
            self._finished = True

        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"], [self._last_thrust]))

        theta0 = self._current_theta
        v_theta0 = self._current_v_theta

        x0_aug = np.concatenate((x0, np.array([theta0, v_theta0])))
        self._acados_ocp_solver.set(0, "lbx", x0_aug)
        self._acados_ocp_solver.set(0, "ubx", x0_aug)

        yref_target = np.zeros((self._ny,))
        yref_target[7] = self.drone_params["mass"] * 9.81  # Hover thrust
        yref_target[8] = (
            5.0  # Target progress speed (v_theta), matches state constraint upper bound
        )

        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref_target)

        yref_e_zero = np.zeros((self._ny_e,))
        self._acados_ocp_solver.set(self._N, "y_ref", yref_e_zero)

        obs_params = self._obstacle_manager.get_obstacle_parameters()
        num_obs = len(self._obstacle_manager.obstacles)
        total_params = 14 + (6 * num_obs)

        # Set spline parameter guess stage-by-stage using predicted theta.
        use_kinematic_guess = self._tick == 0 or self._needs_warm_start_reset
        for j in range(self._N):
            if use_kinematic_guess:
                # Provide a kinematic guess on first tick and after any trajectory rebuild.
                theta_pred = self._current_theta + j * self._dt * max(self._current_v_theta, 0.5)
                theta_pred = float(
                    np.clip(
                        theta_pred,
                        self._trajectory.knot_points[0],
                        self._trajectory.knot_points[-1],
                    )
                )

                xj_guess = x0_aug.copy()

                # 1. Advance position along the path
                xj_guess[0:3] = self._trajectory.evaluate(theta_pred)

                # 2. Point velocity along the path tangent
                t_ref = self._trajectory.evaluate_velocity(theta_pred)
                t_norm = t_ref / (np.linalg.norm(t_ref) + 1e-6)
                xj_guess[6:9] = t_norm * max(self._current_v_theta, 0.5)

                xj_guess[13] = theta_pred
                xj_guess[14] = max(self._current_v_theta, 0.5)

                hover_u = np.array([0.0, 0.0, 0.0, self.drone_params["mass"] * 9.81, 0.0])
                self._acados_ocp_solver.set(j, "x", xj_guess)
                self._acados_ocp_solver.set(j, "u", hover_u)
            else:
                xj_prev = self._acados_ocp_solver.get(j, "x")
                theta_pred = float(xj_prev[13])

            theta_pred = float(
                np.clip(
                    theta_pred, self._trajectory.knot_points[0], self._trajectory.knot_points[-1]
                )
            )

            px_j, py_j, pz_j, theta_offset_j = self._trajectory.get_polynomial_coeffs_at(theta_pred)
            pos_pred = self._trajectory.evaluate(theta_pred)
            q_c_j = self._obstacle_manager.dynamic_contour_weight(pos_pred, target_gate_idx)

            params_j = np.zeros((total_params,))
            params_j[0:4] = px_j.flatten()
            params_j[4:8] = py_j.flatten()
            params_j[8:12] = pz_j.flatten()
            params_j[12] = q_c_j
            params_j[13] = theta_offset_j
            params_j[14:] = obs_params

            self._acados_ocp_solver.set(j, "p", params_j)

            # Phase 4b: steer the progress speed v_theta toward the PMM's time-optimal speed
            # at this point on the path (fast on straights, slower into tight turns) instead of
            # the constant target above. Only the PMM planner exposes a speed profile; the
            # legacy spline keeps the constant target. Clip to the v_theta state bound (= 5 m/s).
            if self.USE_PMM_PLANNER:
                yref_j = yref_target.copy()
                yref_j[8] = float(
                    np.clip(self._trajectory.evaluate_speed(theta_pred), 0.5, yref_target[8])
                )
                self._acados_ocp_solver.set(j, "yref", yref_j)

        # Set parameters for the terminal node (N): extrapolate theta one more step
        xN_prev = self._acados_ocp_solver.get(self._N, "x")
        theta_N = float(
            np.clip(xN_prev[13], self._trajectory.knot_points[0], self._trajectory.knot_points[-1])
        )
        px_N, py_N, pz_N, theta_offset_N = self._trajectory.get_polynomial_coeffs_at(theta_N)
        pos_N = self._trajectory.evaluate(theta_N)
        q_c_N = self._obstacle_manager.dynamic_contour_weight(pos_N, target_gate_idx)
        params_N = np.zeros((total_params,))
        params_N[0:4] = px_N.flatten()
        params_N[4:8] = py_N.flatten()
        params_N[8:12] = pz_N.flatten()
        params_N[12] = q_c_N
        params_N[13] = theta_offset_N
        params_N[14:] = obs_params
        self._acados_ocp_solver.set(self._N, "p", params_N)

        # Solve and extract first control. We run the RTI solver to meet 50 Hz real-time.
        status = self._acados_ocp_solver.solve()
        self._needs_warm_start_reset = False

        if status != 0:
            # QP solver failed (e.g. status 3 = NaN, status 1 = max iter). Fall back to the
            # last known-good command and force a fresh warm start next step.
            self._needs_warm_start_reset = True
            return self._last_u0.copy()

        for j in range(self._N):
            xj = self._acados_ocp_solver.get(j, "x")
            self._predicted_trajectory[j] = xj[0:3]

        u0_aug = self._acados_ocp_solver.get(0, "u")
        x1_opt = self._acados_ocp_solver.get(1, "x")
        self._current_theta = float(x1_opt[13])
        self._current_v_theta = float(x1_opt[14])

        current_a_theta = float(u0_aug[4])

        u0 = u0_aug[0:4]
        self._last_thrust = float(u0[3])  # Save thrust command for next tick
        self._last_u0 = u0.copy()

        p_curr = obs["pos"]
        p_ref = self._trajectory.evaluate(self._current_theta)
        t_ref = self._trajectory.evaluate_velocity(self._current_theta)
        t_norm = t_ref / (np.linalg.norm(t_ref) + 1e-6)

        e_pos = p_curr - p_ref
        e_lag_val = np.dot(e_pos, t_norm)
        e_cont_vec = e_pos - e_lag_val * t_norm
        e_cont_val = np.linalg.norm(e_cont_vec)

        q_c_current = self._obstacle_manager.dynamic_contour_weight(p_ref, target_gate_idx)

        self._log_thrust.append(float(u0[3]))
        self._log_roll.append(float(u0[0]))
        self._log_pitch.append(float(u0[1]))
        self._log_contour.append(float(e_cont_val))
        self._log_lag.append(float(e_lag_val))
        self._log_v_theta.append(self._current_v_theta)
        self._log_a_theta.append(float(current_a_theta))
        self._log_q_c.append(float(q_c_current))

        return u0

    def _maybe_replan_pmm(
        self,
        obs: dict[str, NDArray[np.floating]],
        gates_pos: NDArray[np.floating],
        gates_rpys: NDArray[np.floating] | None,
    ) -> None:
        """Phase 4: off-thread PMM replanning driven by new gate data.

        Each tick this (1) swaps in a finished background plan if one is ready, and (2) starts a
        new background plan when a gate's observed position has changed (its true position was
        just revealed/refined) or the target gate advanced. Planning runs on a worker thread, so
        the 50 Hz control loop never blocks; the drone keeps flying on the current plan until the
        new one is ready. Meanwhile the obstacle manager is updated synchronously every tick, so
        collision avoidance already uses the true gate positions — only the reference centerline
        lags by the (~150 ms) planning time.

        Args:
            obs: Current observation (uses pos, vel, target_gate).
            gates_pos: (N, 3) currently observed gate positions (nominal until revealed).
            gates_rpys: (N, 3) observed gate orientations, or None to derive normals from geometry.
        """
        target = int(obs.get("target_gate", 0))

        # (1) Adopt a finished background plan — but only if it is NOT stale. Every plan is built
        # through gates[target:], i.e. starting at the gate the drone was flying toward when the
        # request was issued (recorded in self._planned_target). A full plan takes ~150 ms, during
        # which the drone may pass that gate (target advances). Swapping such a plan in would route
        # the reference backwards through an already-passed gate and make the drone loop back
        # through it. So we discard any plan whose starting gate has since been passed and keep
        # flying forward on the current plan; the trigger below immediately requests a fresh plan
        # from the new target (because target != self._planned_target). Passed gates still exist as
        # obstacles in the ObstacleManager — they simply stop being reference waypoints.
        new_planner = self._replanner.take()
        if new_planner is not None and target == self._planned_target:
            # Reject a reversing plan (belt-and-suspenders): if the new path's initial tangent
            # opposes the drone's current velocity, adopting it would yank the reference backward
            # (large mismatch -> crash). Only meaningful while moving. With near-field committing
            # (below) this practically never triggers, but it guards the non-committed cases.
            vel = np.array(obs["vel"], dtype=np.float64)
            speed = float(np.linalg.norm(vel))
            # Evaluate the new plan's heading at its START knot, NOT at the global-nearest point.
            # A plan can curve sharply near a freshly revealed gate; the global-nearest point may
            # then land on the far leg of that curve and report a backward tangent, wrongly flagging
            # a perfectly forward plan as "reversing" and trapping the drone on the stale, looping
            # plan. The start knot is where the committed near-field begins, so its tangent always
            # reflects the drone's actual direction of travel.
            theta0 = float(new_planner.knot_points[0])
            tang = np.asarray(new_planner.evaluate_velocity(theta0), dtype=np.float64)
            tang_n = tang / (np.linalg.norm(tang) + 1e-9)
            reversing = speed > 0.3 and float(np.dot(tang_n, vel / speed)) < 0.0
            if not reversing:
                self._trajectory = new_planner
                self._reanchor_progress(obs)
                self._needs_warm_start_reset = True  # previous warm start was for the old path
                logger.info(
                    "REPLAN adopted: target_gate=%d, path_len=%.2f m",
                    target,
                    self._trajectory.total_length,
                )
            else:
                # discard the reversing plan and keep flying the current (forward) plan
                logger.warning("REPLAN discarded: reversing plan (target_gate=%d)", target)
        elif new_planner is not None:
            # A finished plan exists but its starting gate was passed during the planning latency
            # -> stale. Discard; the trigger below immediately requests a fresh plan from the new
            # target (target != self._planned_target).
            logger.warning(
                "REPLAN discarded: stale (built for gate %d, current target %d)",
                self._planned_target,
                target,
            )

        if target < 0:
            return  # final gate passed; nothing left to plan

        # (2) Replan trigger: a gate within the planning window moved beyond the threshold (a
        # nominal->real reveal is a large jump), or we passed a gate (target advanced), which
        # recedes the horizon forward.
        window = slice(target, min(target + self._replan_horizon, len(gates_pos)))
        moved = 0.0
        if window.stop > window.start:
            moved = float(
                np.max(np.linalg.norm(gates_pos[window] - self._planned_gates_pos[window], axis=1))
            )
        if not (target != self._planned_target or moved > self._replan_gate_move):
            return
        if self._replanner.busy():
            return  # a replan is already running; this trigger is re-checked next tick

        reason = "target_advance" if target != self._planned_target else f"gate_moved={moved:.3f}m"
        logger.info(
            "REPLAN trigger @tick=%d: reason=%s, target_gate=%d, window=[%d:%d] (%d gate[s])",
            self._tick,
            reason,
            target,
            window.start,
            window.stop,
            window.stop - window.start,
        )

        # Commit the near-field: the replan changes the path only BEYOND a short look-ahead on
        # the CURRENT trajectory. We start the new plan at the commit point (commit_distance ahead
        # of the drone's current progress) and prepend the segment in between, so the MPCC's
        # immediate reference is identical before and after adoption -> no jump. The commit point
        # is also kept short of the current target gate, otherwise the far plan could route back
        # to it. If there is no meaningful near-field to commit (start of run, near the path end,
        # or already past the target on this path), we fall back to planning from the drone.
        knots = self._trajectory.knot_points
        theta_now = float(np.clip(self._current_theta, knots[0], knots[-1]))
        theta_commit = min(theta_now + self._commit_distance, self._trajectory.total_length)
        theta_target = float(self._trajectory.nearest_theta(gates_pos[target]))
        if theta_target > theta_now:
            # Stop committing well BEFORE the target gate, not right up against it. Leaving only
            # ~0.1 m of approach forced the PMM to overshoot the freshly revealed gate and loop
            # back to satisfy the +x crossing; a larger margin gives it room to align. If the
            # drone is already inside this margin, theta_commit drops below theta_now and the
            # branch below falls back to planning straight from the drone's real state.
            theta_commit = min(theta_commit, theta_target - self._gate_approach_margin)

        committed_pts = None
        committed_speeds = None
        if theta_commit - theta_now > 0.05:
            s_pre = np.linspace(theta_now, theta_commit, 12)
            committed_pts = np.asarray(self._trajectory.evaluate(s_pre), dtype=np.float64)
            committed_speeds = np.array(
                [float(self._trajectory.evaluate_speed(float(s))) for s in s_pre], dtype=np.float64
            )
            start_pos = committed_pts[-1].copy()  # new plan starts at the commit point
            tang = np.asarray(self._trajectory.evaluate_velocity(theta_commit), dtype=np.float64)
            tang_n = tang / (np.linalg.norm(tang) + 1e-9)
            start_vel = tang_n * float(self._trajectory.evaluate_speed(theta_commit))
        else:
            start_pos = np.array(obs["pos"], dtype=np.float64)
            start_vel = np.array(obs["vel"], dtype=np.float64)

        # Far-field suffix (Part 2): reuse the offline backbone beyond the replan window so the
        # high-M global racing line is preserved instead of being re-solved at low M every replan.
        # Anchored just past the last window gate; covers all downstream gates (still nominal until
        # their own reveal) out to the backbone's end. Read here on the control thread (the backbone
        # is never mutated) so the worker only sees frozen arrays.
        committed_suffix_pts = None
        committed_suffix_speeds = None
        if self._backbone is not None and window.stop < len(gates_pos):
            bb = self._backbone
            theta_we = float(bb.nearest_theta(gates_pos[window.stop - 1]))
            theta_suffix = min(theta_we + self._suffix_gap, bb.total_length)
            if bb.total_length - theta_suffix > 0.05:
                n_suf = int(np.clip((bb.total_length - theta_suffix) / 0.05, 10, 300))
                s_suf = np.linspace(theta_suffix, bb.total_length, n_suf)
                committed_suffix_pts = np.asarray(bb.evaluate(s_suf), dtype=np.float64)
                committed_suffix_speeds = np.asarray(bb.evaluate_speed(s_suf), dtype=np.float64)

        # Capture snapshots on this (control) thread so the worker reads no shared mutable state.
        horizon_gates = np.array(gates_pos[window], dtype=np.float64)
        horizon_rpys = (
            np.array(gates_rpys[window], dtype=np.float64) if gates_rpys is not None else None
        )
        # Exclude only the gates THIS replan routes through (the window); other gates + poles still
        # block, so the local plan can't clip a gate it isn't crossing.
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

        # Record what the in-flight plan is based on so the same data does not re-trigger it.
        self._planned_gates_pos = np.array(gates_pos, dtype=np.float64)
        self._planned_target = target

    def _reanchor_progress(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Re-fit the progress state (theta, v_theta) to the current trajectory after a swap.

        Uses a localized search near the start of the trajectory to prevent the reference
        from snapping to future track segments if the path crosses over itself.
        """
        knot_start = self._trajectory.knot_points[0]
        knot_end = self._trajectory.knot_points[-1]

        # Search only the first 3 meters of the new path to avoid crossover traps
        search_limit = min(knot_start + 3.0, knot_end)
        search_thetas = np.linspace(knot_start, search_limit, 60)

        pts = self._trajectory.evaluate(search_thetas)
        dists = np.linalg.norm(pts - obs["pos"], axis=1)
        best_idx = np.argmin(dists)

        self._current_theta = float(search_thetas[best_idx])

        t_new = self._trajectory.evaluate_velocity(self._current_theta)
        t_norm = t_new / (np.linalg.norm(t_new) + 1e-6)
        v_proj = float(np.dot(np.array(obs["vel"], dtype=np.float64), t_norm))
        self._current_v_theta = max(0.01, v_proj)

    def _legacy_rebuild(
        self, obs: dict[str, NDArray[np.floating]], gates_pos: NDArray[np.floating]
    ) -> None:
        """Synchronous rebuild for the non-PMM spline planner (original behavior).

        Rebuilds the reference path through the revealed gate centers the first time any gate
        enters sensor range, then re-anchors the progress state. Used only when
        USE_PMM_PLANNER is False.
        """
        gates_visited_now = np.array(obs["gates_visited"], dtype=bool)
        if not np.any(gates_visited_now & ~self._gates_visited_flags):
            return
        # Only gates from the current target onwards, to avoid routing the path backwards
        # through a gate we just passed.
        target_gate_idx = int(obs.get("target_gate", 0))
        if 0 <= target_gate_idx < len(gates_pos):
            remaining_gates = gates_pos[target_gate_idx:]
        else:
            remaining_gates = gates_pos[-1:]
        # No approach waypoints mid-flight: the drone is already close, so a 0.3 m offset point
        # could land behind it and kink the path.
        self._trajectory.rebuild(obs["pos"], remaining_gates, gate_rpys=None)
        self._reanchor_progress(obs)
        self._gates_visited_flags = gates_visited_now.copy()
        self._needs_warm_start_reset = True

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
        plotting = True
        if plotting is True:
            hover_thrust = self.drone_params["mass"] * 9.81
            max_thrust = self.drone_params["thrust_max"] * 4
            min_thrust = self.drone_params["thrust_min"] * 4

            fig, axs = plt.subplots(6, 1, figsize=(10, 18), sharex=True)

            axs[0].plot(self._log_contour, label="Contour Error (e_c)")
            axs[0].plot(self._log_lag, label="Lag Error (e_l)")
            axs[0].set_ylabel("Error [m]")
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(self._log_roll, label="Roll Command")
            axs[1].plot(self._log_pitch, label="Pitch Command")
            axs[1].axhline(0.5, color="r", linestyle="--", label="Upper Limit")
            axs[1].axhline(-0.5, color="r", linestyle="--", label="Lower Limit")
            axs[1].set_ylabel("Angle [rad]")
            axs[1].legend()
            axs[1].grid(True)

            axs[2].plot(self._log_thrust, label="Thrust Command")
            axs[2].axhline(hover_thrust, color="g", linestyle=":", label="Hover")
            axs[2].axhline(max_thrust, color="r", linestyle="--", label="Max Thrust")
            axs[2].axhline(min_thrust, color="r", linestyle="--", label="Min Thrust")
            axs[2].set_ylabel("Thrust [N]")
            axs[2].legend()
            axs[2].grid(True)

            axs[3].plot(self._log_v_theta, label="Virtual Speed (v_theta)")
            axs[3].axhline(7.5, color="g", linestyle="--", label="Target Speed")
            axs[3].set_ylabel("Speed [m/s]")
            axs[3].legend()
            axs[3].grid(True)

            axs[4].plot(self._log_q_c, label="Contour Weight (q_c)", color="purple")
            axs[4].set_ylabel("Weight")
            axs[4].set_xlabel("Timestep")
            axs[4].legend()
            axs[4].grid(True)

            axs[5].plot(self._log_a_theta, label="Virtual Accel (a_theta)", color="orange")
            axs[5].axhline(5.0, color="r", linestyle="--", label="Upper Limit")
            axs[5].axhline(0.01, color="r", linestyle="--", label="Lower Limit")
            axs[5].set_ylabel("Accel [m/s^2]")
            axs[5].set_xlabel("Timestep")
            axs[5].legend()
            axs[5].grid(True)

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
            self._log_a_theta.clear()
            self._log_q_c.clear()
            self._tick = 0
