"""Obstacle manager for drone racing."""

import casadi as ca
import numpy as np
from crazyflow.sim import Sim
from crazyflow.sim.visualize import draw_capsule, draw_points


class ObstacleManager:
    """Manager for drone obstacles."""

    def __init__(self, safety_margin: float = 0.08):
        """Initialize the obstacle manager.

        Args:
            safety_margin: Extra buffer distance (in meters) around obstacles.
        """
        self.safety_margin = safety_margin
        # Each entry: {"type": "sphere/cylinder", "p1": np.array, "p2": np.array, "r": float}
        self.obstacles = []
        self.gates: list[dict[str, object]] = []
        self._gate_obstacle_indices: list[list[int]] = []
        self._nominal_obstacle_indices: list[int] = []

    def add_sphere(self, center: np.ndarray, radius: float):
        """Add a spherical obstacle."""
        # Force the array to be float64 even if integers are passed
        p = np.array(center, dtype=np.float64)
        self.obstacles.append(
            {
                "type": "sphere",
                "p1": p,
                "p2": p.copy(),  # p1=p2 for a sphere capsule
                "r": radius,
            }
        )

    def add_cylinder(self, start: np.ndarray, end: np.ndarray, radius: float):
        """Add a cylinder (capsule) at any orientation."""
        # Force start and end to be float64
        self.obstacles.append(
            {
                "type": "cylinder",
                "p1": np.array(start, dtype=np.float64),
                "p2": np.array(end, dtype=np.float64),
                "r": radius,
            }
        )

    def add_gate(
        self,
        pos: list | np.ndarray,
        rpy: list | np.ndarray,
        inner_width: float = 0.4,
        outer_width: float = 0.72,
    ) -> None:
        """Models the solid banner of a gate as 4 cylinders.

        Args:
            pos: [x, y, z] center of the gate.
            rpy: [roll, pitch, yaw] in radians.
            inner_width: Width/height of the opening (default 0.4m).
            outer_width: Outer width/height of the frame (default 0.72m).
        """
        center = np.array(pos, dtype=np.float64)
        yaw = rpy[2]

        # Calculate banner midpoint (where the cylinder centerline goes)
        # Offset = (0.2 + 0.36) / 2 = 0.28m
        banner_offset = (inner_width / 4.0) + (outer_width / 4.0)
        # Thickness covers the banner: (0.36 - 0.2) / 2 = 0.08m
        thickness = (outer_width - inner_width) / 4.0

        # Define 4 corners at the banner's centerline
        local_corners = [
            np.array([0, -banner_offset, banner_offset]),  # Top-Left
            np.array([0, banner_offset, banner_offset]),  # Top-Right
            np.array([0, banner_offset, -banner_offset]),  # Bottom-Right
            np.array([0, -banner_offset, -banner_offset]),  # Bottom-Left
        ]

        # Rotation Matrix (Yaw)
        R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

        world_corners = [(R @ p) + center for p in local_corners]

        start_idx = len(self.obstacles)
        self.obstacles.extend(
            [
                {
                    "type": "cylinder",
                    "p1": world_corners[0],
                    "p2": world_corners[1],
                    "r": thickness,
                },
                {
                    "type": "cylinder",
                    "p1": world_corners[1],
                    "p2": world_corners[2],
                    "r": thickness,
                },
                {
                    "type": "cylinder",
                    "p1": world_corners[2],
                    "p2": world_corners[3],
                    "r": thickness,
                },
                {
                    "type": "cylinder",
                    "p1": world_corners[3],
                    "p2": world_corners[0],
                    "r": thickness,
                },
            ]
        )
        self._gate_obstacle_indices.append(list(range(start_idx, start_idx + 4)))
        self.gates.append(
            {
                "pos": center,
                "rpy": np.array(rpy, dtype=np.float64),
                "inner_width": inner_width,
                "outer_width": outer_width,
            }
        )

    def update_gate_positions(
        self, gates_pos: np.ndarray, gates_rpy: np.ndarray | None = None
    ) -> None:
        """Update the obstacle representation of gates without altering trajectory."""
        if len(self.gates) == 0:
            return

        gates_pos_arr = np.asarray(gates_pos, dtype=np.float64)
        gates_rpy_arr = None
        if gates_rpy is not None:
            gates_rpy_arr = np.asarray(gates_rpy, dtype=np.float64)

        for gate_idx, gate in enumerate(self.gates):
            if gate_idx >= gates_pos_arr.shape[0]:
                break

            new_pos = gates_pos_arr[gate_idx]
            new_rpy = gates_rpy_arr[gate_idx] if gates_rpy_arr is not None else gate["rpy"]

            gate["pos"] = np.array(new_pos, dtype=np.float64)
            gate["rpy"] = np.array(new_rpy, dtype=np.float64)

            yaw = float(gate["rpy"][2])
            banner_offset = (gate["inner_width"] / 4.0) + (gate["outer_width"] / 4.0)
            thickness = (gate["outer_width"] - gate["inner_width"]) / 4.0

            local_corners = [
                np.array([0, -banner_offset, banner_offset]),
                np.array([0, banner_offset, banner_offset]),
                np.array([0, banner_offset, -banner_offset]),
                np.array([0, -banner_offset, -banner_offset]),
            ]
            R_mat = np.array(
                [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
            )
            world_corners = [(R_mat @ p) + gate["pos"] for p in local_corners]

            new_obstacles = [
                {
                    "type": "cylinder",
                    "p1": world_corners[0],
                    "p2": world_corners[1],
                    "r": thickness,
                },
                {
                    "type": "cylinder",
                    "p1": world_corners[1],
                    "p2": world_corners[2],
                    "r": thickness,
                },
                {
                    "type": "cylinder",
                    "p1": world_corners[2],
                    "p2": world_corners[3],
                    "r": thickness,
                },
                {
                    "type": "cylinder",
                    "p1": world_corners[3],
                    "p2": world_corners[0],
                    "r": thickness,
                },
            ]

            for obs_idx, obstacle in zip(self._gate_obstacle_indices[gate_idx], new_obstacles):
                self.obstacles[obs_idx] = obstacle

    def update_obstacle_positions(self, obstacles_pos: np.ndarray) -> None:
        """Update the nominal cylindrical obstacle positions without altering trajectory."""
        if len(self._nominal_obstacle_indices) == 0:
            return

        obstacles_pos_arr = np.asarray(obstacles_pos, dtype=np.float64)
        for i, obs_idx in enumerate(self._nominal_obstacle_indices):
            if i >= obstacles_pos_arr.shape[0]:
                break

            pos = obstacles_pos_arr[i]
            if self.obstacles[obs_idx]["type"] != "cylinder":
                continue

            start = self.obstacles[obs_idx]["p1"]
            end = self.obstacles[obs_idx]["p2"]
            height = np.linalg.norm(end - start)
            self.obstacles[obs_idx]["p1"] = np.array([pos[0], pos[1], 0.0], dtype=np.float64)
            self.obstacles[obs_idx]["p2"] = np.array([pos[0], pos[1], height], dtype=np.float64)

    def points_in_obstacles(self, points: np.ndarray, margin: float | None = None) -> np.ndarray:
        """Return a boolean mask of points that intersect obstacles inflated by margin."""
        points_arr = np.asarray(points, dtype=np.float64)
        mask = np.zeros(points_arr.shape[0], dtype=bool)
        margin = self.safety_margin if margin is None else float(margin)

        for idx, point in enumerate(points_arr):
            for obs in self.obstacles:
                r_total = float(obs["r"]) + margin
                if obs["type"] == "sphere":
                    if np.linalg.norm(point - obs["p1"]) <= r_total:
                        mask[idx] = True
                        break

                else:
                    v = obs["p2"] - obs["p1"]
                    w = point - obs["p1"]
                    v_norm_sq = np.dot(v, v)
                    if v_norm_sq == 0.0:
                        closest = obs["p1"]
                    else:
                        t = np.dot(w, v) / v_norm_sq
                        t = np.clip(t, 0.0, 1.0)
                        closest = obs["p1"] + t * v

                    if np.linalg.norm(point - closest) <= r_total:
                        mask[idx] = True
                        break

        return mask

    def initialize_nominal_track(self) -> None:
        """Hardcodes the nominal positions of gates and obstacles for Level 0.

        Args:
            manager: The ObstacleManager instance to populate.
        """
        # reset any existing obstacles and gate metadata
        self.obstacles = []
        self.gates = []
        self._gate_obstacle_indices = []
        self._nominal_obstacle_indices = []

        # --- 1. GATES ---
        # Standard dimensions: outer_width = 0.72m
        # Heights provided are from ground to the center of the gate.
        gates_data = [
            {"pos": [0.5, 0.25, 0.7], "rpy": [0.0, 0.0, -0.78]},
            {"pos": [1.05, 0.75, 1.2], "rpy": [0.0, 0.0, 2.35]},
            {"pos": [-1.0, -0.25, 0.7], "rpy": [0.0, 0.0, 3.14]},
            {"pos": [0.0, -0.75, 1.2], "rpy": [0.0, 0.0, 0.0]},
        ]

        for gate in gates_data:
            self.add_gate(pos=gate["pos"], rpy=gate["rpy"], inner_width=0.4, outer_width=0.72)

        # --- 2. OBSTACLES (Vertical Poles) ---
        # From TOML: diameter = 0.03m (radius = 0.015m)
        # Height is ground (z=0) to the top (z=1.55)
        poles_data = [
            [0.0, 0.75, 1.55],
            [1.0, 0.25, 1.55],
            [-1.5, -0.25, 1.55],
            [-0.5, -0.75, 1.55],
        ]

        for pole_pos in poles_data:
            # We define a cylinder from ground to top height
            start_point = np.array([pole_pos[0], pole_pos[1], 0.0], dtype=np.float64)
            end_point = np.array([pole_pos[0], pole_pos[1], pole_pos[2]], dtype=np.float64)

            obs_idx = len(self.obstacles)
            self.add_cylinder(
                start=start_point,
                end=end_point,
                radius=0.015,  # Half of 0.03m diameter
            )
            self._nominal_obstacle_indices.append(obs_idx)

    def get_obstacle_parameters(self) -> np.ndarray:
        """Flattens current obstacle coordinates into a 1D array for the solver."""
        params = []
        for obs in self.obstacles:
            params.extend(obs["p1"])
            params.extend(obs["p2"])
        return np.array(params, dtype=np.float64)

    def get_collision_expressions(self, x_sym: ca.MX, p_sym: ca.MX) -> ca.MX:
        """Generates symbolic CasADi expressions using dynamic parameters."""
        constraints = []
        drone_pos = x_sym[0:3]

        for i, obs in enumerate(self.obstacles):
            # Extract dynamic p1 and p2 from the parameter vector
            # Each obstacle takes up 6 slots (3 for p1, 3 for p2)
            idx = i * 6
            p1 = p_sym[idx : idx + 3]
            p2 = p_sym[idx + 3 : idx + 6]

            r_total = obs["r"] + self.safety_margin

            if obs["type"] == "sphere":
                dist_sq = ca.sumsqr(drone_pos - p1)
                constraints.append(dist_sq - r_total**2)

            else:  # Cylinder / Capsule
                v = p2 - p1
                w = drone_pos - p1

                t = ca.dot(w, v) / (ca.sumsqr(v) + 1e-9)
                t_clamped = ca.fmax(0, ca.fmin(1, t))

                closest_point = p1 + t_clamped * v
                dist_sq = ca.sumsqr(drone_pos - closest_point)
                constraints.append(dist_sq - r_total**2)

        return ca.vcat(constraints)

    def render(
        self, sim: Sim, rgba: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.3)
    ) -> None:
        """Draws all obstacles in the simulation."""
        for obs in self.obstacles:
            if obs["type"] == "sphere":
                # draw_points expects an [N, 3] array
                point = obs["p1"].reshape(1, 3)
                # size is diameter, so we use 2 * radius
                draw_points(sim, points=point, rgba=np.array(rgba), size=obs["r"] * 2.0)
            else:
                draw_capsule(sim, p1=obs["p1"], p2=obs["p2"], radius=obs["r"], rgba=rgba)
