"""Obstacle handling and collision detection for MPC.

This module provides gate modeling as capsule obstacles, collision detection,
dynamic contour weighting, and optional hard constraint expressions for the
attitude MPC controller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np

if TYPE_CHECKING:
    from crazyflow.sim import Sim


class ObstacleManager:
    """Manages obstacles, gates, collision detection, and MPC cost shaping."""

    def __init__(self, safety_margin: float = 0.08) -> None:
        """Initialize the obstacle manager.

        Args:
            safety_margin: Extra buffer distance (in meters) around obstacles.
        """
        self.safety_margin = safety_margin
        self.obstacles = []
        self.gates = []
        self._gate_obstacle_indices = []
        self._pole_obstacle_indices = []
        self._q_nom = 1.0
        self._q_wp = 150.0
        self._sigma_sq = 0.25**2

    def add_sphere(self, center: np.ndarray, radius: float) -> None:
        """Add a spherical obstacle.

        Args:
            center: Center position [x, y, z].
            radius: Sphere radius in meters.
        """
        p = np.array(center, dtype=np.float64)
        self.obstacles.append({"type": "sphere", "p1": p, "p2": p.copy(), "r": radius})

    def add_cylinder(self, start: np.ndarray, end: np.ndarray, radius: float) -> None:
        """Add a cylindrical obstacle.

        Args:
            start: Start point of cylinder axis [x, y, z].
            end: End point of cylinder axis [x, y, z].
            radius: Cylinder radius in meters.
        """
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
        """Model a gate as 4 capsule obstacles representing the solid banner frame.

        Args:
            pos: [x, y, z] center position of the gate.
            rpy: [roll, pitch, yaw] orientation in radians.
            inner_width: Width/height of the opening (default 0.4m).
            outer_width: Outer width/height of the frame (default 0.72m).
        """
        center = np.array(pos, dtype=np.float64)
        yaw = rpy[2]

        banner_offset = (inner_width / 4.0) + (outer_width / 4.0)
        thickness = (outer_width - inner_width) / 4.0

        local_corners = [
            np.array([0, -banner_offset, banner_offset]),
            np.array([0, banner_offset, banner_offset]),
            np.array([0, banner_offset, -banner_offset]),
            np.array([0, -banner_offset, -banner_offset]),
        ]

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

    def add_pole(self, pos: list | np.ndarray, height: float = 1.55, radius: float = 0.015) -> None:
        """Add a pole (vertical cylindrical obstacle).

        Args:
            pos: [x, y, z_top] center position (z_top is top of pole, from ground to marker).
            height: Total height from ground to top (default 1.55m).
            radius: Pole radius in meters (default 0.015m = 0.03m diameter).
        """
        # Accept several input formats for convenience: list/ndarray or dict-like
        if isinstance(pos, dict):
            if "pos" in pos:
                pos_val = pos["pos"]
            elif "position" in pos:
                pos_val = pos["position"]
            elif all(k in pos for k in ("x", "y", "z")):
                pos_val = [pos["x"], pos["y"], pos["z"]]
            else:
                raise TypeError(
                    "Unsupported dict format for pole position; "
                    "expected keys 'pos', 'position' or 'x','y','z'"
                )
        else:
            pos_val = pos

        pos_arr = np.asarray(pos_val, dtype=np.float64)
        if pos_arr.size < 3:
            raise ValueError("Pole position must be length 3: [x, y, z_top]")

        z_top = pos_arr[2]
        z_bottom = z_top - height

        start_point = np.array([pos_arr[0], pos_arr[1], z_bottom], dtype=np.float64)
        end_point = np.array([pos_arr[0], pos_arr[1], z_top], dtype=np.float64)

        pole_idx = len(self.obstacles)
        self.add_cylinder(start_point, end_point, radius)
        self._pole_obstacle_indices.append(pole_idx)

    def update_gate_positions(self, gate_positions: np.ndarray, gate_rpys: np.ndarray) -> None:
        """Update gate positions without re-planning spline.

        Args:
            gate_positions: (N, 3) array of new gate positions [x, y, z].
            gate_rpys: (N, 3) array of new gate orientations [roll, pitch, yaw].
        """
        gate_positions_arr = np.asarray(gate_positions, dtype=np.float64)
        gate_rpys_arr = np.asarray(gate_rpys, dtype=np.float64)

        for gate_idx, gate in enumerate(self.gates):
            if gate_idx >= gate_positions_arr.shape[0]:
                break

            new_pos = gate_positions_arr[gate_idx]
            new_rpy = gate_rpys_arr[gate_idx]
            inner_width = gate["inner_width"]
            outer_width = gate["outer_width"]

            center = np.array(new_pos, dtype=np.float64)
            yaw = new_rpy[2]

            banner_offset = (inner_width / 4.0) + (outer_width / 4.0)
            thickness = (outer_width - inner_width) / 4.0

            local_corners = [
                np.array([0, -banner_offset, banner_offset]),
                np.array([0, banner_offset, banner_offset]),
                np.array([0, banner_offset, -banner_offset]),
                np.array([0, -banner_offset, -banner_offset]),
            ]

            R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            world_corners = [(R @ p) + center for p in local_corners]

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

            gate["pos"] = center
            gate["rpy"] = np.array(new_rpy, dtype=np.float64)

    def update_pole_positions(self, pole_positions: np.ndarray, height: float = 1.55) -> None:
        """Update pole positions without re-planning spline.

        Args:
            pole_positions: (N, 3) array of new pole positions [x, y, z_top].
            height: Pole height from ground to top (default 1.55m).
        """
        pole_positions_arr = np.asarray(pole_positions, dtype=np.float64)

        for i, pole_idx in enumerate(self._pole_obstacle_indices):
            if i >= pole_positions_arr.shape[0]:
                break

            pos = pole_positions_arr[i]
            z_top = pos[2]
            z_bottom = z_top - height

            start_point = np.array([pos[0], pos[1], z_bottom], dtype=np.float64)
            end_point = np.array([pos[0], pos[1], z_top], dtype=np.float64)

            self.obstacles[pole_idx]["p1"] = start_point
            self.obstacles[pole_idx]["p2"] = end_point

    def snapshot(self) -> ObstacleManager:
        """Return a frozen, independent copy for thread-safe collision queries.

        The control thread updates obstacle positions every tick (update_gate_positions /
        update_pole_positions), while the background PMM replanner reads them through
        points_in_obstacles. Handing the planner this copy lets it see a consistent set of
        positions while it runs off the control thread. Only the geometry needed by
        points_in_obstacles is copied (type, endpoints, radius, margin).
        """
        snap = ObstacleManager(safety_margin=self.safety_margin)
        snap.obstacles = [
            {"type": o["type"], "p1": o["p1"].copy(), "p2": o["p2"].copy(), "r": float(o["r"])}
            for o in self.obstacles
        ]
        return snap

    def points_in_obstacles(self, points: np.ndarray, margin: float | None = None) -> np.ndarray:
        """Return boolean mask of points intersecting obstacles (with margin).

        Args:
            points: (N, 3) array of query points.
            margin: Collision margin. If None, uses self.safety_margin.

        Returns:
            Boolean array of shape (N,), True if point intersects an obstacle.
        """
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

    def distance_to_obstacles(self, position: np.ndarray) -> float:
        """Compute minimum distance from position to any obstacle surface.

        Args:
            position: Query position [x, y, z].

        Returns:
            Minimum distance (can be negative if inside obstacle).
        """
        position = np.asarray(position, dtype=np.float64)
        min_dist = float("inf")

        for obs in self.obstacles:
            if obs["type"] == "sphere":
                dist = np.linalg.norm(position - obs["p1"]) - obs["r"]
            else:
                v = obs["p2"] - obs["p1"]
                w = position - obs["p1"]
                v_norm_sq = np.dot(v, v)
                if v_norm_sq == 0.0:
                    closest = obs["p1"]
                else:
                    t = np.dot(w, v) / v_norm_sq
                    t = np.clip(t, 0.0, 1.0)
                    closest = obs["p1"] + t * v
                dist = np.linalg.norm(position - closest) - obs["r"]

            min_dist = min(min_dist, dist)

        return float(min_dist if min_dist != float("inf") else 0.0)

    def dynamic_contour_weight(self, position: np.ndarray) -> float:
        """Compute dynamic contour weight based on gate proximity.

        Near gates, increases the weighting of contour error in the MPCC cost.
        This provides soft obstacle avoidance without hard constraints.

        Args:
            position: Current drone position [x, y, z].

        Returns:
            Contour weight q_c for the cost function.
        """
        q_c = self._q_nom
        for gate in self.gates:
            dist_sq = np.sum((position - gate["pos"]) ** 2)
            q_c += self._q_wp * np.exp(-0.5 * dist_sq / self._sigma_sq)
        return float(q_c)

    def get_obstacle_parameters(self) -> np.ndarray:
        """Flatten current obstacle coordinates into 1D array for solver.

        Returns:
            Flat array: [p1_x, p1_y, p1_z, p2_x, p2_y, p2_z, ...] for all obstacles.
        """
        params = []
        for obs in self.obstacles:
            params.extend(obs["p1"])
            params.extend(obs["p2"])
        return np.array(params, dtype=np.float64)

    def get_collision_expressions(self, x_sym: ca.MX, p_sym: ca.MX) -> ca.MX:
        """Generate CasADi collision constraint expressions.

        Expressions: distance² - (radius + margin)² ≥ 0 (drone outside obstacle).
        Used for optional collision constraints in future implementations.

        Args:
            x_sym: State vector (x[0:3] is drone position).
            p_sym: Parameter vector with obstacle coordinates.

        Returns:
            Constraint expressions as ca.MX column vector.
        """
        constraints = []
        drone_pos = x_sym[0:3]

        for i, obs in enumerate(self.obstacles):
            idx = i * 6
            p1 = p_sym[idx : idx + 3]
            p2 = p_sym[idx + 3 : idx + 6]

            r_total = obs["r"] + self.safety_margin

            if obs["type"] == "sphere":
                dist_sq = ca.sumsqr(drone_pos - p1)
                constraints.append(dist_sq - r_total**2)
            else:
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
        """Draw all obstacles in the simulation.

        Args:
            sim: Crazyflow simulator instance.
            rgba: Color as (red, green, blue, alpha) with values in [0, 1].
        """
        from crazyflow.sim.visualize import draw_capsule, draw_points

        for obs in self.obstacles:
            if obs["type"] == "sphere":
                point = obs["p1"].reshape(1, 3)
                draw_points(sim, points=point, rgba=np.array(rgba), size=obs["r"] * 2.0)
            else:
                draw_capsule(sim, p1=obs["p1"], p2=obs["p2"], radius=obs["r"], rgba=rgba)


class CollisionConstraintBuilder:
    """Factory for optional hard collision constraint expressions.

    Used for future hard constraint integration. Currently kept for extensibility.
    """

    @staticmethod
    def build_hard_constraints(
        x_sym: ca.MX, p_sym: ca.MX, obstacles: list[dict], safety_margin: float
    ) -> ca.MX:
        """Build CasADi constraint expressions for hard collision avoidance.

        Args:
            x_sym: State vector (x[0:3] is drone position).
            p_sym: Parameter vector with obstacle coordinates.
            obstacles: List of obstacle dictionaries.
            safety_margin: Extra margin around obstacles.

        Returns:
            Column vector of constraint expressions: dist² - r_total² ≥ 0.
        """
        constraints = []
        drone_pos = x_sym[0:3]

        for i, obs in enumerate(obstacles):
            idx = i * 6
            p1 = p_sym[idx : idx + 3]
            p2 = p_sym[idx + 3 : idx + 6]

            r_total = obs["r"] + safety_margin

            if obs["type"] == "sphere":
                dist_sq = ca.sumsqr(drone_pos - p1)
                constraints.append(dist_sq - r_total**2)
            else:
                v = p2 - p1
                w = drone_pos - p1
                t = ca.dot(w, v) / (ca.sumsqr(v) + 1e-9)
                t_clamped = ca.fmax(0, ca.fmin(1, t))
                closest_point = p1 + t_clamped * v
                dist_sq = ca.sumsqr(drone_pos - closest_point)
                constraints.append(dist_sq - r_total**2)

        return ca.vcat(constraints)
