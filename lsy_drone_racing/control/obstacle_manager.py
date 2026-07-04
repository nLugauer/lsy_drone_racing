"""Obstacle handling and collision detection for the attitude MPC.

Models gates as capsule obstacles, exposes collision queries for the PMM planner, dynamic
contour weighting for the MPCC cost, and CasADi hard-constraint expressions. All obstacles
are capsules (cylinders)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np

if TYPE_CHECKING:
    from crazyflow.sim import Sim


class ObstacleManager:
    """Manages obstacles, gates, collision detection, and MPCC cost shaping."""

    def __init__(self, safety_margin: float = 0.08) -> None:
        """Initialize the obstacle manager."""
        self.safety_margin = safety_margin
        self.obstacles: list[dict] = []
        self.gates: list[dict] = []
        self._gate_obstacle_indices = []  # 4 frame-capsule indices per gate (snapshot-excludable)
        self._gate_lower_indices = []  # lower-frame cylinder index per gate (never excluded)
        self._gate_openings = []  # per-gate opening corridor: {center, axis, radius, half_depth}
        self._pole_obstacle_indices = []
        self._q_nom = 1.0
        self._q_wp = 10
        self._sigma_sq = 0.35**2

        self._opening_margin = 0.05  # [m] shrink corridor radius in from the clear half-width
        self._opening_half_depth = 0.25  # [m] half-length of the corridor along the gate normal

    def add_cylinder(self, start: np.ndarray, end: np.ndarray, radius: float) -> None:
        """Add a capsule obstacle with the given endpoints and radius."""
        self.obstacles.append(
            {
                "p1": np.array(start, dtype=np.float64),
                "p2": np.array(end, dtype=np.float64),
                "r": radius,
            }
        )

    def _frame_capsules(
        self, center: np.ndarray, yaw: float, inner_width: float, outer_width: float
    ) -> list[dict]:
        """Build the four capsules that model a gate frame."""
        banner_offset = (inner_width / 4.0) + (outer_width / 4.0)
        thickness = (outer_width - inner_width) / 4.0
        local_corners = [
            np.array([0, -banner_offset, banner_offset]),
            np.array([0, banner_offset, banner_offset]),
            np.array([0, banner_offset, -banner_offset]),
            np.array([0, -banner_offset, -banner_offset]),
        ]
        rot = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        wc = [(rot @ p) + center for p in local_corners]
        return [
            {"p1": wc[i], "p2": wc[(i + 1) % 4], "r": thickness, "kind": "gate_frame"}
            for i in range(4)
        ]

    def _lower_cylinder(self, center: np.ndarray, outer_width: float, radius: float) -> dict:
        """Build the lower capsule obstacle below a gate opening."""
        z_top = max(float(center[2]) - outer_width / 2.0, 0.05)
        return {
            "p1": np.array([center[0], center[1], 0.0], dtype=np.float64),
            "p2": np.array([center[0], center[1], z_top], dtype=np.float64),
            "r": float(radius),
            "kind": "gate_lower",
        }

    def _opening_corridor(self, center: np.ndarray, yaw: float, inner_width: float) -> dict:
        """Build the free-space corridor through a gate opening."""
        return {
            "center": np.array(center, dtype=np.float64),
            "axis": np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64),
            "radius": max(inner_width / 2.0 - self._opening_margin, 0.05),
            "half_depth": self._opening_half_depth,
        }

    def add_gate(
        self,
        pos: list | np.ndarray,
        rpy: list | np.ndarray,
        inner_width: float = 0.4,
        outer_width: float = 0.72,
        lower_frame_radius: float = 0.20,
    ) -> None:
        """Add a gate to the obstacle manager."""
        center = np.array(pos, dtype=np.float64)
        yaw = float(rpy[2])

        start_idx = len(self.obstacles)
        self.obstacles.extend(self._frame_capsules(center, yaw, inner_width, outer_width))
        self._gate_obstacle_indices.append(list(range(start_idx, start_idx + 4)))

        self.obstacles.append(self._lower_cylinder(center, outer_width, lower_frame_radius))
        self._gate_lower_indices.append(start_idx + 4)

        self._gate_openings.append(self._opening_corridor(center, yaw, inner_width))
        self.gates.append(
            {
                "pos": center,
                "rpy": np.array(rpy, dtype=np.float64),
                "inner_width": inner_width,
                "outer_width": outer_width,
                "lower_frame_radius": float(lower_frame_radius),
            }
        )

    def add_pole(self, pos: list | np.ndarray, height: float = 1.55, radius: float = 0.015) -> None:
        """Add a pole obstacle."""
        if isinstance(pos, dict):
            pos = pos["pos"]
        pos_arr = np.asarray(pos, dtype=np.float64)
        z_top = pos_arr[2]

        self._pole_obstacle_indices.append(len(self.obstacles))
        self.add_cylinder(
            np.array([pos_arr[0], pos_arr[1], z_top - height], dtype=np.float64),
            np.array([pos_arr[0], pos_arr[1], z_top], dtype=np.float64),
            radius,
        )

    def update_gate_positions(self, gate_positions: np.ndarray, gate_rpys: np.ndarray) -> None:
        """Update gate positions and orientations in place."""
        gate_positions = np.asarray(gate_positions, dtype=np.float64)
        gate_rpys = np.asarray(gate_rpys, dtype=np.float64)

        for gate_idx, gate in enumerate(self.gates):
            if gate_idx >= gate_positions.shape[0]:
                break
            center = np.array(gate_positions[gate_idx], dtype=np.float64)
            yaw = float(gate_rpys[gate_idx][2])
            inner_width, outer_width = gate["inner_width"], gate["outer_width"]
            lower_radius = gate.get("lower_frame_radius", 0.20)

            # Rewrite in place (same obstacle indices, so the MPCC constraint count is unchanged).
            new_frame = self._frame_capsules(center, yaw, inner_width, outer_width)
            for obs_idx, obstacle in zip(self._gate_obstacle_indices[gate_idx], new_frame):
                self.obstacles[obs_idx] = obstacle
            self.obstacles[self._gate_lower_indices[gate_idx]] = self._lower_cylinder(
                center, outer_width, lower_radius
            )
            self._gate_openings[gate_idx] = self._opening_corridor(center, yaw, inner_width)

            gate["pos"] = center
            gate["rpy"] = np.array(gate_rpys[gate_idx], dtype=np.float64)

    def update_pole_positions(self, pole_positions: np.ndarray, height: float = 1.55) -> None:
        """Update pole positions in place."""
        pole_positions = np.asarray(pole_positions, dtype=np.float64)
        for i, pole_idx in enumerate(self._pole_obstacle_indices):
            if i >= pole_positions.shape[0]:
                break
            x, y, z_top = pole_positions[i]
            self.obstacles[pole_idx]["p1"] = np.array([x, y, z_top - height], dtype=np.float64)
            self.obstacles[pole_idx]["p2"] = np.array([x, y, z_top], dtype=np.float64)

    def snapshot(
        self, exclude_gate_centers: np.ndarray | None = None, exclude_tol: float = 0.3
    ) -> ObstacleManager:
        """Create a frozen snapshot of the current obstacles for the planner."""
        exclude_idx: set[int] = set()
        if exclude_gate_centers is not None:
            ec = np.asarray(exclude_gate_centers, dtype=np.float64).reshape(-1, 3)
            if len(ec) > 0:
                for gi, gate in enumerate(self.gates):
                    if float(np.min(np.linalg.norm(ec - gate["pos"], axis=1))) <= exclude_tol:
                        exclude_idx.update(self._gate_obstacle_indices[gi])

        snap = ObstacleManager(safety_margin=self.safety_margin)
        snap.obstacles = [
            {"p1": o["p1"].copy(), "p2": o["p2"].copy(), "r": float(o["r"]), "kind": o.get("kind")}
            for i, o in enumerate(self.obstacles)
            if i not in exclude_idx
        ]
        snap._gate_openings = [
            {
                "center": op["center"].copy(),
                "axis": op["axis"].copy(),
                "radius": op["radius"],
                "half_depth": op["half_depth"],
            }
            for op in self._gate_openings
        ]
        snap._opening_margin = self._opening_margin
        snap._opening_half_depth = self._opening_half_depth
        return snap

    def _points_in_openings(self, points: np.ndarray) -> np.ndarray:
        """Return a boolean mask for points inside any gate opening."""
        inside = np.zeros(points.shape[0], dtype=bool)
        for op in self._gate_openings:
            d = points - op["center"]
            axial = d @ op["axis"]
            lateral = np.linalg.norm(d - axial[:, None] * op["axis"], axis=1)
            inside |= (np.abs(axial) <= op["half_depth"]) & (lateral <= op["radius"])
        return inside

    def points_in_obstacles(self, points: np.ndarray, margin: float | None = None) -> np.ndarray:
        """Return a boolean mask for points that intersect any obstacle."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        mask = np.zeros(points.shape[0], dtype=bool)
        margin = self.safety_margin if margin is None else float(margin)
        in_opening = self._points_in_openings(points) if self._gate_openings else None

        for obs in self.obstacles:
            if mask.all():
                break
            p1, v = obs["p1"], obs["p2"] - obs["p1"]
            v_norm_sq = float(np.dot(v, v))
            if v_norm_sq == 0.0:
                dist = np.linalg.norm(points - p1, axis=1)
            else:
                t = np.clip((points - p1) @ v / v_norm_sq, 0.0, 1.0)
                dist = np.linalg.norm(points - (p1 + t[:, None] * v), axis=1)
            hit = dist <= float(obs["r"]) + margin
            if in_opening is not None and obs.get("kind") == "gate_frame":
                hit &= ~in_opening  # forgive frame contact inside the opening tube
            mask |= hit
        return mask

    def dynamic_contour_weight(
        self, position: np.ndarray, target_gate_idx: int | None = None
    ) -> float:
        """Compute the contour-weight bump near gates."""
        if target_gate_idx is not None and 0 <= target_gate_idx < len(self.gates):
            gates = [self.gates[target_gate_idx]]
        else:
            gates = self.gates
        q_c = self._q_nom
        for gate in gates:
            dist_sq = float(np.sum((position - gate["pos"]) ** 2))
            q_c += self._q_wp * float(np.exp(-0.5 * dist_sq / self._sigma_sq))
        return float(q_c)

    def get_obstacle_parameters(self) -> np.ndarray:
        """Flatten obstacle endpoints into a parameter vector."""
        params = []
        for obs in self.obstacles:
            params.extend(obs["p1"])
            params.extend(obs["p2"])
        return np.array(params, dtype=np.float64)

    def get_collision_expressions(self, x_sym: ca.MX, p_sym: ca.MX) -> ca.MX:
        """Build collision constraints for the MPC solver."""
        constraints = []
        drone_pos = x_sym[0:3]
        eps = 1e-6
        for i, obs in enumerate(self.obstacles):
            idx = i * 6
            p1, p2 = p_sym[idx : idx + 3], p_sym[idx + 3 : idx + 6]
            v = p2 - p1
            t = ca.fmax(0, ca.fmin(1, ca.dot(drone_pos - p1, v) / (ca.sumsqr(v) + 1e-9)))
            dist_sq = ca.sumsqr(drone_pos - (p1 + t * v))
            constraints.append(ca.sqrt(dist_sq + eps) - (obs["r"] + self.safety_margin))
        return ca.vcat(constraints)

    def render(
        self, sim: Sim, rgba: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.3)
    ) -> None:
        """Render all obstacle capsules in the simulator."""
        from crazyflow.sim.visualize import draw_capsule

        for obs in self.obstacles:
            draw_capsule(sim, p1=obs["p1"], p2=obs["p2"], radius=obs["r"], rgba=rgba)
