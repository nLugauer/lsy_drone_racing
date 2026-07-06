"""Point-mass trajectory planner for the PMM stack.

It builds minimum-time motion primitives, solves a sampled gate-crossing graph, and fits the
result as an arc-length cubic spline for the MPCC controller.

Based on the algorithms in "AlphaPilot: Autonomous Drone Racing", Foehn et al., Autonomous Robots
2021 (min-time primitives + sampled graph), and "Model Predictive Contouring Control for
Time-Optimal Quadrotor Flight", Romero et al., IEEE T-RO 2022 (arc-length parameterization).
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline

if TYPE_CHECKING:
    from collections.abc import Callable

    from lsy_drone_racing.control.obstacle_manager import ObstacleManager

# Shared logger for the whole PMM stack
logger = logging.getLogger("lsy_drone_racing.pmm")


class _Axis1D:
    """Represent a one-axis acceleration profile as a list of (acceleration, duration) phases."""

    def __init__(self, p0: float, v0: float, phases: list[tuple[float, float]]) -> None:
        self.p0 = float(p0)
        self.v0 = float(v0)
        # Non-zero-duration phases only, as (acceleration, duration) tuples
        self.phases = [(float(a), float(dt)) for a, dt in phases if dt > 1e-12]
        # Total duration is the sum of all phase durations
        self.Ttotal = float(sum(dt for _, dt in self.phases))

    def state_at(self, t: float) -> tuple[float, float]:
        """Return (position, velocity) at time t (clamped to [0, T])."""
        # Clamp t to the range
        t = float(np.clip(t, 0.0, self.Ttotal))
        p, v = self.p0, self.v0
        for a, dt in self.phases:
            # Determine the time step to integrate
            dt_step = min(t, dt)
            # Integrate the motion equations for constant acceleration
            p += v * dt_step + 0.5 * a * dt_step * dt_step
            v += a * dt_step
            # Update remaining time
            t -= dt_step
            if t <= 1e-12:
                break
        return p, v


class _CubicAxis1D:
    """Smooth one-axis trajectory stretched to a prescribed duration.

    Matches the desired start and end positions and velocities while arriving exactly at T,
    allowing all three axes to finish simultaneously without introducing waiting segments.
    """

    def __init__(self, p0: float, v0: float, pf: float, vf: float, T: float) -> None:
        self.Ttotal = float(T)
        # Use a cubic Hermite spline to interpolate between the start and end states
        self._pos_spline = CubicHermiteSpline([0.0, self.Ttotal], [p0, pf], [v0, vf])
        self._vel_spline = self._pos_spline.derivative()

    def state_at(self, t: float) -> tuple[float, float]:
        """Return (position, velocity) at time t (clamped to [0, T])."""
        t = float(np.clip(t, 0.0, self.Ttotal))
        return float(self._pos_spline(t)), float(self._vel_spline(t))


def _two_phase(
    p0: float, v0: float, pf: float, vf: float, a1: float, a2: float
) -> tuple[float, float] | None:
    """Return the phase durations (t1, t2) of 'a1 then a2' reaching (pf, vf), or None.

    Given the initial and final states and the two phase accelerations (bang-bang).
    """
    # Write t2 in terms of t1 and solve the quadratic equation for t1: A*t1² + B*t1 + C = 0
    A = a1 * (a2 - a1) / (2.0 * a2)
    B = v0 * (a2 - a1) / a2
    C = (vf * vf - v0 * v0) / (2.0 * a2) - (pf - p0)
    # Compute the discriminant to check for real solutions
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return None
    best = None
    # Compute the two quadratic roots
    sqrt_disc = np.sqrt(disc)
    denom = 2.0 * A
    roots = [(-B + sqrt_disc) / denom, (-B - sqrt_disc) / denom]
    for t1 in roots:
        # Solve for t2
        t2 = (vf - v0 - a1 * t1) / a2
        # Allow small negative values from numerical error
        if t1 >= -1e-9 and t2 >= -1e-9 and (best is None or t1 + t2 < best[0] + best[1]):
            # choose smallest positive solution
            best = (max(t1, 0.0), max(t2, 0.0))
    return best


def _capped(p0: float, v0: float, pf: float, vf: float, u: float, v_sat: float) -> _Axis1D | None:
    """Accelerate to v_sat, cruise, brake to vf; return the profile (or None if it will not fit).

    Bang-singular-bang under a speed cap.
    """
    a_acc = u if v_sat >= v0 else -u
    a_dec = u if vf >= v_sat else -u
    # Time to reach v_sat from v0
    t1 = (v_sat - v0) / a_acc
    # Time to reach vf from v_sat
    t3 = (vf - v_sat) / a_dec
    # distance while (de)accelerating
    d1 = (v_sat * v_sat - v0 * v0) / (2.0 * a_acc)
    d3 = (vf * vf - v_sat * v_sat) / (2.0 * a_dec)
    # Remaining distance cruised at v_sat
    t2 = ((pf - p0) - d1 - d3) / v_sat
    if min(t1, t2, t3) < -1e-9:
        return None
    # Three phases: accelerate, cruise, decelerate
    return _Axis1D(p0, v0, [(a_acc, max(t1, 0.0)), (0.0, max(t2, 0.0)), (a_dec, max(t3, 0.0))])


def min_time_1d(
    p0: float, v0: float, pf: float, vf: float, u: float, v_cap: float = np.inf
) -> _Axis1D:
    """Return the minimum-time 1-D profile: bang-bang, or bang-singular-bang under a speed cap."""
    best = None
    for a1, a2 in ((u, -u), (-u, u)):
        # Solve the two-phase bang-bang problem for the given acceleration order
        ts = _two_phase(p0, v0, pf, vf, a1, a2)
        if ts is not None and (best is None or ts[0] + ts[1] < best[2] + best[3]):
            best = (a1, a2, ts[0], ts[1])
    # Extract best solutions
    a1, a2, t1, t2 = best
    # If the peak speed would exceed the cap, cruise at the cap instead (v_cap=inf disables it).
    if abs(v0 + a1 * t1) > v_cap + 1e-9:
        capped = _capped(p0, v0, pf, vf, u, np.sign(v0 + a1 * t1) * v_cap)
        if capped is not None:
            return capped
    return _Axis1D(p0, v0, [(a1, t1), (a2, t2)])


def _fixed_time_1d(
    p0: float, v0: float, pf: float, vf: float, u: float, T_target: float, v_cap: float = np.inf
) -> _Axis1D | _CubicAxis1D:
    """Return a 1-D profile that reaches (pf, vf) in exactly T_target.

    In a 3-axis primitive the slowest axis sets the total time T*; the faster ("slack") axes are
    stretched to T* so all three finish together.
    """
    # Take the time-optimal solution if it already fills T_target
    full = min_time_1d(p0, v0, pf, vf, u, v_cap)
    if full.Ttotal >= T_target - 1e-6:
        return full
    # Otherwise, stretch the motion to T_target with a cubic Hermite spline
    return _CubicAxis1D(p0, v0, pf, vf, T_target)


def _edge_cost_matrix(
    pA: np.ndarray,
    vA: np.ndarray,
    pB: np.ndarray,
    vB: np.ndarray,
    u_max: np.ndarray,
    v_cap: np.ndarray,
) -> np.ndarray:
    """Minimum edge time T* between every layer-A and layer-B state; a (KA, KB) matrix.

    Vectorized twin of ``min_time_1d`` (identical bang-bang / bang-singular-bang formulas), applied
    to all KA*KB edges and 3 axes at once because the graph scores ~10^5 candidate edges per plan.
    """
    dp = (pB[None, :, :] - pA[:, None, :])[..., None]  # (KA, KB, 3, 1)
    v0 = vA[:, None, :, None]
    vf = vB[None, :, :, None]
    u = u_max[:, None]
    vc = v_cap[:, None]
    # 4 candidates per axis = 2 accel orderings x 2 quadratic roots (min_time_1d's two orderings).
    a1 = u * np.array([1.0, 1.0, -1.0, -1.0])
    a2 = u * np.array([-1.0, -1.0, 1.0, 1.0])
    root = np.array([1.0, -1.0, 1.0, -1.0])
    A = a1 * (a2 - a1) / (2.0 * a2)
    B = v0 * (a2 - a1) / a2
    C = (vf**2 - v0**2) / (2.0 * a2) - dp
    disc = B**2 - 4.0 * A * C
    t1 = (-B + root * np.sqrt(np.maximum(disc, 0.0))) / (2.0 * A)
    t2 = (vf - v0 - a1 * t1) / a2
    valid = (disc >= 0.0) & (t1 >= -1e-4) & (t2 >= -1e-4)
    t1 = np.maximum(t1, 0.0)
    T_bb = np.where(valid, t1 + np.maximum(t2, 0.0), np.inf)

    # Bang-singular-bang time for the candidates whose peak speed exceeds the cap.
    v_peak = v0 + a1 * t1
    v_sat = np.sign(v_peak) * vc
    a_acc = np.where(v_sat >= v0, u, -u)
    a_dec = np.where(vf >= v_sat, u, -u)
    t1c = (v_sat - v0) / a_acc
    t3c = (vf - v_sat) / a_dec
    d1 = (v_sat**2 - v0**2) / (2.0 * a_acc)
    d3 = (vf**2 - v_sat**2) / (2.0 * a_dec)
    t2c = (dp - d1 - d3) / np.where(np.abs(v_sat) < 1e-6, 1e-6, v_sat)
    valid_cap = (t1c >= -1e-4) & (t2c >= -1e-4) & (t3c >= -1e-4)
    T_cap = np.where(
        valid_cap, np.maximum(t1c, 0.0) + np.maximum(t2c, 0.0) + np.maximum(t3c, 0.0), np.inf
    )
    needs_cap = valid & (np.abs(v_peak) > vc)
    # Per axis: min over the 4 candidates; then the edge time is the slowest of the 3 axes.
    return np.where(needs_cap, T_cap, T_bb).min(axis=3).max(axis=2)


class MotionPrimitive:
    """Represent a feasible 3D motion between two states.

    Combines per-axis time-optimal control profiles into a synchronized trajectory.
    """

    def __init__(
        self,
        p0: np.ndarray,
        v0: np.ndarray,
        pf: np.ndarray,
        vf: np.ndarray,
        u_max: np.ndarray,
        v_max: np.ndarray,
    ) -> None:
        """Initialize the primitive from the given boundary states."""
        # Convert inputs to float64 arrays
        p0 = np.asarray(p0, dtype=np.float64)
        v0 = np.asarray(v0, dtype=np.float64)
        pf = np.asarray(pf, dtype=np.float64)
        vf = np.asarray(vf, dtype=np.float64)
        u_max = np.asarray(u_max, dtype=np.float64)
        v_cap = np.asarray(v_max, dtype=np.float64)

        # Per-axis minimum times, then T* = max so all axes finish simultaneously.
        full = [min_time_1d(p0[k], v0[k], pf[k], vf[k], u_max[k], v_cap[k]) for k in range(3)]
        # Slowest axis
        self.Ttotal = max(ax.Ttotal for ax in full)
        self.axes: list[_Axis1D | _CubicAxis1D] = []
        self.n_cubic = 0  # axes stretched to T*
        for k in range(3):
            # If the axis already fills T*, use its time-optimal profile;
            if full[k].Ttotal >= self.Ttotal - 1e-9:
                self.axes.append(full[k])
            # otherwise, stretch it to T*.
            else:
                ax = _fixed_time_1d(p0[k], v0[k], pf[k], vf[k], u_max[k], self.Ttotal, v_cap[k])
                self.axes.append(ax)
                self.n_cubic += isinstance(ax, _CubicAxis1D)

    def state_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the position and velocity at time t."""
        pv = [ax.state_at(t) for ax in self.axes]
        return np.array([p for p, _ in pv]), np.array([v for _, v in pv])

    def sample_positions(self, n: int) -> np.ndarray:
        """Sample positions along the primitive (for collision checks)."""
        return np.array([self.state_at(t)[0] for t in np.linspace(0.0, self.Ttotal, max(n, 2))])


def _cone_directions(
    rng: np.random.Generator, axis: np.ndarray, half_angle: float, n: int
) -> np.ndarray:
    """Sample unit vectors within a cone around the axis.

    Used to generate multiple feasible gate-approach directions for the graph search.
    """
    # Normalize the axis
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, ref)
    e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(axis, e1)

    cos_t = rng.uniform(np.cos(half_angle), 1.0, n)  # area-uniform on the spherical cap
    sin_t = np.sqrt(np.clip(1.0 - cos_t**2, 0.0, 1.0))
    azim = rng.uniform(0.0, 2.0 * np.pi, n)
    return (
        cos_t[:, None] * axis
        + (sin_t * np.cos(azim))[:, None] * e1
        + (sin_t * np.sin(azim))[:, None] * e2
    )


class _GraphPlanner:
    """Find the fastest sequence of gate-crossing states using a layered graph.

    Each gate is represented by multiple sampled position/velocity states.
    Edges correspond to feasible minimum-time motion primitives, and Dijkstra's
    algorithm selects the fastest path through all graph layers.

    Compared to the original AlphaPilot approach, this implementation also adds
    exit layers behind each gate and a penalty for sharp velocity turns.
    """

    def __init__(
        self,
        start_pos: np.ndarray,
        start_vel: np.ndarray,
        gate_centers: list[np.ndarray],
        gate_normals: list[np.ndarray],
        u_max: np.ndarray,
        v_max: float,
        n_samples: int,
        phi_max: float,
        speed_lo_frac: float,
        obstacle_manager: ObstacleManager | None,
        n_collision_pts: int,
        seed: int,
        collision_margin: float = 0.05,
        gate_exit_dist: float = 0.3,
        turn_penalty_weight: float = 0.3,
        max_turn_angle: float = np.pi,
    ) -> None:
        """Initialize the graph planner and sample the gate-crossing layers.

        The non selfexplanatory parameters are:
        n_samples: Number of candidate states sampled per gate; phi_max: Half-angle of the
        sampling cone around each gate normal; speed_lo_frac: Minimum sampled speed as a
        fraction of ``v_max``; obstacle_manager: Obstacle manager used for collision checking;
        n_collision_pts: Number of points sampled when checking a primitive for collisions;
        seed: Random seed for reproducible gate-state sampling; collision_margin: Safety margin
        added around obstacles during collision checking; gate_exit_dist: Distance beyond each
        gate for generating exit-state layers; turn_penalty_weight: Weight of the turn-angle
        penalty in edge costs; max_turn_angle: Hard cap on the per-edge turn angle; edges whose
        entry or exit bend exceeds it are rejected outright (default pi = no rejection).
        """
        self.u_max = u_max
        self.velocity_limits = np.full(3, float(v_max))
        self.obs = obstacle_manager
        self.n_collision_pts = n_collision_pts
        self.collision_margin = float(collision_margin)
        self.turn_penalty_weight = float(turn_penalty_weight)
        self.max_turn_angle = float(max_turn_angle)
        self.stats: dict[str, float] = {"cost": np.inf}
        rng = np.random.default_rng(seed)

        max_speed = float(v_max)
        min_speed = speed_lo_frac * max_speed

        # Build the nodes, Layer 0 is the start state
        start_state = {
            "pos": np.asarray(start_pos, dtype=float),
            "vel": np.asarray(start_vel, dtype=float),
        }
        self.layers: list[list[dict]] = [[start_state]]
        # Sample candidate states for each gate
        for c, nrm in zip(gate_centers, gate_normals):
            # First sample crosses straight along the gate normal at mid speed
            mid_speed = 0.5 * (min_speed + max_speed)
            states = [{"pos": c.copy(), "vel": nrm * (mid_speed)}]
            # Sample additional states within a cone around the gate normal, with random speeds
            dirs = _cone_directions(rng, nrm, phi_max, n_samples - 1)
            speeds = rng.uniform(min_speed, max_speed, n_samples - 1)
            for direction, speed in zip(dirs, speeds):
                states.append({"pos": c.copy(), "vel": direction * speed})
            self.layers.append(states)
            # Add exit layer nodes to ensure proper gate crossing
            if gate_exit_dist > 0.0:
                exit_states = []

                for state in states:
                    direction = state["vel"] / (np.linalg.norm(state["vel"]) + 1e-9)
                    exit_states.append(
                        {"pos": c + gate_exit_dist * direction, "vel": state["vel"].copy()}
                    )
                self.layers.append(exit_states)

        if logger.isEnabledFor(logging.DEBUG):
            spd = [float(np.linalg.norm(s["vel"])) for layer in self.layers[1:] for s in layer]
            logger.debug(
                "sampling: %d gate(s) x %d samples/gate; |v| in [%.2f, %.2f] m/s, phi_max=%.0f deg",
                len(gate_centers),
                n_samples,
                min(spd) if spd else 0.0,
                max(spd) if spd else 0.0,
                np.rad2deg(phi_max),
            )

    def _edge_ok(self, prim: MotionPrimitive) -> bool:
        """Check whether the motion primitive intersects any obstacle."""
        if self.obs is None:
            return True
        # Sample points along the primitive
        pts = prim.sample_positions(self.n_collision_pts)
        return not self.obs.points_in_obstacles(pts, margin=self.collision_margin).any()

    def _build_graph(self) -> tuple[list[dict], list[list[tuple[int, float]]], int]:
        """Flatten the layered states into a single node list and directed adjacency list."""
        # Convert the layered graph into one flat list of nodes; node_ids maps each
        # (layer, local index) pair to a unique node ID.
        node_ids: dict[tuple[int, int], int] = {}
        nodes: list[dict] = []
        for lay_idx, layer in enumerate(self.layers):
            for node_idx, state in enumerate(layer):
                node_ids[(lay_idx, node_idx)] = len(nodes)
                nodes.append(state)
        # A virtual end node is added after the final layer so Dijkstra has a single goal.
        end_node = len(nodes)

        # Create adjacency list
        adj: list[list[tuple[int, float]]] = [[] for _ in range(end_node + 1)]

        for lay_idx in range(len(self.layers) - 1):
            layer_A, layer_B = self.layers[lay_idx], self.layers[lay_idx + 1]
            positions_A = np.asarray([state["pos"] for state in layer_A], dtype=np.float64)
            velocities_A = np.asarray([state["vel"] for state in layer_A], dtype=np.float64)
            positions_B = np.asarray([state["pos"] for state in layer_B], dtype=np.float64)
            velocities_B = np.asarray([state["vel"] for state in layer_B], dtype=np.float64)
            # Compute all edge costs
            cost = _edge_cost_matrix(
                positions_A,
                velocities_A,
                positions_B,
                velocities_B,
                self.u_max,
                self.velocity_limits,
            )

            # Penalize edges that require large changes in flight direction.
            uA = velocities_A / (np.linalg.norm(velocities_A, axis=1, keepdims=True) + 1e-9)
            uB = velocities_B / (np.linalg.norm(velocities_B, axis=1, keepdims=True) + 1e-9)
            path_direction = positions_B[None, :, :] - positions_A[:, None, :]
            path_direction = path_direction / (
                np.linalg.norm(path_direction, axis=2, keepdims=True) + 1e-9
            )
            ang_in = np.arccos(np.clip(np.einsum("ad,abd->ab", uA, path_direction), -1.0, 1.0))
            ang_out = np.arccos(np.clip(np.einsum("bd,abd->ab", uB, path_direction), -1.0, 1.0))
            # Increase the cost of sharp-turn edges, weighted by turn_penalty_weight.
            cost = cost + self.turn_penalty_weight * (ang_in**2 + ang_out**2)
            # Reject edges whose entry or exit bend exceeds max_turn_angle (default pi = no-op).
            cost[np.maximum(ang_in, ang_out) > self.max_turn_angle] = np.inf

            # Build the final graph adjacency list, skipping edges that are infeasible
            for a_idx in range(len(layer_A)):
                ua = node_ids[(lay_idx, a_idx)]
                for b_idx in range(len(layer_B)):
                    edge_cost = float(cost[a_idx, b_idx])
                    if np.isfinite(edge_cost):
                        adj[ua].append((node_ids[(lay_idx + 1, b_idx)], edge_cost))

        # Virtual end node is connected to all nodes in the last layer with zero cost edges
        last = len(self.layers) - 1
        for node_idx in range(len(self.layers[last])):
            adj[node_ids[(last, node_idx)]].append((end_node, 0.0))

        return nodes, adj, end_node

    def _dijkstra(
        self, adj: list[list[tuple[int, float]]], end_node: int, banned: set[tuple[int, int]]
    ) -> tuple[list[int] | None, float]:
        """Shortest path from node 0 to end_node, skipping banned (u, v) edges."""
        best_cost = [np.inf] * (end_node + 1)
        prev: list[int | None] = [None] * (end_node + 1)
        best_cost[0] = 0.0
        pq = [(0.0, 0)]
        while pq:
            cur_cost, cur_node = heapq.heappop(pq)
            # Ignore outdated entries
            if cur_cost > best_cost[cur_node] + 1e-12:
                continue
            if cur_node == end_node:
                break
            for next_node, edge_cost in adj[cur_node]:
                if (cur_node, next_node) in banned:
                    continue  # edge disabled by a previous collision check
                nd = cur_cost + edge_cost
                if nd < best_cost[next_node] - 1e-12:
                    best_cost[next_node] = nd
                    prev[next_node] = cur_node
                    heapq.heappush(pq, (nd, next_node))

        if not np.isfinite(best_cost[end_node]):
            return None, np.inf

        # Reconstruct the route (end_node itself is virtual and left out).
        path_nodes: list[int] = []
        cur = prev[end_node]
        while cur is not None:
            path_nodes.append(cur)
            cur = prev[cur]
        path_nodes.reverse()
        return path_nodes, float(best_cost[end_node])

    def solve(self) -> list[MotionPrimitive] | None:
        """Return the minimum-time primitive chain through the gates, or None if infeasible.

        Find the single fastest route through the graph, then collision-check only the primitives
        on that optimal route. If any of them clips an obstacle, return None so the caller drops to
        the no-check straight-line fallback. The optimal path is used as-is and never patched
        around obstacles by banning edges and re-solving; this matches the working smooth-replan
        behaviour and avoids the contorted detour routes that the MPCC cannot track.
        """
        t0 = time.perf_counter()
        nodes, adj, end_node = self._build_graph()

        path_nodes, cost = self._dijkstra(adj, end_node, banned=set())
        if path_nodes is None:
            self.stats = {"cost": np.inf}
            logger.debug("graph: infeasible, %.1f ms", 1e3 * (time.perf_counter() - t0))
            return None

        # Reconstruct the best route and collision-check only its primitives.
        prims: list[MotionPrimitive] = []
        for i in range(len(path_nodes) - 1):
            a, b = nodes[path_nodes[i]], nodes[path_nodes[i + 1]]
            prim = MotionPrimitive(
                a["pos"], a["vel"], b["pos"], b["vel"], self.u_max, self.velocity_limits
            )
            if not self._edge_ok(prim):
                self.stats = {"cost": np.inf}
                return None  # optimal path clips an obstacle -> trigger fallback
            prims.append(prim)

        self.stats = {"cost": cost}
        logger.debug("graph: solved, cost=%.3f s, %.1f ms", cost, 1e3 * (time.perf_counter() - t0))
        return prims


class PointMassPlanner:
    """Expose the PMM planner through the trajectory-planner interface for use with toggle."""

    def __init__(
        self,
        start_pos: np.ndarray,
        gates_pos: np.ndarray,
        gate_rpys: np.ndarray | None = None,
        start_vel: np.ndarray | None = None,
        obstacle_manager: ObstacleManager | None = None,
        u_max: float | np.ndarray = 10.0,
        v_max: float | np.ndarray = 3.0,
        n_vel_samples: int = 5,
        phi_max: float = np.deg2rad(30.0),
        speed_lo_frac: float = 0.5,
        n_eval_points: int = 500,
        n_path_samples_per_seg: int = 60,
        n_collision_pts: int = 20,
        collision_margin: float = 0.05,
        min_z: float = 0.15,
        tail_extension: float = 0.5,
        seed: int = 0,
        gate_exit_dist: float = 0.3,
        turn_penalty_weight: float = 0.3,
        max_turn_angle: float = np.pi,
        committed_pts: np.ndarray | None = None,
        committed_speeds: np.ndarray | None = None,
        committed_suffix_pts: np.ndarray | None = None,
        committed_suffix_speeds: np.ndarray | None = None,
    ) -> None:
        """Initialize the planner and run the first plan.

        The non selfexplanatory parameters are:
        n_path_samples_per_seg: samples per primitive used for spline reconstruction;
        n_collision_pts: number of points sampled per primitive for collision checking;
        collision_margin: obstacle inflation margin for graph-level collision checking;
        min_z: minimum allowed altitude in final trajectory;
        tail_extension: extension beyond final waypoint to avoid MPCC horizon cutoff;
        seed: random seed for reproducible graph sampling;
        gate_exit_dist: offset behind each gate forcing a straight crossing so the gate counts;
        turn_penalty_weight: cost weight for angular deviation in graph edges;
        max_turn_angle: hard cap on the turn angle at either end of an edge; edges bending more
        than this are rejected outright (default pi = no rejection, only the soft turn penalty);
        committed_pts: previously executed trajectory points for replanning continuity;
        committed_speeds: speeds corresponding to committed trajectory points;
        committed_suffix_pts: fixed trajectory suffix appended after optimized segment;
        committed_suffix_speeds: speeds for the committed suffix segment

        """
        self._u_max = np.full(3, float(u_max)) if np.isscalar(u_max) else np.asarray(u_max, float)
        self._v_max = float(np.max(v_max))
        self._obs = obstacle_manager
        self._n_vel_samples = int(n_vel_samples)
        self._phi_max = float(phi_max)
        self._speed_lo_frac = float(speed_lo_frac)
        self._n_eval_points = int(n_eval_points)
        self._n_path_samples = int(n_path_samples_per_seg)
        self._n_collision_pts = int(n_collision_pts)
        self._collision_margin = float(collision_margin)
        self._min_z = float(min_z)
        self._tail = float(tail_extension)
        self._seed = int(seed)
        self._gate_exit_dist = float(gate_exit_dist)
        self._turn_penalty_weight = float(turn_penalty_weight)
        self._max_turn_angle = float(max_turn_angle)

        # Take snapshot of the parameters
        self._kwargs = dict(
            u_max=self._u_max,
            v_max=self._v_max,
            n_vel_samples=self._n_vel_samples,
            phi_max=self._phi_max,
            speed_lo_frac=self._speed_lo_frac,
            n_eval_points=self._n_eval_points,
            n_path_samples_per_seg=self._n_path_samples,
            n_collision_pts=self._n_collision_pts,
            collision_margin=self._collision_margin,
            min_z=self._min_z,
            tail_extension=self._tail,
            seed=self._seed,
            gate_exit_dist=self._gate_exit_dist,
            turn_penalty_weight=self._turn_penalty_weight,
            max_turn_angle=self._max_turn_angle,
        )

        self.plan(
            start_pos,
            gates_pos,
            gate_rpys,
            start_vel,
            committed_pts,
            committed_speeds,
            committed_suffix_pts,
            committed_suffix_speeds,
        )

    def plan(
        self,
        start_pos: np.ndarray,
        gates_pos: np.ndarray,
        gate_rpys: np.ndarray | None = None,
        start_vel: np.ndarray | None = None,
        committed_pts: np.ndarray | None = None,
        committed_speeds: np.ndarray | None = None,
        committed_suffix_pts: np.ndarray | None = None,
        committed_suffix_speeds: np.ndarray | None = None,
    ) -> None:
        """Run the graph planner and fit the result as a smooth arc-length spline."""
        # Store the committed trajectory and suffix for continuity and replanning
        self._committed_pts = (
            None
            if committed_pts is None
            else np.asarray(committed_pts, dtype=np.float64).reshape(-1, 3)
        )
        self._committed_speeds = (
            None
            if committed_speeds is None
            else np.asarray(committed_speeds, dtype=np.float64).reshape(-1)
        )
        self._committed_suffix_pts = (
            None
            if committed_suffix_pts is None
            else np.asarray(committed_suffix_pts, dtype=np.float64).reshape(-1, 3)
        )
        self._committed_suffix_speeds = (
            None
            if committed_suffix_speeds is None
            else np.asarray(committed_suffix_speeds, dtype=np.float64).reshape(-1)
        )
        start_pos = np.asarray(start_pos, dtype=np.float64)
        gates_pos = np.asarray(gates_pos, dtype=np.float64).reshape(-1, 3)

        # Build gate structure
        centers = [gates_pos[i] for i in range(len(gates_pos))]
        normals = self._gate_normals(start_pos, centers, gate_rpys)
        # Initialize start velocity (on first plan it is 0)
        if start_vel is None or float(np.linalg.norm(start_vel)) < 1e-6:
            d0 = (centers[0] - start_pos) if centers else np.array([1.0, 0.0, 0.0])
            start_vel = d0 / (np.linalg.norm(d0) + 1e-9) * (self._speed_lo_frac * self._v_max)
        start_vel = np.asarray(start_vel, dtype=np.float64)

        t0 = time.perf_counter()
        logger.info("plan START: gates=%d, M=%d", len(centers), self._n_vel_samples)

        # Run the graph planner
        prims, stats = self._run_graph(
            start_pos, start_vel, centers, normals, obstacle_filtering=True
        )
        use_fallback = prims is None
        if use_fallback:
            # Use fallback trajectory that may clip obstacles
            logger.warning(
                "graph infeasible (no collision-free route) -> "
                "single-sample no-check fallback (gates=%d)",
                len(centers),
            )
            prims, stats = self._run_graph(
                start_pos, start_vel, centers, normals, obstacle_filtering=False, single=True
            )

        # Turn the primitives into a smooth arc-length spline for the MPCC to track
        self._build_spline_from_primitives(prims, normals[-1] if normals else None)

        logger.info(
            "plan DONE: %.1f ms, prims=%d, len=%.2f m, cost=%.3f s, fallback=%s",
            1e3 * (time.perf_counter() - t0),
            len(prims or []),
            self._s_total,
            stats.get("cost", float("nan")),
            use_fallback,
        )

    def _gate_normals(
        self, start_pos: np.ndarray, centers: list[np.ndarray], gate_rpys: np.ndarray | None
    ) -> list[np.ndarray]:
        """Return the required crossing direction for each gate."""
        normals = []
        prev = start_pos
        for i, c in enumerate(centers):
            if gate_rpys is not None:
                yaw = float(np.asarray(gate_rpys, float).reshape(-1, 3)[i, 2])
                nrm = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            else:
                nxt = centers[i + 1] if i + 1 < len(centers) else c + (c - prev)
                nrm = nxt - prev
            normals.append(nrm / (np.linalg.norm(nrm) + 1e-9))
            prev = c
        return normals

    def _run_graph(
        self,
        start_pos: np.ndarray,
        start_vel: np.ndarray,
        centers: list[np.ndarray],
        normals: list[np.ndarray],
        obstacle_filtering: bool,
        single: bool = False,
    ) -> tuple[list[MotionPrimitive] | None, dict]:
        """Build and solve the graph for the current planning window."""
        gp = _GraphPlanner(
            start_pos=start_pos,
            start_vel=start_vel,
            gate_centers=centers,
            gate_normals=normals,
            u_max=self._u_max,
            v_max=self._v_max,
            n_samples=1 if single else self._n_vel_samples,
            phi_max=self._phi_max,
            speed_lo_frac=self._speed_lo_frac,
            obstacle_manager=self._obs if obstacle_filtering else None,
            n_collision_pts=self._n_collision_pts,
            seed=self._seed,
            collision_margin=self._collision_margin,
            gate_exit_dist=self._gate_exit_dist,
            turn_penalty_weight=self._turn_penalty_weight,
            max_turn_angle=self._max_turn_angle,
        )
        prims = gp.solve()
        return prims, gp.stats

    def _extend(
        self, pts: list[np.ndarray], spd: list[float], cpts: np.ndarray, cspeeds: np.ndarray | None
    ) -> None:
        """Append committed points and speeds to the path buffer for smooth replanning."""
        pts.extend(cpts)
        if cspeeds is not None and len(cspeeds) == len(cpts):
            spd.extend(float(s) for s in cspeeds)
        else:
            spd.extend([self._v_max] * len(cpts))

    def _build_spline_from_primitives(
        self, prims: list[MotionPrimitive], final_normal: np.ndarray | None
    ) -> None:
        """Sample the primitive path and fit the arc-length spline."""
        pts: list[np.ndarray] = []
        spd: list[float] = []
        # If a committed prefix exists, extend the path with it first, to avoid jumps
        has_prefix = self._committed_pts is not None and len(self._committed_pts) > 0
        if has_prefix:
            self._extend(pts, spd, self._committed_pts, self._committed_speeds)
        else:
            # Without a prefix the first primitive's t=0 sample is needed for starting point
            p0, v0 = prims[0].state_at(0.0)
            pts.append(p0)
            spd.append(float(np.linalg.norm(v0)))

        for prim in prims or []:
            # Discretization of the dynamics
            for t in np.linspace(0.0, prim.Ttotal, max(self._n_path_samples, 2))[1:]:
                p, v = prim.state_at(t)
                pts.append(p)
                spd.append(float(np.linalg.norm(v)))

        # On a local-horizon replan add the backbone to the end
        has_suffix = self._committed_suffix_pts is not None and len(self._committed_suffix_pts) > 0
        if has_suffix:
            self._extend(pts, spd, self._committed_suffix_pts, self._committed_suffix_speeds)

        dense = np.array(pts, dtype=np.float64)
        speed_dense = np.array(spd, dtype=np.float64)

        # Tail extension past the final point so the MPCC horizon never stalls at the endpoint.
        if self._tail > 0.0 and not has_suffix and len(dense) >= 2:
            tang = dense[-1] - dense[-2]
            tang = tang / (np.linalg.norm(tang) + 1e-9)
            if final_normal is not None and np.dot(tang, final_normal) < 0:
                tang = final_normal
            dense = np.vstack([dense, dense[-1] + self._tail * tang])
            speed_dense = np.append(speed_dense, speed_dense[-1])

        # Ground clearance
        dense[:, 2] = np.maximum(dense[:, 2], self._min_z)

        # Arc-length parameterization (MPCC paper Sec. C, eq. (8))
        # Chord length of each segment between consecutive sampled points
        seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        # Keep the first point and drop duplicate points so the arc length strictly increases
        keep = np.concatenate(([True], seg > 1e-6))
        dense, speed_dense = dense[keep], speed_dense[keep]
        # Recompute the segment lengths after dropping duplicates
        seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        # Cumulative arc length along the path = the stored arc length per point
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        # Total path length
        total = float(cum[-1])
        # Degenerate path (collapsed to a point) -> tiny straight stub so the spline stays valid
        if total < 1e-6:
            dense = np.vstack([dense[0], dense[0] + np.array([1e-3, 0.0, 0.0])])
            speed_dense = np.array([speed_dense[0], speed_dense[0]])
            cum = np.array([0.0, 1e-3])
            total = 1e-3

        # Equidistant arc-length grid to resample onto
        s_uniform = np.linspace(0.0, total, self._n_eval_points)
        # Resample position at each arc-length sample (linear interp between the stored points)
        pos_uniform = np.stack([np.interp(s_uniform, cum, dense[:, k]) for k in range(3)], axis=1)
        # Resample speed the same way, clipped to v_max (diagonal motion can exceed the cap)
        speed_uniform = np.clip(np.interp(s_uniform, cum, speed_dense), 0.0, self._v_max)

        # Store the arc-length knots and the total length
        self._s = s_uniform
        self._s_total = total
        # Fit the cubic spline p(s) the MPCC evaluates, plus its derivative dp/ds (unit tangent)
        self._des_pos_spline = CubicSpline(s_uniform, pos_uniform)
        self._des_vel_spline = self._des_pos_spline.derivative()
        self._waypoints_pos = pos_uniform
        self._speed_profile = speed_uniform

    # public API consumed by attitude_mpc.py for MPCC and logging
    @property
    def total_length(self) -> float:
        """Return the total arc length of the planned trajectory."""
        return self._s_total

    @property
    def knot_points(self) -> np.ndarray:
        """Return the spline knot points in arc length."""
        return self._s

    @property
    def waypoints_pos(self) -> np.ndarray:
        """Return the sampled positions along the path."""
        return self._waypoints_pos

    def final_waypoint(self) -> np.ndarray:
        """Return the final position at the end of the spline."""
        return self._des_pos_spline(self._s_total)

    def evaluate(self, s: float | np.ndarray) -> np.ndarray:
        """Return the path position at arc length s."""
        return self._des_pos_spline(s)

    def evaluate_velocity(self, s: float | np.ndarray) -> np.ndarray:
        """Return the path tangent at arc length s."""
        return self._des_vel_spline(s)

    def evaluate_speed(self, s: float | np.ndarray) -> np.ndarray | float:
        """Return the planned speed at arc length s."""
        return np.interp(s, self._s, self._speed_profile)

    def nearest_theta(self, pos: np.ndarray) -> float:
        """Return the arc length of the path point nearest to pos."""
        idx = int(np.argmin(np.linalg.norm(self._waypoints_pos - pos, axis=1)))
        return float(self._s_total * idx / max(self._n_eval_points - 1, 1))

    def get_polynomial_coeffs_at(
        self, theta_pred: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Return the cubic coefficients for the segment at theta_pred."""
        theta_pred = float(np.clip(theta_pred, self._s[0], self._s[-1]))
        seg_idx = int(np.searchsorted(self._s[1:], theta_pred, side="right"))
        seg_idx = min(max(seg_idx, 0), len(self._s) - 2)
        c_seg = self._des_pos_spline.c[:, seg_idx, :]
        return c_seg[:, 0], c_seg[:, 1], c_seg[:, 2], float(self._s[seg_idx])

    def _replan_local(
        self,
        start_pos: np.ndarray,
        gates_pos: np.ndarray,
        gate_rpys: np.ndarray | None = None,
        start_vel: np.ndarray | None = None,
        obstacle_manager: ObstacleManager | None = None,
        committed_pts: np.ndarray | None = None,
        committed_speeds: np.ndarray | None = None,
        committed_suffix_pts: np.ndarray | None = None,
        committed_suffix_speeds: np.ndarray | None = None,
        n_vel_samples: int | None = None,
    ) -> PointMassPlanner:
        """Build a new planner with the same tuning but a fresh path."""
        kwargs = dict(self._kwargs)
        kwargs["obstacle_manager"] = obstacle_manager
        if n_vel_samples is not None:
            kwargs["n_vel_samples"] = int(n_vel_samples)
        return PointMassPlanner(
            start_pos,
            gates_pos,
            gate_rpys,
            start_vel,
            committed_pts=committed_pts,
            committed_speeds=committed_speeds,
            committed_suffix_pts=committed_suffix_pts,
            committed_suffix_speeds=committed_suffix_speeds,
            **kwargs,
        )


class AsyncPMMReplanner:
    """Run a PMM replan on a background thread so the 50 Hz control loop never stalls.

    A full replan is too long to run inside a control tick.
    Instead of blocking, `request` launches the planner build on a daemon thread while the control
    loop keeps flying the current path; each tick it calls `take` to swap in the finished planner
    once it is ready.
    """

    def __init__(self) -> None:
        """Initialize the replanner."""
        self._lock = threading.Lock()
        self._ready: PointMassPlanner | None = None
        self._busy = False

    def busy(self) -> bool:
        """Return True while a background plan is in flight."""
        with self._lock:
            return self._busy

    def request(self, build_fn: Callable[[], PointMassPlanner]) -> bool:
        """Start a background replan if one is not already running."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        threading.Thread(target=self._run, args=(build_fn,), daemon=True).start()
        return True

    def _run(self, build_fn: Callable[[], PointMassPlanner]) -> None:
        """Run the build function and store the result."""
        try:
            result = build_fn()
        except Exception:
            result = None  # keep flying on the current planner
        with self._lock:
            self._ready = result
            self._busy = False

    def take(self) -> PointMassPlanner | None:
        """Return a finished planner if one is ready."""
        with self._lock:
            result, self._ready = self._ready, None
            return result
