"""Point Mass Model (PMM) trajectory planner.

Implements the sampling-based, near time-optimal path planner from Section VI of
Foehn et al., "AlphaPilot: Autonomous Drone Racing" (Autonomous Robots, 2021):

    Waypoints / Gates
            v
    Point Mass Model planner  (this module)
            v
    Geometric reference path  (dense PMM samples)
            v
    Arc-length parameterization  (uniform-arc-length cubic spline)
            v
    MPCC  (attitude_mpc.py, unchanged)

The drone is modelled as a point mass with bounded per-axis acceleration. Minimum-time
motion primitives between states have a closed-form bang-bang (Sec. VI-A, eq. 23) or
bang-singular-bang (eq. 24) solution. A layered graph is built by sampling candidate
velocities at each gate (Sec. VI-B), and Dijkstra finds the minimum-total-time path
through the gates. The resulting trajectory is sampled densely and refit as a cubic
spline parameterized by true arc length (Sec. VI-C, adapted to the MPCC's cubic basis).

IMPORTANT — geometry-only handoff to the MPCC:
    The PMM produces a time-optimal trajectory p*(t), i.e. shape *and* timing. The MPCC
    decides speed itself through its progress state v_theta, so this planner discards the
    PMM time/velocity profile and exposes only the geometric path, re-parameterized by
    arc length. The public API is identical to ``TrajectoryPlanner`` so this class is a
    drop-in replacement consumed by ``attitude_mpc.py`` without changes to the solver.
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from lsy_drone_racing.control.obstacle_manager import ObstacleManager

# Shared logger for the whole PMM stack (planner + the controller's replan trigger).
# Enable verbose planner output at runtime with:
#     logging.getLogger("lsy_drone_racing.pmm").setLevel(logging.DEBUG)
# INFO gives one line per plan (timing, cost, fallback); DEBUG adds graph/sampling/primitive detail.
logger = logging.getLogger("lsy_drone_racing.pmm")


# ----------------------------------------------------------------------------------------
# Phase 1/2: One-dimensional minimum-time double-integrator primitives (Sec. VI-A)
# ----------------------------------------------------------------------------------------
class _Axis1D:
    """A single-axis acceleration profile as a list of (accel, duration) phases."""

    def __init__(self, p0: float, v0: float, phases: list[tuple[float, float]]) -> None:
        self.p0 = float(p0)
        self.v0 = float(v0)
        self.phases = [(float(a), float(dt)) for a, dt in phases if dt > 1e-12]
        self.T = float(sum(dt for _, dt in self.phases))

    def state_at(self, t: float) -> tuple[float, float]:
        """Return (position, velocity) at time t (clamped to [0, T])."""
        t = float(np.clip(t, 0.0, self.T))
        p, v = self.p0, self.v0
        for a, dt in self.phases:
            step = min(t, dt)
            p += v * step + 0.5 * a * step * step
            v += a * step
            t -= step
            if t <= 1e-12:
                break
        return p, v


class _CubicAxis1D:
    """A single-axis cubic Hermite profile reaching (pf, vf) at exactly time T.

    Used as a robust synchronization fallback when alpha-scaled bang-bang cannot stretch a
    maneuver to T* (e.g. a fast gate fly-through where the boundary velocities pin the
    duration). The PMM timing is discarded downstream, so only the smooth geometric shape
    matters here.
    """

    def __init__(self, p0: float, v0: float, pf: float, vf: float, T: float) -> None:
        self.p0, self.v0, self.T = float(p0), float(v0), float(T)
        P = pf - p0 - v0 * T
        V = vf - v0
        self._c = 3.0 * P / T**2 - V / T
        self._d = V / T**2 - 2.0 * P / T**3

    def state_at(self, t: float) -> tuple[float, float]:
        t = float(np.clip(t, 0.0, self.T))
        p = self.p0 + self.v0 * t + self._c * t * t + self._d * t**3
        v = self.v0 + 2.0 * self._c * t + 3.0 * self._d * t * t
        return p, v


def _quad_roots(a: float, b: float, c: float) -> list[float]:
    """Real roots of a x^2 + b x + c = 0 (handles the linear/degenerate cases)."""
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return []
        return [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return []
    sq = np.sqrt(disc)
    return [(-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a)]


def _two_phase_time(
    p0: float, v0: float, pf: float, vf: float, a1: float, a2: float
) -> tuple[float, float, float] | None:
    """Solve a two-phase (accel a1 then a2) maneuver. Returns (T, t1, t2) or None.

    Closed-form from substituting the velocity constraint into the position constraint:
        dp = A t1^2 + B t1 + C, with the coefficients below (see module derivation).
    """
    dp = pf - p0
    A = a1 * (a2 - a1) / (2.0 * a2)
    B = v0 * (a2 - a1) / a2
    C = (vf * vf - v0 * v0) / (2.0 * a2)

    best = None
    for t1 in _quad_roots(A, B, C - dp):
        if t1 < -1e-9:
            continue
        t1 = max(t1, 0.0)
        t2 = (vf - v0 - a1 * t1) / a2
        if t2 < -1e-9:
            continue
        t2 = max(t2, 0.0)
        T = t1 + t2
        if best is None or T < best[0]:
            best = (T, t1, t2)
    return best


def min_time_1d(
    p0: float, v0: float, pf: float, vf: float, u_lo: float, u_hi: float, v_cap: float | None = None
) -> _Axis1D:
    """Minimum-time profile for a 1-D double integrator with accel in [u_lo, u_hi].

    Bang-bang solution (eq. 23). If ``v_cap`` is given and the unconstrained peak speed
    would exceed it, a cruise phase is inserted yielding a bang-singular-bang solution
    (eq. 24).
    """
    # Try both bang-bang orderings (accelerate-first vs decelerate-first) and keep the faster.
    best, best_a = None, None
    for a1, a2 in ((u_hi, u_lo), (u_lo, u_hi)):
        sol = _two_phase_time(p0, v0, pf, vf, a1, a2)
        if sol is not None and (best is None or sol[0] < best[0]):
            best, best_a = sol, (a1, a2)

    if best is None:
        # No feasible bang-bang (should not happen for finite bounds); hold position.
        return _Axis1D(p0, v0, [(0.0, 0.0)])

    T, t1, t2 = best
    a1, a2 = best_a

    # Velocity-cap check: insert a singular (zero-accel) cruise arc if the peak exceeds v_cap.
    if v_cap is not None:
        v_peak = v0 + a1 * t1
        if abs(v_peak) > v_cap + 1e-9:
            cap = min_time_1d_capped(p0, v0, pf, vf, u_lo, u_hi, np.sign(v_peak) * v_cap)
            if cap is not None:
                return cap

    return _Axis1D(p0, v0, [(a1, t1), (a2, t2)])


def min_time_1d_capped(
    p0: float, v0: float, pf: float, vf: float, u_lo: float, u_hi: float, v_sat: float
) -> _Axis1D | None:
    """Bang-singular-bang profile that saturates the velocity at ``v_sat`` (eq. 24)."""
    a_acc = u_hi if v_sat >= v0 else u_lo
    a_dec = u_hi if vf >= v_sat else u_lo
    if abs(a_acc) < 1e-12 or abs(a_dec) < 1e-12 or abs(v_sat) < 1e-12:
        return None

    t1 = (v_sat - v0) / a_acc
    t3 = (vf - v_sat) / a_dec
    d1 = (v_sat * v_sat - v0 * v0) / (2.0 * a_acc)
    d3 = (vf * vf - v_sat * v_sat) / (2.0 * a_dec)
    d2 = (pf - p0) - d1 - d3
    t2 = d2 / v_sat

    if t1 < -1e-9 or t2 < -1e-9 or t3 < -1e-9:
        return None  # cruise not feasible -> caller falls back to bang-bang
    return _Axis1D(p0, v0, [(a_acc, max(t1, 0.0)), (0.0, max(t2, 0.0)), (a_dec, max(t3, 0.0))])


def fixed_time_1d(
    p0: float,
    v0: float,
    pf: float,
    vf: float,
    u_lo: float,
    u_hi: float,
    T_target: float,
    v_cap: float | None = None,
) -> _Axis1D:
    """Profile reaching (pf, vf) in exactly ``T_target`` by scaling the accel bounds by alpha.

    Implements the per-axis synchronization of Sec. VI-A: the slower (critical) axis sets
    T* = T_target, and faster axes are slowed by scaling their acceleration bounds by
    alpha in (0, 1]. Since the minimum time decreases monotonically with alpha, alpha is
    found by bisection (robust; runs offline, not in the 50 Hz loop).
    """
    # Trivial axis already at the target and at rest: just hold for the full duration.
    if abs(pf - p0) < 1e-9 and abs(v0) < 1e-9 and abs(vf) < 1e-9:
        return _Axis1D(p0, 0.0, [(0.0, T_target)])

    full = min_time_1d(p0, v0, pf, vf, u_lo, u_hi, v_cap)
    if full.T >= T_target - 1e-6:
        return full  # already at or above the target time at full authority

    # Faithful attempt: alpha-scaled bang-bang (paper Sec. VI-A). Sweep alpha downward only
    # within the monotonic region (min-time increases as authority drops). Reducing alpha too
    # far yields spurious large-overshoot solutions, so stop as soon as monotonicity breaks.
    def T_of(a: float) -> float:
        return min_time_1d(p0, v0, pf, vf, a * u_lo, a * u_hi, v_cap).T

    prev_a, prev_T = 1.0, full.T
    for a in np.linspace(1.0, 0.05, 40)[1:]:
        T_a = T_of(a)
        if not np.isfinite(T_a) or T_a < prev_T - 1e-9:
            break  # left the well-behaved region
        if T_a >= T_target:  # bracket [a, prev_a]; T decreasing in alpha
            lo, hi = a, prev_a
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if T_of(mid) > T_target:
                    lo = mid
                else:
                    hi = mid
            cand = min_time_1d(p0, v0, pf, vf, hi * u_lo, hi * u_hi, v_cap)
            if abs(cand.T - T_target) < 1e-2:
                return cand
            break
        prev_a, prev_T = a, T_a

    # Robust fallback: cubic Hermite reaching (pf, vf) at exactly T_target.
    return _CubicAxis1D(p0, v0, pf, vf, T_target)


class MotionPrimitive:
    """Time-optimal, axis-synchronized motion primitive between two point-mass states."""

    def __init__(
        self,
        p0: np.ndarray,
        v0: np.ndarray,
        pf: np.ndarray,
        vf: np.ndarray,
        u_max: np.ndarray,
        v_max: np.ndarray | None = None,
    ) -> None:
        p0 = np.asarray(p0, dtype=np.float64)
        v0 = np.asarray(v0, dtype=np.float64)
        pf = np.asarray(pf, dtype=np.float64)
        vf = np.asarray(vf, dtype=np.float64)
        u_max = np.asarray(u_max, dtype=np.float64)
        v_cap = None if v_max is None else np.asarray(v_max, dtype=np.float64)

        # Per-axis minimum times, then T* = max so all axes finish simultaneously.
        full = [
            min_time_1d(
                p0[k], v0[k], pf[k], vf[k], -u_max[k], u_max[k],
                None if v_cap is None else v_cap[k],
            )
            for k in range(3)
        ]
        self.T = max(ax.T for ax in full)

        self.axes: list[_Axis1D | _CubicAxis1D] = []
        self.n_cubic = 0  # number of axes that fell back to cubic-Hermite synchronization
        for k in range(3):
            if full[k].T >= self.T - 1e-9:
                self.axes.append(full[k])
            else:
                ax = fixed_time_1d(
                    p0[k], v0[k], pf[k], vf[k], -u_max[k], u_max[k], self.T,
                    None if v_cap is None else v_cap[k],
                )
                self.axes.append(ax)
                if isinstance(ax, _CubicAxis1D):
                    self.n_cubic += 1
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "cubic-Hermite fallback axis=%s: (p0=%.3f, v0=%.3f) -> "
                            "(pf=%.3f, vf=%.3f), T_sync=%.3fs "
                            "(alpha-scaled bang-bang could not reach T*)",
                            "xyz"[k], p0[k], v0[k], pf[k], vf[k], self.T,
                        )

    def state_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (position, velocity) 3-vectors at time t."""
        pv = [ax.state_at(t) for ax in self.axes]
        return np.array([p for p, _ in pv]), np.array([v for _, v in pv])

    def sample_positions(self, n: int) -> np.ndarray:
        """Sample n positions uniformly in time over [0, T]. Returns (n, 3)."""
        ts = np.linspace(0.0, self.T, max(n, 2))
        return np.array([self.state_at(t)[0] for t in ts])


# ----------------------------------------------------------------------------------------
# Phase 3: Sampling-based receding-horizon graph search (Sec. VI-B)
# ----------------------------------------------------------------------------------------
def _cone_directions(rng: np.random.Generator, axis: np.ndarray, half_angle: float, n: int) -> np.ndarray:
    """Sample n unit vectors uniformly within ``half_angle`` of ``axis``."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    # Orthonormal basis (e1, e2) spanning the plane perpendicular to axis.
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, ref)
    e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(axis, e1)

    cos_t = rng.uniform(np.cos(half_angle), 1.0, n)  # area-uniform on the spherical cap
    sin_t = np.sqrt(np.clip(1.0 - cos_t**2, 0.0, 1.0))
    azim = rng.uniform(0.0, 2.0 * np.pi, n)
    dirs = (
        cos_t[:, None] * axis
        + (sin_t * np.cos(azim))[:, None] * e1
        + (sin_t * np.sin(azim))[:, None] * e2
    )
    return dirs


class _GraphPlanner:
    """Layered graph over sampled gate-crossing velocities, solved with Dijkstra."""

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
    ) -> None:
        self.u_max = u_max
        self.v_axis_cap = np.full(3, float(v_max))  # per-axis velocity cap for the primitives
        self.obs = obstacle_manager
        self.n_collision_pts = n_collision_pts
        self.collision_margin = float(collision_margin)
        rng = np.random.default_rng(seed)

        v_mag = float(v_max)  # largest sampled gate speed (as a norm)
        speed_lo = speed_lo_frac * v_mag

        # Layer 0: the (single) start state. Layers 1..G: sampled velocities at each gate.
        self.layers: list[list[dict]] = [[{"pos": np.asarray(start_pos, float),
                                           "vel": np.asarray(start_vel, float)}]]
        for c, nrm in zip(gate_centers, gate_normals):
            states = [{"pos": c.copy(), "vel": nrm * (0.5 * (speed_lo + v_mag))}]  # nominal sample
            if n_samples > 1:
                dirs = _cone_directions(rng, nrm, phi_max, n_samples - 1)
                speeds = rng.uniform(speed_lo, v_mag, n_samples - 1)
                states += [{"pos": c.copy(), "vel": d * s} for d, s in zip(dirs, speeds)]
            self.layers.append(states)

        self.stats: dict = {}  # populated by solve(): node/edge/rejection counts, path cost
        if logger.isEnabledFor(logging.DEBUG):
            spd = [float(np.linalg.norm(s["vel"])) for layer in self.layers[1:] for s in layer]
            logger.debug(
                "sampling: %d gate(s) x %d samples/gate; |v| in [%.2f, %.2f] m/s, phi_max=%.0f deg",
                len(gate_centers), n_samples,
                min(spd) if spd else 0.0, max(spd) if spd else 0.0, np.rad2deg(phi_max),
            )

    def _edge_ok(self, prim: MotionPrimitive) -> bool:
        if self.obs is None:
            return True
        pts = prim.sample_positions(self.n_collision_pts)
        return not bool(self.obs.points_in_obstacles(pts, margin=self.collision_margin).any())

    def solve(self) -> list[MotionPrimitive] | None:
        """Return the minimum-time list of primitives through all gates, or None."""
        # Flatten nodes; assign a global id. Build a virtual sink after the last gate layer.
        node_ids: dict[tuple[int, int], int] = {}
        nodes: list[dict] = []
        for li, layer in enumerate(self.layers):
            for ni, st in enumerate(layer):
                node_ids[(li, ni)] = len(nodes)
                nodes.append(st)
        sink = len(nodes)

        # Adjacency with the connecting primitive stored on each edge.
        adj: list[list[tuple[int, float, MotionPrimitive | None]]] = [[] for _ in range(sink + 1)]
        n_edges = n_rej_collision = n_rej_infeasible = 0
        for li in range(len(self.layers) - 1):
            for ai, a in enumerate(self.layers[li]):
                ua = node_ids[(li, ai)]
                for bi, b in enumerate(self.layers[li + 1]):
                    prim = MotionPrimitive(
                        a["pos"], a["vel"], b["pos"], b["vel"], self.u_max, self.v_axis_cap
                    )
                    if not np.isfinite(prim.T):
                        n_rej_infeasible += 1
                        continue
                    if not self._edge_ok(prim):
                        n_rej_collision += 1
                        continue
                    n_edges += 1
                    adj[ua].append((node_ids[(li + 1, bi)], prim.T, prim))
        last = len(self.layers) - 1
        for ni in range(len(self.layers[last])):
            adj[node_ids[(last, ni)]].append((sink, 0.0, None))

        # Dijkstra from the start node (0) to the sink.
        dist = [np.inf] * (sink + 1)
        prev: list[tuple[int, MotionPrimitive | None] | None] = [None] * (sink + 1)
        dist[0] = 0.0
        pq = [(0.0, 0)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u] + 1e-12:
                continue
            if u == sink:
                break
            for v, w, prim in adj[u]:
                nd = d + w
                if nd < dist[v] - 1e-12:
                    dist[v] = nd
                    prev[v] = (u, prim)
                    heapq.heappush(pq, (nd, v))

        self.stats = {
            "nodes": len(nodes),
            "edges": n_edges,
            "rejected_collision": n_rej_collision,
            "rejected_infeasible": n_rej_infeasible,
            "cost": float(dist[sink]),
        }
        logger.debug(
            "graph: %d nodes, %d edges kept, %d rejected (collision=%d, infeasible=%d), best T=%.3fs",
            len(nodes), n_edges, n_rej_collision + n_rej_infeasible,
            n_rej_collision, n_rej_infeasible, dist[sink],
        )

        if not np.isfinite(dist[sink]):
            return None

        # Reconstruct primitive sequence (skip the zero-weight sink edge).
        prims: list[MotionPrimitive] = []
        cur = sink
        while prev[cur] is not None:
            u, prim = prev[cur]
            if prim is not None:
                prims.append(prim)
            cur = u
        prims.reverse()
        return prims


# ----------------------------------------------------------------------------------------
# Top-level planner: drop-in replacement for TrajectoryPlanner
# ----------------------------------------------------------------------------------------
class PointMassPlanner:
    """PMM planner exposing the same public API as ``TrajectoryPlanner``.

    Builds a near time-optimal geometric path through the gates and refits it as a
    cubic spline parameterized by true arc length so the MPCC consumes it unchanged.
    """

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
        committed_pts: np.ndarray | None = None,
        committed_speeds: np.ndarray | None = None,
    ) -> None:
        self._u_max = np.full(3, float(u_max)) if np.isscalar(u_max) else np.asarray(u_max, float)
        # v_max is the maximum SPEED (velocity norm). It is used both as the per-axis velocity
        # cap inside the motion primitives and as the largest speed sampled at gates. Kept as a
        # scalar so sampled gate speeds and the exported v_theta reference stay consistent with
        # the MPCC's progress-speed bound (a per-axis cap would let the norm exceed the bound).
        self._v_max = float(np.max(v_max))
        self._obs = obstacle_manager
        self._n_vel_samples = int(n_vel_samples)
        self._phi_max = float(phi_max)
        self._speed_lo_frac = float(speed_lo_frac)
        self._n_eval_points = int(n_eval_points)
        self._n_path_samples = int(n_path_samples_per_seg)
        self._n_collision_pts = int(n_collision_pts)
        # Collision-pruning margin for the PMM graph only. Deliberately smaller than the MPCC's
        # hard-constraint safety margin (~0.14 m): the PMM just needs to avoid gross collisions
        # while sampling candidate lines; the MPCC owns the final, conservative clearance. A large
        # PMM margin over-prunes (it leaves only a few cm of clear gate opening) and forces the
        # single-sample no-collision-check fallback.
        self._collision_margin = float(collision_margin)
        self._min_z = float(min_z)
        self._tail = float(tail_extension)
        self._seed = int(seed)

        # Tuning snapshot so the async replanner can spawn an identically-configured planner.
        self._kwargs = dict(
            u_max=self._u_max, v_max=self._v_max, n_vel_samples=self._n_vel_samples,
            phi_max=self._phi_max, speed_lo_frac=self._speed_lo_frac,
            n_eval_points=self._n_eval_points, n_path_samples_per_seg=self._n_path_samples,
            n_collision_pts=self._n_collision_pts, collision_margin=self._collision_margin,
            min_z=self._min_z, tail_extension=self._tail, seed=self._seed,
        )

        self.plan(start_pos, gates_pos, gate_rpys, start_vel, committed_pts, committed_speeds)

    # -- planning -------------------------------------------------------------------------
    def plan(
        self,
        start_pos: np.ndarray,
        gates_pos: np.ndarray,
        gate_rpys: np.ndarray | None = None,
        start_vel: np.ndarray | None = None,
        committed_pts: np.ndarray | None = None,
        committed_speeds: np.ndarray | None = None,
    ) -> None:
        """Run the PMM graph search and build the arc-length spline.

        ``committed_pts`` / ``committed_speeds`` (optional) are a near-field prefix taken from the
        previously-active trajectory. When supplied, ``start_pos`` is the END of that prefix and
        the prefix is prepended to the new path, so the segment the MPCC is already tracking stays
        continuous across the replan (no reference jump). See AttitudeMPC._maybe_replan_pmm.
        """
        self._committed_pts = (
            None if committed_pts is None
            else np.asarray(committed_pts, dtype=np.float64).reshape(-1, 3)
        )
        self._committed_speeds = (
            None if committed_speeds is None
            else np.asarray(committed_speeds, dtype=np.float64).reshape(-1)
        )
        start_pos = np.asarray(start_pos, dtype=np.float64)
        gates_pos = np.asarray(gates_pos, dtype=np.float64).reshape(-1, 3)

        centers = [gates_pos[i] for i in range(len(gates_pos))]
        normals = self._gate_normals(start_pos, centers, gate_rpys)

        if start_vel is None or float(np.linalg.norm(start_vel)) < 1e-6:
            # Estimate an initial velocity heading toward the first gate.
            d0 = (centers[0] - start_pos) if centers else np.array([1.0, 0.0, 0.0])
            start_vel = d0 / (np.linalg.norm(d0) + 1e-9) * (self._speed_lo_frac * self._v_max)
        start_vel = np.asarray(start_vel, dtype=np.float64)

        t0 = time.perf_counter()
        has_prefix = self._committed_pts is not None and len(self._committed_pts) > 0
        logger.info(
            "plan START: gates=%d, M=%d, committed_prefix=%s",
            len(centers), self._n_vel_samples, has_prefix,
        )

        prims, stats = self._run_graph(start_pos, start_vel, centers, normals, prune=True)
        used_fallback = prims is None
        if used_fallback:
            # Robust fallback: a single nominal sample per gate, no collision pruning, so the
            # planner always returns a usable path (the MPCC's soft constraints handle clearance).
            logger.warning(
                "graph infeasible with pruning -> single-sample NO-COLLISION-CHECK fallback "
                "(M=%d, phi=%.0f deg, gates=%d); path may clip obstacles",
                self._n_vel_samples, np.rad2deg(self._phi_max), len(centers),
            )
            prims, stats = self._run_graph(
                start_pos, start_vel, centers, normals, prune=False, single=True
            )

        self._build_spline_from_primitives(prims, normals[-1] if normals else None)

        n_cubic = sum(getattr(p, "n_cubic", 0) for p in (prims or []))
        logger.info(
            "plan DONE: %.1f ms, prims=%d, len=%.2f m, |v| in [%.2f, %.2f] m/s, "
            "cost=%.3f s, cubic_axes=%d, fallback=%s",
            1e3 * (time.perf_counter() - t0), len(prims or []), self._s_total,
            float(self._speed_profile.min()), float(self._speed_profile.max()),
            stats.get("cost", float("nan")), n_cubic, used_fallback,
        )

    def rebuild(
        self, start_pos: np.ndarray, gates_pos: np.ndarray, gate_rpys: np.ndarray | None = None
    ) -> None:
        """Re-plan in place (signature matches TrajectoryPlanner.rebuild).

        Called when a gate's true position is revealed. Start velocity is estimated
        internally; the MPCC re-anchors theta/v_theta after the call regardless.
        """
        self.plan(start_pos, gates_pos, gate_rpys, start_vel=None)

    def _gate_normals(
        self, start_pos: np.ndarray, centers: list[np.ndarray], gate_rpys: np.ndarray | None
    ) -> list[np.ndarray]:
        """Required crossing direction (unit vector) for each gate.

        The race environment only registers a gate as passed when the drone crosses its plane in
        the gate's +x direction (from the -x side to the +x side; see ``gate_passed`` in
        ``envs/utils.py``). The gate's +x axis in world frame is ``[cos(yaw), sin(yaw), 0]``, so
        the planned crossing velocity MUST point along it.

        We deliberately do NOT flip the normal toward the approach direction. A geometry-aligned
        (flipped) normal can make the planner cross a gate the wrong way; the env then does not
        count it, the gate stays the current target, and the drone gets pulled back through it
        (the loop-back-through-an-already-flown-gate bug). If the drone happens to approach from
        the +x side, the motion primitives will route it around to cross in +x, which is exactly
        what the race rules demand. Only when orientations are unavailable do we fall back to a
        geometric estimate (best effort; the controller always supplies orientations).
        """
        normals = []
        prev = start_pos
        for i, c in enumerate(centers):
            if gate_rpys is not None:
                yaw = float(np.asarray(gate_rpys, float).reshape(-1, 3)[i, 2])
                nrm = np.array([np.cos(yaw), np.sin(yaw), 0.0])  # gate +x = required crossing dir
            else:
                nxt = centers[i + 1] if i + 1 < len(centers) else c + (c - prev)
                nrm = nxt - prev
            nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
            normals.append(nrm)
            prev = c
        return normals

    def _run_graph(
        self,
        start_pos: np.ndarray,
        start_vel: np.ndarray,
        centers: list[np.ndarray],
        normals: list[np.ndarray],
        prune: bool,
        single: bool = False,
    ) -> tuple[list[MotionPrimitive] | None, dict]:
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
            obstacle_manager=None if not prune else self._obs,
            n_collision_pts=self._n_collision_pts,
            seed=self._seed,
            collision_margin=self._collision_margin,
        )
        prims = gp.solve()
        return prims, gp.stats

    def _build_spline_from_primitives(
        self, prims: list[MotionPrimitive], final_normal: np.ndarray | None
    ) -> None:
        """Sample primitives densely (position + speed), resample at uniform arc length, fit.

        Besides the geometric spline, the PMM speed |v(t)| is mapped onto arc length and kept
        as ``self._speed_profile``. For an arc-length path |v| equals ds/dt, i.e. exactly the
        progress speed v_theta the MPCC should target, so it is exported via evaluate_speed().
        """
        # Dense samples of position and speed. Optionally begin with the committed near-field
        # prefix (from the previously-active trajectory) so the path the MPCC is already tracking
        # stays continuous across a replan; the PMM primitives only cover the part beyond it.
        pts: list[np.ndarray] = []
        spd: list[float] = []
        has_prefix = self._committed_pts is not None and len(self._committed_pts) > 0
        if has_prefix:
            pts.extend(list(self._committed_pts))
            if self._committed_speeds is not None and len(self._committed_speeds) == len(self._committed_pts):
                spd.extend([float(s) for s in self._committed_speeds])
            else:
                spd.extend([self._v_max] * len(self._committed_pts))
        if prims:
            # With a prefix the first primitive starts at the commit point (== last prefix
            # point), so its t=0 sample is dropped below to avoid a duplicate.
            if not has_prefix:
                p0, v0 = prims[0].state_at(0.0)
                pts.append(p0)
                spd.append(float(np.linalg.norm(v0)))
        elif not has_prefix:
            pts.append(np.zeros(3))
            spd.append(0.0)
        for prim in prims:
            ts = np.linspace(0.0, prim.T, max(self._n_path_samples, 2))
            for t in ts[1:]:  # drop the shared joint point shared with the previous primitive
                p, v = prim.state_at(t)
                pts.append(p)
                spd.append(float(np.linalg.norm(v)))
        dense = np.array(pts, dtype=np.float64)
        speed_dense = np.array(spd, dtype=np.float64)

        # Tail extension past the final gate so the MPCC horizon never stalls at the endpoint.
        if self._tail > 0.0 and len(dense) >= 2:
            tang = dense[-1] - dense[-2]
            tang = tang / (np.linalg.norm(tang) + 1e-9)
            if final_normal is not None and np.dot(tang, final_normal) < 0:
                tang = final_normal
            dense = np.vstack([dense, dense[-1] + self._tail * tang])
            speed_dense = np.append(speed_dense, speed_dense[-1])  # hold the final speed

        # Ground clearance.
        dense[:, 2] = np.maximum(dense[:, 2], self._min_z)

        # True arc-length parameterization: cumulative chord length over the dense polyline,
        # drop zero-length segments, then resample position and speed uniformly in arc length.
        seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        keep = np.concatenate(([True], seg > 1e-6))
        dense = dense[keep]
        speed_dense = speed_dense[keep]
        seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        total = float(cum[-1])
        if total < 1e-6:  # degenerate path -> tiny straight stub to keep the spline valid
            dense = np.vstack([dense[0], dense[0] + np.array([1e-3, 0.0, 0.0])])
            speed_dense = np.array([speed_dense[0], speed_dense[0]])
            cum = np.array([0.0, 1e-3])
            total = 1e-3

        s_uniform = np.linspace(0.0, total, self._n_eval_points)
        pos_uniform = np.stack([np.interp(s_uniform, cum, dense[:, k]) for k in range(3)], axis=1)
        # Cap the exported speed at the max-speed bound. The per-axis velocity cap can let the
        # velocity norm slightly exceed v_max on diagonal motions; clip so the v_theta reference
        # never asks for more than the planner's (and the MPCC's) speed limit.
        speed_uniform = np.clip(np.interp(s_uniform, cum, speed_dense), 0.0, self._v_max)

        self._s = s_uniform
        self._s_total = total
        self._des_pos_spline = CubicSpline(s_uniform, pos_uniform)
        self._des_vel_spline = self._des_pos_spline.derivative()
        self._waypoints_pos = pos_uniform
        self._speed_profile = speed_uniform

    # -- public API (identical to TrajectoryPlanner) --------------------------------------
    @property
    def total_length(self) -> float:
        """Total arc length of the planned trajectory."""
        return self._s_total

    @property
    def waypoints_pos(self) -> np.ndarray:
        """Fine-grained sampled positions along the path."""
        return self._waypoints_pos

    @property
    def knot_points(self) -> np.ndarray:
        """Spline knot points (arc-length values) used for segment indexing."""
        return self._s

    def final_waypoint(self) -> np.ndarray:
        """Final position at the end of the spline."""
        return self._des_pos_spline(self._s_total)

    def evaluate(self, s: float | np.ndarray) -> np.ndarray:
        """Evaluate the desired path position at arc-length parameter s."""
        return self._des_pos_spline(s)

    def evaluate_velocity(self, s: float | np.ndarray) -> np.ndarray:
        """Evaluate the path tangent (dp/ds) at arc-length parameter s."""
        return self._des_vel_spline(s)

    def evaluate_speed(self, s: float | np.ndarray) -> np.ndarray | float:
        """PMM time-optimal speed (m/s) at arc-length parameter s.

        For an arc-length path this equals the planned ds/dt, i.e. exactly the progress speed
        v_theta the MPCC should aim for (fast on straights, slower into tight turns). Returns
        the raw PMM speed; clip it to the MPCC's v_theta bound at the call site.
        """
        return np.interp(s, self._s, self._speed_profile)

    def _spawn(
        self,
        start_pos: np.ndarray,
        gates_pos: np.ndarray,
        gate_rpys: np.ndarray | None = None,
        start_vel: np.ndarray | None = None,
        obstacle_manager: ObstacleManager | None = None,
        committed_pts: np.ndarray | None = None,
        committed_speeds: np.ndarray | None = None,
        n_vel_samples: int | None = None,
    ) -> PointMassPlanner:
        """Build a NEW planner with identical tuning but a fresh path (does not mutate self).

        Used by AsyncPMMReplanner: the background thread calls this with snapshots captured on
        the control thread, so the worker never reads shared mutable state. ``committed_pts`` /
        ``committed_speeds`` prepend a continuous near-field prefix (see plan()). ``n_vel_samples``
        overrides the per-gate sample count for this plan only (online replans use fewer samples
        than the offline initial plan, trading a little path quality for low latency).
        """
        kwargs = dict(self._kwargs)
        kwargs["obstacle_manager"] = obstacle_manager
        if n_vel_samples is not None:
            kwargs["n_vel_samples"] = int(n_vel_samples)
        return PointMassPlanner(
            start_pos, gates_pos, gate_rpys, start_vel,
            committed_pts=committed_pts, committed_speeds=committed_speeds, **kwargs,
        )

    def nearest_theta(self, pos: np.ndarray) -> float:
        """Arc-length parameter of the path point nearest to ``pos``."""
        idx = self.get_nearest_waypoint_index(pos)
        return float(self._s_total * idx / max(self._n_eval_points - 1, 1))

    def get_nearest_waypoint_index(self, pos: np.ndarray) -> int:
        """Index of the sampled path point nearest to a world-space position."""
        return int(np.argmin(np.linalg.norm(self._waypoints_pos - pos, axis=1)))

    def get_polynomial_coeffs_at(
        self, theta_pred: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Return per-segment cubic coefficients and the segment's left knot.

        Format is identical to ``TrajectoryPlanner.get_polynomial_coeffs_at`` (scipy
        CubicSpline coefficient layout), so the MPCC model parameters are unchanged.
        """
        theta_pred = float(np.clip(theta_pred, self._s[0], self._s[-1]))
        seg_idx = int(np.searchsorted(self._s[1:], theta_pred, side="right"))
        seg_idx = min(max(seg_idx, 0), len(self._s) - 2)

        c_seg = self._des_pos_spline.c[:, seg_idx, :]
        return c_seg[:, 0], c_seg[:, 1], c_seg[:, 2], float(self._s[seg_idx])


# ----------------------------------------------------------------------------------------
# Phase 4: Asynchronous (off-control-thread) replanning
# ----------------------------------------------------------------------------------------
class AsyncPMMReplanner:
    """Runs PMM replans on a background thread so the 50 Hz control loop never blocks.

    A full PMM plan costs ~100-200 ms; running it inline would stall ~5-10 control cycles
    (fatal at racing speed on real hardware). Mirroring the paper's separate planning thread,
    the controller keeps flying on the *current* planner while a replan runs in the background:

        trigger -> request(build_fn)           # control thread, non-blocking
        ... keep using the live planner ...
        take() -> new planner (or None)        # control thread, picks up the result later

    ``build_fn`` is a zero-argument closure that captures snapshots (start state, gates,
    obstacles) taken on the control thread, so the worker never touches shared mutable state.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready: PointMassPlanner | None = None
        self._busy = False

    def busy(self) -> bool:
        """True while a background plan is in flight (used to avoid queueing duplicates)."""
        with self._lock:
            return self._busy

    def request(self, build_fn: Callable[[], PointMassPlanner]) -> bool:
        """Start a background replan. No-op (returns False) if one is already running."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        threading.Thread(target=self._run, args=(build_fn,), daemon=True).start()
        return True

    def _run(self, build_fn: Callable[[], PointMassPlanner]) -> None:
        try:
            result = build_fn()
        except Exception:
            result = None  # on any failure keep flying on the current planner
        with self._lock:
            self._ready = result
            self._busy = False

    def take(self) -> PointMassPlanner | None:
        """Return a finished planner if one is ready (clearing it), else None."""
        with self._lock:
            result, self._ready = self._ready, None
            return result
