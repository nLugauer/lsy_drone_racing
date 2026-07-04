"""Point-mass trajectory planner for the PMM stack.

It builds minimum-time motion primitives, solves a sampled gate-crossing graph, and fits the
result as an arc-length cubic spline for the MPCC controller.
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from collections.abc import Callable

    from lsy_drone_racing.control.obstacle_manager import ObstacleManager

# Shared logger for the whole PMM stack (planner + the controller's replan trigger). Enable with
# logging.getLogger("lsy_drone_racing.pmm").setLevel(logging.INFO).
logger = logging.getLogger("lsy_drone_racing.pmm")


def _jax_edge_cost_base(
    p0: jax.Array, v0: jax.Array, pf: jax.Array, vf: jax.Array, u_max: jax.Array, v_cap: jax.Array
) -> jax.Array:
    """Compute the minimum-time edge cost for a single primitive."""
    dp = pf - p0

    def _solve_axis(
        dp_ax: jax.Array,
        v0_ax: jax.Array,
        vf_ax: jax.Array,
        u_max_ax: jax.Array,
        v_cap_ax: jax.Array,
    ) -> jax.Array:
        # Two bang-bang sequences: (+u, -u) and (-u, +u). dp = A t1^2 + B t1 + C per sequence.
        a1_seq = jnp.array([u_max_ax, -u_max_ax])
        a2_seq = jnp.array([-u_max_ax, u_max_ax])
        A = a1_seq * (a2_seq - a1_seq) / (2.0 * a2_seq)
        B = v0_ax * (a2_seq - a1_seq) / a2_seq
        C = (vf_ax**2 - v0_ax**2) / (2.0 * a2_seq) - dp_ax

        disc = B**2 - 4.0 * A * C
        valid_disc = disc >= 0.0
        sq = jnp.sqrt(jnp.maximum(disc, 0.0))

        # 4 candidate roots for t1 (2 sequences x 2 quadratic roots).
        t1_cands = jnp.array(
            [
                (-B[0] + sq[0]) / (2.0 * A[0]),
                (-B[0] - sq[0]) / (2.0 * A[0]),
                (-B[1] + sq[1]) / (2.0 * A[1]),
                (-B[1] - sq[1]) / (2.0 * A[1]),
            ]
        )
        valid_disc_cands = jnp.array([valid_disc[0], valid_disc[0], valid_disc[1], valid_disc[1]])
        a1_cands = jnp.array([a1_seq[0], a1_seq[0], a1_seq[1], a1_seq[1]])
        a2_cands = jnp.array([a2_seq[0], a2_seq[0], a2_seq[1], a2_seq[1]])

        t2_cands = (vf_ax - v0_ax - a1_cands * t1_cands) / a2_cands
        valid_bb = valid_disc_cands & (t1_cands >= -1e-4) & (t2_cands >= -1e-4)
        t1_cands = jnp.maximum(t1_cands, 0.0)
        t2_cands = jnp.maximum(t2_cands, 0.0)
        T_bb = jnp.where(valid_bb, t1_cands + t2_cands, jnp.inf)

        # Bang-singular-bang: insert a velocity-capped cruise when the peak exceeds v_cap (eq. 24).
        v_peak = v0_ax + a1_cands * t1_cands
        needs_cap = valid_bb & (jnp.abs(v_peak) > v_cap_ax)
        v_sat = jnp.sign(v_peak) * v_cap_ax
        v_sat_safe = jnp.where(jnp.abs(v_sat) < 1e-6, 1e-6, v_sat)
        a_acc = jnp.where(v_sat >= v0_ax, u_max_ax, -u_max_ax)
        a_dec = jnp.where(vf_ax >= v_sat, u_max_ax, -u_max_ax)
        t1_c = (v_sat - v0_ax) / a_acc
        t3_c = (vf_ax - v_sat) / a_dec
        d1 = (v_sat**2 - v0_ax**2) / (2.0 * a_acc)
        d3 = (vf_ax**2 - v_sat**2) / (2.0 * a_dec)
        t2_c = (dp_ax - d1 - d3) / v_sat_safe
        valid_cap = (t1_c >= -1e-4) & (t2_c >= -1e-4) & (t3_c >= -1e-4)
        T_cap = jnp.where(
            valid_cap,
            jnp.maximum(t1_c, 0.0) + jnp.maximum(t2_c, 0.0) + jnp.maximum(t3_c, 0.0),
            jnp.inf,
        )

        return jnp.min(jnp.where(needs_cap, T_cap, T_bb))

    return jnp.max(jax.vmap(_solve_axis)(dp, v0, vf, u_max, v_cap))


_jax_edge_cost = jax.jit(
    jax.vmap(
        jax.vmap(_jax_edge_cost_base, in_axes=(None, None, 0, 0, None, None)),
        in_axes=(0, 0, None, None, None, None),
    )
)


class _Axis1D:
    """Return a one-dimensional acceleration profile."""

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
    """Return a one-dimensional cubic profile used as a fallback for slack axes."""

    def __init__(self, p0: float, v0: float, pf: float, vf: float, T: float) -> None:
        self.p0, self.v0, self.T = float(p0), float(v0), float(T)
        P = pf - p0 - v0 * T
        V = vf - v0
        self._c = 3.0 * P / T**2 - V / T
        self._d = V / T**2 - 2.0 * P / T**3

    def state_at(self, t: float) -> tuple[float, float]:
        """Return (position, velocity) at time t (clamped to [0, T])."""
        t = float(np.clip(t, 0.0, self.T))
        p = self.p0 + self.v0 * t + self._c * t * t + self._d * t**3
        v = self.v0 + 2.0 * self._c * t + 3.0 * self._d * t * t
        return p, v


def _quad_roots(a: float, b: float, c: float) -> list[float]:
    """Return the real roots of a quadratic polynomial."""
    if abs(a) < 1e-12:
        return [] if abs(b) < 1e-12 else [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return []
    sq = np.sqrt(disc)
    return [(-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a)]


def _two_phase_time(
    p0: float, v0: float, pf: float, vf: float, a1: float, a2: float
) -> tuple[float, float, float] | None:
    """Return the two-phase profile for one axis."""
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
    """Return the minimum-time profile for one axis."""
    best, best_a = None, None
    for a1, a2 in ((u_hi, u_lo), (u_lo, u_hi)):
        sol = _two_phase_time(p0, v0, pf, vf, a1, a2)
        if sol is not None and (best is None or sol[0] < best[0]):
            best, best_a = sol, (a1, a2)

    if best is None:  # no feasible bang-bang (should not happen for finite bounds); hold position
        return _Axis1D(p0, v0, [(0.0, 0.0)])

    T, t1, t2 = best
    a1, a2 = best_a
    if v_cap is not None and abs(v0 + a1 * t1) > v_cap + 1e-9:
        cap = min_time_1d_capped(p0, v0, pf, vf, u_lo, u_hi, np.sign(v0 + a1 * t1) * v_cap)
        if cap is not None:
            return cap
    return _Axis1D(p0, v0, [(a1, t1), (a2, t2)])


def min_time_1d_capped(
    p0: float, v0: float, pf: float, vf: float, u_lo: float, u_hi: float, v_sat: float
) -> _Axis1D | None:
    """Return a capped minimum-time profile for one axis."""
    a_acc = u_hi if v_sat >= v0 else u_lo
    a_dec = u_hi if vf >= v_sat else u_lo
    if abs(a_acc) < 1e-12 or abs(a_dec) < 1e-12 or abs(v_sat) < 1e-12:
        return None

    t1 = (v_sat - v0) / a_acc
    t3 = (vf - v_sat) / a_dec
    d1 = (v_sat * v_sat - v0 * v0) / (2.0 * a_acc)
    d3 = (vf * vf - v_sat * v_sat) / (2.0 * a_dec)
    t2 = ((pf - p0) - d1 - d3) / v_sat
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
) -> _Axis1D | _CubicAxis1D:
    """Return a profile that reaches the target state in the requested time."""
    if abs(pf - p0) < 1e-9 and abs(v0) < 1e-9 and abs(vf) < 1e-9:
        return _Axis1D(p0, 0.0, [(0.0, T_target)])
    full = min_time_1d(p0, v0, pf, vf, u_lo, u_hi, v_cap)
    if full.T >= T_target - 1e-6:
        return full
    return _CubicAxis1D(p0, v0, pf, vf, T_target)


class MotionPrimitive:
    """Represent a time-optimal primitive between two point-mass states."""

    def __init__(
        self,
        p0: np.ndarray,
        v0: np.ndarray,
        pf: np.ndarray,
        vf: np.ndarray,
        u_max: np.ndarray,
        v_max: np.ndarray | None = None,
    ) -> None:
        """Initialize the primitive from the given boundary states."""
        p0 = np.asarray(p0, dtype=np.float64)
        v0 = np.asarray(v0, dtype=np.float64)
        pf = np.asarray(pf, dtype=np.float64)
        vf = np.asarray(vf, dtype=np.float64)
        u_max = np.asarray(u_max, dtype=np.float64)
        v_cap = None if v_max is None else np.asarray(v_max, dtype=np.float64)

        # Per-axis minimum times, then T* = max so all axes finish simultaneously.
        full = [
            min_time_1d(
                p0[k], v0[k], pf[k], vf[k], -u_max[k], u_max[k], None if v_cap is None else v_cap[k]
            )
            for k in range(3)
        ]
        self.T = max(ax.T for ax in full)
        self.axes: list[_Axis1D | _CubicAxis1D] = []
        self.n_cubic = 0  # non-critical axes stretched to T* with a cubic Hermite
        for k in range(3):
            if full[k].T >= self.T - 1e-9:
                self.axes.append(full[k])
            else:
                ax = fixed_time_1d(
                    p0[k],
                    v0[k],
                    pf[k],
                    vf[k],
                    -u_max[k],
                    u_max[k],
                    self.T,
                    None if v_cap is None else v_cap[k],
                )
                self.axes.append(ax)
                self.n_cubic += isinstance(ax, _CubicAxis1D)

    def state_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the position and velocity at time t."""
        pv = [ax.state_at(t) for ax in self.axes]
        return np.array([p for p, _ in pv]), np.array([v for _, v in pv])

    def sample_positions(self, n: int) -> np.ndarray:
        """Sample positions uniformly over the primitive duration."""
        return np.array([self.state_at(t)[0] for t in np.linspace(0.0, self.T, max(n, 2))])


def _cone_directions(
    rng: np.random.Generator, axis: np.ndarray, half_angle: float, n: int
) -> np.ndarray:
    """Sample unit vectors around the given axis."""
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
    """Solve a layered graph of sampled gate-crossing states."""

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
        self.u_max = u_max
        self.v_axis_cap = np.full(3, float(v_max))
        self.obs = obstacle_manager
        self.n_collision_pts = n_collision_pts
        self.collision_margin = float(collision_margin)
        self.turn_penalty_weight = float(turn_penalty_weight)
        self.max_turn_angle = float(max_turn_angle)
        self.stats: dict = {}
        rng = np.random.default_rng(seed)

        v_mag = float(v_max)
        speed_lo = speed_lo_frac * v_mag

        # Build the start layer and one crossing layer per gate.
        self.layers: list[list[dict]] = [
            [{"pos": np.asarray(start_pos, float), "vel": np.asarray(start_vel, float)}]
        ]
        for c, nrm in zip(gate_centers, gate_normals):
            states = [{"pos": c.copy(), "vel": nrm * (0.5 * (speed_lo + v_mag))}]
            if n_samples > 1:
                dirs = _cone_directions(rng, nrm, phi_max, n_samples - 1)
                speeds = rng.uniform(speed_lo, v_mag, n_samples - 1)
                states += [{"pos": c.copy(), "vel": d * s} for d, s in zip(dirs, speeds)]
            self.layers.append(states)
            if gate_exit_dist > 0.0:
                self.layers.append(
                    [
                        {
                            "pos": c
                            + gate_exit_dist * s["vel"] / (np.linalg.norm(s["vel"]) + 1e-9),
                            "vel": s["vel"].copy(),
                        }
                        for s in states
                    ]
                )

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
        """Return True when the primitive stays clear of obstacles."""
        if self.obs is None:
            return True
        pts = prim.sample_positions(self.n_collision_pts)
        return not bool(self.obs.points_in_obstacles(pts, margin=self.collision_margin).any())

    def solve(self) -> list[MotionPrimitive] | None:
        """Return the minimum-time primitive chain through the gates."""
        node_ids: dict[tuple[int, int], int] = {}
        nodes: list[dict] = []
        for li, layer in enumerate(self.layers):
            for ni, st in enumerate(layer):
                node_ids[(li, ni)] = len(nodes)
                nodes.append(st)
        sink = len(nodes)

        adj: list[list[tuple[int, float]]] = [[] for _ in range(sink + 1)]
        u_max_jnp = jnp.array(self.u_max)
        v_cap_jnp = jnp.array(self.v_axis_cap)

        for li in range(len(self.layers) - 1):
            layer_A, layer_B = self.layers[li], self.layers[li + 1]
            pA = np.asarray([a["pos"] for a in layer_A], dtype=np.float64)
            pB = np.asarray([b["pos"] for b in layer_B], dtype=np.float64)
            vA = np.asarray([a["vel"] for a in layer_A], dtype=np.float64)
            vB = np.asarray([b["vel"] for b in layer_B], dtype=np.float64)

            # Batched min-time cost on the GPU, transferred to CPU once.
            cost = np.array(
                _jax_edge_cost(
                    jnp.array(pA), jnp.array(vA), jnp.array(pB), jnp.array(vB), u_max_jnp, v_cap_jnp
                )
            )

            # Improvement 2b: penalize edges whose endpoint velocities deviate from the straight
            # chord pA->pB (sharp turns the real quadrotor tracks poorly), and hard-reject bends
            # beyond max_turn_angle. Straight center->exit edges deviate ~0 and stay unpenalized.
            uA = vA / (np.linalg.norm(vA, axis=1, keepdims=True) + 1e-9)
            uB = vB / (np.linalg.norm(vB, axis=1, keepdims=True) + 1e-9)
            chord = pB[None, :, :] - pA[:, None, :]
            chord = chord / (np.linalg.norm(chord, axis=2, keepdims=True) + 1e-9)
            ang_in = np.arccos(np.clip(np.einsum("ad,abd->ab", uA, chord), -1.0, 1.0))
            ang_out = np.arccos(np.clip(np.einsum("bd,abd->ab", uB, chord), -1.0, 1.0))
            cost = cost + self.turn_penalty_weight * (ang_in**2 + ang_out**2)
            cost[np.maximum(ang_in, ang_out) > self.max_turn_angle] = np.inf

            for ai in range(len(layer_A)):
                ua = node_ids[(li, ai)]
                for bi in range(len(layer_B)):
                    T_star = float(cost[ai, bi])
                    if np.isfinite(T_star):
                        adj[ua].append((node_ids[(li + 1, bi)], T_star))

        last = len(self.layers) - 1
        for ni in range(len(self.layers[last])):
            adj[node_ids[(last, ni)]].append((sink, 0.0))

        dist = [np.inf] * (sink + 1)
        prev: list[int | None] = [None] * (sink + 1)
        dist[0] = 0.0
        pq = [(0.0, 0)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u] + 1e-12:
                continue
            if u == sink:
                break
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v] - 1e-12:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if not np.isfinite(dist[sink]):
            return None
        self.stats = {"cost": float(dist[sink])}

        # Reconstruct the best route and check only its primitives.
        path_nodes = []
        cur = prev[sink]
        while cur is not None:
            path_nodes.append(cur)
            cur = prev[cur]
        path_nodes.reverse()

        prims: list[MotionPrimitive] = []
        for i in range(len(path_nodes) - 1):
            a, b = nodes[path_nodes[i]], nodes[path_nodes[i + 1]]
            prim = MotionPrimitive(
                a["pos"], a["vel"], b["pos"], b["vel"], self.u_max, self.v_axis_cap
            )
            if not self._edge_ok(prim):
                return None  # optimal path clips an obstacle -> trigger fallback
            prims.append(prim)
        return prims


def _as_f64(a: np.ndarray | None, cols: int) -> np.ndarray | None:
    """Return an array as float64 with the requested shape."""
    if a is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(-1) if cols == 1 else a.reshape(-1, cols)


class PointMassPlanner:
    """Expose the PMM planner through the trajectory-planner interface."""

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
        """Initialize the planner and run the first plan."""
        self._u_max = np.full(3, float(u_max)) if np.isscalar(u_max) else np.asarray(u_max, float)
        # v_max is a scalar SPEED (velocity norm), used both as the per-axis primitive cap and the
        # largest gate-crossing speed sampled, kept consistent with the MPCC's v_theta bound.
        self._v_max = float(np.max(v_max))
        self._obs = obstacle_manager
        self._n_vel_samples = int(n_vel_samples)
        self._phi_max = float(phi_max)
        self._speed_lo_frac = float(speed_lo_frac)
        self._n_eval_points = int(n_eval_points)
        self._n_path_samples = int(n_path_samples_per_seg)
        self._n_collision_pts = int(n_collision_pts)
        # Pruning margin for the PMM graph only; smaller than the MPCC's hard-constraint margin,
        # which owns the final conservative clearance. A large value over-prunes gate openings.
        self._collision_margin = float(collision_margin)
        self._min_z = float(min_z)
        self._tail = float(tail_extension)
        self._seed = int(seed)
        self._gate_exit_dist = float(gate_exit_dist)
        self._turn_penalty_weight = float(turn_penalty_weight)
        self._max_turn_angle = float(max_turn_angle)

        # Tuning snapshot so the async replanner can spawn an identically-configured planner.
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
        """Build a new path and fit it as an arc-length spline."""
        self._committed_pts = _as_f64(committed_pts, 3)
        self._committed_speeds = _as_f64(committed_speeds, 1)
        self._committed_suffix_pts = _as_f64(committed_suffix_pts, 3)
        self._committed_suffix_speeds = _as_f64(committed_suffix_speeds, 1)
        start_pos = np.asarray(start_pos, dtype=np.float64)
        gates_pos = np.asarray(gates_pos, dtype=np.float64).reshape(-1, 3)

        centers = [gates_pos[i] for i in range(len(gates_pos))]
        normals = self._gate_normals(start_pos, centers, gate_rpys)
        if start_vel is None or float(np.linalg.norm(start_vel)) < 1e-6:
            d0 = (centers[0] - start_pos) if centers else np.array([1.0, 0.0, 0.0])
            start_vel = d0 / (np.linalg.norm(d0) + 1e-9) * (self._speed_lo_frac * self._v_max)
        start_vel = np.asarray(start_vel, dtype=np.float64)

        t0 = time.perf_counter()
        has_prefix = self._committed_pts is not None and len(self._committed_pts) > 0
        logger.info(
            "plan START: gates=%d, M=%d, prefix=%s", len(centers), self._n_vel_samples, has_prefix
        )

        prims, stats = self._run_graph(start_pos, start_vel, centers, normals, prune=True)
        used_fallback = prims is None
        if used_fallback:
            # Single nominal sample per gate, no pruning, so a usable path is always returned (the
            # MPCC's soft constraints handle clearance); the path may clip obstacles.
            logger.warning(
                "graph infeasible with pruning -> single-sample no-check fallback (gates=%d)",
                len(centers),
            )
            prims, stats = self._run_graph(
                start_pos, start_vel, centers, normals, prune=False, single=True
            )

        self._build_spline_from_primitives(prims, normals[-1] if normals else None)
        logger.info(
            "plan DONE: %.1f ms, prims=%d, len=%.2f m, cost=%.3f s, fallback=%s",
            1e3 * (time.perf_counter() - t0),
            len(prims or []),
            self._s_total,
            stats.get("cost", float("nan")),
            used_fallback,
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
        prune: bool,
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
            obstacle_manager=self._obs if prune else None,
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
        """Append committed points and speeds to the path buffer."""
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
        has_prefix = self._committed_pts is not None and len(self._committed_pts) > 0
        if has_prefix:
            self._extend(pts, spd, self._committed_pts, self._committed_speeds)

        if prims and not has_prefix:
            # Without a prefix the first primitive's t=0 sample seeds the polyline; with one it
            # equals the last prefix point and is dropped below.
            p0, v0 = prims[0].state_at(0.0)
            pts.append(p0)
            spd.append(float(np.linalg.norm(v0)))
        elif not prims and not has_prefix:
            pts.append(np.zeros(3))
            spd.append(0.0)
        for prim in prims or []:
            for t in np.linspace(0.0, prim.T, max(self._n_path_samples, 2))[1:]:
                p, v = prim.state_at(t)
                pts.append(p)
                spd.append(float(np.linalg.norm(v)))

        has_suffix = self._committed_suffix_pts is not None and len(self._committed_suffix_pts) > 0
        if has_suffix:
            self._extend(pts, spd, self._committed_suffix_pts, self._committed_suffix_speeds)

        dense = np.array(pts, dtype=np.float64)
        speed_dense = np.array(spd, dtype=np.float64)

        # Tail extension past the final point so the MPCC horizon never stalls at the endpoint.
        # Skipped with a suffix, which already runs to the backbone's own tail.
        if self._tail > 0.0 and not has_suffix and len(dense) >= 2:
            tang = dense[-1] - dense[-2]
            tang = tang / (np.linalg.norm(tang) + 1e-9)
            if final_normal is not None and np.dot(tang, final_normal) < 0:
                tang = final_normal
            dense = np.vstack([dense, dense[-1] + self._tail * tang])
            speed_dense = np.append(speed_dense, speed_dense[-1])

        dense[:, 2] = np.maximum(dense[:, 2], self._min_z)  # ground clearance

        # True arc-length parameterization: cumulative chord length, drop zero-length segments,
        # then resample position and speed uniformly in arc length.
        seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        keep = np.concatenate(([True], seg > 1e-6))
        dense, speed_dense = dense[keep], speed_dense[keep]
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
        # Clip exported speed to v_max (diagonal motions can nudge the norm above the per-axis cap).
        speed_uniform = np.clip(np.interp(s_uniform, cum, speed_dense), 0.0, self._v_max)

        self._s = s_uniform
        self._s_total = total
        self._des_pos_spline = CubicSpline(s_uniform, pos_uniform)
        self._des_vel_spline = self._des_pos_spline.derivative()
        self._waypoints_pos = pos_uniform
        self._speed_profile = speed_uniform

    # -- public API consumed by attitude_mpc.py -------------------------------------------
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

    def _spawn(
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
    """Run PMM replans on a background thread without blocking the control loop."""

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
