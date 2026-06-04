"""Obstacle handling helper for MPC contour weight adjustment.

This module contains a future-ready interface for obstacle awareness and the
current gate-based contour weighting logic used by the attitude MPC.
"""

from __future__ import annotations

import numpy as np


class ObstacleManager:
    """Manages obstacle-related cost shaping for the MPC solver."""

    def __init__(
        self,
        gate_positions: np.ndarray,
        q_nom: float = 1.0,
        q_wp: float = 300.0,
        sigma: float = 0.4,
    ) -> None:
        """Initialize the obstacle manager."""
        self._gate_positions = gate_positions
        self._q_nom = q_nom
        self._q_wp = q_wp
        self._sigma_sq = sigma**2

    def dynamic_contour_weight(self, position: np.ndarray) -> float:
        """Compute a dynamic contour weight based on gate proximity."""
        q_c = self._q_nom
        for gate_pos in self._gate_positions:
            dist_sq = np.sum((position - gate_pos) ** 2)
            q_c += self._q_wp * np.exp(-0.5 * dist_sq / self._sigma_sq)
        return float(q_c)

    def predict_future_obstacles(self, state: np.ndarray, horizon: int) -> np.ndarray:
        """Placeholder for future obstacle predictions.

        The current implementation does not yet use a dynamic prediction model,
        but this method defines the interface for future extension.
        """
        return np.empty((0, 3))
