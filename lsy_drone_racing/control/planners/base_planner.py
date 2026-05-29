"""Base class for all trajectory planners."""

from abc import ABC, abstractmethod

import numpy as np


class BasePlanner(ABC):
    """Abstract base class ensuring a unified interface for all trajectory planners."""

    def __init__(self, freq: int):
        """Initialize the base planner with a reference generation frequency."""
        self.freq = freq
        self.max_ticks = 0

    @abstractmethod
    def plan_trajectory(self) -> None:
        """Executes the planning algorithm to generate the trajectory arrays."""
        pass

    @abstractmethod
    def get_references(
        self, current_tick: int, horizon: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Returns the reference slice for the prediction horizon.

        Returns:
            pos_ref: (N, 3) array of positions
            vel_ref: (N, 3) array of velocities
            yaw_ref: (N,) array of yaws
            pos_e: (3,) terminal position
            vel_e: (3,) terminal velocity
            yaw_e: float terminal yaw
        """
        pass
