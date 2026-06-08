"""Trust Region Bayesian Optimization (TuRBO) for MPCC parameter tuning.

This module implements the TuRBO-1 algorithm to optimize the non-physical
weights of the Model Predictive Contouring Control (MPCC). The implementation
and objective function follow the methodology described by Krinner et al.
"""

from __future__ import annotations

import os

os.environ["SCIPY_ARRAY_API"] = "1"

import copy
import math
from typing import TYPE_CHECKING, Any

import gymnasium
import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine

# Import your controller
from lsy_drone_racing.control.attitude_mpc import AttitudeMPC

if TYPE_CHECKING:
    from ml_collections import ConfigDict

# Define the bounds for your parameters: [Q_c, Q_l, R_u, mu]
# Adjust these based on the physical limits and stability of your quadrotor.
BOUNDS = torch.tensor(
    [[1.0, 1.0, 1.0, 0.01, 1.0], [1000.0, 5000.0, 2000.0, 10.0, 1000.0]], dtype=torch.float64
)


class TurboState:
    """Tracks the Trust Region state for the TuRBO algorithm.

    This state expands or contracts the search region based on the
    recent success or failure of parameter evaluations.

    Attributes:
        dim: Dimensionality of the parameter space.
        batch_size: Number of candidate points evaluated per batch.
        length: Current length of the trust region.
        length_min: Minimum allowable length before triggering a restart.
        length_max: Maximum allowable length.
        failure_counter: Number of consecutive failed batches.
        failure_tolerance: Threshold of failures to shrink the trust region.
        success_counter: Number of consecutive successful batches.
        success_tolerance: Threshold of successes to expand the trust region.
        best_value: The highest reward found so far.
        restart_triggered: Boolean flag indicating if the region has collapsed.
    """

    def __init__(self, dim: int, batch_size: int) -> None:
        """Initialize the TuRBO state."""
        self.dim = dim
        self.batch_size = batch_size
        self.length = 0.8
        self.length_min = 0.5**7
        self.length_max = 1.6
        self.failure_counter = 0
        self.failure_tolerance = math.ceil(dim / batch_size)
        self.success_counter = 0
        self.success_tolerance = 2
        self.best_value = -float("inf")
        self.restart_triggered = False

    def update(self, y_next: Tensor) -> None:
        """Update the trust region state based on the latest batch evaluations.

        Args:
            y_next: A tensor of rewards from the latest evaluated batch.
        """
        if y_next.max().item() > self.best_value + 1e-3:
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter == self.success_tolerance:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:
            self.length /= 2.0
            self.failure_counter = 0

        self.best_value = max(self.best_value, y_next.max().item())
        if self.length < self.length_min:
            self.restart_triggered = True


def evaluate_batch(params_batch: np.ndarray, config: ConfigDict | dict[str, Any]) -> np.ndarray:
    """Evaluates a batch of parameters in parallel using isolated environments."""
    batch_size = params_batch.shape[0]
    config.env.track.drones = [config.env.track.drones[0]]

    # Create completely independent parallel worlds
    env = gymnasium.make_vec(
        "DroneRacing-v0",
        num_envs=batch_size,
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        sensor_range=config.env.sensor_range,
        control_mode="attitude",
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )

    obs, info = env.reset()

    controllers = []
    for i in range(batch_size):
        drone_config = copy.deepcopy(config)

        drone_config.mpcc_tune = {
            "Q_c": float(params_batch[i, 0]),
            "Q_l": float(params_batch[i, 1]),
            "R_u": float(params_batch[i, 2]),
            "mu": float(params_batch[i, 3]),
            "R_T": float(params_batch[i, 4]),
        }

        # VecDroneRaceEnv returns arrays of shape (batch_size, ...)
        single_obs = {k: v[i] for k, v in obs.items()}
        single_info = {k: v[i] for k, v in info.items()}

        ctrl = AttitudeMPC(single_obs, single_info, drone_config)
        controllers.append(ctrl)

    lap_times = np.zeros(batch_size, dtype=np.float64)
    fail_counts = np.zeros(batch_size, dtype=np.float64)
    total_steps = 0
    done_mask = np.zeros(batch_size, dtype=bool)

    while not np.all(done_mask):
        actions = []
        for i, ctrl in enumerate(controllers):
            if not done_mask[i]:
                single_obs = {k: v[i] for k, v in obs.items()}
                single_info = {k: v[i] for k, v in info.items()}

                action = ctrl.compute_control(single_obs, single_info)

                if ctrl.last_solver_status != 0:
                    fail_counts[i] += 1

                actions.append(action)
            else:
                actions.append(np.zeros(4, dtype=np.float32))

        # Shape: (batch_size, 4)
        actions_arr = np.array(actions, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(actions_arr)

        if config.sim.get("gui", False) or config.sim.get("render", False):
            try:
                env.render()
            except Exception as e:
                if not str(e).startswith("No known conversion for Jax type"):
                    raise e

        # In VecDroneRaceEnv, these are arrays of shape (batch_size,)
        current_dones = terminated | truncated
        total_steps += 1

        for i in range(batch_size):
            if not done_mask[i]:
                # 1. Drone crossed finish line
                if int(obs["target_gate"][i]) == -1:
                    lap_times[i] = total_steps / config.env.freq
                    done_mask[i] = True
                # 2. Drone crashed or timed out
                elif current_dones[i]:
                    done_mask[i] = True

    env.close()

    gamma = 100.0
    r_fail = fail_counts / max(total_steps, 1)

    max_time = total_steps / config.env.freq
    lap_times[lap_times == 0] = max_time

    print("\n--- Batch Results ---")
    print(f"Lap Times (s): {np.round(lap_times, 2)}")
    print(f"Solver Fails : {fail_counts}")
    print(f"Rewards      : {np.round(-lap_times - (gamma * r_fail), 2)}")
    print("---------------------\n")

    rewards = -lap_times - (gamma * r_fail)
    return rewards


def generate_batch(
    state: TurboState,
    model: SingleTaskGP,
    x_train: Tensor,
    y_train: Tensor,
    bounds: Tensor,
    n_candidates: int,
) -> Tensor:
    """Generates the next batch of candidates strictly within the Trust Region.

    Args:
        state: The current TuRBO state.
        model: The fitted Gaussian Process surrogate model.
        x_train: Normalized training inputs.
        y_train: Standardized training outputs (rewards).
        bounds: Tensor of shape (2, dim) representing normalized [0, 1] bounds.
        n_candidates: Number of points to generate.

    Returns:
        A tensor of shape (n_candidates, dim) containing normalized candidate parameters.
    """
    x_center = x_train[y_train.argmax(), :].clone()
    weights = bounds[1] - bounds[0]

    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, bounds[0], bounds[1])
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, bounds[0], bounds[1])

    # Use LogEI instead of standard EI
    q_ei = qLogExpectedImprovement(model, best_f=y_train.max())

    x_next, _ = optimize_acqf(
        q_ei, bounds=torch.stack([tr_lb, tr_ub]), q=n_candidates, num_restarts=10, raw_samples=512
    )
    return x_next


def run_turbo(
    config: ConfigDict | dict[str, Any], max_evals: int = 600, batch_size: int = 8
) -> Tensor:
    """Executes the full TuRBO optimization loop.

    Args:
        config: The simulation configuration dictionary.
        max_evals: Maximum number of environment evaluations.
        batch_size: Number of parallel evaluations per TuRBO iteration.

    Returns:
        The optimized parameter tensor found during the run.
    """
    dim = BOUNDS.shape[1]

    sobol = SobolEngine(dimension=dim, scramble=True)
    x_init_norm = sobol.draw(n=16).to(dtype=torch.float64)
    x_init = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * x_init_norm

    print(f"Evaluating initial {len(x_init)} points...")
    y_init_np = evaluate_batch(x_init.numpy(), config)
    y_init = torch.tensor(y_init_np, dtype=torch.float64).unsqueeze(-1)

    x_data = x_init
    y_data = y_init

    state = TurboState(dim, batch_size=batch_size)
    state.best_value = y_data.max().item()

    norm_bounds = torch.stack(
        [torch.zeros(dim, dtype=torch.float64), torch.ones(dim, dtype=torch.float64)]
    )

    while len(x_data) < max_evals:
        # Standardize outputs and normalize inputs
        train_y = (y_data - y_data.mean()) / (y_data.std() + 1e-9)
        train_x = (x_data - BOUNDS[0]) / (BOUNDS[1] - BOUNDS[0])

        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        x_next_norm = generate_batch(state, model, train_x, train_y, norm_bounds, batch_size)
        x_next = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * x_next_norm

        y_next_np = evaluate_batch(x_next.numpy(), config)
        y_next = torch.tensor(y_next_np, dtype=torch.float64).unsqueeze(-1)

        state.update(y_next)
        x_data = torch.cat((x_data, x_next), dim=0)
        y_data = torch.cat((y_data, y_next), dim=0)

        print(
            f"Evals: {len(x_data)} | Best Reward: {state.best_value:.4f} | "
            f"TR Length: {state.length:.3f}"
        )

        if state.restart_triggered:
            print("Trust Region collapsed. Triggering restart...")
            break

    best_idx = y_data.argmax()
    print("Optimization finished.")
    print("Best Parameters:", x_data[best_idx].numpy())

    return x_data[best_idx]


if __name__ == "__main__":
    import logging
    from pathlib import Path

    from lsy_drone_racing.utils import load_config

    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TuRBO_Tuning")

    # Path to the multi-drone configuration file
    config_path = Path(__file__).parents[1] / "config" / "multi_level2.toml"

    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # --- FIX: Patch multi-drone config structure to match single-drone expectations ---
    if "kwargs" in config.env:
        config.env.freq = config.env.kwargs[0]["freq"]
        config.env.sensor_range = config.env.kwargs[0]["sensor_range"]
        config.env.control_mode = config.env.kwargs[0]["control_mode"]

    # Force GUI to false during high-speed parallel tuning to save compute
    config.sim.gui = True
    config.sim.render = True

    logger.info("Starting TuRBO Optimization...")

    # Run the optimizer
    # Ensure batch_size matches or is a factor of the number of parallel workers you want
    best_params = run_turbo(config, max_evals=600, batch_size=2)

    logger.info("========================================")
    logger.info("Optimization Finished!")
    logger.info(f"Best Parameters [Q_c, Q_l, R_u, mu, R_T]:\n{best_params.numpy()}")
    logger.info("========================================")
