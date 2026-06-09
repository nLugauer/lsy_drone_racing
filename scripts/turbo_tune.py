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
from typing import Any

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
from tqdm import tqdm

# Import your controller
from lsy_drone_racing.control.attitude_mpc import AttitudeMPC

# Define the bounds for your parameters: [Q_c, Q_l, R_u, mu, R_T, log10(Z_l), log10(z_l)]
# Adjust these based on the physical limits and stability of your quadrotor.
BOUNDS = torch.tensor(
    [
        [1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0],  # Minimums
        [1000.0, 1000.0, 1000.0, 5.0, 500.0, 5.0, 5.0],  # Maximums
    ],
    dtype=torch.float64,
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


def evaluate_batch(
    params_batch: np.ndarray, config: dict | Any, env: gymnasium.vector.VectorEnv
) -> np.ndarray:
    """Evaluates a batch of parameters using the ALREADY RUNNING parallel environments."""
    batch_size = params_batch.shape[0]

    # We no longer create the env here. We just reset the one passed in.
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
            "Z_l": float(10 ** params_batch[i, 5]),
            "z_l": float(10 ** params_batch[i, 6]),
        }

        single_obs = {k: v[i] for k, v in obs.items()}
        single_info = {k: v[i] for k, v in info.items()}
        ctrl = AttitudeMPC(single_obs, single_info, drone_config)
        controllers.append(ctrl)

    # Tracking variables
    steps_survived = np.zeros(batch_size, dtype=np.float64)
    max_theta = np.zeros(batch_size, dtype=np.float64)
    finished = np.zeros(batch_size, dtype=bool)
    fail_counts = np.zeros(batch_size, dtype=np.float64)

    total_steps = 0
    done_mask = np.zeros(batch_size, dtype=bool)
    num_total_gates = len(config.env.track.gates)

    while not np.all(done_mask):
        actions = []
        for i, ctrl in enumerate(controllers):
            if not done_mask[i]:
                single_obs = {k: v[i] for k, v in obs.items()}
                single_info = {k: v[i] for k, v in info.items()}
                action = ctrl.compute_control(single_obs, single_info)
                if ctrl.last_solver_status != 0:
                    fail_counts[i] += 1

                max_theta[i] = max(max_theta[i], ctrl._current_theta)
                actions.append(action)
            else:
                actions.append(np.zeros(4, dtype=np.float32))

        actions_arr = np.array(actions, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(actions_arr)

        current_dones = terminated | truncated
        total_steps += 1

        for i in range(batch_size):
            if not done_mask[i]:
                steps_survived[i] += 1
                target_gate = int(obs["target_gate"][i])

                if target_gate == -1:
                    finished[i] = True
                    done_mask[i] = True
                elif current_dones[i]:
                    done_mask[i] = True

    # Calculate augmented rewards
    rewards = np.zeros(batch_size, dtype=np.float64)
    for i in range(batch_size):
        time_alive = steps_survived[i] / config.env.freq
        fail_rate = fail_counts[i] / max(steps_survived[i], 1)
        fail_penalty = 75.0 * fail_rate

        if finished[i]:
            rewards[i] = 1000.0 - time_alive - fail_penalty
        else:
            rewards[i] = (max_theta[i] * 10.0) + time_alive - fail_penalty

    log_lines = ["\n--- Batch Results ---"]
    for i in range(batch_size):
        p = params_batch[i]
        # Convert the log-scale parameters back to linear for printing
        Z_l_val = 10 ** p[5]
        z_l_val = 10 ** p[6]

        log_lines.append(
            f"Env {i} Params [Q_c, Q_l, R_u, mu, R_T, Z_l, z_l]: "
            f"[{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}, {p[3]:.2f}, {p[4]:.1f}, {Z_l_val:.1e}, {z_l_val:.1e}] "
            f"-> Reward: {rewards[i]:.2f} (Theta: {max_theta[i]:.2f}m, Finished: {finished[i]})"
        )
    log_lines.append("---------------------")
    tqdm.write("\n".join(log_lines))

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


def run_turbo(config: dict | Any, max_evals: int = 600, batch_size: int = 4) -> Tensor:
    """Runs the TuRBO optimization loop to find the best MPCC parameters."""
    dim = BOUNDS.shape[1]

    print("Initializing parallel environments (this takes ~60s once)...")
    config.env.track.drones = [config.env.track.drones[0]]
    env = gymnasium.make_vec(
        "DroneRacing-v0",
        num_envs=batch_size,
        vectorization_mode="async",
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        sensor_range=config.env.sensor_range,
        control_mode="attitude",
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )

    sobol = SobolEngine(dimension=dim, scramble=True)
    x_init_norm = sobol.draw(n=16).to(dtype=torch.float64)
    x_init = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * x_init_norm

    print(f"Evaluating initial {len(x_init)} points...")

    y_init_list = []
    x_init_np = x_init.numpy()
    for i in range(0, len(x_init_np), batch_size):
        chunk = x_init_np[i : i + batch_size]
        # PASS ENV HERE
        chunk_results = evaluate_batch(chunk, config, env)
        y_init_list.append(chunk_results)

    y_init_np = np.concatenate(y_init_list)
    y_init = torch.tensor(y_init_np, dtype=torch.float64).unsqueeze(-1)

    x_data = x_init
    y_data = y_init

    state = TurboState(dim, batch_size=batch_size)
    state.best_value = y_data.max().item()

    norm_bounds = torch.stack(
        [torch.zeros(dim, dtype=torch.float64), torch.ones(dim, dtype=torch.float64)]
    )

    pbar = tqdm(
        total=max_evals, initial=len(x_data), desc="TuRBO Tuning", dynamic_ncols=True, colour="CYAN"
    )
    pbar.set_postfix(best=state.best_value, tr_length=state.length)

    while len(x_data) < max_evals:
        train_y = (y_data - y_data.mean()) / (y_data.std() + 1e-9)
        train_x = (x_data - BOUNDS[0]) / (BOUNDS[1] - BOUNDS[0])

        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        x_next_norm = generate_batch(state, model, train_x, train_y, norm_bounds, batch_size)
        x_next = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * x_next_norm

        # PASS ENV HERE
        y_next_np = evaluate_batch(x_next.numpy(), config, env)
        y_next = torch.tensor(y_next_np, dtype=torch.float64).unsqueeze(-1)

        state.update(y_next)
        x_data = torch.cat((x_data, x_next), dim=0)
        y_data = torch.cat((y_data, y_next), dim=0)

        current_max = float(y_next.max().item())
        pbar.set_postfix(
            best=f"{state.best_value:.2f}",
            batch_max=f"{current_max:.2f}",
            tr_length=f"{state.length:.3f}",
        )
        pbar.update(batch_size)

        if state.restart_triggered:
            tqdm.write(
                "\nTrust Region collapsed! Injecting fresh points to escape local minimum..."
            )

            # 1. Reset the Trust Region state
            state = TurboState(dim, batch_size=batch_size)
            state.best_value = y_data.max().item()

            # 2. Draw a fresh batch of random points to force exploration
            sobol = SobolEngine(dimension=dim, scramble=True)
            x_new_norm = sobol.draw(n=batch_size).to(dtype=torch.float64)
            x_new = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * x_new_norm

            y_new_np = evaluate_batch(x_new.numpy(), config, env)

            # 3. Evaluate and append the fresh points
            y_new_np, _ = evaluate_batch(x_new.numpy(), config, env)
            y_new = torch.tensor(y_new_np, dtype=torch.float64).unsqueeze(-1)

            x_data = torch.cat((x_data, x_new), dim=0)
            y_data = torch.cat((y_data, y_new), dim=0)

    pbar.close()

    env.close()

    best_idx = y_data.argmax()
    print("Optimization finished.")
    print("Best Parameters:", x_data[best_idx].numpy())

    return x_data[best_idx]


if __name__ == "__main__":
    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import multiprocessing as mp

    # Force Python to spawn fresh processes instead of forking
    mp.set_start_method("spawn", force=True)

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
    config.sim.gui = False
    config.sim.render = False

    logger.info("Starting TuRBO Optimization...")

    # Run the optimizer
    # Ensure batch_size matches or is a factor of the number of parallel workers you want
    best_params = run_turbo(config, max_evals=600, batch_size=4)

    logger.info("========================================")
    logger.info("Optimization Finished!")

    best_np = best_params.numpy()
    Z_l_val = 10 ** best_np[5]
    z_l_val = 10 ** best_np[6]

    logger.info(
        f"Best Parameters:\n"
        f"  Q_c : {best_np[0]:.2f}\n"
        f"  Q_l : {best_np[1]:.2f}\n"
        f"  R_u : {best_np[2]:.2f}\n"
        f"  mu  : {best_np[3]:.2f}\n"
        f"  R_T : {best_np[4]:.2f}\n"
        f"  Z_l : {Z_l_val:.2e}\n"
        f"  z_l : {z_l_val:.2e}"
    )
    logger.info("========================================")
