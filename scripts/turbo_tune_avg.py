"""Trust Region Bayesian Optimization (TuRBO) for MPCC parameter tuning.

This module implements the TuRBO-1 algorithm to optimize the non-physical
weights of the Model Predictive Contouring Control (MPCC). It uses
Sample Average Approximation (SAA) across parallel environments to ensure
the resulting parameters are robust to track variations.
"""

from __future__ import annotations

import copy
import csv
import logging
import math
import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path
from typing import Any

# Essential for BoTorch/PyTorch numerical stability
os.environ["SCIPY_ARRAY_API"] = "1"
# Prevent JAX from hogging all GPU VRAM across multiple processes
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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

from lsy_drone_racing.control.attitude_mpc import AttitudeMPC
from lsy_drone_racing.utils import load_config

# ANSI Color Codes for terminal UI
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

# Define the bounds for parameters: [Q_c, Q_l, R_u, mu, R_T, log10(Z_l), log10(z_l)]
BOUNDS = torch.tensor(
    [
        [1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0],  # Minimums
        [2000.0, 2000.0, 1000.0, 10.0, 1000.0, 5.5, 5.5],  # Maximums
    ],
    dtype=torch.float64,
)


class TurboState:
    """Tracks the Trust Region state for the TuRBO algorithm."""

    def __init__(self, dim: int, batch_size: int) -> None:
        """Initializes the TuRBO state."""
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
        """Update the trust region state based on the latest batch evaluations."""
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


def evaluate_robust(
    p: np.ndarray, config: dict | Any, env: gymnasium.vector.VectorEnv
) -> dict[str, float]:
    """Evaluates ONE parameter set across ALL parallel environments for robustness."""
    num_envs = env.num_envs

    obs, info = env.reset()

    controllers = []
    for i in range(num_envs):
        drone_config = copy.deepcopy(config)
        # Broadcast the exact same parameters to every track
        drone_config.mpcc_tune = {
            "Q_c": float(p[0]),
            "Q_l": float(p[1]),
            "R_u": float(p[2]),
            "mu": float(p[3]),
            "R_T": float(p[4]),
            "Z_l": float(10 ** p[5]),
            "z_l": float(10 ** p[6]),
        }

        single_obs = {k: v[i] for k, v in obs.items()}
        single_info = {k: v[i] for k, v in info.items()}
        ctrl = AttitudeMPC(single_obs, single_info, drone_config)
        controllers.append(ctrl)

    steps_survived = np.zeros(num_envs, dtype=np.float64)
    max_theta = np.zeros(num_envs, dtype=np.float64)
    finished = np.zeros(num_envs, dtype=bool)
    fail_counts = np.zeros(num_envs, dtype=np.float64)
    done_mask = np.zeros(num_envs, dtype=bool)

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

        for i in range(num_envs):
            if not done_mask[i]:
                steps_survived[i] += 1
                target_gate = int(obs["target_gate"][i])

                if target_gate == -1:
                    finished[i] = True
                    done_mask[i] = True
                elif current_dones[i]:
                    done_mask[i] = True

    rewards = np.zeros(num_envs, dtype=np.float64)
    time_alive_arr = np.zeros(num_envs, dtype=np.float64)

    for i in range(num_envs):
        time_alive = steps_survived[i] / config.env.freq
        time_alive_arr[i] = time_alive

        fail_rate = fail_counts[i] / max(steps_survived[i], 1)
        fail_penalty = 75.0 * fail_rate

        if finished[i]:
            rewards[i] = 1000.0 - time_alive - fail_penalty
        else:
            rewards[i] = (max_theta[i] * 10.0) + time_alive - fail_penalty

    Z_l_val = 10 ** p[5]
    z_l_val = 10 ** p[6]
    mean_reward = float(np.mean(rewards))
    min_reward = float(np.min(rewards))
    success_rate = float(np.sum(finished) / num_envs)

    tqdm.write(
        f"\n--- Robust Eval [{num_envs} tracks] ---\n"
        f"Params: [Q_c:{p[0]:.0f}, Q_l:{p[1]:.0f}, R_u:{p[2]:.0f}, "
        f"mu:{p[3]:.2f}, R_T:{p[4]:.0f}, Z_l:{Z_l_val:.1e}, z_l:{z_l_val:.1e}]\n"
        f"Mean Reward: {mean_reward:.2f} | Worst: {min_reward:.2f} | "
        f"Finishes: {success_rate * 100:.0f}%\n"
        "-----------------------------"
    )

    return {
        "mean_reward": mean_reward,
        "min_reward": min_reward,
        "mean_time": float(np.mean(time_alive_arr)),
        "success_rate": success_rate,
    }


def generate_batch(
    state: TurboState,
    model: SingleTaskGP,
    x_train: Tensor,
    y_train: Tensor,
    bounds: Tensor,
    n_candidates: int,
) -> Tensor:
    """Generates the next batch of candidates strictly within the Trust Region."""
    x_center = x_train[y_train.argmax(), :].clone()
    weights = bounds[1] - bounds[0]

    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, bounds[0], bounds[1])
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, bounds[0], bounds[1])

    q_ei = qLogExpectedImprovement(model, best_f=y_train.max())
    x_next, _ = optimize_acqf(
        q_ei, bounds=torch.stack([tr_lb, tr_ub]), q=n_candidates, num_restarts=10, raw_samples=512
    )
    return x_next


def run_turbo(config: dict | Any, max_evals: int = 600, num_envs: int = 4) -> Tensor:
    """Runs the TuRBO optimization loop to find robust MPCC parameters."""
    dim = BOUNDS.shape[1]
    turbo_batch = 1  # We propose 1 point, evaluate it on num_envs, return 1 average reward

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"turbo_log_{timestamp}.csv"

    # Initialize CSV Logger
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Eval_ID",
                "Q_c",
                "Q_l",
                "R_u",
                "mu",
                "R_T",
                "Z_l",
                "z_l",
                "Mean_Reward",
                "Min_Reward",
                "Success_Rate",
                "Mean_Time",
                "TR_Length",
            ]
        )

    print(f"Initializing {num_envs} parallel environments (takes ~60s once)...")
    config.env.track.drones = [config.env.track.drones[0]]
    env = gymnasium.make_vec(
        "DroneRacing-v0",
        num_envs=num_envs,
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

    n_init = 16
    sobol = SobolEngine(dimension=dim, scramble=True)
    x_init_norm = sobol.draw(n=n_init).to(dtype=torch.float64)
    x_init = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * x_init_norm

    print(f"Evaluating initial {len(x_init)} points (averaged across {num_envs} tracks)...")

    y_init_list = []
    x_data = x_init.clone()

    # Initial Evaluations
    for i in range(len(x_init)):
        p = x_init[i].numpy()
        metrics = evaluate_robust(p, config, env)
        y_init_list.append([metrics["mean_reward"]])

        with open(log_file, mode="a", newline="") as file:
            csv.writer(file).writerow(
                [
                    i,
                    p[0],
                    p[1],
                    p[2],
                    p[3],
                    p[4],
                    10 ** p[5],
                    10 ** p[6],
                    metrics["mean_reward"],
                    metrics["min_reward"],
                    metrics["success_rate"],
                    metrics["mean_time"],
                    0.8,
                ]
            )

    y_data = torch.tensor(y_init_list, dtype=torch.float64)

    state = TurboState(dim, batch_size=turbo_batch)
    state.best_value = y_data.max().item()
    norm_bounds = torch.stack(
        [torch.zeros(dim, dtype=torch.float64), torch.ones(dim, dtype=torch.float64)]
    )

    pbar = tqdm(
        total=max_evals, initial=len(x_data), desc="TuRBO Tuning", dynamic_ncols=True, colour="CYAN"
    )
    pbar.set_postfix(best=state.best_value, tr_length=state.length)

    eval_id = len(x_data)

    while len(x_data) < max_evals:
        train_y = (y_data - y_data.mean()) / (y_data.std() + 1e-9)
        train_x = (x_data - BOUNDS[0]) / (BOUNDS[1] - BOUNDS[0])

        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        x_next_norm = generate_batch(state, model, train_x, train_y, norm_bounds, turbo_batch)
        x_next = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * x_next_norm

        p = x_next[0].numpy()
        metrics = evaluate_robust(p, config, env)

        y_next = torch.tensor([[metrics["mean_reward"]]], dtype=torch.float64)
        state.update(y_next)

        x_data = torch.cat((x_data, x_next), dim=0)
        y_data = torch.cat((y_data, y_next), dim=0)

        with open(log_file, mode="a", newline="") as file:
            csv.writer(file).writerow(
                [
                    eval_id,
                    p[0],
                    p[1],
                    p[2],
                    p[3],
                    p[4],
                    10 ** p[5],
                    10 ** p[6],
                    metrics["mean_reward"],
                    metrics["min_reward"],
                    metrics["success_rate"],
                    metrics["mean_time"],
                    state.length,
                ]
            )

        tr_color = RED if state.length < 0.05 else YELLOW
        pbar.set_postfix(
            best=f"{GREEN}{state.best_value:.2f}{RESET}",
            last_mean=f"{metrics['mean_reward']:.2f}",
            SR=f"{metrics['success_rate'] * 100:.0f}%",
            tr_length=f"{tr_color}{state.length:.3f}{RESET}",
        )
        pbar.update(1)
        eval_id += 1

        if state.restart_triggered:
            tqdm.write(
                "\nTrust Region collapsed! Injecting fresh points to escape local minimum..."
            )
            state = TurboState(dim, batch_size=turbo_batch)
            state.best_value = y_data.max().item()

            sobol = SobolEngine(dimension=dim, scramble=True)
            x_new_norm = sobol.draw(n=turbo_batch).to(dtype=torch.float64)
            x_new = BOUNDS[0] + (BOUNDS[1] - BOUNDS[0]) * x_new_norm

            p_new = x_new[0].numpy()
            new_metrics = evaluate_robust(p_new, config, env)

            y_new = torch.tensor([[new_metrics["mean_reward"]]], dtype=torch.float64)
            x_data = torch.cat((x_data, x_new), dim=0)
            y_data = torch.cat((y_data, y_new), dim=0)

            with open(log_file, mode="a", newline="") as file:
                csv.writer(file).writerow(
                    [
                        eval_id,
                        p_new[0],
                        p_new[1],
                        p_new[2],
                        p_new[3],
                        p_new[4],
                        10 ** p_new[5],
                        10 ** p_new[6],
                        new_metrics["mean_reward"],
                        new_metrics["min_reward"],
                        new_metrics["success_rate"],
                        new_metrics["mean_time"],
                        state.length,
                    ]
                )

            pbar.update(1)
            eval_id += 1

    pbar.close()
    env.close()

    best_idx = int(y_data.argmax())
    print(f"Optimization finished. Logs saved to {log_file}")
    return x_data[best_idx]


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TuRBO_Tuning")

    config_path = Path(__file__).parents[1] / "config" / "multi_level2.toml"
    config = load_config(config_path)

    if "kwargs" in config.env:
        config.env.freq = config.env.kwargs[0]["freq"]
        config.env.sensor_range = config.env.kwargs[0]["sensor_range"]
        config.env.control_mode = config.env.kwargs[0]["control_mode"]

    config.sim.gui = False
    config.sim.render = False

    logger.info("Starting TuRBO Optimization...")

    # num_envs is fully adjustable here. Defaulting to 4 for a 4-core CPU SAA.
    best_params = run_turbo(config, max_evals=600, num_envs=4)

    best_np = best_params.numpy()
    logger.info(
        f"\n========================================\n"
        f"Optimization Finished! Best Parameters:\n"
        f"  Q_c : {best_np[0]:.2f}\n"
        f"  Q_l : {best_np[1]:.2f}\n"
        f"  R_u : {best_np[2]:.2f}\n"
        f"  mu  : {best_np[3]:.2f}\n"
        f"  R_T : {best_np[4]:.2f}\n"
        f"  Z_l : {10 ** best_np[5]:.2e}\n"
        f"  z_l : {10 ** best_np[6]:.2e}\n"
        f"========================================"
    )
