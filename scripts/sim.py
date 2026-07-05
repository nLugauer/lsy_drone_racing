"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller

# Make the repo-root `analysis/` package importable regardless of the cwd
# from which sim.py is launched.
sys.path.insert(0, str(Path(__file__).parents[1]))
from analysis.sim_logging import SimRecorder  # noqa: E402

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv


logger = logging.getLogger(__name__)


def simulate(
    config: str = "level3.toml",
    controller: str | None = None,
    n_runs: int = 1,
    render: bool | None = None,
    log_tag: str | None = None,
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
        n_runs: The number of episodes.
        render: Enable/disable rendering the simulation.
        log_tag: If set, telemetry for every episode is written to
            `results/<log_tag>_lvl<L>_seed<S>_run<i>/` for the analysis
            pipeline (see analysis/). Use a label such as "pmm" or "cubic"
            so runs can be compared. If None, no telemetry is written.

    Returns:
        A list of episode times.
    """
    # Remember the config filename (for the run label) before it is replaced
    # by the loaded config object below.
    config_name = config
    # Load configuration and check if firmare should be used.
    config = load_config(Path(__file__).parents[1] / "config" / config)
    if render is None:
        render = config.sim.render
    else:
        config.sim.render = render
    # Load the controller module
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)  # This returns a class, not an instance
    # Create the racing environment
    env: DroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
        max_episode_steps=5000,
    )
    env = JaxToNumpy(env)

    ep_times = []
    for run_idx in range(n_runs):  # Run n_runs episodes with the controller
        obs, info = env.reset()
        controller: Controller = controller_cls(obs, info, config)
        recorder = (
            SimRecorder(config, config_name, run_idx=run_idx, tag=log_tag)
            if log_tag is not None
            else None
        )
        i = 0
        fps = 6000

        while True:
            curr_time = i / config.env.freq

            t_ctrl = time.perf_counter()
            action = controller.compute_control(obs, info)
            control_ms = (time.perf_counter() - t_ctrl) * 1e3

            obs, reward, terminated, truncated, info = env.step(action)
            if recorder is not None:
                recorder.record_tick(curr_time, obs, control_ms)
            # Update the controller internal state and models.
            controller_finished = controller.step_callback(
                action, obs, reward, terminated, truncated, info
            )
            # Add up reward, collisions
            if terminated or truncated or controller_finished:
                break
            if config.sim.render:  # Render the sim if selected.
                if ((i * fps) % config.env.freq) < fps:
                    controller.render_callback(env.unwrapped.sim)
                    env.render()
                    time.sleep(1.0 / fps)
            i += 1

        # Write telemetry BEFORE episode_callback(), which clears the
        # controller's _log_* lists.
        if recorder is not None:
            recorder.finish(controller, obs, curr_time)
        controller.episode_callback()  # Update the controller internal state and models.
        log_episode_stats(obs, info, config, curr_time)
        controller.episode_reset()
        ep_times.append(curr_time if obs["target_gate"] == -1 else None)

    # Close the environment
    env.close()
    return ep_times


def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:  # The drone has passed the final gate
        gates_passed = len(config.env.track.gates)
    finished = gates_passed == len(config.env.track.gates)
    logger.info(
        f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {gates_passed}\n"
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
