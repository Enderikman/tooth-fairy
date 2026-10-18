"""Trains the "dental-suction-hold" policy with PPO (rsl_rl) in Isaac Lab.

Run on the GPU VM, from the repo root, inside the `leisaac` conda env:
    export LEISAAC_ASSETS_ROOT=$HOME/hackathon/leisaac/assets
    python -m isaac.dental_rl.train --headless                       2048 envs, 1500 iterations
    python -m isaac.dental_rl.train --headless --num-envs 64 --max-iterations 5     smoke test

Each run writes to isaac/dental_rl/runs/<timestamp>/: tensorboard logs, model_<iteration>.pt
checkpoints and, at the end, policy.pt (TorchScript actor, observation -> mean action).

A loss curve proves nothing about the task; read Episode/success_rate in tensorboard and then
run isaac/dental_rl/play.py, which measures how close the tip actually gets.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    """Command line of this script plus Isaac Lab's app launcher options."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--num-envs", type=int, default=2048, help="parallel environments"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=1500, help="PPO iterations"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for torch, numpy and the environment"
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


# Isaac modules may only be imported once the app runs, so it is started here, above the imports.
# play.py imports build_runner from this file; it has started its own app by then and parses its
# own command line, which is why both only happen when this file is the script being run.
if __name__ == "__main__":
    args = parse_args()
    app = AppLauncher(args).app

import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from isaac.dental_rl.env import (
    OBS_DIM,
    DentalSuctionHoldEnv,
    DentalSuctionHoldEnvCfg,
    RslRlVecEnv,
)

RUNS_DIR = Path(__file__).resolve().parent / "runs"
POLICY_FILE = "policy.pt"
SAVE_INTERVAL = 50  # iterations between checkpoints

# PPO setup, following the Isaac Lab manipulation defaults with a smaller network: the observation
# is 24 numbers of robot state, not an image, and the arm has five joints.
TRAIN_CFG = {
    "num_steps_per_env": 24,
    "save_interval": SAVE_INTERVAL,
    "logger": "tensorboard",
    "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
    "policy": {
        "class_name": "ActorCritic",
        "actor_hidden_dims": [256, 128, 64],
        "critic_hidden_dims": [256, 128, 64],
        "activation": "elu",
        "init_noise_std": 1.0,
    },
    "algorithm": {
        "class_name": "PPO",
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "clip_param": 0.2,
        "gamma": 0.99,
        "lam": 0.95,
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "entropy_coef": 0.005,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    },
}


def build_runner(
    num_envs: int, seed: int, log_dir: str | None, device: str
) -> tuple[OnPolicyRunner, RslRlVecEnv]:
    """Creates the environment and an rsl_rl runner for it.

    Args:
        num_envs: Number of parallel environments.
        seed: Seed for the environment and the networks.
        log_dir: Directory for tensorboard logs and checkpoints, or None for no logging.
        device: Simulation and training device, e.g. "cuda:0".

    Returns:
        The runner and the wrapped environment.
    """
    cfg = DentalSuctionHoldEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    env = RslRlVecEnv(DentalSuctionHoldEnv(cfg))
    runner = OnPolicyRunner(env, dict(TRAIN_CFG), log_dir=log_dir, device=device)
    return runner, env


def export_policy(runner: OnPolicyRunner, path: Path) -> None:
    """Saves the actor as a TorchScript module taking one observation tensor.

    Args:
        runner: Runner holding the trained policy.
        path: File to write (policy.pt).
    """
    actor = runner.alg.policy.actor.eval()
    example = torch.zeros(1, OBS_DIM, device=runner.device)
    torch.jit.save(torch.jit.trace(actor, example), path)
    print(f"exported {path}")


def main() -> None:
    """Runs PPO for --max-iterations and exports the trained actor."""
    log_dir = RUNS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"run directory: {log_dir}")

    runner, env = build_runner(args.num_envs, args.seed, str(log_dir), args.device)
    runner.learn(
        num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
    )
    export_policy(runner, log_dir / POLICY_FILE)
    env.close()
    app.close()
    sys.stdout.flush()  # os._exit skips the normal flush
    os._exit(0)  # SimulationApp.close() hangs in standalone Isaac Lab scripts


if __name__ == "__main__":
    main()
