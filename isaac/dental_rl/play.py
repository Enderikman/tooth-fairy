"""Rolls the trained suction policy out and writes rollout.json for the web viewer.

Run on the GPU VM, from the repo root, inside the `leisaac` conda env:
    export LEISAAC_ASSETS_ROOT=$HOME/hackathon/leisaac/assets
    python -m isaac.dental_rl.play --checkpoint isaac/dental_rl/runs/<run>/policy.pt --headless

One environment plays one episode; the targets are the four tooth targets (46, 43, 33, 36)
cycled over the episodes, so the recording shows the same four suction positions the MuJoCo demo
visits. Positions in the file are in the jaw_sim world frame (robot base at the origin), metres,
obtained by subtracting the Isaac environment origin.

Printed at the end: how many episodes held the tip on the target (same rule as training) and how
far the tip ended from it. Those two numbers, not the training curve, say whether it works.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

DEFAULT_EPISODES = 8

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="policy.pt (TorchScript) or model_<it>.pt",
)
parser.add_argument(
    "--episodes", type=int, default=DEFAULT_EPISODES, help="episodes to record"
)
parser.add_argument(
    "--num-envs",
    type=int,
    default=0,
    help="parallel environments (default: --episodes)",
)
parser.add_argument(
    "--out", type=str, default="rollout.json", help="rollout file to write"
)
parser.add_argument("--seed", type=int, default=0, help="seed for the environment")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json
import os
import sys
from pathlib import Path

import torch

from isaac.dental_rl.corridor import JAW_SCALE
from isaac.dental_rl.env import HOLD_STEPS, REACH_TOL_M, ROLLOUT_JOINTS
from isaac.dental_rl.train import build_runner
from jaw_sim.zones import CORRIDOR_WIDTH_MM

MM_PER_M = 1000.0
TORCHSCRIPT_SUFFIX = "policy.pt"


class RolloutRecorder:
    """Collects one episode per environment and writes the viewer's rollout.json.

    Attributes:
        env: The environment being recorded.
        episodes: Per-environment lists of step dictionaries.
    """

    def __init__(self, env):
        """Prepares empty per-environment step lists.

        Args:
            env: The DentalSuctionHoldEnv being played.
        """
        self.env = env
        self.episodes = [[] for _ in range(env.num_envs)]

    def record(self, rewards: torch.Tensor) -> None:
        """Appends the current simulator state of every environment.

        Args:
            rewards: (num_envs,) rewards of the step that has just been taken.
        """
        robot = self.env.robot
        origins = self.env.scene.env_origins
        joints = robot.data.joint_pos[:, self.env.rollout_joint_ids].cpu().numpy()
        gripper_pos = (
            (robot.data.body_pos_w[:, self.env.gripper_body_id] - origins).cpu().numpy()
        )
        gripper_quat = robot.data.body_quat_w[:, self.env.gripper_body_id].cpu().numpy()
        tip = (self.env.tip_pos_w - origins).cpu().numpy()
        blocked = self.env.blocked.cpu().numpy()
        reward = rewards.cpu().numpy()
        for index, steps in enumerate(self.episodes):
            steps.append(
                {
                    "q": [round(float(q), 6) for q in joints[index]],
                    "gripper_pos_w": [round(float(v), 6) for v in gripper_pos[index]],
                    "gripper_quat_w": [round(float(v), 6) for v in gripper_quat[index]],
                    "tip_world_m": [round(float(v), 6) for v in tip[index]],
                    "reward": round(float(reward[index]), 4),
                    "blocked": bool(blocked[index]),
                }
            )

    def write(
        self, path: Path, targets_local: torch.Tensor, tooth_targets: torch.Tensor
    ) -> None:
        """Writes the rollout contract file.

        Args:
            path: Destination file.
            targets_local: (num_envs, 3) target of every episode, jaw_sim world frame.
            tooth_targets: (4, 3) the four tooth targets, jaw_sim world frame.
        """
        success = self.env.success.cpu().numpy()
        rollout = {
            "dt": round(float(self.env.step_dt), 6),
            "joint_names": list(ROLLOUT_JOINTS),
            "jaw_scale": JAW_SCALE,
            "corridor_width_mm": CORRIDOR_WIDTH_MM,
            "targets_world_m": [
                [round(float(v), 6) for v in point]
                for point in tooth_targets.cpu().numpy()
            ],
            "episodes": [
                {
                    "target_world_m": [
                        round(float(v), 6) for v in targets_local[index].cpu().numpy()
                    ],
                    "success": bool(success[index]),
                    "steps": steps,
                }
                for index, steps in enumerate(self.episodes)
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rollout))
        print(
            f"wrote {path} ({len(self.episodes)} episodes, {len(self.episodes[0])} steps each)"
        )


def load_policy(checkpoint: str, runner):
    """Returns a callable observation -> action for the checkpoint.

    Args:
        checkpoint: Path to policy.pt (TorchScript) or to an rsl_rl model_<it>.pt.
        runner: Runner whose network is loaded when the checkpoint is a state dict.

    Returns:
        A function mapping an observation tensor to the deterministic action.
    """
    if checkpoint.endswith(TORCHSCRIPT_SUFFIX):
        policy = torch.jit.load(checkpoint, map_location=runner.device)
        policy.eval()
        return policy
    runner.load(checkpoint, load_optimizer=False)
    inference = runner.get_inference_policy(device=runner.device)
    return lambda obs: inference({"policy": obs})


def main() -> None:
    """Plays one episode per environment and writes the rollout file."""
    episodes = max(args.episodes, 1)
    num_envs = args.num_envs if args.num_envs > 0 else episodes
    runner, wrapper = build_runner(
        num_envs, args.seed, log_dir=None, device=args.device
    )
    env = wrapper.env
    policy = load_policy(args.checkpoint, runner)

    tooth_targets = env.corridor.tooth_targets()
    env.fixed_targets = tooth_targets[
        torch.arange(num_envs, device=env.device) % len(tooth_targets)
    ]
    obs, _ = env.reset()

    recorder = RolloutRecorder(env)
    # The environment resets itself on the step that reaches the time limit; stop one step earlier
    # so that the recording and the final tip error all belong to the same episode.
    for _ in range(int(env.max_episode_length) - 2):
        with torch.inference_mode():
            actions = policy(obs["policy"])
        obs, rewards, _, _, _ = env.step(actions)
        recorder.record(rewards)

    distance_mm = torch.linalg.norm(env.target_pos_w - env.tip_pos_w, dim=-1) * MM_PER_M
    success_rate = float(env.success.float().mean())
    print(
        f"episodes: {num_envs}, hold tolerance {REACH_TOL_M * MM_PER_M:.1f} mm for {HOLD_STEPS} steps"
    )
    print(f"success rate: {success_rate:.3f}")
    print(
        f"mean final tip error: {float(distance_mm.mean()):.2f} mm (demo scale, {JAW_SCALE}x real size)"
    )
    recorder.write(Path(args.out), env.fixed_targets, tooth_targets)

    env.close()
    app.close()
    sys.stdout.flush()  # os._exit skips the normal flush
    os._exit(0)  # SimulationApp.close() hangs in standalone Isaac Lab scripts


if __name__ == "__main__":
    main()
