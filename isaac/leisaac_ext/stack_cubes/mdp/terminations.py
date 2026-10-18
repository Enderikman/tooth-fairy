from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def cubes_stacked(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfgs: list[SceneEntityCfg],
    cube_size: float = 0.03,
    table_z: float = 0.041,
    xy_threshold: float = 0.015,
    z_threshold: float = 0.008,
    max_speed: float = 0.05,
) -> torch.Tensor:
    """True where all cubes form one vertical tower, in any order, resting on the table.

    Cubes are sorted by height per env; each must sit one cube_size above the one below it,
    horizontally within xy_threshold, the lowest must rest on the table, and none may be moving.
    Positions are root (centre) positions relative to the env origin.
    """
    cubes: list[RigidObject] = [env.scene[cfg.name] for cfg in cube_cfgs]
    pos = torch.stack([c.data.root_pos_w for c in cubes], dim=1) - env.scene.env_origins[:, None, :]
    speed = torch.stack([torch.linalg.vector_norm(c.data.root_lin_vel_w, dim=-1) for c in cubes], dim=1)

    order = torch.argsort(pos[..., 2], dim=1)
    pos = torch.gather(pos, 1, order.unsqueeze(-1).expand(-1, -1, 3))
    lower, upper = pos[:, :-1], pos[:, 1:]

    aligned = torch.linalg.vector_norm(upper[..., :2] - lower[..., :2], dim=-1) < xy_threshold
    spaced = torch.abs(upper[..., 2] - lower[..., 2] - cube_size) < z_threshold
    grounded = torch.abs(pos[:, 0, 2] - (table_z + 0.5 * cube_size)) < z_threshold
    resting = speed < max_speed
    return aligned.all(dim=1) & spaced.all(dim=1) & grounded & resting.all(dim=1)
