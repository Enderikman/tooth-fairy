"""Scripted demo generator for LeIsaac-SO101-StackCubes-v0.

Picks the blue cube onto the red one, then the green cube onto the blue one, using the same
action interface and gripper offsets as LeIsaac's PickOrangeStateMachine (8-D action:
EE position + quaternion in the robot base frame + binary gripper, solved by differential IK).
"""

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul
from leisaac.tasks.stack_cubes.mdp import cubes_stacked
from leisaac.tasks.stack_cubes.stack_cubes_env_cfg import CUBE_NAMES, CUBE_SIZE, TABLE_Z

from .base import StateMachineBase

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0
# From PickOrangeStateMachine: the IK body ("gripper") sits this far above / offset from the grasp point.
_GRIPPER_OFFSET = 0.1
_GRASP_XY_OFFSET = (-0.03, -0.01)
_HOVER = 0.08  # clearance above a cube or the tower while moving
_PLACE_CLEARANCE = 0.004  # release this far above the exact stacking height

# Phase boundaries in env steps (60 Hz). Each pick-and-place cycle ends at _CYCLE_END.
_APPROACH_END = 120  # first cycle only: blend from the initial EE position to the hover pose
_HOVER_END = 200
_LOWER_END = 260
_CLOSE_END = 320
_LIFT_END = 400
_TRANSPORT_END = 480
_PLACE_END = 560
_RELEASE_END = 600
_CYCLE_END = 660
_SETTLE_STEPS = 90  # after the last cycle: hold above the tower so the cubes come to rest


class StackCubesStateMachine(StateMachineBase):
    """Two pick-and-place cycles: (blue -> on red), (green -> on blue)."""

    BASE = CUBE_NAMES[0]
    MOVES = CUBE_NAMES[1:]  # picked in this order; level k lands k cubes above the base

    def __init__(self) -> None:
        self._cycle = 0
        self._step = 0
        self._settle = 0
        self._done = False
        self._initial_ee_pos: torch.Tensor | None = None
        self._pick_pos: torch.Tensor | None = None

    def setup(self, env) -> None:
        pass

    def check_success(self, env) -> bool:
        stacked = cubes_stacked(
            env,
            cube_cfgs=[SceneEntityCfg(name) for name in CUBE_NAMES],
            cube_size=CUBE_SIZE,
            table_z=TABLE_Z,
        )
        return bool(stacked.all().item())

    def get_action(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=10.0)  # as in PickOrangeStateMachine
        n, device = env.num_envs, env.device

        base_quat_w = robot.data.root_quat_w
        base_pos_w = robot.data.root_pos_w
        # Identity world orientation = top-down grasp in PickOrangeStateMachine's convention.
        target_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1)
        target_quat = quat_mul(quat_inv(base_quat_w), target_quat_w)

        if self._initial_ee_pos is None:
            self._initial_ee_pos = robot.data.body_pos_w[:, -1, :].clone()

        target_w, gripper = self._target(env, n, device)
        target_local = quat_apply(quat_inv(base_quat_w), target_w - base_pos_w)
        return torch.cat([target_local, target_quat, gripper], dim=-1)

    def _target(self, env, n, device) -> tuple[torch.Tensor, torch.Tensor]:
        open_ = torch.full((n, 1), _GRIPPER_OPEN, device=device)
        close = torch.full((n, 1), _GRIPPER_CLOSE, device=device)

        tower = env.scene[self.BASE].data.root_pos_w.clone()
        place = tower.clone()
        place[:, 2] += (self._cycle + 1) * CUBE_SIZE + _PLACE_CLEARANCE
        place = self._grasp_frame(place)

        if self._done or self._cycle >= len(self.MOVES):  # settle phase
            return self._up(place, _HOVER), open_

        cube = self._grasp_frame(env.scene[self.MOVES[self._cycle]].data.root_pos_w.clone())
        s = self._step
        if s < _HOVER_END:
            hover = self._up(cube, _HOVER)
            if self._cycle == 0 and s < _APPROACH_END:
                a = s / _APPROACH_END
                return (1.0 - a) * self._initial_ee_pos + a * hover, open_
            return hover, open_
        if s < _LOWER_END:
            self._pick_pos = cube  # latched: the cube moves with the gripper from here on
            return cube, open_
        if s < _CLOSE_END:
            return self._pick_pos, close
        if s < _LIFT_END:
            return self._up(self._pick_pos, _HOVER + (self._cycle + 1) * CUBE_SIZE), close
        if s < _TRANSPORT_END:
            return self._up(place, _HOVER), close
        if s < _PLACE_END:
            return place, close
        if s < _RELEASE_END:
            return place, open_
        return self._up(place, _HOVER), open_

    @staticmethod
    def _grasp_frame(pos_w: torch.Tensor) -> torch.Tensor:
        """Cube centre -> IK body target for a top-down grasp."""
        pos_w[:, 0] += _GRASP_XY_OFFSET[0]
        pos_w[:, 1] += _GRASP_XY_OFFSET[1]
        pos_w[:, 2] += _GRIPPER_OFFSET
        return pos_w

    @staticmethod
    def _up(pos: torch.Tensor, dz: float) -> torch.Tensor:
        out = pos.clone()
        out[:, 2] += dz
        return out

    def advance(self) -> None:
        if self._cycle >= len(self.MOVES):
            self._settle += 1
            self._done = self._settle >= _SETTLE_STEPS
            return
        self._step += 1
        if self._step >= _CYCLE_END:
            self._cycle += 1
            self._step = _APPROACH_END  # later cycles start at the hover phase
            self._pick_pos = None

    def reset(self) -> None:
        self.__init__()

    @property
    def is_episode_done(self) -> bool:
        return self._done
