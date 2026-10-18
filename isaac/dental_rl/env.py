"""The "dental-suction-hold" task: hold the saliva ejector tip on a point inside the lower jaw.

Scene (one per environment): the LeIsaac SO-101, placed exactly like isaac/dental/run_isaac.py
places it, so its link poses match the MJCF the jaw_sim brain uses. The jaw itself is *not* a
physics object: MuJoCo and PhysX both collide meshes by their convex hull, which for a horseshoe
would fill the whole mouth. The corridor rule of jaw_sim/zones.py is the jaw, and the policy is
punished when the tip leaves it (see `TorchCorridor`).

Episode: 6 s at 50 Hz control (200 Hz physics, decimation 4). The arm starts parked in front of
the incisors, a target is drawn from the corridor, and the policy moves the five arm joints by
small position deltas. An episode counts as a success once the tip has been inside the tolerance
for HOLD_STEPS control steps in a row, which is what "hold" means for the suction task: reaching
the point for a single step is not enough to keep a drilling site dry.

Everything in this file works in the Isaac world frame; observations and the rollout file use the
environment-local frame (world minus the environment origin), which is the jaw_sim world frame.
"""

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from leisaac.assets.robots.lerobot import SO101_FOLLOWER_CFG
from tensordict import TensorDict

from dental_sim.scene import TUBE_AXIS_XY
from isaac.dental_rl.corridor import JAW_SCALE, TorchCorridor
from jaw_sim.controller import GRIPPER_CLOSED_ON_TUBE
from jaw_sim.ejector import SalivaEjector
from jaw_sim.scene import GRIP_CENTER_Z

# Robot placement, copied from isaac/dental/run_isaac.py (fitted so the USD arm matches the MJCF).
ISAAC_ROOT_POS = (0.01265, -0.02005, -0.03292)  # m
ISAAC_ROOT_YAW_DEG = 89.832
GROUND_Z = ISAAC_ROOT_POS[2]

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_JOINT = "gripper"
ROLLOUT_JOINTS = (
    *ARM_JOINTS,
    GRIPPER_JOINT,
)  # order used by the rollout file and the viewer
GRIPPER_BODY = "gripper"  # link the ejector is rigidly attached to

# Park pose: the tip waits 30 mm in front of the incisors with the hook lying flat, the pose
# jaw_sim/controller.py:park_pose(+1) asks for. Solved once with dental_sim's SuctionIK on the
# MuJoCo scene of jaw_sim (tip error 5e-7 m); hard-coded so training needs no MuJoCo model.
PARK_JOINT_POS = (
    -0.030811,
    -1.627238,
    0.913775,
    0.713463,
    1.570796,
)  # rad, ARM_JOINTS order
RESET_JOINT_NOISE_RAD = 0.05  # uniform noise added to the park pose on every reset

PHYSICS_DT = 1.0 / 200.0
DECIMATION = 4  # -> 50 Hz control
EPISODE_LENGTH_S = 6.0
ENV_SPACING = 1.0  # m between environment origins; the arm reaches at most ~0.5 m

ACTION_SCALE = 0.03  # rad of joint position target per control step and unit action
# Tolerances are given at real size and scaled with the demo jaw (jaw_sim.jaw.JAW_SCALE).
REACH_TOL_M = 0.005 * JAW_SCALE  # 5 mm at real size: "the tip is on the target"
HOLD_STEPS = (
    25  # consecutive control steps inside the tolerance that count as a success (0.5 s)
)
DIST_W = 10.0  # reward per metre of tip-target distance
REACH_BONUS = 1.0  # reward per step while inside the tolerance
ZONE_PENALTY = 2.0  # reward lost per step with the tip in the forbidden zone
ACTION_RATE_W = 0.01  # penalty on squared action change, keeps the motion smooth

# Observation layout: joint positions, joint velocities, tip, target, target - tip, previous action.
OBS_TERM_SIZES = (len(ARM_JOINTS), len(ARM_JOINTS), 3, 3, 3, len(ARM_JOINTS))
OBS_DIM = sum(OBS_TERM_SIZES)


def _root_quat() -> tuple[float, float, float, float]:
    """Robot root orientation (w, x, y, z) for ISAAC_ROOT_YAW_DEG."""
    yaw = np.deg2rad(ISAAC_ROOT_YAW_DEG)
    return (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))


@configclass
class DentalSuctionHoldEnvCfg(DirectRLEnvCfg):
    """Configuration of the suction-hold task."""

    decimation = DECIMATION
    episode_length_s = EPISODE_LENGTH_S
    action_space = len(ARM_JOINTS)
    observation_space = OBS_DIM
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=PHYSICS_DT, render_interval=DECIMATION)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=ENV_SPACING, replicate_physics=True
    )
    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ISAAC_ROOT_POS,
            rot=_root_quat(),
            joint_pos={
                name: q for name, q in zip(ARM_JOINTS, PARK_JOINT_POS, strict=True)
            }
            | {GRIPPER_JOINT: GRIPPER_CLOSED_ON_TUBE},
        ),
    )


class DentalSuctionHoldEnv(DirectRLEnv):
    """Moves the ejector tip onto a corridor target and keeps it there.

    Attributes:
        robot: The SO-101 articulation of all environments.
        corridor: Batched corridor rule used for targets, zone penalty and the blocked flag.
        tip_pos_w: (num_envs, 3) current ejector tip position in the Isaac world frame.
        target_pos_w: (num_envs, 3) current target position in the Isaac world frame.
        blocked: (num_envs,) True where the tip is currently inside the forbidden zone.
        fixed_targets: If set, resets reuse these targets instead of sampling new ones (play.py).
    """

    cfg: DentalSuctionHoldEnvCfg

    def __init__(
        self, cfg: DentalSuctionHoldEnvCfg, render_mode: str | None = None, **kwargs
    ):
        """Builds the scene and all per-environment buffers.

        Args:
            cfg: Task configuration.
            render_mode: Passed through to DirectRLEnv.
            **kwargs: Passed through to DirectRLEnv.
        """
        super().__init__(cfg, render_mode, **kwargs)
        self.corridor = TorchCorridor(self.device)
        self._arm_joint_ids, _ = self.robot.find_joints(
            list(ARM_JOINTS), preserve_order=True
        )
        self._gripper_joint_id, _ = self.robot.find_joints([GRIPPER_JOINT])
        self.rollout_joint_ids, _ = self.robot.find_joints(
            list(ROLLOUT_JOINTS), preserve_order=True
        )
        self.gripper_body_id = self.robot.body_names.index(GRIPPER_BODY)

        _, tip_in_gripper = SalivaEjector().mount(
            np.array([*TUBE_AXIS_XY, GRIP_CENTER_Z])
        )
        self._tip_in_gripper = torch.tensor(
            tip_in_gripper, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self._park_pose = torch.tensor(
            PARK_JOINT_POS, dtype=torch.float32, device=self.device
        )
        limits = self.robot.data.soft_joint_pos_limits[:, self._arm_joint_ids]
        self._joint_lower, self._joint_upper = limits[..., 0], limits[..., 1]

        zeros = torch.zeros(self.num_envs, len(ARM_JOINTS), device=self.device)
        self.actions, self._previous_actions, self._joint_targets = (
            zeros.clone(),
            zeros.clone(),
            zeros.clone(),
        )
        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.tip_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.blocked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hold_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fixed_targets: torch.Tensor | None = None
        self._update_tip()

    def _setup_scene(self) -> None:
        """Ground, light, SO-101; no jaw body, the corridor rule stands in for it."""
        self.robot = Articulation(self.cfg.robot)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(),
            translation=(0.0, 0.0, GROUND_Z),
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9))
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Turns the action into joint position targets, clipped to the joint limits."""
        self._previous_actions = self.actions
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self._joint_targets = torch.clamp(
            self._joint_targets + ACTION_SCALE * self.actions,
            self._joint_lower,
            self._joint_upper,
        )

    def _apply_action(self) -> None:
        """Writes the arm targets and keeps the gripper closed on the tube."""
        self.robot.set_joint_position_target(
            self._joint_targets, joint_ids=self._arm_joint_ids
        )
        self.robot.set_joint_position_target(
            torch.full((self.num_envs, 1), GRIPPER_CLOSED_ON_TUBE, device=self.device),
            joint_ids=self._gripper_joint_id,
        )

    def _get_observations(self) -> dict:
        """State observation, all positions in the environment-local (jaw_sim world) frame."""
        tip_local = self.tip_pos_w - self.scene.env_origins
        target_local = self.target_pos_w - self.scene.env_origins
        obs = torch.cat(
            (
                self.robot.data.joint_pos[:, self._arm_joint_ids],
                self.robot.data.joint_vel[:, self._arm_joint_ids],
                tip_local,
                target_local,
                target_local - tip_local,
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Distance, reach bonus, forbidden-zone penalty and an action-rate penalty."""
        self._update_tip()
        distance = torch.linalg.norm(self.target_pos_w - self.tip_pos_w, dim=-1)
        reached = distance < REACH_TOL_M
        self.blocked = self.corridor.forbidden(self.tip_pos_w - self.scene.env_origins)
        self._hold_counter = torch.where(
            reached, self._hold_counter + 1, torch.zeros_like(self._hold_counter)
        )
        self.success |= self._hold_counter >= HOLD_STEPS
        action_rate = torch.sum((self.actions - self._previous_actions) ** 2, dim=-1)
        return (
            -DIST_W * distance
            + REACH_BONUS * reached.float()
            - ZONE_PENALTY * self.blocked.float()
            - ACTION_RATE_W * action_rate
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Only the time limit ends an episode; holding is what is being learnt."""
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(timed_out), timed_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        """Parks the arm with a little noise and draws a new target for the reset environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        count = len(env_ids)

        noise = RESET_JOINT_NOISE_RAD * (
            2.0 * torch.rand(count, len(ARM_JOINTS), device=self.device) - 1.0
        )
        arm_pos = torch.clamp(
            self._park_pose + noise,
            self._joint_lower[env_ids],
            self._joint_upper[env_ids],
        )
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._arm_joint_ids] = arm_pos
        joint_pos[:, self._gripper_joint_id] = GRIPPER_CLOSED_ON_TUBE
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._joint_targets[env_ids] = arm_pos

        if self.fixed_targets is None:
            self.target_pos_w[env_ids] = (
                self.corridor.sample_targets(count) + self.scene.env_origins[env_ids]
            )
        else:
            self.target_pos_w[env_ids] = (
                self.fixed_targets[env_ids] + self.scene.env_origins[env_ids]
            )

        self.extras["success_rate"] = self.success[env_ids].float().mean()
        self.extras["log"] = {"Episode/success_rate": self.extras["success_rate"]}
        self.actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._hold_counter[env_ids] = 0
        self.success[env_ids] = False
        self._update_tip()

    def _update_tip(self) -> None:
        """Ejector tip in the world frame: gripper link pose times the mount offset from jaw_sim."""
        gripper_pos = self.robot.data.body_pos_w[:, self.gripper_body_id]
        gripper_quat = self.robot.data.body_quat_w[:, self.gripper_body_id]
        self.tip_pos_w = gripper_pos + quat_apply(gripper_quat, self._tip_in_gripper)


class RslRlVecEnv:
    """rsl_rl's VecEnv interface on top of a DirectRLEnv.

    isaaclab_rl's wrapper cannot be imported in this environment (its tensordict/torch versions
    clash), and rsl_rl only needs a handful of attributes, so this is the whole bridge.

    Attributes:
        env: The wrapped Isaac Lab environment.
    """

    def __init__(self, env: DentalSuctionHoldEnv):
        """Caches the attributes rsl_rl reads from the environment.

        Args:
            env: The environment to wrap.
        """
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = env.cfg.action_space
        self.max_episode_length = env.max_episode_length
        self.device = env.device
        self.cfg = env.cfg

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Current episode length of every environment."""
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length_buf = value

    def get_observations(self) -> TensorDict:
        """Returns the current observations as the single group "policy"."""
        return self._pack(self.env._get_observations())

    def reset(self) -> tuple[TensorDict, dict]:
        """Resets all environments.

        Returns:
            Observations and an empty extras dictionary.
        """
        obs, _ = self.env.reset()
        return self._pack(obs), {}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Steps the environment and converts the gym-style result for rsl_rl.

        Args:
            actions: (num_envs, num_actions) policy output.

        Returns:
            Observations, rewards, done flags and extras (with "time_outs" for bootstrapping).
        """
        obs, reward, terminated, truncated, extras = self.env.step(actions)
        extras["time_outs"] = truncated
        return self._pack(obs), reward, (terminated | truncated).to(torch.long), extras

    def close(self) -> None:
        """Closes the wrapped environment."""
        self.env.close()

    def _pack(self, obs: dict) -> TensorDict:
        """Wraps the observation dictionary in the TensorDict rsl_rl 3.x expects."""
        return TensorDict(obs, batch_size=[self.num_envs], device=self.device)
