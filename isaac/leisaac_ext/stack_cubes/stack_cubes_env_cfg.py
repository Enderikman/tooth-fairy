"""SO-101 single-arm task: stack three 3 cm cubes into one tower.

Scene: LeIsaac "table_with_cube" USD (table + red cube "cube") plus two spawned cubes.
Geometry from the USD (env frame, metres): table top z=0.041, red cube centre (0.346, -0.362, 0.0615),
robot base at (0.35, -0.64, 0.01) facing +y, so the cubes sit ~0.28 m in front of it.
"""

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets.episode_data import EpisodeData
from leisaac.assets.scenes.simple import TABLE_WITH_CUBE_CFG, TABLE_WITH_CUBE_USD_PATH
from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
from leisaac.utils.domain_randomization import domain_randomization, randomize_object_uniform
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot

from ..template import (
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
from . import mdp

CUBE_SIZE = 0.03
TABLE_Z = 0.041
CUBE_Z = 0.0615  # same spawn height as the USD cube; settles onto the table on reset
RED_XY = (0.346, -0.362)
CUBE_SPACING = 0.07  # along x, i.e. left/right as seen from the robot
CUBE_NAMES = ("cube", "cube_blue", "cube_green")  # "cube" is the red cube created from the scene USD


def _spawned_cube(prim_name: str, x: float, rgb: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),  # matches the USD cube
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.95, dynamic_friction=0.95),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(x, RED_XY[1], CUBE_Z)),
    )


@configclass
class StackCubesSceneCfg(SingleArmTaskSceneCfg):
    """Table with the red USD cube, plus a blue and a green cube left and right of it."""

    scene: AssetBaseCfg = TABLE_WITH_CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    cube_blue: RigidObjectCfg = _spawned_cube("BlueCube", RED_XY[0] - CUBE_SPACING, (0.1, 0.25, 0.9))
    cube_green: RigidObjectCfg = _spawned_cube("GreenCube", RED_XY[0] + CUBE_SPACING, (0.1, 0.75, 0.2))

    # Same framing as LeIsaac's LiftCube task, which uses this table scene.
    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.6, -0.75, 0.38), rot=(0.77337, 0.55078, -0.2374, -0.20537), convention="opengl"
        ),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=40.6,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )


@configclass
class TerminationsCfg(SingleArmTerminationsCfg):
    success = DoneTerm(
        func=mdp.cubes_stacked,
        params={
            "cube_cfgs": [SceneEntityCfg(name) for name in CUBE_NAMES],
            "cube_size": CUBE_SIZE,
            "table_z": TABLE_Z,
        },
    )


@configclass
class StackCubesEnvCfg(SingleArmTaskEnvCfg):
    """Stack three cubes; keeps the template's front + wrist cameras (GR00T SO-100 config uses both)."""

    scene: StackCubesSceneCfg = StackCubesSceneCfg(env_spacing=8.0)

    observations: SingleArmObservationsCfg = SingleArmObservationsCfg()

    terminations: TerminationsCfg = TerminationsCfg()

    task_description: str = "Stack the blue cube on the red cube, then the green cube on the blue cube."

    def __post_init__(self) -> None:
        super().__post_init__()

        self.episode_length_s = 60.0  # two pick-and-place cycles take longer than the template's 25 s

        self.viewer.eye = (-0.4, -0.6, 0.5)
        self.viewer.lookat = (0.9, 0.0, -0.3)

        self.scene.robot.init_state.pos = (0.35, -0.64, 0.01)

        parse_usd_and_create_subassets(TABLE_WITH_CUBE_USD_PATH, self)

        # +-2 cm keeps the 7 cm-spaced cubes from overlapping; +-15 deg yaw stays graspable with a
        # fixed top-down gripper orientation (used by the scripted demo generator).
        cube_range = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (0.0, 0.0),
            "yaw": (-15 * torch.pi / 180, 15 * torch.pi / 180),
        }
        domain_randomization(
            self,
            random_options=[randomize_object_uniform(name, pose_range=cube_range) for name in CUBE_NAMES],
        )

    def build_lerobot_frame(self, episode_data: EpisodeData, dataset_cfg: LeRobotDatasetCfg) -> dict:
        """Export joint-space actions regardless of how the episode was driven.

        Scripted demos command end-effector poses through differential IK, so their raw actions are
        8-D poses. GR00T's SO-100/101 config expects 6 joint targets, so the recorded joint position
        targets (the IK output, or the leader-arm command when teleoperated) become the action.
        """
        frame = super().build_lerobot_frame(episode_data, dataset_cfg)
        joint_target = episode_data._data["obs"]["joint_pos_target"][-1].unsqueeze(0)
        frame["action"] = convert_leisaac_action_to_lerobot(joint_target).squeeze(0)
        return frame
