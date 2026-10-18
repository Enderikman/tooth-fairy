"""Dental suction demo in Isaac Sim: LeIsaac's SO-101 driven by the jaw_sim brain.

Run on the GPU VM (from the repo root, inside the `leisaac` conda env):
    python -m isaac.dental.run_isaac --mode route                      automatic route, loops
    python -m isaac.dental.run_isaac --mode teeth                      operate, then 46 -> 43 -> 33 -> 36 -> park
    python -m isaac.dental.run_isaac --mode teeth --video out/dental.mp4 --seconds 30 --headless
or simply `bash isaac/run.sh dental [route|teeth]`.

Per frame:  DentalBrain.step(dt) -> six joint targets (rad) -> SO-101 articulation position targets.
The brain (jaw_sim: forbidden zones, IK, routes) is unchanged; the same targets are what the real
SO-101 follower receives through LeRobot.

Frames: everything is placed in jaw_sim's world frame (robot base at the origin, +x toward the
patient, +z up). LeIsaac's SO-101 USD has a different base orientation, so the robot root is
rotated/shifted by ISAAC_ROOT_* (fitted once, see below) to make both worlds coincide.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--mode", choices=("route", "teeth"), default="route")
parser.add_argument("--seconds", type=float, default=0.0, help="stop after this much sim time (0 = run until closed)")
parser.add_argument("--video", type=str, default="", help="write an mp4 of the front camera to this path")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

# Isaac modules can only be imported after the app has started.
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from pxr import Gf, Sdf, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from leisaac.assets.robots.lerobot import SO101_FOLLOWER_CFG  # noqa: E402

from dental_sim.scene import TUBE_AXIS_XY  # noqa: E402
from isaac.dental.brain import JOINTS, DentalBrain  # noqa: E402
from jaw_sim.ejector import STL_TO_GRIPPER, SalivaEjector  # noqa: E402
from jaw_sim.jaw import JAWS, MM  # noqa: E402
from jaw_sim.scene import FORBIDDEN_GROUP, GRIP_CENTER_Z  # noqa: E402

PHYSICS_DT = 1.0 / 200.0
RENDER_EVERY = 4  # physics steps per rendered frame (50 fps)
# LeIsaac SO-101 root pose that makes its link poses match the vendored MJCF used by the brain.
# Fitted by least squares over 25 random joint configurations (gripper link pose):
# residual 3.2 mm mean / 6.8 mm max (the two models' link lengths differ slightly), gripper link
# frames agree within 1.6 deg, so jaw_sim's gripper-frame offsets (ejector) are reused as they are.
# Fit: p_isaac = A p_mujoco + t, A = rot_z(-89.832 deg), t = (20.01, 12.71, 32.92) mm; root = (A^T, -A^T t).
ISAAC_ROOT_POS = (0.01265, -0.02005, -0.03292)  # m
ISAAC_ROOT_YAW_DEG = 89.832
GROUND_Z = ISAAC_ROOT_POS[2]  # the USD base sits on its own origin, 33 mm below the MJCF base origin
JAW_COLOR = (0.95, 0.93, 0.85)
EJECTOR_COLOR = (0.55, 0.75, 0.95)
ZONE_COLOR = (1.0, 0.1, 0.1)
ALLOWED_COLOR, BLOCKED_COLOR = (0.1, 0.9, 0.2), (1.0, 0.1, 0.1)
TEETH_SCRIPT = ("46", "43", "33", "36", "park")
CAMERA_EYE, CAMERA_TARGET = (0.20, -0.55, 0.42), (0.20, 0.0, 0.16)
# Gravity sag: the simulated STS3215 position servos (LeIsaac gains) settle ~2 deg short of their
# target under the arm's weight. A slow integral trim per joint removes that steady-state error.
# The brain's targets stay unchanged; on the real arm the same trim can run in the LeRobot loop.
SAG_TRIM_GAIN = 2.0  # 1/s
SAG_TRIM_LIMIT = 0.15  # rad


class IsaacDentalScene:
    """Builds the USD stage: ground, light, SO-101, jaws, ejector, forbidden-zone dots, target marker."""

    def __init__(self, sim: sim_utils.SimulationContext, brain: DentalBrain):
        self.stage = sim.stage
        self.brain = brain
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/Ground", ground, translation=(0.0, 0.0, GROUND_Z))
        sim_utils.DomeLightCfg(intensity=1500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=1500.0))
        self.robot = self._add_robot()
        for jaw in JAWS:
            self._add_mesh(f"/World/Jaws/{jaw.name}", jaw.file, jaw.rotation, jaw.position(), JAW_COLOR)
        self._add_forbidden_points()
        self.marker = self._add_marker()

    def add_ejector(self) -> None:
        """Ejector mesh in its own prim; follow_gripper() moves it with the gripper link every frame.

        (LeIsaac's robot USD is instanceable, so meshes cannot be added under its links.)
        The mesh points are stored in the gripper link frame, using jaw_sim's mount offsets.
        """
        ejector = SalivaEjector()
        mesh_pos, tip_pos = ejector.mount(np.array([*TUBE_AXIS_XY, GRIP_CENTER_Z]))
        self.tip_in_gripper = tip_pos
        self.ejector = UsdGeom.Xform.Define(self.stage, Sdf.Path("/World/Ejector"))  # typed, so its transform renders
        self._add_mesh("/World/Ejector/mesh", ejector.assembled_mesh(), STL_TO_GRIPPER, mesh_pos, EJECTOR_COLOR)
        self.ejector_translate = self.ejector.AddTranslateOp()
        self.ejector_orient = self.ejector.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        self.gripper_index = self.robot.body_names.index("gripper")

    def follow_gripper(self) -> np.ndarray:
        """Moves the ejector to the gripper link pose; returns the ejector tip in world (m)."""
        pos = self.robot.data.body_pos_w[0, self.gripper_index].cpu().numpy().astype(float)
        w, x, y, z = self.robot.data.body_quat_w[0, self.gripper_index].cpu().numpy().astype(float)
        self.ejector_translate.Set(Gf.Vec3d(*pos))
        self.ejector_orient.Set(Gf.Quatd(w, x, y, z))
        rotation = np.array(Gf.Matrix3d(Gf.Quatd(w, x, y, z))).T  # Gf matrices are row-vector convention
        return pos + rotation @ self.tip_in_gripper

    def set_marker(self, position_m, blocked: bool) -> None:
        xform = UsdGeom.Xformable(self.marker)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, position_m)))
        UsdGeom.Gprim(self.marker).GetDisplayColorAttr().Set([Gf.Vec3f(*(BLOCKED_COLOR if blocked else ALLOWED_COLOR))])

    def _add_robot(self) -> Articulation:
        cfg = SO101_FOLLOWER_CFG.replace(prim_path="/World/Robot")
        yaw = np.deg2rad(ISAAC_ROOT_YAW_DEG)
        cfg.init_state.pos = ISAAC_ROOT_POS
        cfg.init_state.rot = (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))  # w, x, y, z
        cfg.init_state.joint_pos = {name: float(q) for name, q in self.brain.start_pose().items()}
        return Articulation(cfg)

    def _add_mesh(self, path: str, stl_file, rotation: np.ndarray, position: np.ndarray, color) -> None:
        """Visual-only USD mesh from an STL in mm, transformed into its parent's frame (metres)."""
        mesh = trimesh.load(stl_file)
        points = (np.asarray(mesh.vertices) * MM) @ np.asarray(rotation).T + np.asarray(position)
        usd_mesh = UsdGeom.Mesh.Define(self.stage, Sdf.Path(path))
        usd_mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
        usd_mesh.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
        usd_mesh.CreateFaceVertexIndicesAttr(mesh.faces.ravel().tolist())
        usd_mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        usd_mesh.CreateSubdivisionSchemeAttr("none")

    def _add_forbidden_points(self) -> None:
        """Small red dots at every grid point the forbidden zones reject (same grid as jaw_sim)."""
        model, data = self.brain.model, self.brain.data
        zone_sites = np.flatnonzero(model.site_group == FORBIDDEN_GROUP)
        points = UsdGeom.Points.Define(self.stage, Sdf.Path("/World/ForbiddenZones"))
        points.CreatePointsAttr([Gf.Vec3f(*map(float, data.site_xpos[i])) for i in zone_sites])
        points.CreateWidthsAttr([0.0015] * len(zone_sites))
        points.CreateDisplayColorAttr([Gf.Vec3f(*ZONE_COLOR)])

    def _add_marker(self):
        sphere = UsdGeom.Sphere.Define(self.stage, Sdf.Path("/World/Target"))
        sphere.CreateRadiusAttr(0.004)
        sphere.CreateDisplayColorAttr([Gf.Vec3f(*ALLOWED_COLOR)])
        return sphere.GetPrim()


class VideoRecorder:
    """Front camera -> mp4 (via ffmpeg). Only used with --video."""

    def __init__(self, path: str):
        from isaaclab.sensors import Camera, CameraCfg

        self.path = path
        self.frames: list[np.ndarray] = []
        self.camera = Camera(
            CameraCfg(
                prim_path="/World/VideoCamera",
                update_period=0,
                height=720,
                width=1280,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 10.0)),
            )
        )

    def aim(self) -> None:
        eye = torch.tensor([CAMERA_EYE], device=self.camera.device)
        target = torch.tensor([CAMERA_TARGET], device=self.camera.device)
        self.camera.set_world_poses_from_view(eye, target)

    def capture(self, dt: float) -> None:
        self.camera.update(dt)
        self.frames.append(self.camera.data.output["rgb"][0, ..., :3].cpu().numpy())

    def save(self, fps: int) -> None:
        import subprocess

        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        h, w, _ = self.frames[0].shape
        ffmpeg = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", self.path],
            stdin=subprocess.PIPE,
        )
        for frame in self.frames:
            ffmpeg.stdin.write(frame.astype(np.uint8).tobytes())
        ffmpeg.stdin.close()
        ffmpeg.wait()
        print(f"wrote {self.path} ({len(self.frames)} frames)")


class TeethScript:
    """Mode "teeth": operate, then visit each tooth in TEETH_SCRIPT, one after the other."""

    def __init__(self, brain: DentalBrain):
        self.brain = brain
        self.queue = list(TEETH_SCRIPT)
        brain.command("operate")

    def update(self) -> None:
        if self.queue and not self.brain.busy:
            name = self.queue.pop(0)
            print(f"t={self.brain.time:5.1f}s  command {name}")
            self.brain.command(name)


def main() -> int:
    brain = DentalBrain("drag" if args.mode == "teeth" else "route")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args.device))
    sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)
    scene = IsaacDentalScene(sim, brain)
    recorder = VideoRecorder(args.video) if args.video else None
    sim.reset()
    scene.add_ejector()
    if recorder:
        recorder.aim()
    script = TeethScript(brain) if args.mode == "teeth" else None

    robot = scene.robot
    joint_ids = [robot.joint_names.index(name) for name in JOINTS]
    frame_dt = PHYSICS_DT * RENDER_EVERY
    trim = torch.zeros(1, len(JOINTS), device=sim.device)
    step = 0
    while app.is_running():
        if script:
            script.update()
        targets = brain.step(frame_dt)
        q = torch.tensor([[targets[name] for name in JOINTS]], device=sim.device)
        error = q - robot.data.joint_pos[:, joint_ids]
        trim = (trim + SAG_TRIM_GAIN * frame_dt * error).clamp(-SAG_TRIM_LIMIT, SAG_TRIM_LIMIT)
        robot.set_joint_position_target(q + trim, joint_ids=joint_ids)
        status = brain.status()
        scene.set_marker(np.array(status["target_mm"]) * MM, status["blocked"])
        for _ in range(RENDER_EVERY):
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(PHYSICS_DT)
        isaac_tip = scene.follow_gripper()
        sim.render()
        if recorder:
            recorder.capture(frame_dt)
        step += 1
        if step % 50 == 0:
            # MuJoCo brain tip vs. where the ejector tip really is in Isaac (different models + physics)
            gap_mm = np.linalg.norm(isaac_tip * 1000 - np.array(status["tip_mm"]))
            joint_error = (robot.data.joint_pos[0, joint_ids] - q[0]).abs().max().item()
            print(
                f"t={brain.time:5.1f}s  {status['label']:<12} blocked={status['blocked']}  "
                f"brain tip {status['tip_mm']}  isaac tip {(isaac_tip * 1000).round(1).tolist()}  "
                f"gap {gap_mm:5.1f} mm  max joint error {np.degrees(joint_error):4.1f} deg"
            )
        if args.seconds and brain.time >= args.seconds:
            break
    if recorder:
        recorder.save(fps=round(1 / frame_dt))
    sys.stdout.flush()
    os._exit(0)  # SimulationApp.close() hangs in standalone Isaac Lab 2.3 scripts


if __name__ == "__main__":
    main()
