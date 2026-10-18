"""Command line for the dental suction simulation.

    uv run python -m dental_sim demo      viewer, arm runs the suction route
    uv run python -m dental_sim manual    viewer, move the joints yourself (Control panel)
    uv run python -m dental_sim check     headless run with a numeric report (and --png images)
"""

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .controller import ARM_JOINTS, CONTROL_HZ, Autopilot
from .scene import DentalSceneBuilder

CAMERAS = ("overview", "mouth_cam")
IMAGE_SIZE = (600, 800)  # (height, width) of saved camera images
ARM_BODIES = ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")


class DentalSimulation:
    """One compiled scene, its autopilot and the simulation state, ready to view or run."""

    def __init__(self):
        self.model = DentalSceneBuilder().build()
        self.autopilot = Autopilot(self.model)
        self.data = mujoco.MjData(self.model)
        self._place_arm_at_start_pose()

    def view(self, autopilot: bool) -> None:
        """Opens the interactive viewer; blocks until the window is closed."""
        if autopilot:
            mujoco.set_mjcb_control(self.autopilot)
        try:
            mujoco.viewer.launch(self.model, self.data)
        finally:
            mujoco.set_mjcb_control(None)

    def save_images(self, directory: Path) -> list[Path]:
        """Renders every camera in CAMERAS to a PNG in `directory`."""
        from PIL import Image

        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        with mujoco.Renderer(self.model, *IMAGE_SIZE) as renderer:
            for camera in CAMERAS:
                renderer.update_scene(self.data, camera=camera)
                path = directory / f"{camera}.png"
                Image.fromarray(renderer.render()).save(path)
                paths.append(path)
        return paths

    def joint_angles(self) -> dict[str, float]:
        """Current arm joint angles in radians."""
        return {name: float(self.data.qpos[self.model.joint(name).qposadr[0]]) for name in ARM_JOINTS}

    def _place_arm_at_start_pose(self) -> None:
        for name, q in self.autopilot.start_pose().items():
            self.data.qpos[self.model.joint(name).qposadr[0]] = q
            self.data.ctrl[self.model.actuator(name).id] = q
        mujoco.mj_forward(self.model, self.data)


class RunReport:
    """Collects accuracy and safety numbers while the simulation steps."""

    def __init__(self, sim: DentalSimulation):
        model = sim.model
        self.sim = sim
        self.tip = model.site("suction_tip").id
        self.marker = model.body("target").mocapid[0]
        self.tube = model.geom("suction_tube").id
        head = model.body("phantom_head").id
        arm = {model.body(name).id for name in ARM_BODIES}
        self.mouth_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == head}
        self.arm_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in arm and g != self.tube}
        self.waypoint_names = {w.name for w in sim.autopilot.trajectory.waypoints}

        self.tracking_errors: list[float] = []
        self.ik_errors: list[float] = []
        self.settled_errors: dict[str, float] = {}  # last error measured while dwelling at each waypoint
        self.tube_contact_steps = 0
        self.arm_contact_steps = 0

    def record_step(self) -> None:
        """Call after every physics step."""
        data, pilot = self.sim.data, self.sim.autopilot
        error = float(np.linalg.norm(data.site_xpos[self.tip] - data.mocap_pos[self.marker]))
        self.tracking_errors.append(error)
        self.ik_errors.append(pilot.ik_error)
        if pilot.label in self.waypoint_names:
            self.settled_errors[pilot.label] = error
        for contact in data.contact[: data.ncon]:
            pair = {contact.geom1, contact.geom2}
            if self.tube in pair and pair & self.mouth_geoms:
                self.tube_contact_steps += 1
            elif pair & self.arm_geoms and pair & self.mouth_geoms:
                self.arm_contact_steps += 1

    def print(self) -> None:
        mm = 1e3
        data = self.sim.data
        print(f"simulated {data.time:.2f} s, timestep {self.sim.model.opt.timestep * mm:.1f} ms, control {CONTROL_HZ:.0f} Hz")
        print(f"IK residual       max {max(self.ik_errors) * mm:6.2f} mm")
        print(
            f"tip tracking      mean {np.mean(self.tracking_errors) * mm:6.2f} mm"
            f"   max {max(self.tracking_errors) * mm:6.2f} mm"
        )
        for name, error in self.settled_errors.items():
            print(f"  settled at {name:<20} {error * mm:6.2f} mm")
        print(f"tube-mouth contact steps {self.tube_contact_steps}, arm-mouth contact steps {self.arm_contact_steps}")
        joints = ", ".join(f"{name}={q:+.3f}" for name, q in self.sim.joint_angles().items())
        print(f"final joints [rad]: {joints}")


class CheckCommand:
    """Headless run of the approach plus one full route, then a printed report."""

    def __init__(self, seconds: float | None, image_dir: Path | None):
        self.seconds = seconds
        self.image_dir = image_dir

    def run(self) -> int:
        sim = DentalSimulation()
        trajectory = sim.autopilot.trajectory
        duration = self.seconds or trajectory.approach_s + trajectory.loop_s
        report = RunReport(sim)
        mujoco.set_mjcb_control(sim.autopilot)
        try:
            while sim.data.time < duration:
                mujoco.mj_step(sim.model, sim.data)
                report.record_step()
        finally:
            mujoco.set_mjcb_control(None)
        report.print()
        if self.image_dir:
            for path in sim.save_images(self.image_dir):
                print(f"wrote {path}")
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dental_sim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="viewer; the arm moves the suction tip through the mouth")
    commands.add_parser("manual", help="viewer without autopilot; move joints with the Control sliders")
    check = commands.add_parser("check", help="headless run; prints tracking error and contacts")
    check.add_argument("--seconds", type=float, help="default: approach + one full route")
    check.add_argument("--png", type=Path, help="directory for camera images")
    return parser


def main() -> int:
    """Entry point; returns the process exit code."""
    args = build_parser().parse_args()
    if args.command == "check":
        return CheckCommand(args.seconds, args.png).run()
    DentalSimulation().view(autopilot=args.command == "demo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
