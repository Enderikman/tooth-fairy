"""Command line for the jaw workspace (your STL jaws + saliva ejector + forbidden zones).

    uv run mjpython -m jaw_sim drag     viewer; drag the green marker, the arm follows where allowed
    uv run mjpython -m jaw_sim route    viewer; automatic route: insert the hook, sweep left/right
    uv run python   -m jaw_sim check    headless route run: how often the zones blocked (and --png)

The viewer needs `mjpython` (macOS), which comes with the mujoco package.

In the viewer:
    move the marker  double-click the green marker, then hold Ctrl and drag with the RIGHT mouse button
    rotate the marker double-click it, hold Ctrl and drag with the LEFT mouse button: rolls the hook
                     (the marker's red stick shows which way the hook points)
    Enter            calibration: start, then record one reference point per press (drag mode)
    top-left text    live tip coordinates in the world and in both jaw STL frames (mm)
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .calibration import CalibrationRecorder
from .controller import DraggedTarget, GuardedController, RouteTarget, marker_quat
from .jaw import JAWS
from .scene import FORBIDDEN_GROUP, JawSceneBuilder
from .zones import ForbiddenZones

CAMERAS = ("overview", "mouth_cam")
IMAGE_SIZE = (600, 800)  # (height, width)
SCREEN_FPS = 60.0
KEY_ENTER = 257  # GLFW key code (layout independent, unlike letter/symbol keys)
AXIS_LENGTH = 0.015  # m, drawn STL axes at each jaw origin
AXIS_RGBA = ([1, 0, 0, 1], [0, 0.8, 0, 1], [0, 0.3, 1, 1])  # x red, y green, z blue


class JawSimulation:
    """Scene + zones + controller + state, wired together for one mode ("drag" or "route")."""

    def __init__(self, mode: str):
        self.mode = mode
        self.zones = ForbiddenZones()
        self.model = JawSceneBuilder(self.zones).build()
        self.data = mujoco.MjData(self.model)
        self.source = DraggedTarget(self.model) if mode == "drag" else RouteTarget(self.model)
        self.controller = GuardedController(self.model, self.source, self.zones)
        self.calibration = CalibrationRecorder()
        self.tip = self.model.site("suction_tip").id
        self.marker = self.model.body("target").mocapid[0]
        self._place_arm_at_start()

    def view(self) -> None:
        """Interactive viewer: physics in real time, on-screen coordinates, keyboard buttons."""
        mujoco.set_mjcb_control(self.controller)
        try:
            with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self._on_key) as viewer:
                self._draw_jaw_axes(viewer)
                while viewer.is_running():
                    frame_start = time.time()
                    with viewer.lock():
                        self._step_for(1.0 / SCREEN_FPS)
                        overlay = self._overlay()
                    # outside the lock: set_texts waits for the GUI thread, which itself needs the lock
                    viewer.set_texts(overlay)
                    viewer.sync()
                    time.sleep(max(0.0, 1.0 / SCREEN_FPS - (time.time() - frame_start)))
        finally:
            mujoco.set_mjcb_control(None)

    def run(self, seconds: float) -> None:
        """Steps the physics headless for `seconds` of simulated time."""
        mujoco.set_mjcb_control(self.controller)
        try:
            self._step_for(seconds)
        finally:
            mujoco.set_mjcb_control(None)

    def save_images(self, directory: Path) -> list[Path]:
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

    def _step_for(self, seconds: float) -> None:
        end = self.data.time + seconds
        while self.data.time < end:
            mujoco.mj_step(self.model, self.data)

    def _on_key(self, key: int) -> None:
        """Keyboard buttons; called by the viewer on every key press."""
        if self.mode != "drag":
            return
        if key == KEY_ENTER:
            self.calibration.press_enter(self.data.mocap_pos[self.marker])

    def _overlay(self) -> tuple:
        """Top-left text: labels in the left column, values (mm, deg) in the right column."""
        tip = self.data.site_xpos[self.tip]
        target = self.controller.pose.tip
        rows = [
            ("mode", f"{self.mode}: {self.controller.pose.label}"),
            ("status", "BLOCKED (forbidden zone)" if self.controller.blocked else "allowed"),
            ("hook roll", f"{np.degrees(self.controller.last_allowed.roll):.0f} deg"),
            ("tip world", _xyz(tip * 1000)),
            ("target world", _xyz(target * 1000)),
        ]
        rows += [(f"tip {jaw.name} STL", _xyz(jaw.to_stl(tip))) for jaw in JAWS]
        if self.mode == "drag":
            rows.append(("calibration", self.calibration.status()))
        left = "\n".join(label for label, _ in rows)
        right = "\n".join(value for _, value in rows)
        return (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right)

    def _draw_jaw_axes(self, viewer) -> None:
        """Red/green/blue arrows = STL x/y/z at each jaw origin, to read the coordinates against."""
        scene = viewer.user_scn
        for jaw in JAWS:
            origin = jaw.position()
            for axis in range(3):
                geom = scene.geoms[scene.ngeom]
                color = np.array(AXIS_RGBA[axis], dtype=np.float32)
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), color)
                tip = origin + AXIS_LENGTH * jaw.rotation[:, axis]
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.0008, origin, tip)
                scene.ngeom += 1

    def _place_arm_at_start(self) -> None:
        for name, q in self.controller.start_pose().items():
            self.data.qpos[self.model.joint(name).qposadr[0]] = q
            self.data.ctrl[self.model.actuator(name).id] = q
        start = self.source.start()
        self.data.mocap_pos[self.marker] = start.tip
        self.data.mocap_quat[self.marker] = marker_quat(start.roll)
        mujoco.mj_forward(self.model, self.data)


def _xyz(v: np.ndarray) -> str:
    return f"x {v[0]:7.1f}   y {v[1]:7.1f}   z {v[2]:7.1f}  mm"


def check(image_dir: Path | None) -> int:
    """Runs the approach plus two route loops headless and reports how the zones reacted."""
    sim = JawSimulation("route")
    route = sim.source.route
    sim.run(route.approach_s + 2 * route.loop_s)
    ctrl = sim.controller
    error = np.linalg.norm(sim.data.site_xpos[sim.tip] - ctrl.last_allowed.tip)
    print(f"simulated {sim.data.time:.1f} s, {ctrl.ticks} control ticks")
    print(f"target blocked by forbidden zones in {ctrl.blocked_ticks} ticks ({100 * ctrl.blocked_ticks / ctrl.ticks:.0f} %)")
    print(f"final tip error {error * 1000:.1f} mm, hook roll {np.degrees(ctrl.last_allowed.roll):.0f} deg")
    print(f"forbidden markers drawn: {int((sim.model.site_group == FORBIDDEN_GROUP).sum())}")
    if image_dir:
        for path in sim.save_images(image_dir):
            print(f"wrote {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="jaw_sim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("drag", help="viewer; move the green marker, the arm follows where allowed")
    commands.add_parser("route", help="viewer; automatic route with forbidden zones")
    check_parser = commands.add_parser("check", help="headless route run with a short report")
    check_parser.add_argument("--png", type=Path, help="directory for camera images")
    args = parser.parse_args()

    if args.command == "check":
        return check(args.png)
    JawSimulation(args.command).view()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
