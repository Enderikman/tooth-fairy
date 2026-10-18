"""Command line for the jaw workspace (your STL jaws + saliva ejector + forbidden zones).

    uv run mjpython -m jaw_sim drag            viewer; drag the green marker, the arm follows where allowed
    uv run mjpython -m jaw_sim drag --voice    same, plus spoken commands from the default microphone
    uv run mjpython -m jaw_sim route           viewer; automatic route: enter, suction at 46, sweep to 43
    uv run python   -m jaw_sim check           headless: route plus every command, how often the zones blocked
    uv run mjpython -m jaw_sim leader          viewer; the simulated arm copies the real SO-101 leader arm

The viewer needs `mjpython` (macOS), which comes with the mujoco package.

Commands (drag mode) come from three places and all end up in JawSimulation.command(name):
    hardware buttons   a = calibrate, b = operate, c = stop, d = park (jaw_sim/buttons.py, work in any window)
    voice              "46", "stopp", "parken", ... (jaw_sim/voice.py, needs --voice)
    viewer keys        6 7 8 9 = tooth 46 43 33 36, . = park, Enter = calibrate (window must be in front)
Tooth commands only move the arm in operating mode (button b); stop leaves operating mode.
Space pauses the physics: that is the emergency stop of the simulation.

In the viewer:
    move the marker  double-click the green marker, then hold Ctrl and drag with the RIGHT mouse button
    rotate the marker double-click it, hold Ctrl and drag with the LEFT mouse button: rolls the hook
                     (the marker's red stick shows which way the hook points)
    top-left text    live tip coordinates in the world and in both jaw STL frames (mm)
"""

import argparse
import queue
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .buttons import ButtonListener
from .calibration import CalibrationRecorder, JawCalibration
from .controller import DraggedTarget, GuardedController, RouteTarget, marker_quat
from .jaw import JAWS
from .leader import LeaderArm, LeaderTarget, find_port
from .scene import FORBIDDEN_GROUP, JawSceneBuilder
from .voice import VoiceListener
from .zones import TOOTH_ACROSS, ForbiddenZones

CAMERAS = ("overview", "mouth_cam")
IMAGE_SIZE = (600, 800)  # (height, width)
SCREEN_FPS = 60.0
KEY_ENTER = 257  # GLFW key code (layout independent, unlike letter/symbol keys)
AXIS_LENGTH = 0.015  # m, drawn STL axes at each jaw origin
AXIS_RGBA = ([1, 0, 0, 1], [0, 0.8, 0, 1], [0, 0.3, 1, 1])  # x red, y green, z blue
# GLFW key codes of the viewer keys: 6 7 8 9 and . (letters, 0-5 and , are taken by the viewer)
COMMAND_KEYS = {54: "46", 55: "43", 56: "33", 57: "36", 46: "park", KEY_ENTER: "calibrate"}
COMMAND_KEY_HELP = "buttons a=calibrate b=operate c=stop d=park | keys 6=46 7=43 8=33 9=36 .=park"
COMMAND_STEP_S = 0.1  # headless check: simulated time per step while waiting for a command
COMMAND_CAP_S = 30.0  # headless check: give up on a command after this much simulated time


class JawSimulation:
    """Scene + zones + controller + state, wired together for one mode ("drag" or "route")."""

    def __init__(self, mode: str, voice: bool = False, leader: "LeaderArm | None" = None):
        self.mode = mode
        self.calibration_points = JawCalibration.load_or_default()
        self.zones = ForbiddenZones(self.calibration_points)
        self.model = JawSceneBuilder(self.zones).build()
        self.data = mujoco.MjData(self.model)
        if mode == "drag":
            self.source = DraggedTarget(self.model, self.zones)
        elif mode == "leader":
            self.source = LeaderTarget(self.model, leader)
        else:
            self.source = RouteTarget(self.model, self.zones)
        self.controller = GuardedController(self.model, self.source, self.zones)
        self.calibration = CalibrationRecorder()
        self.tip = self.model.site("suction_tip").id
        self.marker = self.model.body("target").mocapid[0]
        self.operating = False  # tooth commands are only executed while True (button b)
        self.last_command = "-"
        self.commands: queue.Queue[str] = queue.Queue()  # filled by the buttons and the voice listener
        self.voice = VoiceListener(self.commands) if voice else None
        self.listeners = [ButtonListener(self.commands)] + ([self.voice] if voice else [])
        self._place_arm_at_start()

    def view(self) -> None:
        """Interactive viewer: physics in real time, on-screen text, buttons, keys and voice."""
        mujoco.set_mjcb_control(self.controller)
        for listener in self.listeners:
            listener.start()
        try:
            with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self._on_key) as viewer:
                self._draw_jaw_axes(viewer)
                while viewer.is_running():
                    frame_start = time.time()
                    with viewer.lock():
                        self._run_queued_commands()
                        self._step_for(1.0 / SCREEN_FPS)
                        overlay = self._overlay()
                    # outside the lock: set_texts waits for the GUI thread, which itself needs the lock
                    viewer.set_texts(overlay)
                    viewer.sync()
                    time.sleep(max(0.0, 1.0 / SCREEN_FPS - (time.time() - frame_start)))
        finally:
            mujoco.set_mjcb_control(None)
            for listener in self.listeners:
                listener.stop()

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

    def command(self, name: str) -> None:
        """Runs one command, whether it came from a button, a key or a voice phrase.

        Args:
            name: "calibrate", "operate", "stop", "park", or a tooth ("46", "43", "33", "36").
                Tooth commands are ignored unless the sim is operating (after "operate").
                In leader mode only "calibrate" does something: the leader steers, not the commands.

        Raises:
            ValueError: If the simulation is in "route" mode, or the name is unknown.
        """
        if self.mode == "route":
            raise ValueError("commands need drag or leader mode")
        self.last_command = name
        if name == "calibrate":
            # drag: the marker is where the user put it; leader: the real tip position counts
            point = self.data.mocap_pos[self.marker] if self.mode == "drag" else self.data.site_xpos[self.tip]
            self.calibration.press_enter(point)
            self.controller.guarded = not self.calibration.active  # touching the teeth is the point of calibrating
        elif self.mode == "leader":
            return
        elif name == "operate":
            self.operating = True
        elif name == "stop":
            self.source.stop()
            self.operating = False
        elif name == "park":
            self.source.park(self.data.time)
        elif name in TOOTH_ACROSS:
            if self.operating:
                self.source.goto(name, self.data.time)
        else:
            raise ValueError(f"unknown command {name!r}")

    def _run_queued_commands(self) -> None:
        """Executes what the buttons and the voice listener put on the queue since the last frame."""
        while not self.commands.empty():
            self.command(self.commands.get())

    def _on_key(self, key: int) -> None:
        """Viewer keys; called by the viewer on every key press."""
        if self.mode != "route" and key in COMMAND_KEYS:
            self.command(COMMAND_KEYS[key])

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
        if self.mode == "leader":
            rows.append(("calibration", self.calibration.status() + "  (a / Enter)"))
            rows.append(("leader joints", "  ".join(f"{name} {np.degrees(a):.0f}" for name, a in self.source.angles.items())))
        if self.mode == "drag":
            rows.append(("operating", "yes" if self.operating else "no (press b)"))
            rows.append(("calibration", self.calibration.status()))
            rows.append(("last command", f"{self.last_command}" + (f"   running: {self.controller.pose.label}" if self.source.busy else "")))
            rows.append(("commands", COMMAND_KEY_HELP))
            if self.voice:
                rows.append(("voice heard", self.voice.error or self.voice.last_text))
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
    """Runs the route and then every command headless, and reports how the zones reacted."""
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
    _check_commands()
    return 0


def _check_commands() -> None:
    """Sends every command to a drag-mode simulation and prints one report line per command."""
    sim = JawSimulation("drag")
    ctrl = sim.controller
    for name in ("46", "operate", "46", "43", "33", "36", "stop", "park"):
        blocked_before = ctrl.blocked_ticks
        deadline = sim.data.time + COMMAND_CAP_S
        sim.command(name)
        while sim.source.busy and sim.data.time < deadline:
            sim.run(COMMAND_STEP_S)
        error = np.linalg.norm(sim.data.site_xpos[sim.tip] - ctrl.last_allowed.tip)
        print(f"command {name:>7}: blocked {ctrl.blocked_ticks - blocked_before} ticks, tip error {error * 1000:.1f} mm, moving {sim.source.busy}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="jaw_sim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    drag = commands.add_parser("drag", help="viewer; move the green marker, the arm follows where allowed")
    drag.add_argument("--voice", action="store_true", help="also listen for spoken commands on the default microphone")
    leader = commands.add_parser("leader", help="viewer; the simulated arm copies the real leader arm")
    leader.add_argument("--port", help="serial port of the leader's bus board (default: the one /dev/tty.usbmodem*)")
    commands.add_parser("route", help="viewer; automatic route with forbidden zones")
    check_parser = commands.add_parser("check", help="headless route run with a short report")
    check_parser.add_argument("--png", type=Path, help="directory for camera images")
    args = parser.parse_args()

    if args.command == "check":
        return check(args.png)
    arm = LeaderArm(args.port or find_port()) if args.command == "leader" else None
    JawSimulation(args.command, voice=getattr(args, "voice", False), leader=arm).view()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
