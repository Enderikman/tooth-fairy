"""CLI: uv run python -m stack_sim {demo,manual,check}

`demo` and `manual` open the MuJoCo viewer; `check` runs the same scene headless, reports
where the three cubes ended up and exits non-zero if they are not a finished stack.
"""

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .controller import ARM_JOINTS, CONTROL_HZ, Autopilot
from .scene import LAYOUT, Layout, build_model

SETTLE_S = 1.0  # extra sim time after the plan, to see whether the stack stays up
ALIGNMENT_TOL = 0.012  # horizontal offset allowed between stacked cube centres
LEVEL_TOL = 0.006  # height error allowed per stack level
RESTING_SPEED = 0.010  # m/s below which a cube counts as resting
IMAGE_SIZE = (720, 960)  # (height, width) of the saved camera images
CAMERAS = ("overview", "front", "wrist")

# Bodies that should never touch anything: the arm links above the hand. The gripper and the
# moving jaw do touch cubes on purpose, but they should not scrape the table.
ARM_LINK_BODIES = ("shoulder", "upper_arm", "lower_arm", "wrist")
HAND_BODIES = ("gripper", "moving_jaw_so101_v1")


class CubeState:
    """Where one cube ended up and whether it is still moving."""

    def __init__(self, name: str, position: np.ndarray, speed: float):
        self.name = name
        self.position = position
        self.speed = speed

    def __str__(self) -> str:
        return (
            f"{self.name:<11} x={self.position[0]:+.4f} y={self.position[1]:+.4f} "
            f"z={self.position[2]:+.4f} m   |v|={self.speed * 1e3:5.1f} mm/s"
        )


class RunMetrics:
    """Tracking accuracy and unwanted contacts collected while stepping the simulation."""

    def __init__(self):
        self.tracking_errors: list[float] = []
        self.ik_errors: list[float] = []
        self.unwanted_contacts = 0

    def add_step(self, tracking_error: float, ik_error: float) -> None:
        """Records one physics step."""
        self.tracking_errors.append(tracking_error)
        self.ik_errors.append(ik_error)

    def summary_lines(self) -> list[str]:
        """Human-readable accuracy summary, in millimetres."""
        return [
            f"IK residual       max {max(self.ik_errors) * 1e3:6.2f} mm",
            f"grasp tracking    mean {np.mean(self.tracking_errors) * 1e3:6.2f} mm"
            f"   max {max(self.tracking_errors) * 1e3:6.2f} mm",
            f"unwanted contact steps (arm links, hand vs table) {self.unwanted_contacts}",
        ]


class StackVerdict:
    """Result of inspecting the cubes: is this a finished stack, and if not, why not."""

    def __init__(self, complete: bool, problems: list[str], height: float):
        self.complete = complete
        self.problems = problems
        self.height = height

    def __str__(self) -> str:
        if self.complete:
            return "stack complete: yes"
        return "stack complete: no - " + "; ".join(self.problems)


class StackInspector:
    """Checks the final cube poses against what a finished stack looks like."""

    def __init__(self, layout: Layout):
        self.layout = layout

    def inspect(self, states: list[CubeState]) -> StackVerdict:
        """Cubes must sit on consecutive levels, share one vertical axis, and be at rest."""
        by_height = sorted(states, key=lambda state: state.position[2])
        axis = np.mean([state.position[:2] for state in states], axis=0)
        problems: list[str] = []
        for level, state in enumerate(by_height):
            expected_z = self.layout.level_height(level)
            offset = float(np.linalg.norm(state.position[:2] - axis))
            if abs(state.position[2] - expected_z) > LEVEL_TOL:
                problems.append(
                    f"{state.name} at z={state.position[2] * 1e3:.1f} mm, "
                    f"level {level} wants {expected_z * 1e3:.1f} mm"
                )
            if offset > ALIGNMENT_TOL:
                problems.append(f"{state.name} off the stack axis by {offset * 1e3:.1f} mm")
            if state.speed > RESTING_SPEED:
                problems.append(f"{state.name} still moving at {state.speed * 1e3:.0f} mm/s")
        height = by_height[-1].position[2] + self.layout.size / 2
        return StackVerdict(not problems, problems, height)


class StackSimulation:
    """One scene plus its autopilot: run it headless, or hand it to the viewer."""

    def __init__(self, layout: Layout = LAYOUT):
        self.layout = layout
        self.model = build_model(layout)
        self.autopilot = Autopilot(self.model, layout)
        self.data = self._fresh_data()
        self.tool_site = self.model.site("gripperframe").id
        self.target_mocap = self.model.body("target").mocapid[0]

    def launch_viewer(self, with_autopilot: bool) -> None:
        """Opens the MuJoCo viewer window; blocks until it is closed."""
        if not with_autopilot:
            mujoco.viewer.launch(self.model, self.data)
            return
        mujoco.set_mjcb_control(self.autopilot)
        try:
            mujoco.viewer.launch(self.model, self.data)
        finally:
            mujoco.set_mjcb_control(None)

    def run(self, duration: float) -> RunMetrics:
        """Steps the physics with the autopilot installed until `duration` of sim time passed."""
        metrics = RunMetrics()
        unwanted = self._unwanted_contact_pairs()
        mujoco.set_mjcb_control(self.autopilot)
        try:
            while self.data.time < duration:
                mujoco.mj_step(self.model, self.data)
                metrics.add_step(self._tracking_error(), self.autopilot.ik_error)
                metrics.unwanted_contacts += self._count_contacts(unwanted)
        finally:
            mujoco.set_mjcb_control(None)
        return metrics

    def cube_states(self) -> list[CubeState]:
        """Current position and speed of every cube."""
        states = []
        for name in self.layout.names:
            body = self.model.body(name).id
            dofadr = self.model.joint(f"{name}_free").dofadr[0]
            speed = float(np.linalg.norm(self.data.qvel[dofadr : dofadr + 3]))
            states.append(CubeState(name, self.data.xpos[body].copy(), speed))
        return states

    def joint_line(self) -> str:
        """Final arm joint angles, in the same units LeRobot's joint targets use (radians)."""
        angles = (
            f"{name}={self.data.qpos[self.model.joint(name).qposadr[0]]:+.3f}" for name in ARM_JOINTS
        )
        return "final joints [rad]: " + ", ".join(angles)

    def save_images(self, directory: Path) -> list[Path]:
        """Renders every fixed camera at the current state into `directory`."""
        from PIL import Image

        directory.mkdir(parents=True, exist_ok=True)
        written = []
        with mujoco.Renderer(self.model, *IMAGE_SIZE) as renderer:
            for camera in CAMERAS:
                renderer.update_scene(self.data, camera=camera)
                path = directory / f"{camera}.png"
                Image.fromarray(renderer.render()).save(path)
                written.append(path)
        return written

    def _fresh_data(self) -> mujoco.MjData:
        """MjData with the arm already standing in the autopilot's rest pose."""
        data = mujoco.MjData(self.model)
        for name, angle in self.autopilot.initial_qpos().items():
            data.qpos[self.model.joint(name).qposadr[0]] = angle
            data.ctrl[self.model.actuator(name).id] = angle
        mujoco.mj_forward(self.model, data)
        return data

    def _tracking_error(self) -> float:
        """Distance between the commanded grasp point and where the jaws actually are."""
        rotation = self.data.site_xmat[self.tool_site].reshape(3, 3)
        grasp_point = self.data.site_xpos[self.tool_site] + rotation[:, 2] * self.autopilot.jaw.offset
        return float(np.linalg.norm(grasp_point - self.data.mocap_pos[self.target_mocap]))

    def _geoms_of(self, body_names) -> set[int]:
        """Ids of all geoms belonging to the named bodies."""
        bodies = {self.model.body(name).id for name in body_names}
        return {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in bodies}

    def _unwanted_contact_pairs(self) -> tuple[set[int], set[int], int]:
        """(arm link geoms, hand geoms, table geom) used to spot collisions that should not happen."""
        return self._geoms_of(ARM_LINK_BODIES), self._geoms_of(HAND_BODIES), self.model.geom("table").id

    def _count_contacts(self, unwanted: tuple[set[int], set[int], int]) -> int:
        """Contacts in the current step that mean the arm hit something it should not have."""
        links, hand, table = unwanted
        count = 0
        for contact in self.data.contact[: self.data.ncon]:
            pair = {contact.geom1, contact.geom2}
            if pair & links:
                count += 1
            elif pair & hand and table in pair:
                count += 1
        return count


class CheckCommand:
    """Headless stacking run with a printed report and a pass/fail exit code."""

    def __init__(self, layout: Layout, seconds: float | None, png_dir: Path | None):
        self.simulation = StackSimulation(layout)
        self.layout = layout
        self.duration = seconds or self.simulation.autopilot.duration + SETTLE_S
        self.png_dir = png_dir

    def run(self) -> int:
        """Runs, prints, and returns the process exit code (0 = the stack is complete)."""
        metrics = self.simulation.run(self.duration)
        states = self.simulation.cube_states()
        verdict = StackInspector(self.layout).inspect(states)
        self._print_report(metrics, states, verdict)
        if self.png_dir:
            for path in self.simulation.save_images(self.png_dir):
                print(f"wrote {path}")
        return 0 if verdict.complete else 1

    def _print_report(self, metrics: RunMetrics, states: list[CubeState], verdict: StackVerdict) -> None:
        """One block of plain text: timing, accuracy, cube positions, verdict."""
        model, data = self.simulation.model, self.simulation.data
        print(
            f"simulated {data.time:.2f} s, timestep {model.opt.timestep * 1e3:.1f} ms, "
            f"control {CONTROL_HZ:.0f} Hz"
        )
        for line in metrics.summary_lines():
            print(line)
        for state in states:
            print(f"  {state}")
        print(
            f"stack height {verdict.height * 1e3:.1f} mm "
            f"({len(states)} cubes of {self.layout.size * 1e3:.0f} mm)"
        )
        print(self.simulation.joint_line())
        print(verdict)


def build_parser() -> argparse.ArgumentParser:
    """Command line: one sub-command per mode, all of them accept `--seed`."""
    parser = argparse.ArgumentParser(prog="stack_sim", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("demo", "open the viewer; the arm stacks the three cubes"),
        ("manual", "open the viewer without the autopilot; move the joints via Control sliders"),
        ("check", "headless run; prints the cube positions and whether the stack is complete"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--seed", type=int, help="randomise the cube start poses")
        if name == "check":
            command.add_argument("--seconds", type=float, help="default: the whole plan plus settling")
            command.add_argument("--png", type=Path, help="directory for camera snapshots")
    return parser


def main() -> int:
    """Entry point; returns the process exit code."""
    args = build_parser().parse_args()
    layout = LAYOUT.jittered(args.seed)
    if args.command == "check":
        return CheckCommand(layout, args.seconds, args.png).run()
    StackSimulation(layout).launch_viewer(with_autopilot=args.command == "demo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
