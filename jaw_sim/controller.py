"""Moves the saliva ejector tip to a target, but only where ForbiddenZones allows it.

Every 1/CONTROL_HZ seconds GuardedController asks a target source for (tip point, hook roll),
asks the forbidden zones whether the point is allowed, and only then solves IK and sends joint
targets. Two target sources:
    DraggedTarget  you move the green marker with the mouse (a "virtual leader arm"); rotating the
                   marker about the shaft direction rolls the hook (the marker's short stick shows the hook)
    EjectorRoute   automatic: enter the mouth, hang the hook behind the lower teeth, sweep left/right

Hook roll = wrist_roll angle. 0 = hook hangs straight down; +90 deg = hook lies flat, pointing to
the patient's left (world -y). Inserting needs the hook flat (it is 46 mm tall, the mouth opening
35 mm); once inside it is rolled down.
"""

from typing import NamedTuple

import mujoco
import numpy as np

from dental_sim.controller import ARM_JOINTS, CONTROL_HZ, SuctionIK
from so101_common import min_jerk

from .ejector import SalivaEjector
from .jaw import BITE_Z, FRONT_X
from .zones import ForbiddenZones

ALLOWED_RGBA = [0.1, 0.9, 0.2, 0.6]
BLOCKED_RGBA = [1.0, 0.1, 0.1, 0.9]
GRIPPER_CLOSED_ON_TUBE = -0.10  # rad, gripper jaws resting on the 6.4 mm shaft


def marker_quat(roll: float) -> np.ndarray:
    """Marker orientation whose stick points like the hook at this roll (rotation about world x by -roll)."""
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([1.0, 0.0, 0.0]), -roll)
    return quat


def roll_from_marker(quat: np.ndarray) -> float:
    """Inverse of marker_quat: the hook roll the marker's current orientation asks for."""
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, quat)
    rotation = rotation.reshape(3, 3)
    return float(np.arctan2(rotation[1, 2], rotation[2, 2]))


# Automatic route (world metres / radians). The shaft passes level through the gap between the
# tooth rows; the hook hangs into the space behind the lower front teeth.
SHAFT_Z = BITE_Z + 0.008  # shaft axis height: 8 mm above the bite plane, clear of the upper tooth tips
INSIDE_X = FRONT_X + 0.040  # tip 40 mm behind the incisor edge
OUTSIDE_X = FRONT_X - 0.030  # tip 30 mm in front of the incisors
FLAT = np.deg2rad(90)  # hook lying flat, for passing the teeth
HANG = np.deg2rad(30)  # hook hanging down, tilted 30 deg so the tip stays above the mouth floor
SWEEP_Y = (-0.011, 0.006)  # tip y at the left and right end of the sweep (patient left = -y)
MOVE_S = 1.6
DWELL_S = 0.6


class Pose(NamedTuple):
    """Where the tip should be and how the hook is rolled."""

    label: str
    tip: np.ndarray
    roll: float


def hook_offset(roll: float) -> np.ndarray:
    """Tip position relative to the point on the (level) shaft axis directly above or beside it."""
    drop = SalivaEjector.hook_drop()
    return np.array([0.0, -drop * np.sin(roll), -drop * np.cos(roll)])


def pose_from_shaft(label: str, x: float, shaft_y: float, roll: float) -> Pose:
    """Pose whose shaft axis passes through (x, shaft_y, SHAFT_Z)."""
    return Pose(label, np.array([x, shaft_y, SHAFT_Z]) + hook_offset(roll), roll)


def pose_from_tip(label: str, x: float, tip_y: float, roll: float) -> Pose:
    """Pose with the tip at lateral position tip_y; the shaft moves sideways to make that happen."""
    shaft_y = tip_y - hook_offset(roll)[1]
    return pose_from_shaft(label, x, shaft_y, roll)


class EjectorRoute:
    """Keyframes for the automatic mode; smooth (min-jerk) motion between them.

    Approach (once): outside, hook flat -> inside -> rolled down.
    Loop (forever): sweep the tip between the left and right end of SWEEP_Y.
    """

    def __init__(self):
        left, right = SWEEP_Y
        hang_left = pose_from_tip("left", INSIDE_X, left, HANG)
        # enter with the shaft where it will be when the hook is rolled down at the left end
        shaft_y = hang_left.tip[1] - hook_offset(HANG)[1]
        self.approach = [
            pose_from_shaft("outside", OUTSIDE_X, shaft_y, FLAT),
            pose_from_shaft("inside, hook flat", INSIDE_X, shaft_y, FLAT),
            hang_left,
        ]
        self.loop = [hang_left, pose_from_tip("right", INSIDE_X, right, HANG)]

    def start(self) -> Pose:
        return self.approach[0]

    @property
    def approach_s(self) -> float:
        return (len(self.approach) - 1) * MOVE_S

    @property
    def loop_s(self) -> float:
        return len(self.loop) * (MOVE_S + DWELL_S)

    def at(self, t: float) -> Pose:
        """Pose at time t."""
        if t < self.approach_s:
            i = int(t // MOVE_S)
            return _blend(self.approach[i], self.approach[i + 1], (t - i * MOVE_S) / MOVE_S, "approach")
        segment = MOVE_S + DWELL_S
        loop_t = (t - self.approach_s) % self.loop_s
        i = int(loop_t // segment)
        into = loop_t - i * segment
        here, there = self.loop[i], self.loop[(i + 1) % len(self.loop)]
        if into < DWELL_S:
            return here
        return _blend(here, there, (into - DWELL_S) / MOVE_S, f"-> {there.label}")


def _blend(a: Pose, b: Pose, s: float, label: str) -> Pose:
    """Min-jerk interpolation of position and roll, s in 0..1."""
    w = min_jerk(min(max(s, 0.0), 1.0))
    return Pose(label, a.tip + w * (b.tip - a.tip), a.roll + w * (b.roll - a.roll))


class RouteTarget:
    """Target source for the automatic mode; also moves the marker along the route."""

    def __init__(self, model: mujoco.MjModel):
        self.marker = model.body("target").mocapid[0]
        self.route = EjectorRoute()

    def start(self) -> Pose:
        return self.route.start()

    def target(self, data: mujoco.MjData) -> Pose:
        pose = self.route.at(data.time)
        data.mocap_pos[self.marker] = pose.tip
        data.mocap_quat[self.marker] = marker_quat(pose.roll)
        return pose


class DraggedTarget:
    """Target = wherever the user dragged the green marker; hook roll = how the marker is rotated."""

    def __init__(self, model: mujoco.MjModel):
        self.marker = model.body("target").mocapid[0]
        self.lower, self.upper = model.jnt_range[model.joint("wrist_roll").id]

    def start(self) -> Pose:
        return EjectorRoute().start()

    def target(self, data: mujoco.MjData) -> Pose:
        roll = np.clip(roll_from_marker(data.mocap_quat[self.marker]), self.lower, self.upper)
        return Pose("dragged", data.mocap_pos[self.marker].copy(), float(roll))


class GuardedController:
    """MuJoCo control callback: target source -> forbidden-zone check -> IK -> joint targets."""

    def __init__(self, model: mujoco.MjModel, source: RouteTarget | DraggedTarget, zones: ForbiddenZones):
        self.source = source
        self.zones = zones
        self.ik = SuctionIK(model)
        self.roll_index = model.joint("wrist_roll").qposadr[0]
        self.actuator = {model.actuator(i).name: i for i in range(model.nu)}
        self.marker_geom = model.geom("target_marker").id
        start = source.start()
        self.last_allowed = start
        self._solve(start, iters=300)
        self.next_tick = 0.0
        self.pose = start
        self.blocked = False
        self.blocked_ticks = 0
        self.ticks = 0

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if data.time < self.next_tick - 1.0 / CONTROL_HZ:  # viewer Reset rewound time
            self.next_tick = 0.0
        if data.time < self.next_tick:
            return
        self.next_tick = data.time + 1.0 / CONTROL_HZ

        self.pose = self.source.target(data)
        self.blocked = not self.zones.allows(self.pose.tip)
        if not self.blocked:
            self.last_allowed = self.pose
        self.ticks += 1
        self.blocked_ticks += self.blocked
        model.geom_rgba[self.marker_geom] = BLOCKED_RGBA if self.blocked else ALLOWED_RGBA

        self._solve(self.last_allowed)
        for name, q in zip(ARM_JOINTS, self.ik.q, strict=True):
            data.ctrl[self.actuator[name]] = q
        data.ctrl[self.actuator["wrist_roll"]] = self.last_allowed.roll
        data.ctrl[self.actuator["gripper"]] = GRIPPER_CLOSED_ON_TUBE

    def start_pose(self) -> dict[str, float]:
        """Joint angles that put the tip at the start pose."""
        joints = dict(zip(ARM_JOINTS, self.ik.q, strict=True))
        return joints | {"wrist_roll": self.last_allowed.roll, "gripper": GRIPPER_CLOSED_ON_TUBE}

    def _solve(self, pose: Pose, iters: int = 30) -> None:
        """IK for the tip, with wrist_roll fixed at the pose's roll (it moves the tip too)."""
        self.ik.data.qpos[self.roll_index] = pose.roll
        self.ik.solve(pose.tip, iters=iters)
