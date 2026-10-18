"""Moves the saliva ejector tip to a target, but only where ForbiddenZones allows it.

Every 1/CONTROL_HZ seconds GuardedController asks a target source for (tip point, hook roll),
asks the forbidden zones whether the point is allowed, and only then solves IK and sends joint
targets. Two target sources:
    DraggedTarget  you move the green marker with the mouse (a "virtual leader arm"); rotating the
                   marker about the shaft direction rolls the hook (the marker's short stick shows the hook)
    EjectorRoute   automatic: enter the mouth, lower the hook next to tooth 46, sweep to 43 and back

DraggedTarget also accepts commands ("go to tooth 46", "park", "stop"): a command plans a short
list of poses (lift the hook flat, travel, lower it again) that the marker follows until it is
done, after which mouse dragging continues from the pose the plan ended on.

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
from .jaw import BITE_Z, FRONT_X, LOWER
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
# tooth rows; the hook is lowered into the corridor next to the teeth (jaw_sim/zones.py).
SHAFT_Z = BITE_Z + 0.008  # shaft axis height: 8 mm above the bite plane, clear of the upper tooth tips
OUTSIDE_X = FRONT_X - 0.030  # tip 30 mm in front of the incisors
FLAT = np.deg2rad(90)  # hook lying flat, for passing the teeth
HANG = np.deg2rad(30)  # hook hanging down, tilted 30 deg so the tip stays above the mouth floor
ROUTE_TEETH = ("46", "43")  # the automatic route sweeps between these (same side of the arch)
MOVE_S = 1.6
DWELL_S = 0.6


class Pose(NamedTuple):
    """Where the tip should be and how the hook is rolled.

    joints: set when the source already knows the arm joint angles (leader arm); then no IK is
    needed and the arm copies them exactly. None = find the joints by IK from tip and roll.
    """

    label: str
    tip: np.ndarray
    roll: float
    joints: np.ndarray | None = None  # ARM_JOINTS order, radians


def park_pose(side: int) -> Pose:
    """Pose waiting in front of the incisors, hook flat and rolled to the given side (+1 / -1)."""
    tip = np.array([OUTSIDE_X, -side * SalivaEjector.hook_drop(), SHAFT_Z])
    return Pose("parked", tip, FLAT * side)


def lifted_pose(label: str, tip: np.ndarray, side: int) -> Pose:
    """Pose above tip with the hook flat, the only height at which it may travel horizontally."""
    return Pose(label, np.array([tip[0], tip[1], SHAFT_Z]), FLAT * side)


def pose_side(value: float) -> int:
    """Roll sign (+1 / -1) of a roll angle or of a tip y coordinate."""
    return 1 if value >= 0.0 else -1


def tooth_pose(zones: ForbiddenZones, tooth: str) -> Pose:
    """Suction pose next to a tooth ("46", "43", "33", "36"): tip in the corridor, hook rolled to that side."""
    tip = LOWER.to_world(zones.lower.tooth_target(tooth))
    return Pose(tooth, tip, HANG * pose_side(-tip[1]))


class EjectorRoute:
    """Keyframes for the automatic mode; smooth (min-jerk) motion between them.

    Approach (once): parked outside, hook flat -> flat above the first tooth -> lowered.
    Loop (forever): sweep the tip between the ROUTE_TEETH suction points.
    """

    def __init__(self, zones: ForbiddenZones):
        first, second = (tooth_pose(zones, tooth) for tooth in ROUTE_TEETH)
        side = pose_side(first.roll)
        self.approach = [park_pose(side), lifted_pose(first.label, first.tip, side), first]
        self.loop = [first, second]

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

    def __init__(self, model: mujoco.MjModel, zones: ForbiddenZones):
        self.marker = model.body("target").mocapid[0]
        self.route = EjectorRoute(zones)

    def start(self) -> Pose:
        return self.route.start()

    def target(self, data: mujoco.MjData) -> Pose:
        pose = self.route.at(data.time)
        data.mocap_pos[self.marker] = pose.tip
        data.mocap_quat[self.marker] = marker_quat(pose.roll)
        return pose


class DraggedTarget:
    """Target = wherever the user dragged the green marker; hook roll = how the marker is rotated.

    Commands (goto / park) replace the dragging for a few seconds: they plan a list of poses and
    the marker is driven along it, so the user can pick the drag up again at the end pose.
    """

    def __init__(self, model: mujoco.MjModel, zones: ForbiddenZones):
        self.marker = model.body("target").mocapid[0]
        self.zones = zones
        self.lower, self.upper = model.jnt_range[model.joint("wrist_roll").id]
        self.plan: list[Pose] = []
        self.plan_start = 0.0
        self.held = park_pose(1)

    @property
    def busy(self) -> bool:
        """True while a planned motion is still running."""
        return bool(self.plan)

    def start(self) -> Pose:
        """Starting pose of the marker: parked in front of the mouth."""
        return park_pose(1)

    def goto(self, tooth: str, now: float) -> None:
        """Plan a safe path from the marker's current pose to the suction pose next to this tooth."""
        self._plan_to(tooth_pose(self.zones, tooth), now)

    def park(self, now: float) -> None:
        """Plan a safe path out of the mouth to the park pose on the side the hook is rolled to."""
        self._plan_to(park_pose(pose_side(self.held.roll)), now)

    def stop(self) -> None:
        """Cancel the running plan; the marker (and so the arm) holds its current pose."""
        self.plan = []

    def target(self, data: mujoco.MjData) -> Pose:
        """Pose for this control step: the planned pose if a plan runs, else the dragged marker."""
        if not self.plan:
            self.held = self._marker_pose(data)
            return self.held
        index = int((data.time - self.plan_start) // MOVE_S)
        if index >= len(self.plan) - 1:
            pose = self.plan[-1]
            self.plan = []
        else:
            fraction = (data.time - self.plan_start) / MOVE_S - index
            pose = _blend(self.plan[index], self.plan[index + 1], fraction, self.plan[index + 1].label)
        data.mocap_pos[self.marker] = pose.tip
        data.mocap_quat[self.marker] = marker_quat(pose.roll)
        self.held = pose
        return pose

    def _marker_pose(self, data: mujoco.MjData) -> Pose:
        """Pose the marker currently shows, with the roll clamped to the wrist joint range."""
        roll = np.clip(roll_from_marker(data.mocap_quat[self.marker]), self.lower, self.upper)
        return Pose("dragged", data.mocap_pos[self.marker].copy(), float(roll))

    def _plan_to(self, goal: Pose, now: float) -> None:
        """Store the pose list from the latest marker pose to goal; one MOVE_S per segment."""
        here = self.held
        side_here = pose_side(here.roll)
        side_goal = pose_side(goal.roll)
        plan = [here, lifted_pose(here.label, here.tip, side_here)]
        if side_here != side_goal:
            plan += [park_pose(side_here), park_pose(side_goal)]
        plan += [lifted_pose(goal.label, goal.tip, side_goal), goal]
        self.plan = plan
        self.plan_start = now


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
        self.guarded = True  # False while calibrating: the tip must be allowed to touch the teeth

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if data.time < self.next_tick - 1.0 / CONTROL_HZ:  # viewer Reset rewound time
            self.next_tick = 0.0
        if data.time < self.next_tick:
            return
        self.next_tick = data.time + 1.0 / CONTROL_HZ

        self.pose = self.source.target(data)
        self.blocked = self.guarded and not self.zones.allows(self.pose.tip)
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
        """Joint angles for the pose: copied when the pose has them, else IK for the tip at its roll."""
        if pose.joints is not None:
            self.ik.q = pose.joints.copy()
            return
        self.ik.data.qpos[self.roll_index] = pose.roll
        self.ik.solve(pose.tip, iters=iters)
