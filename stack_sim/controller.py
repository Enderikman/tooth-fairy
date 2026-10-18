"""Scripted pick-and-place: cube poses in, SO-101 joint position targets out.

What leaves this module is exactly what LeRobot sends to a real SO-101: one goal position per
joint at a fixed control rate. The physics actuators track those targets, and the cubes are
moved by nothing but jaw contact and friction (no welds, no mocap attachment).

Tool frame: the vendored site `gripperframe` sits between the jaws with its x-axis along the
approach axis. The plan is written in terms of a *grasp point* - the place where the centre of
a held cube is - which sits on the +z axis of that site frame, `JawCalibration.offset` away.

Layering:
    JawProbe      measures the jaw geometry of the compiled model (offset, closing angle)
    ToolPose      one commanded tool state (grasp point, yaw, jaw opening)
    Segment       one timed move between two tool poses
    StackPlan     turns live cube poses into segments: pick a cube, then place it
    ArmIK         damped-least-squares inverse kinematics for the five arm joints
    Autopilot     ties it together and writes data.ctrl at CONTROL_HZ
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from so101_common import min_jerk

from .scene import LAYOUT, TABLE_TOP_Z, Layout

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_JOINT = "gripper"
TOOL_SITE = "gripperframe"
CONTROL_HZ = 50.0

# --- rest pose ----------------------------------------------------------------------------
# Grasp point of the rest pose: high above the table and off to the side, so the arm neither
# starts on top of a cube nor brushes the finished stack on its way home.
HOME_POINT = np.array([0.195, -0.095, 0.085])
# The solver has to start from an elbow-up guess; from the all-zero pose (arm stretched
# horizontally) it converges into a folded configuration that cannot reach the table.
ELBOW_UP_SEED = np.array([0.0, -0.5, 1.0, 0.5, 0.0])
HOME_HOLD_S = 1.5

# --- grasp tuning ---------------------------------------------------------------------------
SQUEEZE_RAD = 0.06  # commanded past first contact; the position servo then pushes on the cube
EXTRA_OPENING_RAD = 0.35  # ~25 mm of extra gap, so a cube drops in without being nudged
PRE_GRASP_HEIGHT = 0.050  # how far above a cube the approach starts
LIFT_MARGIN = 0.020  # a carried cube passes this far above the tallest obstacle
DROP_GAP = 0.002  # a cube is released this far above its resting height
MAX_GRASP_CORRECTION = 0.012  # clamp on the measured cube-in-jaws offset, in case a grasp fails

# --- segment durations [s] --------------------------------------------------------------------
REACH_S = 1.30
DESCEND_S = 0.70
CLOSE_S = 0.50
LIFT_S = 0.70
CARRY_S = 1.60
PLACE_S = 0.80
RELEASE_S = 0.40
RETREAT_S = 0.60
CUBE_CYCLE_S = REACH_S + DESCEND_S + CLOSE_S + LIFT_S + CARRY_S + PLACE_S + RELEASE_S + RETREAT_S

# --- inverse kinematics -----------------------------------------------------------------------
IK_ITERATIONS = 40
IK_SETTLE_ITERATIONS = 400  # one-off solve for the rest pose, which starts far from the target
IK_DAMPING = 1e-2
IK_MAX_POSITION_STEP = 0.02  # m per iteration: keeps the solver on the straight line to the goal
IK_MAX_ROTATION_STEP = 0.3  # rad per iteration
IK_MAX_JOINT_STEP = 0.15  # rad per iteration
IK_POSITION_TOL = 1e-4
IK_ROTATION_TOL = 1e-3

# --- jaw probe --------------------------------------------------------------------------------
PROBE_BISECTION_STEPS = 24
PROBE_MAX_EXTRA_OFFSET = 0.04  # search range above half a cube for the grasp-point offset
PROBE_OFFSET_MARGIN = 5e-4  # clearance left to the fixed finger at the grasp pose
PROBE_OPEN_MARGIN = 0.1  # stay this far inside the gripper's upper limit while probing

QUARTER_TURN = np.pi / 2


def fold_to_quarter_turn(yaw: float) -> float:
    """Folds an angle into [-45, 45) degrees, because a cube looks the same every 90 degrees."""
    return (yaw + QUARTER_TURN / 2) % QUARTER_TURN - QUARTER_TURN / 2


def yaw_of(quat: np.ndarray) -> float:
    """Rotation of a quaternion about z, folded into [-45, 45) degrees."""
    w, x, y, z = quat
    return fold_to_quarter_turn(float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    """Scales `vector` down so its length is at most `limit`."""
    length = float(np.linalg.norm(vector))
    if length <= limit or length < 1e-12:
        return vector
    return vector * (limit / length)


@dataclass(frozen=True)
class JawCalibration:
    """Where a cube of one size fits between the jaws of the compiled model."""

    offset: float  # grasp centre, measured from the tool site along its +z axis
    touch: float  # gripper joint angle at which the moving jaw first touches that cube

    @property
    def closed(self) -> float:
        """Jaw target that clamps the cube: past first contact, so the servo keeps pushing."""
        return self.touch - SQUEEZE_RAD

    @property
    def open(self) -> float:
        """Jaw target used while approaching and after releasing."""
        return self.touch + EXTRA_OPENING_RAD


class JawProbe:
    """Measures the jaw geometry of a compiled model by test-placing a cube between the jaws.

    The SO-101 collision meshes are convex hulls, so the usable clamping surfaces are not the
    finger faces you see in the viewer: the hull of the fixed finger bulges across the jaw
    opening. Rather than hard-coding numbers that would silently rot if the model or the cube
    size changed, this class puts a cube at a candidate grasp point, asks MuJoCo for the contact
    distances and bisects for the pose and angle at which each jaw just touches.
    """

    def __init__(self, model: mujoco.MjModel, layout: Layout = LAYOUT):
        self.model = model
        self.layout = layout
        self.data = mujoco.MjData(model)
        self.site = model.site(TOOL_SITE).id
        self.gripper_qposadr = model.joint(GRIPPER_JOINT).qposadr[0]
        self.gripper_range = model.joint(GRIPPER_JOINT).range
        cube_name = layout.names[0]
        self.cube_geom = model.geom(cube_name).id
        self.cube_qposadr = model.joint(f"{cube_name}_free").qposadr[0]
        self.fixed_jaw = self._geoms_of("gripper")
        self.moving_jaw = self._geoms_of("moving_jaw_so101_v1")

    def measure(self) -> JawCalibration:
        """Grasp-point offset and first-contact jaw angle for this model's cube size."""
        wide_open = self.gripper_range[1] - PROBE_OPEN_MARGIN
        offset = self._smallest_clear(
            lo=self.layout.size / 2,
            hi=self.layout.size / 2 + PROBE_MAX_EXTRA_OFFSET,
            clearance=lambda value: self._clearance(value, wide_open, self.fixed_jaw),
        )
        offset += PROBE_OFFSET_MARGIN
        touch = self._smallest_clear(
            lo=self.gripper_range[0],
            hi=wide_open,
            clearance=lambda value: self._clearance(offset, value, self.moving_jaw),
        )
        return JawCalibration(offset=offset, touch=touch)

    def _geoms_of(self, body_name: str) -> set[int]:
        """Ids of all geoms belonging to one body."""
        body = self.model.body(body_name).id
        return {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == body}

    def _clearance(self, offset: float, grip: float, jaw: set[int]) -> float:
        """Signed distance between `jaw` and a cube held at `offset` (negative = penetrating)."""
        model, data = self.model, self.data
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0
        data.qpos[self.gripper_qposadr] = grip
        mujoco.mj_kinematics(model, data)
        rotation = data.site_xmat[self.site].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.flatten())
        data.qpos[self.cube_qposadr : self.cube_qposadr + 3] = data.site_xpos[self.site] + rotation @ [0, 0, offset]
        data.qpos[self.cube_qposadr + 3 : self.cube_qposadr + 7] = quat
        mujoco.mj_forward(model, data)
        distances = [
            contact.dist
            for contact in data.contact[: data.ncon]
            if self.cube_geom in (contact.geom1, contact.geom2) and {contact.geom1, contact.geom2} & jaw
        ]
        return min(distances) if distances else 0.0

    def _smallest_clear(self, lo: float, hi: float, clearance) -> float:
        """Smallest value in [lo, hi] whose `clearance` is non-negative.

        `clearance` must be increasing: penetration shrinks as the grasp point moves out of the
        jaw and as the moving jaw opens, which is what both searches rely on.
        """
        for _ in range(PROBE_BISECTION_STEPS):
            middle = 0.5 * (lo + hi)
            if clearance(middle) < 0:
                lo = middle
            else:
                hi = middle
        return hi


@dataclass(frozen=True)
class ToolPose:
    """One commanded tool state: where the grasp centre is, how it is turned, how open the jaws are."""

    point: np.ndarray  # world position of the grasp centre
    yaw: float  # rotation of the jaws about the vertical approach axis [rad]
    grip: float  # gripper joint target [rad]

    def rotation(self) -> np.ndarray:
        """Desired world orientation of the tool site: approach axis down, jaws closing along `yaw`."""
        approach = np.array([0.0, 0.0, -1.0])  # site x
        closing = np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])  # site z
        return np.column_stack([approach, np.cross(closing, approach), closing])

    def site_target(self, offset: float) -> tuple[np.ndarray, np.ndarray]:
        """Position and orientation the tool site must reach for the grasp centre to land on `point`."""
        rotation = self.rotation()
        return self.point - rotation[:, 2] * offset, rotation

    def blend(self, other: "ToolPose", fraction: float) -> "ToolPose":
        """Min-jerk interpolation towards `other`; `fraction` runs from 0 to 1."""
        weight = min_jerk(fraction)
        return ToolPose(
            self.point + weight * (other.point - self.point),
            self.yaw + weight * (other.yaw - self.yaw),
            self.grip + weight * (other.grip - self.grip),
        )


@dataclass(frozen=True)
class Segment:
    """One timed move: reach `pose` within `duration`, starting from wherever the tool is."""

    label: str
    pose: ToolPose
    duration: float


class StackPlan:
    """Turns live cube poses into segments, one half-cycle at a time.

    Planning is split into a pick batch and a place batch on purpose. The place batch is built
    after the cube has actually been lifted, so it can measure how the cube ended up sitting in
    the jaws and aim the drop accordingly. That absorbs both the random start poses and the
    small shift a cube makes while the jaws close on it.
    """

    def __init__(self, model: mujoco.MjModel, jaw: JawCalibration, layout: Layout = LAYOUT):
        self.jaw = jaw
        self.layout = layout
        self.cube_bodies = [model.body(name).id for name in layout.names]
        self.placed = 0  # cubes already on the stack
        self.holding = False

    def home_pose(self) -> ToolPose:
        """Rest pose with open jaws."""
        return ToolPose(HOME_POINT.copy(), 0.0, self.jaw.open)

    def finished(self) -> bool:
        """True once every cube has been placed and nothing is held."""
        return self.placed >= len(self.cube_bodies) and not self.holding

    def total_duration(self) -> float:
        """Sim time the whole plan needs, including the trip home at the end."""
        return len(self.cube_bodies) * CUBE_CYCLE_S + HOME_HOLD_S

    def next_segments(self, data: mujoco.MjData, pose: ToolPose) -> list[Segment]:
        """Next batch of segments; `pose` is the tool pose currently commanded."""
        if self.finished():
            return [Segment("home", self.home_pose(), HOME_HOLD_S)]
        if self.holding:
            segments = self._place_segments(data, pose)
        else:
            segments = self._pick_segments(data)
        self.holding = not self.holding
        return segments

    def _travel_height(self) -> float:
        """Grasp-point height while carrying: clears both the stack so far and a loose cube."""
        size = self.layout.size
        tallest_obstacle = max(self.placed * size, size)
        return TABLE_TOP_Z + tallest_obstacle + size / 2 + LIFT_MARGIN

    def _cube_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, float]:
        """World position and folded yaw of the cube currently being handled."""
        body = self.cube_bodies[self.placed]
        return data.xpos[body].copy(), yaw_of(data.xquat[body])

    def _pick_segments(self, data: mujoco.MjData) -> list[Segment]:
        """Approach the next cube where it lies, close the jaws on it, lift it to travel height."""
        name = self.layout.names[self.placed]
        cube, yaw = self._cube_pose(data)
        travel_z = self._travel_height()
        above = np.array([cube[0], cube[1], max(cube[2] + PRE_GRASP_HEIGHT, travel_z)])
        lifted = np.array([cube[0], cube[1], travel_z])
        return [
            Segment(f"reach {name}", ToolPose(above, yaw, self.jaw.open), REACH_S),
            Segment(f"descend {name}", ToolPose(cube, yaw, self.jaw.open), DESCEND_S),
            Segment(f"close {name}", ToolPose(cube, yaw, self.jaw.closed), CLOSE_S),
            Segment(f"lift {name}", ToolPose(lifted, yaw, self.jaw.closed), LIFT_S),
        ]

    def _place_segments(self, data: mujoco.MjData, pose: ToolPose) -> list[Segment]:
        """Carry to the stack and drop, corrected for how the cube actually sits in the jaws."""
        name = self.layout.names[self.placed]
        cube, cube_yaw = self._cube_pose(data)

        # How far the cube is from the commanded grasp point, expressed in the tool frame. The
        # cube is held rigidly, so this offset travels with the tool and can be cancelled out
        # by aiming the tool that much to the other side of the stack.
        grasp_yaw = pose.yaw
        offset_in_tool = clip_norm(_rotation_2d(-grasp_yaw) @ (cube[:2] - pose.point[:2]), MAX_GRASP_CORRECTION)
        height_in_tool = float(np.clip(cube[2] - pose.point[2], -MAX_GRASP_CORRECTION, MAX_GRASP_CORRECTION))
        drop_yaw = -fold_to_quarter_turn(cube_yaw - grasp_yaw)  # lands the cube square on the stack

        target_xy = np.array(self.layout.stack) - _rotation_2d(drop_yaw) @ offset_in_tool
        drop_z = self.layout.level_height(self.placed) + DROP_GAP - height_in_tool
        drop = np.array([target_xy[0], target_xy[1], drop_z])
        over = np.array([target_xy[0], target_xy[1], self._travel_height()])

        self.placed += 1
        return [
            Segment(f"carry {name}", ToolPose(over, drop_yaw, self.jaw.closed), CARRY_S),
            Segment(f"place {name}", ToolPose(drop, drop_yaw, self.jaw.closed), PLACE_S),
            Segment(f"release {name}", ToolPose(drop, drop_yaw, self.jaw.open), RELEASE_S),
            Segment(f"retreat {name}", ToolPose(over, drop_yaw, self.jaw.open), RETREAT_S),
        ]


def _rotation_2d(angle: float) -> np.ndarray:
    """2x2 rotation matrix about z."""
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin], [sin, cos]])


class ArmIK:
    """Damped least squares on the full tool-site pose: five arm joints, six residuals.

    The SO-101 has one vertical and three parallel horizontal joint axes plus a roll axis along
    the tool, which is exactly enough for a top-down pose at any yaw inside the joint limits.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site(TOOL_SITE).id
        self.qposadr = np.array([model.joint(name).qposadr[0] for name in ARM_JOINTS])
        self.dofadr = np.array([model.joint(name).dofadr[0] for name in ARM_JOINTS])
        joint_ids = [model.joint(name).id for name in ARM_JOINTS]
        self.lower = model.jnt_range[joint_ids, 0]
        self.upper = model.jnt_range[joint_ids, 1]
        self.q = ELBOW_UP_SEED.copy()
        self._jac_pos = np.zeros((3, model.nv))
        self._jac_rot = np.zeros((3, model.nv))
        self._quat = np.zeros(4)

    def reset(self, q: np.ndarray | None = None) -> None:
        """Re-seeds the solver, by default with the elbow-up guess."""
        self.q = ELBOW_UP_SEED.copy() if q is None else np.asarray(q, float).copy()

    def solve(self, position: np.ndarray, rotation: np.ndarray, iterations: int = IK_ITERATIONS) -> float:
        """Updates `self.q` (warm-started from the last call) and returns the position error [m]."""
        error = np.inf
        for _ in range(iterations):
            self._forward()
            position_error = position - self.data.site_xpos[self.site]
            rotation_error = self._rotation_error(rotation)
            error = float(np.linalg.norm(position_error))
            if error < IK_POSITION_TOL and np.linalg.norm(rotation_error) < IK_ROTATION_TOL:
                break
            self._step(position_error, rotation_error)
        return error

    def _forward(self) -> None:
        """Kinematics of the arm at the current solver guess (cubes stay at their model pose)."""
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.qposadr] = self.q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def _rotation_error(self, rotation: np.ndarray) -> np.ndarray:
        """World-frame rotation vector from the current site orientation to `rotation`."""
        current = self.data.site_xmat[self.site].reshape(3, 3)
        mujoco.mju_mat2Quat(self._quat, (rotation @ current.T).flatten())
        sin_half = float(np.linalg.norm(self._quat[1:]))
        if sin_half < 1e-9:
            return np.zeros(3)
        angle = 2 * np.arctan2(sin_half, self._quat[0])
        return self._quat[1:] / sin_half * angle

    def _step(self, position_error: np.ndarray, rotation_error: np.ndarray) -> None:
        """One damped-least-squares joint update, with the step sizes capped."""
        mujoco.mj_jacSite(self.model, self.data, self._jac_pos, self._jac_rot, self.site)
        jacobian = np.vstack([self._jac_pos[:, self.dofadr], self._jac_rot[:, self.dofadr]])
        residual = np.concatenate(
            [
                clip_norm(position_error, IK_MAX_POSITION_STEP),
                clip_norm(rotation_error, IK_MAX_ROTATION_STEP),
            ]
        )
        damped = jacobian @ jacobian.T + IK_DAMPING**2 * np.eye(6)
        dq = jacobian.T @ np.linalg.solve(damped, residual)
        self.q = np.clip(self.q + np.clip(dq, -IK_MAX_JOINT_STEP, IK_MAX_JOINT_STEP), self.lower, self.upper)


class Autopilot:
    """Runs the stacking plan at CONTROL_HZ and writes joint targets into `data.ctrl`.

    Install it with `mujoco.set_mjcb_control(autopilot)`; it is called every physics step and
    only acts on control ticks.
    """

    def __init__(self, model: mujoco.MjModel, layout: Layout = LAYOUT):
        self.model = model
        self.layout = layout
        self.jaw = JawProbe(model, layout).measure()
        self.ik = ArmIK(model)
        self.actuators = {model.actuator(i).name: i for i in range(model.nu)}
        self.mocap = model.body("target").mocapid[0]
        self.label = "home"
        self.ik_error = 0.0
        self.next_tick = 0.0
        self._restart()
        # Converge once on the rest pose so the arm starts from a solved configuration.
        self.ik.solve(*self.pose.site_target(self.jaw.offset), iterations=IK_SETTLE_ITERATIONS)
        self.home_q = self.ik.q.copy()

    @property
    def duration(self) -> float:
        """Sim time the plan needs from a fresh start."""
        return self.plan.total_duration()

    def initial_qpos(self) -> dict[str, float]:
        """Joint angles of the rest pose, for seeding `MjData` before the first step."""
        return dict(zip(ARM_JOINTS, self.home_q)) | {GRIPPER_JOINT: self.jaw.open}

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """MuJoCo control callback: advance the plan and command the joints."""
        if data.time < self.next_tick - 1.0 / CONTROL_HZ:
            self._restart()  # the viewer's Reset rewound time: start the plan over
        if data.time < self.next_tick:
            return
        self.next_tick = data.time + 1.0 / CONTROL_HZ
        self._advance_segments(data)
        self._command(data, self._current_target(data))

    def _restart(self) -> None:
        """Drops any progress and parks the plan at the rest pose."""
        self.plan = StackPlan(self.model, self.jaw, self.layout)
        self.pose = self.plan.home_pose()  # pose the active segment starts from
        self.segments: list[Segment] = []
        self.segment_index = 0
        self.segment_start = 0.0
        self.next_tick = 0.0
        self.ik.reset()

    def _advance_segments(self, data: mujoco.MjData) -> None:
        """Retires finished segments and asks the plan for more when the batch runs out."""
        while True:
            if self.segment_index >= len(self.segments):
                self.segments = self.plan.next_segments(data, self.pose)
                self.segment_index = 0
                self.segment_start = data.time
                return
            active = self.segments[self.segment_index]
            if data.time < self.segment_start + active.duration:
                return
            self.pose = active.pose
            self.segment_start += active.duration
            self.segment_index += 1

    def _current_target(self, data: mujoco.MjData) -> ToolPose:
        """Interpolated tool pose for the current sim time."""
        active = self.segments[self.segment_index]
        self.label = active.label
        fraction = (data.time - self.segment_start) / active.duration
        return self.pose.blend(active.pose, min(1.0, fraction))

    def _command(self, data: mujoco.MjData, target: ToolPose) -> None:
        """Solves IK for `target` and writes one position target per joint."""
        position, rotation = target.site_target(self.jaw.offset)
        self.ik_error = self.ik.solve(position, rotation)
        data.mocap_pos[self.mocap] = target.point
        for name, angle in zip(ARM_JOINTS, self.ik.q):
            data.ctrl[self.actuators[name]] = angle
        data.ctrl[self.actuators[GRIPPER_JOINT]] = target.grip
