"""The real SO-101 follower arm, without a leader: moved by hand to calibrate, then driven by the sim.

Two states, switched with the usual commands (buttons b / c, see jaw_sim/__main__.py):
    released (start, after `stop`)   torque off; you move the real arm by hand, the simulated arm
                                     copies it joint by joint (same servo reading as jaw_sim/leader.py)
    driving  (after `operate`)       torque on; the sim plans the tooth/park commands with the forbidden
                                     zones, and the real arm follows the sim's joint targets at 50 Hz
                                     (the sim's actuator targets, in LeRobot-calibrated ticks)

One-point jaw calibration: hold the real hook tip on the incisal edge between 31 and 41 of the real
jaw and press `calibrate` twice (start, record). The simulated jaw jumps so that its incisal edge is
at that tip, keeping its orientation: incisors toward the robot, incisal plane level. Everything
else (arch size, corridor, depth) comes from the hand-measured default jaw, or from calibration.json
if you also record points 2 and 3. The offset is saved to jaw_offset.json and reloaded next time.

Needs JAW_SCALE=1: the real jaw is real size.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import scservo_sdk

from dental_sim.controller import ARM_JOINTS, SuctionIK

from .controller import DraggedTarget, GuardedController, Pose, marker_quat
from .leader import HALF_TURN, MOTOR_IDS, SIGNS, TICKS_PER_TURN, LeaderArm, LeaderTarget
from .zones import ForbiddenZones

ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
DRIVEN_JOINTS = ARM_JOINTS + ("wrist_roll",)  # the gripper keeps the position you closed it to by hand
OFFSET_FILE = Path(__file__).resolve().parent.parent / "jaw_offset.json"
REACH_TOLERANCE = 0.005  # m, a plan keyframe the IK misses by more than this makes the command refused


def load_world_offset() -> np.ndarray:
    """The saved jaw offset (metres), zero when there is none."""
    if not OFFSET_FILE.exists():
        return np.zeros(3)
    return np.array(json.loads(OFFSET_FILE.read_text())["world_offset_m"], dtype=float)


def save_world_offset(offset: np.ndarray) -> Path:
    OFFSET_FILE.write_text(json.dumps({"world_offset_m": [round(float(v), 4) for v in offset]}, indent=2))
    return OFFSET_FILE


class FollowerArm(LeaderArm):
    """Reads the follower's servos like the leader, and can also hold and move them."""

    def __init__(self, port: str):
        super().__init__(port)
        self.goals = scservo_sdk.GroupSyncWrite(self.port, self.packets, ADDR_GOAL_POSITION, 2)
        self.sent: dict[str, int] = {}
        self.torque(False)

    def torque(self, on: bool) -> None:
        """Torque on = the servos hold their current position and follow goals; off = movable by hand."""
        if on:  # goal := present first, otherwise a stale goal register would make the arm jump
            self.read_ticks()
            self.sent = {}
            self._write_goals(self.ticks)
        for motor_id in MOTOR_IDS.values():
            self.packets.write1ByteTxRx(self.port, motor_id, ADDR_TORQUE_ENABLE, int(on))

    def drive(self, angles: dict[str, float]) -> None:
        """Sends joint angles (radians, sim convention) as goal positions; unchanged goals are not resent."""
        ticks = {}
        for name, angle in angles.items():
            turn = SIGNS[name] * angle * TICKS_PER_TURN / (2 * np.pi)
            ticks[name] = round(HALF_TURN + turn) % TICKS_PER_TURN
        if ticks != {name: self.sent.get(name) for name in ticks}:
            self._write_goals(ticks)

    def _write_goals(self, ticks: dict[str, int]) -> None:
        self.goals.clearParam()
        for name, value in ticks.items():
            self.goals.addParam(MOTOR_IDS[name], [scservo_sdk.SCS_LOBYTE(value), scservo_sdk.SCS_HIBYTE(value)])
        self.goals.txPacket()
        self.sent |= ticks


class FollowerTarget:
    """Target source of the follower mode: the real arm while released, the command planner while driving."""

    def __init__(self, model: mujoco.MjModel, follower: FollowerArm, zones: ForbiddenZones):
        self.follower = follower
        self.mirror = LeaderTarget(model, follower, label="follower")
        self.planner = DraggedTarget(model, zones)
        self.marker = self.planner.marker
        self.actuator = {name: model.actuator(name).id for name in DRIVEN_JOINTS}
        self.check_ik = SuctionIK(model)  # tries a plan's keyframes before the real arm is sent along them
        self.roll_index = model.joint("wrist_roll").qposadr[0]
        self.controller: GuardedController | None = None  # set by engage
        self.driving = False
        self.commanded = False  # a plan has run since engage: joint targets go to the real arm
        self.refused = ""  # why the last command was not executed, empty if it was

    @property
    def angles(self) -> dict[str, float]:
        """Last read joint angles of the real arm."""
        return self.mirror.angles

    @property
    def busy(self) -> bool:
        return self.driving and self.planner.busy

    def start(self) -> Pose:
        return self.mirror.start()

    def real_tip(self) -> np.ndarray:
        """Where the real hook tip is now (forward kinematics of the last servo reading)."""
        return self.mirror.pose().tip

    def engage(self, data: mujoco.MjData, controller: GuardedController) -> None:
        """Torque on, holding the pose the real arm is in; the planner and the controller continue from it.

        Joint targets are only sent once a command has started a plan: the real pose may lie just
        outside the sim's joint ranges (the folded rest pose does), and holding must not move the arm.
        """
        pose = self.mirror.read()
        self.planner.stop()
        self.planner.held = pose
        controller.last_allowed = pose
        data.mocap_pos[self.marker] = pose.tip
        data.mocap_quat[self.marker] = marker_quat(pose.roll)
        self.follower.torque(True)
        self.controller = controller
        self.driving = True
        self.commanded = False
        self.refused = ""

    def release(self) -> None:
        """Torque off: cancels any running plan and frees the arm to be moved by hand."""
        self.planner.stop()
        self.follower.torque(False)
        self.driving = False

    def stop(self) -> None:
        self.release()

    def goto(self, tooth: str, now: float) -> None:
        self.planner.goto(tooth, now)
        self._run_plan_if_reachable()

    def park(self, now: float) -> None:
        self.planner.park(now)
        self._run_plan_if_reachable()

    def _run_plan_if_reachable(self) -> None:
        """Keeps the plan the planner just made only if the IK reaches every keyframe from the current joints.

        From a folded pose (e.g. the rest pose) the IK runs into joint limits and would send the real
        arm somewhere unplanned; such a command is refused and the arm has to be moved by hand first.
        """
        self.refused = ""
        self.check_ik.q = self.controller.ik.q.copy()
        for pose in self.planner.plan[1:]:
            self.check_ik.data.qpos[self.roll_index] = pose.roll
            error = self.check_ik.solve(pose.tip, iters=300)
            if error > REACH_TOLERANCE:
                self.refused = f"out of reach from here by {error * 1000:.0f} mm; move the arm by hand (c, then b)"
                self.planner.stop()
                return
        self.commanded = True

    def target(self, data: mujoco.MjData) -> Pose:
        """Released: the real pose (marker on its tip). Driving: the planned pose while a command runs,
        else the pose the last command ended on (the marker is not dragged in this mode); the real arm
        gets the joint targets the controller computed in the previous tick (data.ctrl)."""
        if not self.driving:
            return self.mirror.target(data)
        self.mirror.read()  # keeps the on-screen joint readout live
        if self.commanded:
            self.follower.drive({name: float(data.ctrl[index]) for name, index in self.actuator.items()})
        if self.planner.busy:
            return self.planner.target(data)
        data.mocap_pos[self.marker] = self.planner.held.tip
        return self.planner.held
