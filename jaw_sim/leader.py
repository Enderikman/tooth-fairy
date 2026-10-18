"""The real SO-101 leader arm as the target source: the simulated arm copies it, joint by joint.

The leader's Feetech STS3215 servos are read over the USB bus board. One-time setup: calibrate the
leader with LeRobot (e.g. `solo robo --calibrate leader`, which guides you through it). That
calibration writes a homing offset into every servo so that the middle pose reads HALF_TURN ticks.
That middle pose is also the simulated arm's zero, so
    joint angle = (ticks - HALF_TURN) * 2 pi / TICKS_PER_TURN * SIGNS[joint]
and `uv run mjpython -m jaw_sim leader` makes the simulated arm follow the leader.

The forbidden zones still apply: when the leader goes where the tip is not allowed, the simulated
arm stops at the last allowed pose and the marker turns red (GuardedController).
"""

import glob

import mujoco
import numpy as np
import scservo_sdk

from dental_sim.controller import ARM_JOINTS

from .controller import GRIPPER_CLOSED_ON_TUBE, Pose

BAUD_RATE = 1_000_000
PROTOCOL_END = 0  # STS servos: protocol version 0
ADDR_PRESENT_POSITION = 56
TICKS_PER_TURN = 4096
HALF_TURN = 2047  # what a LeRobot-calibrated servo reads in the middle pose
MOTOR_IDS = {"shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3, "wrist_flex": 4, "wrist_roll": 5, "gripper": 6}
# +1 when the servo counts up in the same direction as the simulated joint, -1 otherwise
# (checked against the real leader on 2026-10-18; wrist_roll and gripper not checked yet).
SIGNS = {"shoulder_pan": 1, "shoulder_lift": 1, "elbow_flex": 1, "wrist_flex": 1, "wrist_roll": 1, "gripper": 1}


def find_port() -> str:
    """The USB serial port of the bus board (the only /dev/tty.usbmodem* device)."""
    ports = glob.glob("/dev/tty.usbmodem*")
    if len(ports) != 1:
        raise RuntimeError(f"expected one /dev/tty.usbmodem* port, found {ports}; pass --port")
    return ports[0]


class LeaderArm:
    """Reads the six servo positions of the leader arm."""

    def __init__(self, port: str):
        self.port = scservo_sdk.PortHandler(port)
        self.packets = scservo_sdk.PacketHandler(PROTOCOL_END)
        if not self.port.openPort() or not self.port.setBaudRate(BAUD_RATE):
            raise RuntimeError(f"cannot open {port}")
        self.ticks = self.read_ticks()

    def read_ticks(self) -> dict[str, int]:
        """Raw servo positions, 0 .. TICKS_PER_TURN - 1.

        A corrupted packet (happens now and then at 50 Hz) keeps that servo's previous value;
        only the very first read must succeed for every servo.
        """
        ticks = dict(getattr(self, "ticks", {}))
        for name, motor_id in MOTOR_IDS.items():
            value, result, error = self.packets.read2ByteTxRx(self.port, motor_id, ADDR_PRESENT_POSITION)
            if result == scservo_sdk.COMM_SUCCESS and not error:
                ticks[name] = value
            elif name not in ticks:
                raise RuntimeError(f"servo {motor_id} ({name}): {self.packets.getTxRxResult(result)}")
        self.ticks = ticks
        return ticks

    def angles(self) -> dict[str, float]:
        """Joint angles in radians, zero at the LeRobot-calibrated middle pose."""
        angles = {}
        for name, ticks in self.read_ticks().items():
            turn = (ticks - HALF_TURN + TICKS_PER_TURN / 2) % TICKS_PER_TURN - TICKS_PER_TURN / 2
            angles[name] = SIGNS[name] * turn * 2 * np.pi / TICKS_PER_TURN
        return angles


class LeaderTarget:
    """Target source: the leader's joint angles, plus where they put the tip (for the zone check)."""

    def __init__(self, model: mujoco.MjModel, leader: LeaderArm, label: str = "leader"):
        self.model = model
        self.leader = leader
        self.label = label
        self.scratch = mujoco.MjData(model)  # forward kinematics only, never stepped
        self.tip = model.site("suction_tip").id
        self.marker = model.body("target").mocapid[0]
        self.qpos_index = {name: model.joint(name).qposadr[0] for name in MOTOR_IDS}
        self.lower = {name: model.jnt_range[model.joint(name).id][0] for name in MOTOR_IDS}
        self.upper = {name: model.jnt_range[model.joint(name).id][1] for name in MOTOR_IDS}
        self.angles = {name: 0.0 for name in MOTOR_IDS}

    def start(self) -> Pose:
        return self.read()

    def target(self, data: mujoco.MjData) -> Pose:
        """Current leader pose; the green marker is moved to its tip so the zone colour is visible."""
        pose = self.read()
        data.mocap_pos[self.marker] = pose.tip
        return pose

    def read(self) -> Pose:
        """Reads the servos and returns the pose they are in (joint ranges of the sim model applied)."""
        self.angles = {name: float(np.clip(a, self.lower[name], self.upper[name])) for name, a in self.leader.angles().items()}
        return self.pose()

    def pose(self) -> Pose:
        """Pose of the arm at the last read angles: tip from forward kinematics, joints copied."""
        for name, angle in self.angles.items():
            self.scratch.qpos[self.qpos_index[name]] = angle
        self.scratch.qpos[self.qpos_index["gripper"]] = GRIPPER_CLOSED_ON_TUBE
        mujoco.mj_kinematics(self.model, self.scratch)
        joints = np.array([self.angles[name] for name in ARM_JOINTS])
        return Pose(self.label, self.scratch.site_xpos[self.tip].copy(), self.angles["wrist_roll"], joints)
