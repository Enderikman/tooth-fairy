"""The dental "brain" without a window: jaw_sim's zones, targets and IK, producing joint targets.

Isaac Sim (and later the real SO-101) only needs one thing from the brain: six joint position
targets in radians, 50 times per second. jaw_sim computes exactly that inside a MuJoCo model,
so this class runs that MuJoCo model headless and hands out its actuator targets. Nothing in
jaw_sim is copied or changed; this file only wires its classes together without the viewer,
buttons and voice listener (those need a desktop and a microphone).

Joint names and units match LeRobot's SO-101 follower and LeIsaac's SO-101 articulation:
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper; radians.
"""

import mujoco
import numpy as np

from jaw_sim.calibration import JawCalibration
from jaw_sim.controller import DraggedTarget, GuardedController, RouteTarget, marker_quat
from jaw_sim.scene import JawSceneBuilder
from jaw_sim.zones import ForbiddenZones

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class DentalBrain:
    """jaw_sim's control stack on a headless MuJoCo model.

    mode "route": the automatic route (jaw_sim RouteTarget).
    mode "drag":  commanded moves (jaw_sim DraggedTarget): call command("operate"), then a tooth
                  name such as "46", or "park" / "stop", exactly like the buttons in jaw_sim.
    """

    def __init__(self, mode: str):
        self.mode = mode
        self.zones = ForbiddenZones(JawCalibration.load_or_default())
        self.model = JawSceneBuilder(self.zones).build()
        self.data = mujoco.MjData(self.model)
        if mode == "drag":
            self.source = DraggedTarget(self.model, self.zones)
        else:
            self.source = RouteTarget(self.model, self.zones)
        self.controller = GuardedController(self.model, self.source, self.zones)
        self.tip = self.model.site("suction_tip").id
        self.marker = self.model.body("target").mocapid[0]
        self.actuator = {name: self.model.actuator(name).id for name in JOINTS}
        self.operating = False
        self._place_arm_at_start()

    @property
    def time(self) -> float:
        return float(self.data.time)

    def step(self, seconds: float) -> dict[str, float]:
        """Advances the brain by `seconds` and returns the current joint targets (rad)."""
        mujoco.set_mjcb_control(self.controller)
        try:
            end = self.data.time + seconds
            while self.data.time < end:
                mujoco.mj_step(self.model, self.data)
        finally:
            mujoco.set_mjcb_control(None)
        return self.joint_targets()

    def joint_targets(self) -> dict[str, float]:
        return {name: float(self.data.ctrl[i]) for name, i in self.actuator.items()}

    def start_pose(self) -> dict[str, float]:
        """Joint angles the arm should start in (the robot is placed there before the first step)."""
        return dict(self.controller.start_pose())

    def command(self, name: str) -> None:
        """Same command names as jaw_sim: "operate", "stop", "park", or a tooth ("46", "43", "33", "36")."""
        if self.mode != "drag":
            raise ValueError(f"commands need drag mode, not {self.mode!r}")
        if name == "operate":
            self.operating = True
        elif name == "stop":
            self.source.stop()
            self.operating = False
        elif name == "park":
            self.source.park(self.data.time)
        elif self.operating:
            self.source.goto(name, self.data.time)

    @property
    def busy(self) -> bool:
        """True while a commanded move is still running (drag mode)."""
        return bool(getattr(self.source, "busy", False))

    def status(self) -> dict:
        """Numbers for logs and on-screen text: tip, target, blocked flag (mm)."""
        return {
            "label": self.controller.pose.label,
            "blocked": bool(self.controller.blocked),
            "tip_mm": (self.data.site_xpos[self.tip] * 1000).round(1).tolist(),
            "target_mm": (np.asarray(self.controller.pose.tip) * 1000).round(1).tolist(),
        }

    def _place_arm_at_start(self) -> None:
        for name, q in self.start_pose().items():
            self.data.qpos[self.model.joint(name).qposadr[0]] = q
            self.data.ctrl[self.actuator[name]] = q
        start = self.source.start()
        self.data.mocap_pos[self.marker] = start.tip
        self.data.mocap_quat[self.marker] = marker_quat(start.roll)
        mujoco.mj_forward(self.model, self.data)
