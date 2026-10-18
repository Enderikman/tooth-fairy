"""Where the suction tip should be, and the joint angles that put it there.

Data flow, 50 times per second:
    SuctionTrajectory.at(time) -> target tip position
    SuctionIK.solve(target)    -> joint angles
    Autopilot                  -> writes those angles into data.ctrl (the servo targets)

Joint position targets at a fixed rate are exactly what LeRobot sends to the real SO-101,
so the same Autopilot output could drive the hardware.
"""

from typing import NamedTuple

import mujoco
import numpy as np

from so101_common import min_jerk

from .scene import MOUTH, Mouth

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
# wrist_roll spins the tube about its own axis, which changes nothing for suction: held at 0.
# gripper: -0.17 rad is fully closed; -0.10 rests the jaws on the 9 mm tube.
HELD_JOINTS = {"wrist_roll": 0.0, "gripper": -0.10}

CONTROL_HZ = 50.0
MOVE_S = 1.6  # time to travel between two waypoints
DWELL_S = 0.6  # time spent at each waypoint

TIP_CLEARANCE = 0.011  # tip centre this far inside the line of tooth centres (tongue side)
TIP_HEIGHT = 0.008  # tip this far from the bite plane, toward the jaw being treated
MOLAR_DEPTH_FRACTION = 0.8  # molar waypoints at 80 % of the arch depth
FRONT_DEPTH = 0.014  # front waypoint, behind the incisors
STANDBY_GAP = 0.03  # standby tip position in front of the incisors


class Waypoint(NamedTuple):
    """A named suction position in world coordinates."""

    name: str
    position: np.ndarray


class SuctionRoute:
    """The fixed route through the mouth, derived from the mouth geometry."""

    def __init__(self, mouth: Mouth = MOUTH):
        self.mouth = mouth

    def standby(self) -> np.ndarray:
        """Tip position just outside the mouth, level with the bite plane."""
        return np.array([self.mouth.front_x - STANDBY_GAP, 0.0, self.mouth.center_z])

    def waypoints(self) -> list[Waypoint]:
        """Lower jaw: left molars -> front -> right molars, then upper jaw: right -> front -> left."""
        mouth = self.mouth
        molar_x = mouth.front_x + MOLAR_DEPTH_FRACTION * mouth.depth
        molar_y = mouth.half_width(MOLAR_DEPTH_FRACTION * mouth.depth) - TIP_CLEARANCE
        front_x = mouth.front_x + FRONT_DEPTH

        route = []
        for jaw, sign in (("lower", -1.0), ("upper", 1.0)):
            z = mouth.center_z + sign * TIP_HEIGHT
            row = [
                Waypoint(f"{jaw} left molars", np.array([molar_x, -molar_y, z])),
                Waypoint(f"{jaw} front", np.array([front_x, 0.0, z])),
                Waypoint(f"{jaw} right molars", np.array([molar_x, molar_y, z])),
            ]
            route += row if jaw == "lower" else row[::-1]
        return route


class SuctionTrajectory:
    """Smooth tip motion: approach into the mouth once, then loop through the waypoints forever.

    The approach starts wherever the tip currently is. It first backs out horizontally to the
    standby distance (staying level keeps the tube between the teeth), moves to standby, then
    enters to the first waypoint. Every move uses a min-jerk profile: starts and stops gently.
    """

    def __init__(self, start: np.ndarray, route: SuctionRoute):
        self.waypoints = route.waypoints()
        standby = route.standby()
        back_out = np.array([standby[0], start[1], start[2]])
        path = [start, back_out, standby, self.waypoints[0].position]
        # drop zero-length legs (e.g. when already at standby)
        self.approach = [path[0]] + [p for prev, p in zip(path, path[1:]) if np.linalg.norm(p - prev) > 1e-3]

    @property
    def approach_s(self) -> float:
        """Duration of the approach."""
        return (len(self.approach) - 1) * MOVE_S

    @property
    def loop_s(self) -> float:
        """Duration of one full pass through all waypoints."""
        return len(self.waypoints) * (MOVE_S + DWELL_S)

    def at(self, t: float) -> tuple[str, np.ndarray]:
        """(label, tip target) at time t. Label is "approach", "-> <waypoint>" or the waypoint name."""
        if t < self.approach_s:
            leg = int(t // MOVE_S)
            p0, p1 = self.approach[leg], self.approach[leg + 1]
            return "approach", p0 + min_jerk((t - leg * MOVE_S) / MOVE_S) * (p1 - p0)

        # The loop time is shifted by one move so the loop begins dwelling at waypoint 0,
        # where the approach just ended.
        segment = MOVE_S + DWELL_S
        loop_t = (t - self.approach_s + MOVE_S) % self.loop_s
        index = int(loop_t // segment)
        into_segment = loop_t - index * segment
        previous = self.waypoints[index - 1].position
        current = self.waypoints[index]
        if into_segment >= MOVE_S:
            return current.name, current.position
        return f"-> {current.name}", previous + min_jerk(into_segment / MOVE_S) * (current.position - previous)


class SuctionIK:
    """Inverse kinematics: joint angles that put the suction tip on a target, tube kept level.

    Method: damped least squares (a numerically safe version of "move the joints along the
    Jacobian until the error is gone"), on a private copy of the model state so the running
    simulation is never disturbed. Four joints, four goals: tip x, y, z and tube pitch.
    """

    MAX_TIP_STEP = 0.01  # m per iteration; following the straight line avoids folding into joint limits
    MAX_JOINT_STEP = 0.2  # rad per iteration
    PITCH_WEIGHT = 0.05  # keeping the tube level matters less than reaching the target
    TOLERANCE = 2e-4  # m

    def __init__(self, model: mujoco.MjModel, tilt_deg: float = 0.0):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site("suction_tip").id
        joints = [model.joint(name) for name in ARM_JOINTS]
        self.qpos_index = np.array([j.qposadr[0] for j in joints])
        self.dof_index = np.array([j.dofadr[0] for j in joints])
        self.lower = model.jnt_range[[j.id for j in joints], 0]
        self.upper = model.jnt_range[[j.id for j in joints], 1]
        self.target_axis_z = -np.sin(np.deg2rad(tilt_deg))  # 0 = tube horizontal
        self.q = np.zeros(len(ARM_JOINTS))  # current solution, reused as the next starting guess
        self._jac_pos = np.zeros((3, model.nv))
        self._jac_rot = np.zeros((3, model.nv))

    def reset(self, q: np.ndarray | None = None) -> None:
        """Restart the solver from q (default: all zeros)."""
        self.q = np.zeros(len(ARM_JOINTS)) if q is None else q.copy()

    def solve(self, target: np.ndarray, iters: int = 30, damping: float = 1e-3) -> float:
        """Improves self.q toward the target; returns the remaining tip error in metres."""
        error = np.inf
        for _ in range(iters):
            tip, axis = self._forward_kinematics()
            tip_error = target - tip
            error = float(np.linalg.norm(tip_error))
            jacobian, residual = self._linearise(tip_error, error, axis)
            dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping**2 * np.eye(4), residual)
            dq = np.clip(dq, -self.MAX_JOINT_STEP, self.MAX_JOINT_STEP)
            self.q = np.clip(self.q + dq, self.lower, self.upper)
            if error < self.TOLERANCE:  # error was measured before this last small correction
                break
        return error

    def _forward_kinematics(self) -> tuple[np.ndarray, np.ndarray]:
        """Tip position and tube direction for the current self.q."""
        self.data.qpos[self.qpos_index] = self.q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        tip = self.data.site_xpos[self.site]
        axis = self.data.site_xmat[self.site].reshape(3, 3)[:, 0]
        return tip, axis

    def _linearise(self, tip_error: np.ndarray, error: float, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """4x4 Jacobian and residual: rows 0-2 tip position, row 3 tube pitch (weighted)."""
        mujoco.mj_jacSite(self.model, self.data, self._jac_pos, self._jac_rot, self.site)
        jac_pos = self._jac_pos[:, self.dof_index]
        jac_rot = self._jac_rot[:, self.dof_index]
        jac_axis_z = np.cross(jac_rot.T, axis)[:, 2]  # how each joint tilts the tube up/down
        jacobian = np.vstack([jac_pos, self.PITCH_WEIGHT * jac_axis_z])
        step = tip_error * min(1.0, self.MAX_TIP_STEP / max(error, 1e-9))
        residual = np.concatenate([step, [self.PITCH_WEIGHT * (self.target_axis_z - axis[2])]])
        return jacobian, residual


class Autopilot:
    """MuJoCo control callback: every 1/CONTROL_HZ s, compute and apply new joint targets.

    Install with `mujoco.set_mjcb_control(autopilot)`; MuJoCo then calls it on every physics step.
    Also moves the green "target" marker so the viewer shows where the tip is headed.
    """

    def __init__(self, model: mujoco.MjModel, route: SuctionRoute | None = None):
        self.route = route or SuctionRoute()
        self.ik = SuctionIK(model)
        self.trajectory = SuctionTrajectory(self.route.standby(), self.route)
        self.actuator = {model.actuator(i).name: i for i in range(model.nu)}
        self.marker = model.body("target").mocapid[0]
        self.next_tick = 0.0
        self.label = ""  # what the autopilot is doing right now (for reports)
        self.ik_error = 0.0
        # Solve once for standby so the arm can start there, outside the mouth.
        self.ik.solve(self.route.standby(), iters=300)

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._time_went_backwards(data):
            self._replan_from_current_pose(data)
        if data.time < self.next_tick:
            return
        self.next_tick = data.time + 1.0 / CONTROL_HZ

        self.label, target = self.trajectory.at(data.time)
        self.ik_error = self.ik.solve(target)
        data.mocap_pos[self.marker] = target
        for name, q in zip(ARM_JOINTS, self.ik.q):
            data.ctrl[self.actuator[name]] = q
        for name, q in HELD_JOINTS.items():
            data.ctrl[self.actuator[name]] = q

    def start_pose(self) -> dict[str, float]:
        """Joint angles to initialise the simulation with (arm at standby)."""
        return dict(zip(ARM_JOINTS, self.ik.q)) | HELD_JOINTS

    def _time_went_backwards(self, data: mujoco.MjData) -> bool:
        """True after the viewer's Reset button (time rewinds to 0)."""
        return data.time < self.next_tick - 1.0 / CONTROL_HZ

    def _replan_from_current_pose(self, data: mujoco.MjData) -> None:
        """Reset puts the arm at joint zero; restart the approach from wherever the tip now is."""
        self.ik.reset(data.qpos[self.ik.qpos_index])
        self.trajectory = SuctionTrajectory(data.site_xpos[self.ik.site].copy(), self.route)
        self.next_tick = 0.0
