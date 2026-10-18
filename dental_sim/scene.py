"""MuJoCo scene for the dental suction task: SO-101 arm + suction tube + phantom head.

The arm is the official TheRobotStudio MJCF (models/so101), loaded unmodified. Everything dental
(mouth, tube, markers, cameras) is added in code through MuJoCo's MjSpec API, so the vendored
robot files never need editing.

Coordinate frame used everywhere in this package:
    origin = robot base, +x = direction the arm points at zero pose, +z = up.
    The phantom faces the robot (looks toward -x), so the patient's left is world -y.
"""

from dataclasses import dataclass
from typing import NamedTuple

import mujoco
import numpy as np

from so101_common import ROBOT_XML, add_grid_floor

# Collision filtering. MuJoCo lets two geoms touch when
# (a.contype & b.conaffinity) or (b.contype & a.conaffinity) is non-zero.
# The arm uses the MJCF default (contype=1, conaffinity=1). With the values below the tube
# touches only the mouth (never the gripper that holds it) and the mouth touches tube and arm.
TUBE_CONTYPE, TUBE_CONAFFINITY = 2, 2
MOUTH_CONTYPE, MOUTH_CONAFFINITY = 2, 3
VISUAL_ONLY = {"contype": 0, "conaffinity": 0, "group": 1}

TOOTH_RGBA = [0.95, 0.93, 0.85, 1]
GUM_RGBA = [0.85, 0.45, 0.50, 1]
TONGUE_RGBA = [0.80, 0.35, 0.40, 1]
SKIN_RGBA = [0.95, 0.80, 0.70, 0.25]  # translucent so the mouth stays visible from outside
STAND_RGBA = [0.5, 0.5, 0.55, 1]
TUBE_RGBA = [0.55, 0.75, 0.95, 0.9]
TIP_RGBA = [1, 0.2, 0.2, 1]
TARGET_RGBA = [0.1, 0.9, 0.2, 0.6]

GUM_RADIUS = 0.006
ARCH_SAMPLES = 801  # resolution used to space the teeth evenly along the curved arch

# Suction tube, expressed in the vendored "gripper" body frame. In that frame the tool axis is -z and
# the jaw centre (the vendored "gripperframe" site) sits at z = -0.098.
TUBE_AXIS_XY = (-0.0079, -0.0002)  # same x/y as the gripperframe site, i.e. centred between the jaws
TUBE_START_Z = -0.060  # inside the jaws
TUBE_TIP_Z = -0.185  # 87 mm beyond the jaw centre
TUBE_RADIUS = 0.0045  # 9 mm diameter
TUBE_MASS = 0.01
TOOL_AXIS_QUAT = [0.707107, 0, 0.707107, 0]  # turns the site's x-axis onto the tool axis

OFFSCREEN_SIZE = (1280, 960)  # largest image the offscreen renderer may produce (width, height)


class ToothSlot(NamedTuple):
    """Where one tooth sits on the arch, seen from above."""

    x: float
    y: float
    tangent_angle: float  # rotation about z so the tooth lines up with the arch
    rank: int  # 0 = central incisor ... teeth_per_side - 1 = last molar


@dataclass(frozen=True)
class Mouth:
    """Simplified, open adult mouth (all lengths in metres).

    Each dental arch is a parabola seen from above: the incisors are closest to the robot and
    the arch opens away from it. Upper and lower arch are identical, separated vertically.
    """

    front_x: float = 0.33  # incisor line, distance in front of the robot base
    center_z: float = 0.22  # height of the bite plane; below ~0.20 the arm cannot hold the tube level in front of the lips
    depth: float = 0.045  # incisors to second molars
    width: float = 0.050  # molar to molar, tooth centres
    gum_gap: float = 0.050  # upper gum line to lower gum line
    tooth_height: float = 0.009
    teeth_per_side: int = 7

    def half_width(self, depth_behind_incisors: float) -> float:
        """Half the arch width at a given depth behind the incisors."""
        return 0.5 * self.width * np.sqrt(max(depth_behind_incisors, 0.0) / self.depth)

    def gum_z(self, sign: float) -> float:
        """Gum line height; sign +1 for the upper jaw, -1 for the lower jaw."""
        return self.center_z + sign * 0.5 * self.gum_gap

    def tooth_slots(self) -> list[ToothSlot]:
        """Tooth centres spaced at equal distances along one arch, from patient right to left."""
        y = np.linspace(-0.5 * self.width, 0.5 * self.width, ARCH_SAMPLES)
        depth = self.depth * (y / (0.5 * self.width)) ** 2
        arc_length = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(depth), np.diff(y)))])

        count = 2 * self.teeth_per_side
        slots = []
        for index in range(count):
            # each tooth sits in the middle of its share of the arch
            distance = (index + 0.5) * arc_length[-1] / count
            i = min(np.searchsorted(arc_length, distance), len(y) - 1)
            j = max(i - 1, 0)
            angle = np.arctan2(y[i] - y[j], depth[i] - depth[j])
            steps_from_arch_end = min(index, count - 1 - index)
            rank = self.teeth_per_side - 1 - steps_from_arch_end
            slots.append(ToothSlot(self.front_x + depth[i], y[i], angle, rank))
        return slots

    def tooth_half_size(self, rank: int) -> tuple[float, float]:
        """(along-arch, across-arch) half sizes: incisors are small, molars larger."""
        f = rank / (self.teeth_per_side - 1)
        return 0.0030 + 0.0020 * f, 0.0030 + 0.0025 * f


MOUTH = Mouth()


class DentalSceneBuilder:
    """Assembles the complete MuJoCo model for one mouth geometry.

    Usage: `model = DentalSceneBuilder(Mouth()).build()`.
    Named elements other code relies on: site "suction_tip", geom "suction_tube",
    body "phantom_head", mocap body "target", cameras "overview" and "mouth_cam".
    """

    def __init__(self, mouth: Mouth = MOUTH):
        self.mouth = mouth
        self.spec = mujoco.MjSpec.from_file(str(ROBOT_XML))

    def build(self) -> mujoco.MjModel:
        """Adds all scene elements and compiles the model."""
        self._configure_rendering()
        self._add_room()
        head = self.spec.worldbody.add_body(name="phantom_head")
        self._add_mouth(head)
        self._add_suction_tube()
        self._add_target_marker()
        return self.spec.compile()

    def _configure_rendering(self) -> None:
        """Model name, lighting and default camera framing for the viewer."""
        spec = self.spec
        spec.modelname = "so101_dental_suction"
        spec.visual.global_.azimuth = 135
        spec.visual.global_.elevation = -20
        spec.visual.global_.offwidth, spec.visual.global_.offheight = OFFSCREEN_SIZE
        spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
        spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
        # The default free camera looks at stat.center from about 1.5 * stat.extent away.
        spec.stat.center = [0.22, 0.0, 0.17]
        spec.stat.extent = 0.25

    def _add_mouth(self, head: mujoco.MjsBody) -> None:
        """Everything attached to the phantom head. Subclasses replace this to use other jaws."""
        self._add_jaw(head, "upper", +1.0)
        self._add_jaw(head, "lower", -1.0)
        self._add_tongue(head)
        self._add_face(head)

    def _add_room(self) -> None:
        """Floor, a light from above and a fixed side camera."""
        add_grid_floor(self.spec)
        world = self.spec.worldbody
        world.add_light(pos=[0, 0, 2.0], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        world.add_camera(name="overview", pos=[0.24, -0.62, 0.48], xyaxes=[1, 0, 0, 0, 0.6, 0.8])

    def _add_jaw(self, head: mujoco.MjsBody, jaw: str, sign: float) -> None:
        """Teeth plus a gum line for one jaw; sign +1 = upper (teeth hang down), -1 = lower."""
        mouth = self.mouth
        gum_z = mouth.gum_z(sign)
        tooth_z = gum_z - sign * 0.5 * mouth.tooth_height
        slots = mouth.tooth_slots()
        collides = {"contype": MOUTH_CONTYPE, "conaffinity": MOUTH_CONAFFINITY}

        for index, slot in enumerate(slots):
            along, across = mouth.tooth_half_size(slot.rank)
            quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), slot.tangent_angle)
            head.add_geom(
                name=f"tooth_{jaw}_{index:02d}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[slot.x, slot.y, tooth_z],
                quat=quat,
                size=[along, across, 0.5 * mouth.tooth_height],
                rgba=TOOTH_RGBA,
                **collides,
            )
        for index, (a, b) in enumerate(zip(slots, slots[1:])):
            head.add_geom(
                name=f"gum_{jaw}_{index:02d}",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[a.x, a.y, gum_z, b.x, b.y, gum_z],
                size=[GUM_RADIUS, 0, 0],
                rgba=GUM_RGBA,
                **collides,
            )

    def _add_tongue(self, head: mujoco.MjsBody) -> None:
        """Visual-only tongue resting between the lower molars."""
        x, z = self.mouth.front_x, self.mouth.center_z
        head.add_geom(
            name="tongue",
            type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            pos=[x + 0.030, 0, z - 0.022],
            size=[0.024, 0.016, 0.006],
            rgba=TONGUE_RGBA,
            **VISUAL_ONLY,
        )

    def _add_face(self, head: mujoco.MjsBody) -> None:
        """Visual-only translucent skull, the stand it sits on, and a camera into the mouth."""
        x, z = self.mouth.front_x, self.mouth.center_z
        head.add_geom(
            name="skull",
            type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            pos=[x + 0.075, 0, z + 0.045],
            size=[0.085, 0.072, 0.105],
            rgba=SKIN_RGBA,
            **VISUAL_ONLY,
        )
        stand_half_height = 0.5 * (z - 0.06)
        head.add_geom(
            name="neck_stand",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[x + 0.08, 0, stand_half_height],
            size=[0.035, stand_half_height, 0],
            rgba=STAND_RGBA,
            **VISUAL_ONLY,
        )
        head.add_camera(name="mouth_cam", pos=[x - 0.10, 0.0, z + 0.06], xyaxes=[0, -1, 0, 0.5, 0, 1])

    def _add_suction_tube(self) -> None:
        """Rigid suction tube held between the jaws, plus the "suction_tip" site at its open end.

        The tube is attached to the gripper body directly (no grasp physics): on the real arm it
        is clamped or sits in a printed holder.
        """
        gripper = self.spec.body("gripper")
        x, y = TUBE_AXIS_XY
        gripper.add_geom(
            name="suction_tube",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[x, y, TUBE_START_Z, x, y, TUBE_TIP_Z],
            size=[TUBE_RADIUS, 0, 0],
            rgba=TUBE_RGBA,
            mass=TUBE_MASS,
            contype=TUBE_CONTYPE,
            conaffinity=TUBE_CONAFFINITY,
            group=1,
        )
        gripper.add_site(
            name="suction_tip",
            pos=[x, y, TUBE_TIP_Z],
            quat=TOOL_AXIS_QUAT,
            size=[0.004, 0, 0],
            rgba=TIP_RGBA,
            group=1,
        )

    def _add_target_marker(self) -> None:
        """Green sphere the controller moves to wherever the tip should be (a mocap body)."""
        target = self.spec.worldbody.add_body(
            name="target", mocap=True, pos=[self.mouth.front_x, 0, self.mouth.center_z]
        )
        target.add_geom(
            name="target_marker",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.004, 0, 0],
            rgba=TARGET_RGBA,
            **VISUAL_ONLY,
        )
