"""Scene for the block-stacking task: SO-101 arm, table, three cubes, stack marker.

The arm comes from the official TheRobotStudio MJCF (models/so101), loaded unmodified.
Everything else is added programmatically through MjSpec, so the vendored files stay pristine.

World frame: the arm base sits at the origin on the table top (z = 0), the arm points along
+x in its zero pose, z is up. The cubes start on the table in front of the arm and are stacked
on the marked pad at `Layout.stack`.
"""

from dataclasses import dataclass, replace

import mujoco
import numpy as np

from so101_common import ROBOT_XML, add_grid_floor

# --- table and room ---------------------------------------------------------------------
TABLE_TOP_Z = 0.0  # the plane everything is manipulated on; the arm base stands on it
FLOOR_Z = -0.70  # ground under the table, i.e. the table height of a normal desk
TABLE_CENTER_XY = (0.10, 0.0)  # shifted forward so the whole workspace is supported
TABLE_HALF_SIZE = (0.36, 0.32)
TABLE_THICKNESS = 0.024
TABLE_FRICTION = (1.0, 0.02, 0.001)  # (slide, spin, roll)
LEG_RADIUS = 0.016
LEG_POSITIONS_XY = ((0.42, 0.28), (0.42, -0.28), (-0.22, 0.28), (-0.22, -0.28))

# --- cubes ------------------------------------------------------------------------------
# 25 mm is a size the SO-101 jaws can really close on, and three of them stacked stay inside
# the arm's top-down reach (the tool cannot be driven much above z = 0.09 pointing straight down).
DEFAULT_CUBE_SIZE = 0.025
DEFAULT_CUBE_MASS = 0.022  # ~1400 kg/m3: a small wooden/plastic block
CUBE_FRICTION = (1.2, 0.05, 0.002)  # sticky enough for a two-finger friction grasp
CUBE_SOLIMP = (0.95, 0.99, 0.0005, 0.5, 2)  # stiff contact, so a clamped cube barely sinks in
CUBE_SPAWN_GAP = 2e-4  # cubes start a hair above the table and settle in the first steps
CUBE_COLORS = (
    ("red", (0.85, 0.20, 0.18, 1)),
    ("green", (0.20, 0.70, 0.30, 1)),
    ("blue", (0.20, 0.40, 0.85, 1)),
)

# --- contact solver ---------------------------------------------------------------------
# Grasping needs the friction constraints to be about as stiff as the normal ones. With
# MuJoCo's default impratio of 1 the cube creeps out from between the jaws under the clamp
# force instead of being held by friction.
CONTACT_IMPRATIO = 15
CONTACT_CONE = mujoco.mjtCone.mjCONE_ELLIPTIC

# --- start-pose randomisation -------------------------------------------------------------
JITTER_POSITION_M = 0.010
JITTER_YAW_DEG = 20.0

# --- cameras and visuals ------------------------------------------------------------------
OVERVIEW_EYE = (0.44, -0.46, 0.30)
OVERVIEW_LOOK_AT = (0.19, -0.01, 0.04)
OVERVIEW_FOVY = 50
FRONT_EYE = (0.62, 0.0, 0.09)
FRONT_LOOK_AT = (0.205, 0.0, 0.045)
# Wrist camera, in gripper-body coordinates: the tool axis is -z there and the jaws close
# along +-x, so the camera sits off to the side and looks down past the jaws.
WRIST_EYE = (-0.02, -0.075, -0.015)
WRIST_LOOK_AT = (0.012, 0.0, -0.135)
WRIST_FOVY = 55
FREE_CAMERA_CENTER = (0.18, 0.0, 0.05)  # the viewer frames this at ~1.5 * extent
FREE_CAMERA_EXTENT = 0.28


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> list[float]:
    """Camera `xyaxes` (the right and up vectors) for a camera at `eye` facing `target`."""
    eye, target, up = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    backward = eye - target
    backward /= np.linalg.norm(backward)
    right = np.cross(up, backward)
    right /= np.linalg.norm(right)
    return [*right, *np.cross(backward, right)]


@dataclass(frozen=True)
class Layout:
    """Cube geometry plus where the cubes start and where the stack is built (metres, radians)."""

    size: float = DEFAULT_CUBE_SIZE
    mass: float = DEFAULT_CUBE_MASS
    starts: tuple[tuple[float, float, float], ...] = (
        (0.175, -0.145, 0.25),
        (0.250, -0.050, -0.30),
        (0.190, 0.150, 0.50),
    )
    stack: tuple[float, float] = (0.205, 0.0)

    @property
    def names(self) -> tuple[str, ...]:
        """Body/geom name of every cube, in the order they are stacked."""
        return tuple(f"cube_{color}" for color, _ in CUBE_COLORS[: len(self.starts)])

    def level_height(self, level: int) -> float:
        """Height of the centre of the cube sitting on `level` of the finished stack."""
        return TABLE_TOP_Z + level * self.size + self.size / 2

    def jittered(self, seed: int | None) -> "Layout":
        """A copy with randomly perturbed cube start poses; `seed=None` returns the nominal layout."""
        if seed is None:
            return self
        rng = np.random.default_rng(seed)
        starts = tuple(
            (
                x + rng.uniform(-JITTER_POSITION_M, JITTER_POSITION_M),
                y + rng.uniform(-JITTER_POSITION_M, JITTER_POSITION_M),
                yaw + np.deg2rad(rng.uniform(-JITTER_YAW_DEG, JITTER_YAW_DEG)),
            )
            for x, y, yaw in self.starts
        )
        return replace(self, starts=starts)


LAYOUT = Layout()


class SceneBuilder:
    """Assembles the MjSpec for one `Layout`: vendored arm, table, cubes, markers, cameras."""

    def __init__(self, layout: Layout = LAYOUT):
        self.layout = layout
        self.spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
        self.world = self.spec.worldbody

    def build(self) -> mujoco.MjSpec:
        """Returns the finished spec. Call once per builder."""
        self._set_options()
        self._add_room()
        self._add_cameras()
        self._add_table()
        self._add_stack_pad()
        self._add_cubes()
        self._add_target_marker()
        return self.spec

    def _set_options(self) -> None:
        """Model name, contact solver settings and viewer defaults."""
        spec = self.spec
        spec.modelname = "so101_block_stacking"
        spec.option.impratio = CONTACT_IMPRATIO
        spec.option.cone = CONTACT_CONE
        spec.visual.global_.azimuth = 150
        spec.visual.global_.elevation = -25
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 960
        spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
        spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
        spec.stat.center = list(FREE_CAMERA_CENTER)
        spec.stat.extent = FREE_CAMERA_EXTENT

    def _add_room(self) -> None:
        """Ground plane well below the table, plus one overhead and one fill light."""
        add_grid_floor(self.spec, z=FLOOR_Z)
        self.world.add_light(
            pos=[0.2, 0, 1.2],
            dir=[0, 0, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        )
        self.world.add_light(pos=[0.1, -0.5, 0.8], dir=[0.1, 0.5, -0.8], diffuse=[0.35, 0.35, 0.35])

    def _add_cameras(self) -> None:
        """Fixed `overview` and `front` cameras, plus a `wrist` camera on the gripper body."""
        self.world.add_camera(
            name="overview",
            pos=list(OVERVIEW_EYE),
            xyaxes=look_at(OVERVIEW_EYE, OVERVIEW_LOOK_AT),
            fovy=OVERVIEW_FOVY,
        )
        self.world.add_camera(
            name="front",
            pos=list(FRONT_EYE),
            xyaxes=look_at(FRONT_EYE, FRONT_LOOK_AT),
        )
        self.spec.body("gripper").add_camera(
            name="wrist",
            pos=list(WRIST_EYE),
            xyaxes=look_at(WRIST_EYE, WRIST_LOOK_AT, up=(0, 0, 1)),
            fovy=WRIST_FOVY,
        )

    def _add_table(self) -> None:
        """Table top at z = 0 (the arm is bolted to it) on four visual-only legs."""
        half_thickness = TABLE_THICKNESS / 2
        self.world.add_geom(
            name="table",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[TABLE_CENTER_XY[0], TABLE_CENTER_XY[1], TABLE_TOP_Z - half_thickness],
            size=[TABLE_HALF_SIZE[0], TABLE_HALF_SIZE[1], half_thickness],
            rgba=[0.72, 0.60, 0.44, 1],
            friction=list(TABLE_FRICTION),
            condim=4,
        )
        leg_half_length = 0.5 * (TABLE_TOP_Z - TABLE_THICKNESS - FLOOR_Z)
        for x, y in LEG_POSITIONS_XY:
            self.world.add_geom(
                name=f"leg_{x:+.2f}_{y:+.2f}",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[x, y, FLOOR_Z + leg_half_length],
                size=[LEG_RADIUS, leg_half_length, 0],
                rgba=[0.55, 0.45, 0.33, 1],
                contype=0,
                conaffinity=0,
                group=1,
            )

    def _add_stack_pad(self) -> None:
        """Flat, non-colliding disc on the table that marks where the stack is built."""
        x, y = self.layout.stack
        self.world.add_geom(
            name="stack_pad",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[x, y, TABLE_TOP_Z + 2e-4],
            size=[self.layout.size * 0.9, 2e-4, 0],
            rgba=[0.95, 0.85, 0.25, 0.8],
            contype=0,
            conaffinity=0,
            group=1,
        )

    def _add_cubes(self) -> None:
        """One free-floating cube per start pose; they are moved by contact only."""
        size = self.layout.size
        for (color, rgba), (x, y, yaw) in zip(CUBE_COLORS, self.layout.starts):
            body = self.world.add_body(
                name=f"cube_{color}",
                pos=[x, y, TABLE_TOP_Z + size / 2 + CUBE_SPAWN_GAP],
                quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
            )
            body.add_freejoint(name=f"cube_{color}_free")
            body.add_geom(
                name=f"cube_{color}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[size / 2, size / 2, size / 2],
                rgba=rgba,
                mass=self.layout.mass,
                friction=list(CUBE_FRICTION),
                condim=4,
                solimp=list(CUBE_SOLIMP),
            )

    def _add_target_marker(self) -> None:
        """Mocap sphere the autopilot parks on its commanded grasp point (viewer feedback only)."""
        x, y = self.layout.stack
        target = self.world.add_body(name="target", mocap=True, pos=[x, y, self.layout.size / 2])
        target.add_geom(
            name="target_marker",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.004, 0, 0],
            rgba=[0.1, 0.9, 0.2, 0.6],
            contype=0,
            conaffinity=0,
            group=1,
        )


def build_spec(layout: Layout = LAYOUT) -> mujoco.MjSpec:
    """Scene spec for `layout` (editable: add geoms, then compile yourself)."""
    return SceneBuilder(layout).build()


def build_model(layout: Layout = LAYOUT) -> mujoco.MjModel:
    """Compiled scene for `layout`."""
    return SceneBuilder(layout).build().compile()
