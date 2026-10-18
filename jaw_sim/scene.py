"""MuJoCo scene: SO-101 + saliva ejector + your jaw STL files + forbidden-zone markers.

Reuses DentalSceneBuilder from dental_sim (arm, floor, target marker, cameras) and replaces two parts:
    the mouth         -> models/jaw/*.stl, placed by jaw_sim/jaw.py
    the suction tube  -> the saliva ejector from models/sucker/, see jaw_sim/ejector.py

Jaws and ejector are visual only (they do not collide). MuJoCo collides meshes by their convex
hull, which for a horseshoe-shaped jaw or a hook would fill the whole mouth; keeping the tip out
of the teeth is the job of the forbidden zones instead.
"""

import itertools

import mujoco
import numpy as np

from dental_sim.scene import TIP_RGBA, TOOL_AXIS_QUAT, TUBE_AXIS_XY, TUBE_RGBA, VISUAL_ONLY, DentalSceneBuilder, Mouth

from .ejector import STL_TO_GRIPPER, SalivaEjector
from .jaw import BITE_Z, FRONT_X, JAW_SCALE, JAWS, LOWER, MM, JawModel
from .zones import ForbiddenZones

JAW_RGBA = [0.95, 0.93, 0.85, 1]
FORBIDDEN_RGBA = [1.0, 0.1, 0.1, 0.35]
FORBIDDEN_MARKER_RADIUS = 0.0008 * JAW_SCALE  # m, 0.8 mm at real size
FORBIDDEN_GROUP = 2  # site group; toggle in the viewer under Rendering > Site groups
ZONE_GRID_STEP = 0.004 * JAW_SCALE  # spacing of the forbidden-zone sample points (m); 4 mm at real size
ZONE_GRID_MARGIN = 0.008 * JAW_SCALE  # how far the sampled box reaches beyond the jaws (m)
HOOK_STICK_LENGTH = 0.012  # m, the stick on the target marker showing the hook direction

GRIP_CENTER_Z = -0.098  # gripper frame: point between the jaws (same as the vendored "gripperframe" site)
EJECTOR_MASS = 0.01  # kg

# The route sizes of dental_sim's Mouth, taken from the lower jaw scan (used for camera framing;
# the saliva ejector route itself lives in jaw_sim/controller.py).
TOOTH_CENTRE_WIDTH = 0.85  # tooth-centre arch width as a fraction of the outer arch width
TOOTH_CENTRE_DEPTH = 0.85  # incisors -> last molar centres, fraction of the scan depth
JAW_MOUTH = Mouth(
    front_x=FRONT_X,
    center_z=BITE_Z - SalivaEjector.hook_drop(),
    depth=TOOTH_CENTRE_DEPTH * LOWER.depth_mm * MM,
    width=TOOTH_CENTRE_WIDTH * LOWER.width_mm * MM,
)


class JawSceneBuilder(DentalSceneBuilder):
    """DentalSceneBuilder with STL jaws, the saliva ejector, and red dots where zones say "not allowed"."""

    def __init__(self, zones: ForbiddenZones):
        super().__init__(JAW_MOUTH)
        self.zones = zones
        self.ejector = SalivaEjector()

    def _configure_rendering(self) -> None:
        super()._configure_rendering()
        self.spec.modelname = "so101_jaw"

    def _add_mouth(self, head: mujoco.MjsBody) -> None:
        for jaw in JAWS:
            self._add_jaw_mesh(head, jaw)
        self._add_face(head)
        self._add_zone_markers(head)

    def _add_jaw_mesh(self, head: mujoco.MjsBody, jaw: JawModel) -> None:
        """Loads one STL (millimetres) and places it as described in jaw_sim/jaw.py."""
        self.spec.add_mesh(
            name=f"{jaw.name}_teeth",
            file=str(jaw.sim_mesh()),
            scale=[MM * JAW_SCALE] * 3,
            inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,  # scans are open surfaces, not solids
        )
        head.add_geom(
            name=f"{jaw.name}_teeth",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"{jaw.name}_teeth",
            pos=jaw.position(),
            quat=_quat(jaw.rotation),
            rgba=JAW_RGBA,
            **VISUAL_ONLY,
        )

    def _add_suction_tube(self) -> None:
        """Saliva ejector clamped in the gripper; "suction_tip" site at the open hook end.

        The site's x-axis points along the shaft, so the IK's "keep the tube level" rule keeps the
        shaft level and the hook hanging straight down.
        """
        grip_center = np.array([*TUBE_AXIS_XY, GRIP_CENTER_Z])
        mesh_pos, tip_pos = self.ejector.mount(grip_center)
        self.spec.add_mesh(name="saliva_ejector", file=str(self.ejector.assembled_mesh()), scale=[MM, MM, MM])
        gripper = self.spec.body("gripper")
        gripper.add_geom(
            name="suction_tube",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="saliva_ejector",
            pos=mesh_pos,
            quat=_quat(STL_TO_GRIPPER),
            rgba=TUBE_RGBA,
            mass=EJECTOR_MASS,
            **VISUAL_ONLY,
        )
        gripper.add_site(name="suction_tip", pos=tip_pos, quat=TOOL_AXIS_QUAT, size=[0.003, 0, 0], rgba=TIP_RGBA, group=1)

    def _add_target_marker(self) -> None:
        """Green target sphere plus a short stick pointing where the hook points (shows the roll)."""
        super()._add_target_marker()
        self.spec.body("target").add_geom(
            name="target_hook_direction",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, 0, 0, -HOOK_STICK_LENGTH],
            size=[0.001, 0, 0],
            rgba=TIP_RGBA,
            **VISUAL_ONLY,
        )

    def _add_zone_markers(self, head: mujoco.MjsBody) -> None:
        """Samples a box around both jaws and draws a small red dot at every forbidden point."""
        corners = np.array([jaw.to_world(c) for jaw in JAWS for c in itertools.product(*jaw.bounds_mm.T)])
        low = corners.min(axis=0) - ZONE_GRID_MARGIN
        high = corners.max(axis=0) + ZONE_GRID_MARGIN
        axes = [np.arange(lo, hi + 1e-9, ZONE_GRID_STEP) for lo, hi in zip(low, high, strict=True)]
        for point in itertools.product(*axes):
            if not self.zones.allows(np.array(point)):
                head.add_site(pos=list(point), size=[FORBIDDEN_MARKER_RADIUS, 0, 0], rgba=FORBIDDEN_RGBA, group=FORBIDDEN_GROUP)


def _quat(rotation: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> MuJoCo quaternion (w, x, y, z)."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.flatten())
    return quat
