"""The saliva ejector the robot holds: two printed halves joined into one hook-shaped tube.

The STL files (models/sucker/, millimetres) contain the two halves laid flat on the print bed,
mirror images of each other, cut faces down at z = 0:
    y = along the straight shaft (hose end at y = -65, hook at the top, up to y = +60)
    x = direction the hook bends toward (hook end at about x = +46)
Half B rotated 180 degrees about the y axis lands exactly on top of half A: together they are
the full tube, centred on z = 0.

Mounting in the gripper (see SalivaEjector.mount):
    the gripper jaws close on the shaft at GRIP_POINT_MM, the shaft points along the tool axis,
    and the hook bends downward in the world when the arm holds the shaft level.
The control point ("suction_tip" site) is the open end of the hook, TIP_POINT_MM.
"""

from pathlib import Path

import numpy as np
import trimesh

from .jaw import MM

SUCKER_DIR = Path(__file__).resolve().parent.parent / "models" / "sucker"
HALF_A = SUCKER_DIR / "saliva_ejector_solid_half_A.stl"
HALF_B = SUCKER_DIR / "saliva_ejector_solid_half_B.stl"
ASSEMBLED = SUCKER_DIR / "assembled" / "saliva_ejector.stl"  # generated from the two halves

GRIP_POINT_MM = np.array([0.0, -40.0, 0.0])  # on the shaft, 25 mm from the hose end
TIP_POINT_MM = np.array([46.0, 45.5, 0.0])  # centre of the open hook end (measured from half A)

# Ejector STL axes expressed in the gripper body frame (columns = where STL x, y, z point).
# Gripper frame facts (vendored SO-101 model): tool axis = gripper -z; with the tool held level
# and wrist_roll = 0, gripper +x points up in the world, so STL x (hook) -> gripper -x = down.
STL_TO_GRIPPER = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=float)


class SalivaEjector:
    """Builds the assembled mesh and computes where it sits in the gripper."""

    def assembled_mesh(self) -> Path:
        """Joins half A and half B into one STL, created on first use or when a half changes."""
        newest_half = max(HALF_A.stat().st_mtime, HALF_B.stat().st_mtime)
        if not ASSEMBLED.exists() or ASSEMBLED.stat().st_mtime < newest_half:
            ASSEMBLED.parent.mkdir(exist_ok=True)
            half_b = trimesh.load(HALF_B)
            half_b.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
            trimesh.util.concatenate([trimesh.load(HALF_A), half_b]).export(ASSEMBLED)
        return ASSEMBLED

    def mount(self, grip_center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(mesh position, tip position) in the gripper frame, in metres.

        grip_center: point between the gripper jaws (gripper frame) where GRIP_POINT_MM must sit.
        """
        mesh_pos = grip_center - STL_TO_GRIPPER @ (GRIP_POINT_MM * MM)
        tip_pos = mesh_pos + STL_TO_GRIPPER @ (TIP_POINT_MM * MM)
        return mesh_pos, tip_pos

    @staticmethod
    def hook_drop() -> float:
        """How far (m) the tip hangs below the shaft axis when the shaft is held level."""
        return TIP_POINT_MM[0] * MM

    @staticmethod
    def shaft_to_tip() -> float:
        """How far (m) the tip is ahead of the grip point, measured along the shaft."""
        return (TIP_POINT_MM[1] - GRIP_POINT_MM[1]) * MM
