"""Where the two jaw STL files sit in the simulated world, and how to convert between the frames.

Works for any pair of jaw scans that follow this STL convention (the one your files use):
    x = patient left/right
    y = up from the print bed; the teeth point to +y (tooth tips = largest y)
    z = front/back; incisors toward -z (incisor edge = smallest z)
Size and position are measured from the files on every start, so swapping in another jaw only
means replacing models/jaw/upper_teeth.stl and lower_teeth.stl. Units in the files: millimetres.

World frame (metres): robot base at the origin, +x = where the arm points, +z = up.
The lower jaw is placed teeth-up, the upper jaw is flipped teeth-down above it, incisors facing
the robot at FRONT_X, with OPENING between the tooth tips, centred on BITE_Z.
"""

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import trimesh

JAW_DIR = Path(__file__).resolve().parent.parent / "models" / "jaw"
SIM_MESH_DIR = JAW_DIR / "simplified"  # generated, see JawModel.sim_mesh
SIM_MESH_FACES = 80_000  # MuJoCo accepts at most 200 000 faces per STL; scans often have more
MM = 0.001  # metres per millimetre

FRONT_X = 0.33  # world x of the incisor edge
BITE_Z = 0.22  # world height half-way between upper and lower tooth tips
# Show-scale: the jaws are drawn and placed this many times larger than the scans (1.0 = real size).
# `JAW_SCALE=1 uv run ...` overrides it; the follower mode (real arm, real jaw) requires 1.
JAW_SCALE = float(os.environ.get("JAW_SCALE", "3.0"))
OPENING = 0.035 * JAW_SCALE  # gap between upper and lower tooth tips (mouth opening)

# Where the real jaw turned out to be, relative to the drawn default place (metres, world frame).
# Set by the follower mode's one-point calibration (jaw_sim/follower.py); zero in every other mode.
_world_offset = np.zeros(3)


def world_offset() -> np.ndarray:
    return _world_offset.copy()


def set_world_offset(offset: np.ndarray) -> None:
    _world_offset[:] = offset


# Rotation STL -> world, as a 3x3 matrix whose columns are where the STL x, y, z axes end up.
TEETH_UP = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)  # STL y -> world +z, STL z -> world +x
TEETH_DOWN = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)  # STL y -> world -z (flipped)


@dataclass(frozen=True, eq=False)  # eq=False: compared/hashed by identity (the rotation array is not hashable)
class JawModel:
    """One jaw STL file, placed in the world from its own measured size."""

    name: str  # "upper" or "lower"
    file: Path
    rotation: np.ndarray  # STL -> world rotation

    @cached_property
    def bounds_mm(self) -> np.ndarray:
        """[[min x, min y, min z], [max x, max y, max z]] of the STL, in mm, at JAW_SCALE."""
        return trimesh.load(self.file).bounds * JAW_SCALE

    @property
    def tip_height_mm(self) -> float:
        """STL y of the highest tooth tip."""
        return float(self.bounds_mm[1, 1])

    @property
    def incisor_z_mm(self) -> float:
        """STL z of the incisor edge (front of the arch)."""
        return float(self.bounds_mm[0, 2])

    @property
    def width_mm(self) -> float:
        """Outer width of the arch, left to right."""
        return float(self.bounds_mm[1, 0] - self.bounds_mm[0, 0])

    @property
    def depth_mm(self) -> float:
        """Incisor edge to the back of the last molars."""
        return float(self.bounds_mm[1, 2] - self.bounds_mm[0, 2])

    def position(self) -> np.ndarray:
        """World position of the STL origin, so incisors sit at FRONT_X and tips leave OPENING."""
        teeth_point_down = self.rotation[2, 1] < 0
        tip_z = BITE_Z + 0.5 * OPENING if teeth_point_down else BITE_Z - 0.5 * OPENING
        down_or_up = -1.0 if teeth_point_down else 1.0
        return np.array(
            [
                FRONT_X - self.incisor_z_mm * MM,
                0.0,
                tip_z - down_or_up * self.tip_height_mm * MM,
            ]
        ) + _world_offset

    def to_world(self, point_mm: np.ndarray) -> np.ndarray:
        """STL point in mm (at JAW_SCALE) -> world point in m."""
        return self.position() + self.rotation @ (np.asarray(point_mm) * MM)

    def to_stl(self, point_world: np.ndarray) -> np.ndarray:
        """World point in m -> STL point in mm (the frame your zone functions work in)."""
        return self.rotation.T @ (np.asarray(point_world) - self.position()) / MM

    def sim_mesh(self) -> Path:
        """Copy of the STL MuJoCo can load, simplified if it has too many faces (~0.01 mm change)."""
        path = SIM_MESH_DIR / self.file.name
        if not path.exists() or path.stat().st_mtime < self.file.stat().st_mtime:
            SIM_MESH_DIR.mkdir(exist_ok=True)
            mesh = trimesh.load(self.file)
            if len(mesh.faces) > SIM_MESH_FACES:
                mesh = mesh.simplify_quadric_decimation(face_count=SIM_MESH_FACES)
            mesh.export(path)
        return path


LOWER = JawModel("lower", JAW_DIR / "lower_teeth.stl", rotation=TEETH_UP)
UPPER = JawModel("upper", JAW_DIR / "upper_teeth.stl", rotation=TEETH_DOWN)
JAWS = (UPPER, LOWER)
