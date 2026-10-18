"""Forbidden zones: where the suction tip may and may not go.

The rule (lower jaw only, from three calibration points, see jaw_sim/calibration.py):
    1. Mouth frame: origin = incisal edge between 31 and 41, x = across the arch (46 -> 36),
       y = toward the back of the mouth, depth = straight down from the incisal plane.
    2. Lingual tooth surface = parabola  y = a * x**2,  a fitted from the gum-line points of 36 and 46.
    3. Allowed corridor = the strip up to CORRIDOR_WIDTH_MM from that parabola on the tongue side,
       from 46 round to 36.
    4. Maximum depth below the incisal plane grows with the jaw size:
           depth_max = DEPTH_PER_JC * JC + DEPTH_OFFSET_MM,   JC = arch width * arch depth  (mm**2)
       (JC 300 mm**2, a small child: 5 mm;  JC 1848 mm**2, our printed adult jaw: 20 mm).
    Above the incisal plane everything is allowed, otherwise the hook could not enter the mouth.

Who calls this file
-------------------
1. jaw_sim/controller.py, GuardedController.__call__ (50 times per second while the sim runs):
       allowed = self.zones.allows(target)
   False -> the arm stays at the last allowed point and the green marker turns red.
2. jaw_sim/scene.py, JawSceneBuilder._add_zone_markers (once, when the scene is built):
   asks allows() for every point of a 4 mm grid around the jaws and draws a red dot where it is False.
   So after a new calibration, restart the viewer to see the new corridor.
"""

from dataclasses import dataclass

import numpy as np

from .calibration import JawCalibration
from .jaw import LOWER

CORRIDOR_WIDTH_MM = 20.0  # tongue-side strip next to the teeth where the tip may go
DEPTH_PER_JC = 0.0097  # mm of allowed depth per mm**2 of jaw size
DEPTH_OFFSET_MM = 2.093
ARCH_SAMPLE_MM = 0.5  # spacing of the parabola samples used for the distance test
STL_UP = np.array([0.0, 1.0, 0.0])  # the lower jaw STL has its tooth tips toward +y

SUCTION_DEPTH_FRACTION = 0.8  # the tip sits at this fraction of the allowed depth when suctioning
# Where each voice-command tooth sits along the arch: -1 = 46 (calibration point) .. +1 = 36.
TOOTH_ACROSS = {"46": -1.0, "43": -0.6, "33": 0.6, "36": 1.0}

@dataclass(frozen=True)
class MouthFrame:
    """Converts lower-jaw STL points (mm) into mouth coordinates (x across, y back, depth down)."""

    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    up: np.ndarray

    @classmethod
    def from_points(cls, incisal: np.ndarray, p36: np.ndarray, p46: np.ndarray, up: np.ndarray) -> "MouthFrame":
        """Frame with the incisal edge as origin; the two molar points give the direction of the arch.

        Only the yaw comes from the points; the incisal plane is taken perpendicular to `up`,
        i.e. the jaw is assumed to lie level.
        """
        x_axis = _unit(_horizontal(p36 - p46, up))
        y_axis = _unit(_horizontal((p36 + p46) / 2 - incisal, up))
        return cls(np.asarray(incisal, dtype=float), x_axis, y_axis, _unit(up))

    def to_mouth(self, point: np.ndarray) -> tuple[float, float, float]:
        """STL point -> (x, y, depth) in mm; depth > 0 is below the incisal plane."""
        offset = np.asarray(point, dtype=float) - self.origin
        return float(offset @ self.x_axis), float(offset @ self.y_axis), float(-(offset @ self.up))

    def to_stl(self, x: float, y: float, depth: float) -> np.ndarray:
        """Inverse of to_mouth."""
        return self.origin + x * self.x_axis + y * self.y_axis - depth * self.up


class LowerJawCorridor:
    """The allowed strip between the lower teeth and the tongue, sized from the calibration."""

    def __init__(self, calibration: JawCalibration):
        self.frame = MouthFrame.from_points(calibration.incisal, calibration.tooth_36, calibration.tooth_46, STL_UP)
        x36, y36, _ = self.frame.to_mouth(calibration.tooth_36)
        x46, y46, _ = self.frame.to_mouth(calibration.tooth_46)
        self.a = float(np.mean([y36 / x36**2, y46 / x46**2]))
        self.half_width = float(np.mean([abs(x36), abs(x46)]))
        self.jaw_size = (abs(x36) + abs(x46)) * float(np.mean([y36, y46]))  # JC, mm**2
        self.max_depth = DEPTH_PER_JC * self.jaw_size + DEPTH_OFFSET_MM
        xs = np.arange(-self.half_width, self.half_width + ARCH_SAMPLE_MM / 2, ARCH_SAMPLE_MM)
        self.arch = np.column_stack([xs, self.a * xs**2])  # sampled lingual tooth line, 46 -> 36
        self.arch_depth = float(np.mean([y36, y46]))

    def forbidden(self, point_stl: np.ndarray) -> bool:
        """True if the tip may not be at this lower-jaw STL point (mm).

        The rule only describes the mouth: points in front of the incisors, beside or behind the
        arch are outside the modelled region and left alone.
        """
        x, y, depth = self.frame.to_mouth(point_stl)
        if depth <= 0:
            return False  # above the teeth: open mouth
        if not self._in_mouth_region(x, y):
            return False
        if depth > self.max_depth:
            return True
        if y < self.a * x**2:
            return True  # tooth side of the lingual line
        distance = np.min(np.linalg.norm(self.arch - (x, y), axis=1))
        return bool(distance > CORRIDOR_WIDTH_MM)

    def _in_mouth_region(self, x: float, y: float) -> bool:
        """Inside the box the calibration describes: the arch plus one corridor width around it."""
        margin = CORRIDOR_WIDTH_MM
        return 0 <= y <= self.arch_depth + margin and abs(x) <= self.half_width + margin

    def corridor_point(self, across: float, depth: float) -> np.ndarray:
        """STL point in the middle of the corridor, `across` = -1 (46) .. 0 (incisors) .. +1 (36)."""
        x = across * self.half_width
        tangent = np.array([1.0, 2 * self.a * x])
        normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)  # points to the tongue
        mid_x, mid_y = np.array([x, self.a * x**2]) + normal * CORRIDOR_WIDTH_MM / 2
        return self.frame.to_stl(mid_x, mid_y, depth)

    def tooth_target(self, tooth: str) -> np.ndarray:
        """STL point where the tip suctions next to this tooth (a key of TOOTH_ACROSS)."""
        return self.corridor_point(TOOTH_ACROSS[tooth], SUCTION_DEPTH_FRACTION * self.max_depth)


class ForbiddenZones:
    """Answers one question for any world point: is the suction tip allowed there?"""

    def __init__(self, calibration: JawCalibration):
        self.lower = LowerJawCorridor(calibration)

    def allows(self, point_world: np.ndarray) -> bool:
        """point_world: (x, y, z) in metres, world frame (robot base = origin)."""
        return not self.lower.forbidden(LOWER.to_stl(point_world))


def _horizontal(vector: np.ndarray, up: np.ndarray) -> np.ndarray:
    """The part of the vector perpendicular to `up`."""
    up = _unit(up)
    return vector - (vector @ up) * up


def _unit(vector: np.ndarray) -> np.ndarray:
    return np.asarray(vector, dtype=float) / np.linalg.norm(vector)
