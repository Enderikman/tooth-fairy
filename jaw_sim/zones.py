"""Forbidden zones: where the suction tip may and may not go.  <-- YOUR ALGORITHM GOES HERE

Who calls this file
-------------------
1. jaw_sim/controller.py, GuardedController.__call__ (50 times per second while the sim runs):
       allowed = self.zones.allows(target)
   False -> the arm stays at the last allowed point and the green marker turns red.
2. jaw_sim/scene.py, JawSceneBuilder._add_zone_markers (once, when the scene is built):
   asks allows() for every point of a 4 mm grid around the jaws and draws a red dot where it is False.
   So after editing this file, restart the viewer to see your new zones.

Coordinates your functions receive
----------------------------------
Each function gets the tip position in that jaw's own STL coordinates, in millimetres,
exactly as in your CAD/slicer (conversion done by JawModel.to_stl in jaw_sim/jaw.py):
    x = patient left/right
    y = height from the STL base toward the tooth tips
    z = front (incisors, smallest z) to back
The viewer shows the live tip position in both jaw frames (top-left text).

Try it
------
    uv run mjpython -m jaw_sim drag     drag the green marker, see where it turns red
    uv run python -m jaw_sim check      headless: how often the automatic route was blocked
"""

from functools import cache

import numpy as np
import trimesh

from .jaw import LOWER, UPPER, JawModel

# Example rule only (replace with your functions): a point is forbidden if it lies inside the
# plaster cast, i.e. below the scanned surface, or closer to it than SAFETY_MARGIN_MM.
# It needs no tuning, so it works for any jaw scan in the convention described in jaw_sim/jaw.py.
SAFETY_MARGIN_MM = 3.0
HEIGHTMAP_CELL_MM = 1.0  # resolution of the surface height lookup


class ForbiddenZones:
    """Answers one question for any world point: is the suction tip allowed there?"""

    def allows(self, point_world: np.ndarray) -> bool:
        """True if the point is outside every forbidden zone.

        point_world: (x, y, z) in metres, world frame (robot base = origin).
        Converts the point into each jaw's STL frame and asks your two functions.
        """
        x, y, z = LOWER.to_stl(point_world)
        if self.lower_jaw_forbidden(x, y, z):
            return False
        x, y, z = UPPER.to_stl(point_world)
        if self.upper_jaw_forbidden(x, y, z):
            return False
        return True

    def lower_jaw_forbidden(self, x: float, y: float, z: float) -> bool:
        """Function 1: is this point forbidden because of the lower jaw?

        Args:
            x, y, z: point in mm, lower_teeth.stl coordinates.
        Returns:
            True = forbidden, False = allowed.

        EXAMPLE (replace): inside the lower cast or within the safety margin of its surface.
        """
        return y < surface_height(LOWER, x, z) + SAFETY_MARGIN_MM

    def upper_jaw_forbidden(self, x: float, y: float, z: float) -> bool:
        """Function 2: is this point forbidden because of the upper jaw?

        Args:
            x, y, z: point in mm, upper_teeth.stl coordinates (its y also points toward its own tooth tips).
        Returns:
            True = forbidden, False = allowed.

        EXAMPLE (replace): same rule as the lower jaw.
        """
        return y < surface_height(UPPER, x, z) + SAFETY_MARGIN_MM


def surface_height(jaw: JawModel, x: float, z: float) -> float:
    """Example helper: height (STL y, mm) of the scanned surface above the point (x, z).

    The cast is solid below its surface. Returns -inf where there is no cast (outside the arch,
    or in the open middle of a horseshoe-shaped cast). The map is widened by SAFETY_MARGIN_MM
    sideways, so the margin applies to the sides of the teeth as well as their tops.
    """
    heights, origin = _height_map(jaw)
    col, row = np.floor((np.array([x, z]) - origin) / HEIGHTMAP_CELL_MM).astype(int)
    if not (0 <= col < heights.shape[0] and 0 <= row < heights.shape[1]):
        return -np.inf
    return float(heights[col, row])


@cache
def _height_map(jaw: JawModel) -> tuple[np.ndarray, np.ndarray]:
    """Highest scanned point per HEIGHTMAP_CELL_MM square (seen from above), widened by the margin.

    Built once per jaw from the STL vertices. Returns (heights[x cell, z cell], (x, z) of cell 0).
    """
    vertices = trimesh.load(jaw.file).vertices
    margin_cells = int(np.ceil(SAFETY_MARGIN_MM / HEIGHTMAP_CELL_MM))
    origin = vertices[:, [0, 2]].min(axis=0) - margin_cells * HEIGHTMAP_CELL_MM
    cells = np.floor((vertices[:, [0, 2]] - origin) / HEIGHTMAP_CELL_MM).astype(int)
    shape = cells.max(axis=0) + margin_cells + 1
    heights = np.full(shape, -np.inf)
    np.maximum.at(heights, (cells[:, 0], cells[:, 1]), vertices[:, 1])

    widened = heights.copy()
    for dx in range(-margin_cells, margin_cells + 1):
        for dz in range(-margin_cells, margin_cells + 1):
            if dx * dx + dz * dz <= margin_cells * margin_cells:
                widened = np.maximum(widened, np.roll(heights, (dx, dz), axis=(0, 1)))
    return widened, origin
