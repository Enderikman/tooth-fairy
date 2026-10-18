"""The lower-jaw corridor rule of jaw_sim/zones.py, evaluated for many points at once on the GPU.

jaw_sim answers "may the tip be here?" for one point with numpy, which is what the MuJoCo demo
needs. Training asks the same question for thousands of simulated arms in every control step, so
this module ports the rule to batched torch tensors. The geometry itself is never re-derived: the
parabola coefficient, the corridor width, the arch samples and the maximum depth are read from
`jaw_sim.zones.LowerJawCorridor`, and the jaw placement from `jaw_sim.jaw.LOWER`.

Frames
------
world (m)   robot base at the origin, +x toward the arm, +z up; what the policy and Isaac see.
STL  (mm)   the lower-jaw scan frame, already multiplied by JAW_SCALE.
mouth(mm)   x across the arch (46 -> 36), y toward the back, depth down from the incisal plane.

The world -> mouth conversion is affine, so it is stored as one 3x3 matrix and one offset and
applied to an (N, 3) tensor in a single matmul.
"""

from __future__ import annotations

import numpy as np
import torch

from jaw_sim.calibration import JawCalibration
from jaw_sim.jaw import JAW_SCALE, LOWER, MM
from jaw_sim.zones import CORRIDOR_WIDTH_MM, TOOTH_ACROSS, ForbiddenZones

TEETH = ("46", "43", "33", "36")  # the four tooth targets, 46 -> 36 along the arch
TARGET_DEPTH_FRACTION = (
    0.4,
    0.9,
)  # training targets sit between these fractions of the allowed depth
TARGET_ACROSS = (
    -1.0,
    1.0,
)  # ... and anywhere along the arch, -1 = tooth 46, +1 = tooth 36


class TorchCorridor:
    """Batched forbidden-zone test and target sampler for the lower-jaw corridor.

    Attributes:
        device: Device all returned tensors live on.
        zones: The numpy `ForbiddenZones` this corridor was built from (used for sampling).
    """

    def __init__(self, device: torch.device | str, dtype: torch.dtype = torch.float32):
        """Builds the torch copy of the default (printed adult jaw) corridor.

        Args:
            device: Device for all cached tensors and results.
            dtype: Floating point type for all cached tensors and results.
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self.zones = ForbiddenZones(JawCalibration.load_or_default())
        corridor = self.zones.lower
        frame = corridor.frame

        # world (m) -> mouth (mm): stl = R^T (p - jaw_position) / MM, then project on the mouth axes.
        axes = np.stack([frame.x_axis, frame.y_axis, -frame.up])  # rows: x, y, depth
        self._world_to_mouth = self._tensor(axes @ LOWER.rotation.T / MM)
        self._mouth_offset = self._tensor(
            -axes @ (LOWER.rotation.T @ LOWER.position()) / MM - axes @ frame.origin
        )

        self.parabola_a = float(corridor.a)
        self.half_width_mm = float(corridor.half_width)
        self.arch_depth_mm = float(corridor.arch_depth)
        self.max_depth_mm = float(corridor.max_depth)
        self.corridor_width_mm = float(CORRIDOR_WIDTH_MM)
        self._arch = self._tensor(
            corridor.arch
        )  # (samples, 2) lingual tooth line in mouth mm

    def to_mouth(self, points_world_m: torch.Tensor) -> torch.Tensor:
        """Converts world points to mouth coordinates.

        Args:
            points_world_m: (N, 3) points in the world frame, metres.

        Returns:
            (N, 3) tensor of (x across, y back, depth below the incisal plane), millimetres.
        """
        return (
            points_world_m.to(self.dtype) @ self._world_to_mouth.T + self._mouth_offset
        )

    def distance_to_arch(self, points_world_m: torch.Tensor) -> torch.Tensor:
        """Shortest distance (mm) from each point to the sampled lingual tooth line, in the mouth plane.

        Args:
            points_world_m: (N, 3) points in the world frame, metres.

        Returns:
            (N,) distances in millimetres; depth is ignored, as in jaw_sim.
        """
        mouth = self.to_mouth(points_world_m)
        return torch.cdist(mouth[:, :2], self._arch).min(dim=1).values

    def forbidden(self, points_world_m: torch.Tensor) -> torch.Tensor:
        """True where the suction tip may not be, following jaw_sim.zones.LowerJawCorridor.forbidden.

        Args:
            points_world_m: (N, 3) points in the world frame, metres.

        Returns:
            (N,) boolean tensor, True = forbidden.
        """
        mouth = self.to_mouth(points_world_m)
        across, back, depth = mouth[:, 0], mouth[:, 1], mouth[:, 2]
        margin = self.corridor_width_mm
        in_mouth = (
            (depth > 0.0)
            & (back >= 0.0)
            & (back <= self.arch_depth_mm + margin)
            & (across.abs() <= self.half_width_mm + margin)
        )
        too_deep = depth > self.max_depth_mm
        tooth_side = back < self.parabola_a * across**2
        too_far = torch.cdist(mouth[:, :2], self._arch).min(dim=1).values > margin
        return in_mouth & (too_deep | tooth_side | too_far)

    def sample_targets(
        self, count: int, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Draws target points on the middle line of the corridor.

        The point itself is computed by jaw_sim (`corridor_point`) and only then moved to the
        device: sampling happens once per reset, so numpy is fast enough and the rule stays in
        one place.

        Args:
            count: How many targets to draw.
            generator: Optional torch generator, for reproducible resets.

        Returns:
            (count, 3) target positions in the world frame, metres.
        """
        unit = (
            torch.rand(count, 2, generator=generator, device=self.device).cpu().numpy()
        )
        low_across, high_across = TARGET_ACROSS
        low_depth, high_depth = TARGET_DEPTH_FRACTION
        across = low_across + unit[:, 0] * (high_across - low_across)
        depth = (low_depth + unit[:, 1] * (high_depth - low_depth)) * self.max_depth_mm
        points = [
            LOWER.to_world(self.zones.lower.corridor_point(a, d))
            for a, d in zip(across, depth, strict=True)
        ]
        return self._tensor(np.asarray(points))

    def tooth_targets(self) -> torch.Tensor:
        """The four demo targets (teeth 46, 43, 33, 36).

        Returns:
            (4, 3) suction positions in the world frame, metres, in TEETH order.
        """
        points = [
            LOWER.to_world(self.zones.lower.tooth_target(tooth)) for tooth in TEETH
        ]
        return self._tensor(np.asarray(points))

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        """Moves a numpy array onto this corridor's device and dtype."""
        return torch.as_tensor(np.asarray(array), dtype=self.dtype, device=self.device)


__all__ = ["CORRIDOR_WIDTH_MM", "JAW_SCALE", "TEETH", "TOOTH_ACROSS", "TorchCorridor"]
