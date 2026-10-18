"""Pieces shared by the SO-101 simulations (dental_sim, stack_sim).

Only things that are literally identical between scenes live here: the path to the vendored
robot model, the checkerboard ground plane, and the min-jerk timing profile.
"""

from pathlib import Path

import mujoco

ROBOT_XML = Path(__file__).resolve().parent / "models" / "so101" / "so101_new_calib.xml"


def add_grid_floor(spec: mujoco.MjSpec, z: float = 0.0) -> None:
    """Checkerboard ground plane at height `z` (texture `grid`, material `floor`, geom `floor`)."""
    spec.add_texture(
        name="grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        markrgb=[0.8, 0.8, 0.8],
        width=300,
        height=300,
    )
    mat = spec.add_material(name="floor", texrepeat=[5, 5], texuniform=True, reflectance=0.1)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0, 0, z],
        size=[0, 0, 0.05],
        material="floor",
    )


def min_jerk(s: float) -> float:
    """Normalised 0..1 min-jerk profile (zero velocity and acceleration at both ends)."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5
