"""Web viewer for recorded dental-suction-hold rollouts (rerun).

Reads the rollout file play.py writes (see isaac/dental_rl/README or the contract below) and logs
it into a rerun recording:
    static   both jaw meshes, red dots where the forbidden zones say "not allowed", the four
             tooth targets as labelled spheres
    "step"   per control step (time = step * dt): the arm, drawn by running the MuJoCo scene of
             jaw_sim through forward kinematics on the recorded joint angles, the suction tip and
             its growing trail, the episode target, reward and tip distance as plots
    "episode" a second timeline, one tick per episode, so the viewer can step through episodes

Usage:
    uv run python -m isaac.dental_rl.viz --rollout out/rollout.json           # writes rollout.rrd
    uv run python -m isaac.dental_rl.viz --rollout out/rollout.json --serve   # serves a web viewer

Rollout file (metres, radians, jaw_sim world frame = robot base at the origin):
    {"dt": 0.02, "joint_names": [...6 names...], "jaw_scale": 3.0, "corridor_width_mm": 60.0,
     "targets_world_m": [[x, y, z], ...],            # optional, the four tooth targets
     "episodes": [{"target_world_m": [x, y, z], "success": true,
                   "steps": [{"q": [...6...], "gripper_pos_w": [...], "gripper_quat_w": [...],
                              "tip_world_m": [x, y, z], "reward": 0.1, "blocked": false}]}]}
"""

import argparse
import itertools
import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import mujoco
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import trimesh

from jaw_sim.calibration import JawCalibration
from jaw_sim.jaw import JAW_SCALE, JAWS, LOWER, MM, JawModel
from jaw_sim.scene import JawSceneBuilder
from jaw_sim.zones import TOOTH_ACROSS, ForbiddenZones

APP_ID = "dental-suction-hold"
STEP_TIMELINE = "step"
EPISODE_TIMELINE = "episode"

# Entity paths; everything under WORLD_PATH shares the jaw_sim world frame (metres).
WORLD_PATH = "world"
JAW_PATH = f"{WORLD_PATH}/jaw"
FORBIDDEN_PATH = f"{WORLD_PATH}/forbidden"
TOOTH_TARGET_PATH = f"{WORLD_PATH}/tooth_targets"
ARM_PATH = f"{WORLD_PATH}/arm"
TARGET_PATH = f"{WORLD_PATH}/target"
TIP_PATH = f"{WORLD_PATH}/tip"
TRAIL_PATH = f"{WORLD_PATH}/trail"
REWARD_PATH = "plot/reward"
DISTANCE_PATH = "plot/tip_distance_mm"
EPISODE_LOG_PATH = "log/episode"

# Forbidden-zone dots: the same 4 mm grid jaw_sim/scene.py draws in MuJoCo, at show scale.
ZONE_GRID_STEP_MM = 4.0 * JAW_SCALE
ZONE_GRID_MARGIN_MM = 8.0 * JAW_SCALE
ZONE_MAX_POINTS = 5000  # the viewer stays smooth up to a few thousand dots
ZONE_POINT_RADIUS_MM = 0.8 * JAW_SCALE
ZONE_SEED = 0  # fixed seed, so the thinned-out dot cloud looks the same on every run

HOLD_TOLERANCE_MM = (
    5.0 * JAW_SCALE
)  # tip counts as "on target" within this distance (5 mm real size)
TARGET_RADIUS_MM = (
    HOLD_TOLERANCE_MM  # the target sphere is drawn as big as the tolerance
)
TOOTH_TARGET_RADIUS_MM = 0.5 * TARGET_RADIUS_MM
TIP_RADIUS_MM = 1.5 * JAW_SCALE
TRAIL_RADIUS_MM = 0.5 * JAW_SCALE
TRAIL_MAX_POINTS = (
    2000  # oldest trail points are dropped past this, keeping long episodes cheap
)

COLOR_FORBIDDEN = (230, 40, 40, 110)
COLOR_ON_TARGET = (60, 220, 90)
COLOR_OFF_TARGET = (250, 160, 40)
COLOR_TOOTH_TARGET = (120, 180, 255)
COLOR_TIP = (80, 220, 240)
COLOR_TIP_BLOCKED = (255, 60, 60)
COLOR_TRAIL = (120, 200, 255)

COLLISION_GEOM_GROUP = (
    3  # MuJoCo geom group of the arm's collision hulls; only visuals are drawn
)
JAW_BODY = "phantom_head"  # the body carrying the jaw meshes; they are logged once, as static data
DEFAULT_WEB_PORT = 9090  # the viewer page
DEFAULT_GRPC_PORT = (
    9876  # the recording stream the viewer page connects to (rerun's default)
)
PORT_SEARCH_RANGE = 20  # how many ports to try when the preferred one is taken
SCENE_PLOT_COLUMN_SHARES = [
    2,
    1,
]  # the 3D mouth gets twice the width of the plot column
PLOT_LOG_ROW_SHARES = [2, 1]  # plot above, episode log below
SERVE_POLL_S = 0.5  # how often the serving loop checks for Ctrl-C


@dataclass(frozen=True)
class Step:
    """One recorded control step."""

    q: np.ndarray  # joint positions in rollout joint order, radians
    tip_world_m: np.ndarray  # suction tip in the world frame, metres
    reward: float
    blocked: bool

    @classmethod
    def from_json(cls, entry: dict) -> "Step":
        """Builds a step from one entry of an episode's "steps" list."""
        return cls(
            q=np.asarray(entry["q"], dtype=float),
            tip_world_m=np.asarray(entry["tip_world_m"], dtype=float),
            reward=float(entry.get("reward", 0.0)),
            blocked=bool(entry.get("blocked", False)),
        )


@dataclass(frozen=True)
class Episode:
    """One recorded episode: a target point and the steps spent reaching and holding it."""

    index: int
    target_world_m: np.ndarray
    success: bool
    steps: list[Step]

    @classmethod
    def from_json(cls, index: int, entry: dict) -> "Episode":
        """Builds an episode from one entry of the rollout's "episodes" list."""
        return cls(
            index=index,
            target_world_m=np.asarray(entry["target_world_m"], dtype=float),
            success=bool(entry.get("success", False)),
            steps=[Step.from_json(step) for step in entry["steps"]],
        )


@dataclass(frozen=True)
class Rollout:
    """The contents of a rollout.json file."""

    dt: float
    joint_names: list[str]
    jaw_scale: float
    corridor_width_mm: float
    tooth_targets_world_m: np.ndarray | None
    episodes: list[Episode]

    @classmethod
    def load(cls, path: Path) -> "Rollout":
        """Reads a rollout file written by play.py.

        Args:
            path: Path of the rollout JSON file.

        Returns:
            The parsed rollout.

        Raises:
            ValueError: If the file holds no episodes.
        """
        raw = json.loads(path.read_text())
        episodes = [
            Episode.from_json(i, entry) for i, entry in enumerate(raw["episodes"])
        ]
        if not episodes:
            raise ValueError(f"{path} contains no episodes")
        targets = raw.get("targets_world_m")
        return cls(
            dt=float(raw["dt"]),
            joint_names=list(raw["joint_names"]),
            jaw_scale=float(raw.get("jaw_scale", JAW_SCALE)),
            corridor_width_mm=float(raw.get("corridor_width_mm", 0.0)),
            tooth_targets_world_m=None
            if targets is None
            else np.asarray(targets, dtype=float),
            episodes=episodes,
        )


@dataclass(frozen=True)
class GeomMesh:
    """One mesh geom of the MuJoCo scene, in its own geom frame."""

    geom_id: int
    name: str
    vertices: np.ndarray  # (n, 3) metres, geom frame
    faces: np.ndarray  # (m, 3) vertex indices
    normals: np.ndarray  # (n, 3) unit vertex normals
    color: np.ndarray  # RGBA, 0..1


class ArmKinematics:
    """Forward kinematics and mesh data of the SO-101 with the saliva ejector.

    Uses the MuJoCo scene of jaw_sim, so the drawn arm is the same model the rollout was planned
    against. Only the moving visual geoms are kept: collision hulls and the static jaws are not
    part of the arm.
    """

    def __init__(self, joint_names: list[str]):
        """Builds the scene and resolves where the rollout's joints live in qpos.

        Args:
            joint_names: Joint names of the rollout, in the order its `q` vectors use.
        """
        zones = ForbiddenZones(JawCalibration.load_or_default())
        self.model = JawSceneBuilder(zones).build()
        self.data = mujoco.MjData(self.model)
        model_joints = {self.model.joint(i).name for i in range(self.model.njnt)}
        self._qpos_address = {
            name: int(self.model.joint(name).qposadr[0])
            for name in joint_names
            if name in model_joints
        }
        self.geoms = self._collect_geoms()

    def _collect_geoms(self) -> list[GeomMesh]:
        """Visual mesh geoms of the arm and the ejector, with their vertices in the geom frame."""
        geoms = []
        for geom_id in range(self.model.ngeom):
            if self.model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if self.model.geom_group[geom_id] == COLLISION_GEOM_GROUP:
                continue
            body = self.model.body(self.model.geom_bodyid[geom_id]).name
            if body == JAW_BODY:
                continue  # the jaws never move and are logged from their STL files instead
            mesh_id = int(self.model.geom_dataid[geom_id])
            vertices, faces = self._mesh_arrays(mesh_id)
            # most arm geoms are unnamed in the MJCF; the mesh name plus the id keeps paths unique
            name = (
                self.model.geom(geom_id).name
                or f"{self.model.mesh(mesh_id).name}_{geom_id}"
            )
            geoms.append(
                GeomMesh(
                    geom_id=geom_id,
                    name=f"{body}/{name}",
                    vertices=vertices,
                    faces=faces,
                    normals=vertex_normals(vertices, faces),
                    color=self.model.geom_rgba[geom_id].copy(),
                )
            )
        return geoms

    def _mesh_arrays(self, mesh_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Vertices (metres, geom frame) and triangle indices of one compiled MuJoCo mesh."""
        vert_start = int(self.model.mesh_vertadr[mesh_id])
        vert_count = int(self.model.mesh_vertnum[mesh_id])
        face_start = int(self.model.mesh_faceadr[mesh_id])
        face_count = int(self.model.mesh_facenum[mesh_id])
        vertices = self.model.mesh_vert[vert_start : vert_start + vert_count].astype(
            float
        )
        faces = self.model.mesh_face[face_start : face_start + face_count].astype(
            np.uint32
        )
        return vertices, faces

    def place(self, q: np.ndarray, joint_names: list[str]) -> None:
        """Runs forward kinematics for one recorded step.

        Args:
            q: Joint positions in radians.
            joint_names: Names belonging to `q`, in the same order.
        """
        for name, angle in zip(joint_names, q, strict=False):
            address = self._qpos_address.get(name)
            if address is not None:
                self.data.qpos[address] = angle
        mujoco.mj_kinematics(self.model, self.data)

    def geom_pose(self, geom: GeomMesh) -> tuple[np.ndarray, np.ndarray]:
        """World position and 3x3 rotation of a geom after the last `place` call."""
        return self.data.geom_xpos[geom.geom_id].copy(), self.data.geom_xmat[
            geom.geom_id
        ].reshape(3, 3).copy()


class RolloutLogger:
    """Logs a rollout into the active rerun recording."""

    def __init__(self, rollout: Rollout):
        """Prepares the kinematics and the forbidden zones used for the static scene.

        Args:
            rollout: The rollout to log.
        """
        self.rollout = rollout
        self.zones = ForbiddenZones(JawCalibration.load_or_default())
        self.kinematics = ArmKinematics(rollout.joint_names)

    def log(self) -> None:
        """Logs the static scene and then every episode."""
        self._log_jaws()
        self._log_forbidden_zone()
        self._log_tooth_targets()
        self._log_arm_meshes()
        for episode in self.rollout.episodes:
            self._log_episode(episode)

    def _log_jaws(self) -> None:
        """Both jaw scans as static meshes, placed the way jaw_sim/jaw.py places them."""
        for jaw in JAWS:
            mesh = trimesh.load(jaw.sim_mesh())
            vertices = jaw_vertices_world(jaw, np.asarray(mesh.vertices, dtype=float))
            faces = np.asarray(mesh.faces, dtype=np.uint32)
            rr.log(
                f"{JAW_PATH}/{jaw.name}",
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=faces,
                    vertex_normals=vertex_normals(vertices, faces),
                    albedo_factor=(242, 237, 217),
                ),
                static=True,
            )

    def _log_forbidden_zone(self) -> None:
        """Red dots on a grid through the lower jaw, wherever the zones forbid the tip."""
        points = self._forbidden_points()
        rr.log(
            FORBIDDEN_PATH,
            rr.Points3D(
                points, radii=ZONE_POINT_RADIUS_MM * MM, colors=COLOR_FORBIDDEN
            ),
            static=True,
        )

    def _forbidden_points(self) -> np.ndarray:
        """Samples a box around the lower jaw and keeps the points the zones reject."""
        corners = np.array(
            [LOWER.to_world(corner) for corner in itertools.product(*LOWER.bounds_mm.T)]
        )
        margin = ZONE_GRID_MARGIN_MM * MM
        step = ZONE_GRID_STEP_MM * MM
        axes = [
            np.arange(low - margin, high + margin + step / 2, step)
            for low, high in zip(corners.min(axis=0), corners.max(axis=0), strict=True)
        ]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        forbidden = np.array([not self.zones.allows(point) for point in grid])
        points = grid[
            forbidden
        ]  # allows() is True outside the mouth box, so these are inside it
        if len(points) > ZONE_MAX_POINTS:
            keep = np.random.default_rng(ZONE_SEED).choice(
                len(points), ZONE_MAX_POINTS, replace=False
            )
            points = points[np.sort(keep)]
        return points

    def _log_tooth_targets(self) -> None:
        """The four tooth suction points as labelled spheres."""
        labels = list(TOOTH_ACROSS)
        if self.rollout.tooth_targets_world_m is not None:
            points = self.rollout.tooth_targets_world_m
            labels = labels[: len(points)]
        else:
            points = np.array(
                [
                    LOWER.to_world(self.zones.lower.tooth_target(tooth))
                    for tooth in labels
                ]
            )
        rr.log(
            TOOTH_TARGET_PATH,
            rr.Points3D(
                points,
                radii=TOOTH_TARGET_RADIUS_MM * MM,
                colors=COLOR_TOOTH_TARGET,
                labels=labels,
            ),
            static=True,
        )

    def _log_arm_meshes(self) -> None:
        """Every moving mesh once, in its own geom frame; the steps only move the frames."""
        for geom in self.kinematics.geoms:
            rr.log(
                f"{ARM_PATH}/{geom.name}",
                rr.Mesh3D(
                    vertex_positions=geom.vertices,
                    triangle_indices=geom.faces,
                    vertex_normals=geom.normals,
                    albedo_factor=geom.color,
                ),
                static=True,
            )

    def _log_episode(self, episode: Episode) -> None:
        """Logs one episode on both timelines."""
        rr.set_time(EPISODE_TIMELINE, sequence=episode.index)
        trail: list[np.ndarray] = []
        for number, step in enumerate(episode.steps):
            rr.set_time(STEP_TIMELINE, duration=number * self.rollout.dt)
            trail.append(step.tip_world_m)
            del trail[:-TRAIL_MAX_POINTS]
            self._log_step(episode, step, trail)
        rr.log(
            EPISODE_LOG_PATH,
            rr.TextLog(
                f"episode {episode.index}: {'HOLD' if episode.success else 'MISS'} "
                f"target {np.round(episode.target_world_m, 4).tolist()} m, "
                f"{len(episode.steps)} steps, "
                f"final distance {tip_distance_mm(episode.steps[-1].tip_world_m, episode.target_world_m):.1f} mm",
                level=rr.TextLogLevel.INFO if episode.success else rr.TextLogLevel.WARN,
            ),
        )

    def _log_step(self, episode: Episode, step: Step, trail: list[np.ndarray]) -> None:
        """Logs the arm pose, the tip, the target and the plots of one step."""
        self.kinematics.place(step.q, self.rollout.joint_names)
        for geom in self.kinematics.geoms:
            position, rotation = self.kinematics.geom_pose(geom)
            rr.log(
                f"{ARM_PATH}/{geom.name}",
                rr.Transform3D(translation=position, mat3x3=rotation),
            )
        distance_mm = tip_distance_mm(step.tip_world_m, episode.target_world_m)
        on_target = distance_mm <= HOLD_TOLERANCE_MM
        rr.log(
            TARGET_PATH,
            rr.Points3D(
                [episode.target_world_m],
                radii=TARGET_RADIUS_MM * MM,
                colors=COLOR_ON_TARGET if on_target else COLOR_OFF_TARGET,
            ),
        )
        rr.log(
            TIP_PATH,
            rr.Points3D(
                [step.tip_world_m],
                radii=TIP_RADIUS_MM * MM,
                colors=COLOR_TIP_BLOCKED if step.blocked else COLOR_TIP,
            ),
        )
        rr.log(
            TRAIL_PATH,
            rr.LineStrips3D(
                [np.array(trail)], radii=TRAIL_RADIUS_MM * MM, colors=COLOR_TRAIL
            ),
        )
        rr.log(REWARD_PATH, rr.Scalars(step.reward))
        rr.log(DISTANCE_PATH, rr.Scalars(distance_mm))


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted unit normal per vertex.

    Args:
        vertices: (n, 3) vertex positions.
        faces: (m, 3) triangle vertex indices.

    Returns:
        (n, 3) unit normals; zero-length normals (unused vertices) are left as they are.
    """
    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(lengths > 0, lengths, 1.0)


def jaw_vertices_world(jaw: JawModel, vertices_mm: np.ndarray) -> np.ndarray:
    """STL vertices of a jaw (millimetres, scan size) as world points in metres at show scale."""
    return jaw.position() + (vertices_mm * JAW_SCALE * MM) @ jaw.rotation.T


def tip_distance_mm(tip_world_m: np.ndarray, target_world_m: np.ndarray) -> float:
    """Distance between tip and target in millimetres of the (show-scaled) simulated world."""
    return float(np.linalg.norm(tip_world_m - target_world_m) / MM)


def free_port(preferred: int) -> int:
    """First free TCP port at or above `preferred`.

    A rerun viewer already running on this machine holds the default gRPC port, so the stream has
    to move out of its way.

    Args:
        preferred: Port to try first.

    Returns:
        A port nothing listens on.

    Raises:
        OSError: If no port in the search range is free.
    """
    for port in range(preferred, preferred + PORT_SEARCH_RANGE):
        with (
            socket.socket() as probe
        ):  # no SO_REUSEADDR: a wildcard listener must count as taken
            try:
                probe.bind(("", port))
            except OSError:
                continue
            return port
    raise OSError(f"no free port in {preferred}..{preferred + PORT_SEARCH_RANGE}")


def blueprint() -> rrb.Blueprint:
    """Viewer layout: the mouth in 3D next to the reward/distance plot and the episode log."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin=f"/{WORLD_PATH}", name="mouth"),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="/plot", name="reward and distance"),
                rrb.TextLogView(origin="/log", name="episodes"),
                row_shares=PLOT_LOG_ROW_SHARES,
            ),
            column_shares=SCENE_PLOT_COLUMN_SHARES,
        ),
        collapse_panels=True,
    )


def serve(web_port: int, grpc_port: int, open_browser: bool) -> None:
    """Opens the web viewer on the recording about to be logged.

    Must be called before logging: the viewer shows what the gRPC sink receives.

    Args:
        web_port: Port the viewer page is served on.
        grpc_port: Port the recording is streamed on.
        open_browser: Whether to open the page in the default browser.
    """
    stream_url = rr.serve_grpc(grpc_port=grpc_port, default_blueprint=blueprint())
    rr.serve_web_viewer(
        web_port=web_port, open_browser=open_browser, connect_to=stream_url
    )
    print(f"rerun web viewer on http://localhost:{web_port}/?url={quote(stream_url, safe='')}")


def wait_for_ctrl_c() -> None:
    """Keeps the web server alive until the user stops it."""
    print("Ctrl-C to stop")
    try:
        while True:
            time.sleep(SERVE_POLL_S)
    except KeyboardInterrupt:
        print("stopped")


def main() -> None:
    """Command line entry point: logs a rollout and either saves it or serves it."""
    parser = argparse.ArgumentParser(
        description="Show a dental-suction-hold rollout in rerun."
    )
    parser.add_argument(
        "--rollout", type=Path, required=True, help="rollout.json written by play.py"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="serve a web viewer instead of saving an .rrd",
    )
    parser.add_argument(
        "--web-port", type=int, default=DEFAULT_WEB_PORT, help="port of the web viewer"
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=DEFAULT_GRPC_PORT,
        help="port of the recording stream",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="serve, but do not open a browser window",
    )
    args = parser.parse_args()

    rollout = Rollout.load(args.rollout)
    rr.init(APP_ID)
    if args.serve:
        serve(
            free_port(args.web_port), free_port(args.grpc_port), not args.no_browser
        )  # sink first: nothing is logged into the void
        RolloutLogger(rollout).log()
        wait_for_ctrl_c()
        return
    RolloutLogger(rollout).log()
    recording = args.rollout.with_suffix(".rrd")
    rr.save(recording, default_blueprint=blueprint())
    print(f"rerun {recording}")


if __name__ == "__main__":
    main()
