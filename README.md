# tooth-fairy

## Dental suction arm: SO-101 simulation

MuJoCo simulation of the SO-101 follower arm holding a suction tube in a dental phantom.
Purpose and scope: [CHARTER.md](CHARTER.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 and all packages are installed on first run.

```bash
uv sync
```

## Run

| Command | What it does |
| --- | --- |
| `uv run python -m dental_sim demo` | Viewer window; the arm moves the suction tip lower left molars -> front -> right molars -> upper right -> front -> upper left, looping |
| `uv run python -m dental_sim manual` | Viewer window without autopilot; move the joints yourself |
| `uv run python -m dental_sim check --png out/` | Headless: one loop, prints tip tracking error and tube/teeth contacts, saves camera images |

## Viewer controls (MuJoCo simulate)

- The window opens on a distant free camera: scroll to zoom in on the arm and mouth.
- Free camera: left drag = rotate, right drag = move, scroll = zoom. Double-click a body to select it, `Ctrl`+double-click to track it.
- Left panel `Rendering` > `Camera`: switch to the fixed cameras `overview` or `mouth_cam` (looks into the mouth).
- Right panel `Control`: one slider per motor (`manual` mode only; in `demo` the autopilot overwrites them).
- Right panel `Joint`: live joint angles in radians.
- Left panel `Simulation`: `Pause`/`Run`, `Reset`. Reset snaps the arm to joint zero, which overlaps the phantom;
  in `demo` the autopilot backs the tube out (about 1 s of contact) and re-enters.
- Left panel `Rendering` / `Group enable`: geom group 3 shows the arm's collision meshes.

## Layout

```
models/so101/         official SO-101 MJCF + meshes, unmodified (see SOURCE.txt)
so101_common.py       shared by both sims: robot path, checkerboard floor, min-jerk profile
dental_sim/scene.py       Mouth (geometry), DentalSceneBuilder: phantom head, suction tube, target marker
dental_sim/controller.py  SuctionRoute, SuctionTrajectory, SuctionIK, Autopilot -> joint targets at 50 Hz
dental_sim/__main__.py    DentalSimulation, RunReport, CheckCommand, CLI (demo / manual / check)
stack_sim/scene.py    SceneBuilder: table, three 25 mm cubes, stack pad, overview/front/wrist cameras
stack_sim/controller.py   JawProbe, StackPlan, ArmIK, Autopilot -> joint targets at 50 Hz
stack_sim/__main__.py     StackSimulation + CheckCommand, CLI (demo / manual / check)
isaac/                    Isaac Sim + LeIsaac + GR00T workflow for an NVIDIA GPU machine, see isaac/README.md
```

## jaw_sim: your jaw STLs + saliva ejector + forbidden zones

| Command | What it does |
| --- | --- |
| `uv run mjpython -m jaw_sim drag` | Viewer; you steer the ejector tip with the green marker. The arm follows only where allowed; the marker turns red when blocked |
| `uv run mjpython -m jaw_sim route` | Viewer; automatic: insert the hook flat, roll it down behind the lower front teeth, sweep left/right |
| `uv run python -m jaw_sim check --png out/` | Headless route run; prints how often the zones blocked the target |

Viewer (needs `mjpython`, installed with mujoco):
- Move the marker: double-click it, then Ctrl + right-drag. Roll the hook: Ctrl + left-drag (red stick on the marker = hook direction).
- Top-left text: tip position in world mm and in both jaw STL frames (mm), hook roll, allowed/blocked.
- Red/green/blue arrows at each jaw origin = that STL's x/y/z axes.
- Enter: calibration mode. Asks for 4 reference points (`jaw_sim/calibration.py`); place the marker, press Enter per point; saved to `calibration.json`.

Code:
- Your algorithm: `jaw_sim/zones.py`, `lower_jaw_forbidden(x, y, z)` and `upper_jaw_forbidden(x, y, z)`, millimetres in each STL's own frame.
  Current bodies are an example: forbidden inside the cast or within 3 mm of its surface (works for any jaw scan).
- `jaw_sim/jaw.py` places any `upper_teeth.stl` / `lower_teeth.stl` from their measured size (teeth toward +y, incisors toward -z, mm).
  Files above 200 000 faces (MuJoCo's limit) get a simplified copy in `models/jaw/simplified/`.
- `jaw_sim/ejector.py` joins `models/sucker/saliva_ejector_solid_half_A/B.stl` into one tube; the gripper holds the straight shaft,
  25 mm from the hose end; the control point is the open hook end.
- Jaws and ejector are visual only (MuJoCo would collide them as their convex hulls). Only the tip is checked against the zones,
  not the rest of the hook.

## stack_sim: block stacking

Second task on the same arm: pick up three cubes and stack them. Same interface as `dental_sim`
(50 Hz joint position targets), so it transfers to the follower arm the same way.

| Command | What it does |
| --- | --- |
| `uv run python -m stack_sim demo` | Viewer window; the arm stacks red, green, blue on the marked pad |
| `uv run python -m stack_sim manual` | Viewer window without autopilot |
| `uv run python -m stack_sim check --png out/` | Headless: runs the plan, prints cube positions, stack height and verdict, saves images; exit code 1 if the stack failed |

All three accept `--seed N`, which jitters the cube start poses by +/-10 mm and +/-20 deg.

Per cube the autopilot does: pre-grasp above the cube, descend, close the jaws, lift, carry over
the stack, descend to the next level, open, retreat. Targets are top-down with `wrist_roll` matched
to the cube's yaw. The cubes are moved by jaw contact and friction only - no welds, no mocap.

Known limitations:

- Top-down reach runs out around z = 0.09 m, which is why the cubes are 25 mm: a fourth level
  would be outside the workspace.
- The scene sets `impratio=15` and an elliptic friction cone. With MuJoCo's defaults the cube
  creeps out from between the jaws under the clamp force.
- The SO-101 collision meshes are convex hulls, so the jaw faces are a shallow wedge rather than
  flat pads. `JawProbe` measures that wedge at startup; a cube is pinched near its upper edges,
  not face-on. Real jaws (or printed pads) would grip differently.
- Cube poses are read from the simulator state, i.e. perfect perception. On hardware this needs
  a camera and a pose estimate.
- Cube mass (22 g) and friction are plausible, not measured.

## Model notes

- Joint names match LeRobot's SO-101 follower: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.
  Units differ: MuJoCo uses radians with zero at mid-range ("new calibration"); LeRobot reports
  degrees or normalized values depending on configuration. Convert before sending sim targets to hardware.
- `wrist_roll` is coaxial with the tube and held at 0; the gripper stays closed on the tube.
  The tube is rigidly attached to the gripper body in sim; on hardware it needs a clamp or printed holder.
- Mouth geometry is a primitive placeholder (parabolic arches, 14 teeth per jaw, 50 mm gum gap) set in
  `Mouth` in `dental_sim/scene.py`. A scanned or printed jaw STL can replace it as a mesh geom.
- Servo gains/force limits come from the vendored model (STS3215 estimates), not from a measured arm.
