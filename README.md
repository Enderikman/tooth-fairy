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
| `uv run mjpython -m jaw_sim route` | Viewer; automatic: park in front of the mouth, enter flat, lower the hook next to tooth 46, sweep 46 <-> 43 |
| `uv run python -m jaw_sim check --png out/` | Headless: runs the route, then the commands 46/43/33/36/park; prints blocked ticks and tip error per command |
| `solo robo --calibrate leader` | One-time LeRobot calibration of the leader arm (guided; writes homing offsets into the servos so the middle pose reads 2047) |
| `uv run mjpython -m jaw_sim leader` | Viewer; the simulated arm copies the real leader arm joint by joint (Feetech bus over `/dev/tty.usbmodem*`, `--port` to pick one). Zones still apply; `a`/Enter calibrates the jaw with the real tip position (zones off while calibrating). If a joint turns the wrong way, flip it in `SIGNS` in `jaw_sim/leader.py` |
| `JAW_SCALE=1 uv run mjpython -m jaw_sim follower` | Viewer; the real follower arm alone, no leader (same bus board, `--port` to pick one). Starts with torque off: move the arm by hand, the simulated arm copies it. `n` then `m` with the hook tip on the real jaw's incisal edge places the simulated jaw there (one-point calibration, see below). `b` switches torque on and the tooth/park commands drive the real arm along the sim's zone-checked plan; `c` cancels and releases it |

Viewer (needs `mjpython`, installed with mujoco):
- Move the marker: double-click it, then Ctrl + right-drag. Roll the hook: Ctrl + left-drag (red stick on the marker = hook direction).
- Top-left text: tip position in world mm and in both jaw STL frames (mm), hook roll, allowed/blocked.
- Red/green/blue arrows at each jaw origin = that STL's x/y/z axes.
- Enter: calibration mode, 3 points in order: incisal edge between 31 and 41 (on the tooth), tooth 36 lingual gum line, tooth 46 lingual gum line.
  Place the marker on each point and press Enter; after the third point the result is saved to `calibration.json`. Restart the viewer to use it.
  Without a `calibration.json` the hand-measured printed adult jaw is used as default.
- Commands (drag mode) all go through `JawSimulation.command(name)`:
  - Hardware buttons (MK424Pro pad, types the letters; `jaw_sim/buttons.py`, works in any window): `a` calibrate (same as Enter), `b` operate, `c` stop, `d` park.
  - Voice (`uv run mjpython -m jaw_sim drag --voice`, default microphone, local Parakeet v3 via onnx-asr; `jaw_sim/voice.py`): teeth as two digits "vier sechs" / "four six", "stopp", "parken", "start", "kalibrieren". About one second latency, so the `c` button is the fast stop.
  - Viewer keys (window in front): `6`/`7`/`8`/`9` = tooth 46/43/33/36, `.` = park, `Space` = pause (emergency stop of the simulation).
  - Tooth commands only move the arm in operating mode (`b`); `stop` cancels the move and leaves operating mode.

Follower mode (`JAW_SCALE=1 uv run mjpython -m jaw_sim follower`, real arm + real-size jaw, no leader):
- Put the real jaw in front of the arm the way the sim draws it: incisors toward the robot, incisal plane level. Only the position is calibrated, not the orientation.
- Released (start, after `c`): torque off. Move the arm by hand; the screen shows the real joints and where the hook tip is.
- One-point calibration, two separate keys so a bouncing button cannot restart it: `n` = (re)start, point 1 requested; hold the tip on the
  incisal edge between 31 and 41 and press `m` = record. The simulated jaw and its red zone dots jump so that the incisal edge is at the tip;
  everything else is the default jaw (`DEFAULT_*` in `jaw_sim/calibration.py`). `a`/Enter does nothing in this mode.
  Press `c` to keep those defaults, or go on with `m` at points 2 and 3 to record a full `calibration.json` (needs a restart, as in drag mode).
  The offset is saved in `jaw_offset.json` and reloaded on the next start.
- `b`: torque on, the arm holds its pose. `6`/`7`/`8`/`9`/`.` (or voice/buttons) then plan the move in the sim and the real servos follow the
  sim's joint targets at 50 Hz; the gripper is never driven, close it on the ejector by hand. `c` at any time: plan cancelled, torque off.
- A command is refused (shown on screen) when the IK cannot reach every keyframe of the plan from the current joints, which happens from the
  folded rest pose: unfold the arm by hand near the jaw first. The mouse-dragged marker is ignored in this mode; only commands move the arm.
- `jaw_sim/follower.py`: `FollowerArm` (torque / goal-position writes next to `LeaderArm`'s reads), `FollowerTarget` (released = mirror, driving = planner).

Forbidden zones (lower jaw only, from the three calibration points):
- Mouth frame: origin = incisal edge between 31 and 41, x across the arch from 46 to 36, y toward the back of the mouth, depth straight down from the incisal plane.
- Lingual tooth surface is a parabola `y = a * x**2`, with `a` fitted from the gum-line points of 36 and 46.
- Allowed corridor: a 20 mm wide strip on the tongue side of that parabola, running from 46 round to 36.
- Maximum depth below the incisal plane scales with jaw size: `depth_max = 0.0097 * JC + 2.093` mm, where `JC` = arch width * arch depth in mm^2
  (JC 300 mm^2, a small child: 5 mm; JC 1848 mm^2, the printed adult jaw: 20 mm).
- Above the incisal plane everything is allowed, otherwise the hook could not enter the mouth.

Code:
- `jaw_sim/zones.py`: `ForbiddenZones`/`LowerJawCorridor` implement the corridor rule above from a `JawCalibration`.
- `jaw_sim/calibration.py`: `JawCalibration` (the three stored points) and `CalibrationRecorder` (the Enter-key flow).
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
