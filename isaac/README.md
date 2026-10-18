# Isaac Sim on the GPU VM: dental demo + stack-3-cubes GR00T pipeline

Nothing in this folder can run on the Mac except `mac_leader.sh`: Isaac Sim needs Linux plus an NVIDIA GPU with RT cores.
The MuJoCo sims in the repo root are the local, GPU-free counterpart.

## Dental demo (jaw_sim brain in Isaac Sim)

```bash
ssh isaac
cd ~/hackathonNebius_2026
bash isaac/run.sh dental route                                   # automatic route, WebRTC stream
bash isaac/run.sh dental teeth                                   # operate, then teeth 46 -> 43 -> 33 -> 36 -> park
LIVESTREAM=0 bash isaac/run.sh dental teeth --seconds 25 --video out/dental_teeth.mp4   # headless video
```

- `dental/brain.py` (`DentalBrain`): runs jaw_sim unchanged (forbidden zones, routes, IK) on a headless MuJoCo model
  and returns six joint targets in radians. These are the same values the real SO-101 follower takes via LeRobot.
- `dental/run_isaac.py`: LeIsaac's SO-101 USD + your jaw STLs + the saliva ejector + forbidden-zone dots + target
  marker (green / red when blocked). Every 20 ms: brain targets -> articulation position targets.
- Frames: LeIsaac's robot USD is rotated 90 deg and offset 33 mm against the MJCF the brain uses. The robot root is
  placed with a fitted transform so both worlds coincide (25 random poses, 3.2 mm mean / 6.8 mm max residual).
- A slow per-joint integral trim removes the ~2 deg gravity sag of the simulated servos.
  Measured on the `teeth` run: Isaac ejector tip within 0.9-6.2 mm of the brain's tip.
- After editing jaw_sim on the Mac: `rsync -a --exclude .venv --exclude .git ./ isaac:~/hackathonNebius_2026/`.

## Stack-3-cubes GR00T pipeline

Pipeline: Isaac Sim scene -> demonstrations (scripted or teleoperated) -> LeRobot dataset ->
GR00T N1.6 fine-tune -> closed-loop evaluation in Isaac Sim -> real SO-101.

## 1. Get a GPU machine

| Need | Nebius platform | Notes |
| --- | --- | --- |
| Isaac Sim (rendering, cameras) | `gpu-l40s-a` / `gpu-l40s-d` (L40S 48 GB) or `gpu-rtx6000` (RTX PRO 6000) | RT cores required |
| GR00T fine-tuning | same L40S/RTX PRO, or `gpu-h100-sxm` | 40 GB+ VRAM recommended |

H100/H200/B200 have no RT cores: GR00T training works there, Isaac Sim does not.
One L40S or RTX PRO VM can do everything.

Nebius console: Compute -> Virtual machines -> Create: 1 GPU of the platform above, boot image
`ubuntu24.04-cuda12` (NVIDIA driver preinstalled), disk 300 GB+, public IP, your SSH key.
For the Isaac Sim window, allow inbound TCP 49100 and UDP 47998 from your IP (WebRTC livestream).

If the organisers hand out the Nebius Physical AI Workbench (`npa` CLI, github.com/nebius/nebius-physical-ai)
instead of a raw VM: its `npa workbench leisaac` only launches the pinned LeIsaac tasks (PickOrange, LiftCube),
not our StackCubes task. Use a VM for this workflow.

## 2. Install (on the VM)

```bash
rsync -a --exclude .venv --exclude isaac/.vendor ./ <user>@<vm-ip>:~/hackathonNebius_2026/   # from the Mac
ssh <user>@<vm-ip>
cd ~/hackathonNebius_2026 && bash isaac/setup_vm.sh all        # ~1 h first time, log: ~/hackathon/setup.log
```

Installs, pinned: Isaac Sim 5.1.0, Isaac Lab (LeIsaac submodule, v2.3.0), LeIsaac `24d3bcd`, LeRobot 0.3.3,
GR00T N1.6 (`Isaac-GR00T@e8e625f`, the version LeIsaac's `gr00tn1.6` client targets), SO-101 + table scene USDs.
It symlinks our task into LeIsaac and finishes with an import check that `LeIsaac-SO101-StackCubes-v0` is registered.

## 3. Look at the scene

Install the "Isaac Sim WebRTC Streaming Client" (v1.1.5, macOS arm64 build on the
[Isaac Sim download page](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)), then:

```bash
bash isaac/run.sh teleop keyboard     # on the VM; connect the client to <vm-ip>
```

First launch compiles shaders for several minutes. In the client, enter only the VM IP (no port).
`run.sh` advertises the VM's public IP to the stream automatically (`PUBLIC_IP=...` overrides).
Apple Silicon users have reported grey/black screens with this client. Fallback: a remote desktop
(e.g. NoMachine) on the VM and `LIVESTREAM=window bash isaac/run.sh ...`. `LIVESTREAM=0` runs headless.
Keyboard teleop key bindings are printed by LeIsaac at start-up.

## 4. Demonstrations

| Command | What it does |
| --- | --- |
| `bash isaac/run.sh datagen 50` | Scripted state machine: blue onto red, green onto blue; records until 50 successes |
| `bash isaac/run.sh replay 0 1` | Replays recorded episodes |
| `bash isaac/run.sh teleop remote` | Human demos with the SO-101 leader arm plugged into the Mac (below) |

Leader arm on the Mac, simulation on the VM:

```bash
bash isaac/mac_leader.sh                         # Mac: reads the leader arm, publishes on :5556
ssh -R 5556:localhost:5556 <user>@<vm-ip>        # Mac: tunnel, keep open
bash isaac/run.sh teleop remote                  # VM (inside that ssh session)
```

## 5. Train and evaluate GR00T

```bash
bash isaac/run.sh convert         # HDF5 -> LeRobot v2 at ~/.cache/huggingface/lerobot/local/so101_stack_cubes
bash isaac/run.sh finetune 5000   # GR00T-N1.6-3B, NEW_EMBODIMENT, SO-100 modality config (front + wrist camera)
bash isaac/run.sh serve           # terminal 1: policy server on :5555 (latest checkpoint)
bash isaac/run.sh eval 10         # terminal 2: 10 closed-loop episodes in Isaac Sim, prints success
```

For the real arm, the same checkpoint and server work with GR00T's
`gr00t/eval/real_robot/SO100/eval_so100.py` client (see Isaac-GR00T `examples/SO100/README.md`).

## Our code

| File | Purpose |
| --- | --- |
| `leisaac_ext/stack_cubes/` | Task `LeIsaac-SO101-StackCubes-v0`: table scene, red cube from USD + spawned blue/green 3 cm cubes, +-2 cm / +-15 deg randomisation, success = 3 cubes stacked in any order, resting on the table |
| `leisaac_ext/stack_cubes_state_machine.py` | Scripted demo generator (IK pose targets, same offsets as LeIsaac's PickOrange machine) |
| `leisaac_ext/register_stack_cubes.patch` | Registers the state machine in LeIsaac's `generate.py` |
| `setup_vm.sh`, `run.sh` | VM install and workflow commands |
| `mac_leader.sh` | Leader-arm bridge on the Mac (tested up to opening the serial port) |

Dataset actions are exported as joint targets (`joint_pos_target`), not the state machine's EE poses,
so scripted and teleoperated demos share the 6-D action space GR00T's SO-100 config expects.

## Status

Not yet run on a GPU: the task, state machine, setup and run scripts are written against the pinned
sources and checked for syntax only. Expect to tune on first contact:
- `_GRASP_XY_OFFSET` / `_GRIPPER_OFFSET` / phase timings in the state machine (copied from PickOrange, which grasps oranges, not cubes).
- Whether the identity EE orientation gives a top-down grasp that fits a 3 cm cube.
- Success thresholds in `cubes_stacked` (xy 1.5 cm, z 8 mm).
