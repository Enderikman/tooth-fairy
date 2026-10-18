#!/usr/bin/env bash
# Isaac Sim workflows on the GPU VM: dental demo (jaw_sim brain) and the stack-3-cubes GR00T pipeline.
# Run `bash isaac/run.sh help`.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${WORK:-$HOME/hackathon}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
LEISAAC_DIR="$WORK/leisaac"
GR00T_DIR="$WORK/Isaac-GR00T"
TASK=LeIsaac-SO101-StackCubes-v0
DATA_DIR="$WORK/datasets"
HDF5="$DATA_DIR/stack_cubes.hdf5"
REPO_ID="${REPO_ID:-local/so101_stack_cubes}"
LEROBOT_DS="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/$REPO_ID"
CKPT_DIR="$WORK/checkpoints/stack_cubes"
PROMPT="Stack the blue cube on the red cube, then the green cube on the blue cube."
# LIVESTREAM=1 (default) streams the Isaac Sim window over WebRTC (Isaac Sim WebRTC Streaming Client,
# TCP 49100 + UDP 47998); the VM's public IP is advertised, auto-detected unless PUBLIC_IP is set.
# LIVESTREAM=window opens a normal window (remote desktop on the VM); LIVESTREAM=0 runs headless.
case "${LIVESTREAM:-1}" in
  1)
    PUBLIC_IP="${PUBLIC_IP:-$(curl -fsS --max-time 5 https://api.ipify.org || true)}"
    kit="--no-window --enable omni.kit.livestream.webrtc"
    [ -n "$PUBLIC_IP" ] && kit+=" --/app/livestream/publicEndpointAddress=$PUBLIC_IP --/app/livestream/port=49100"
    VIEW=(--kit_args="$kit") ;;
  window) VIEW=() ;;
  *) VIEW=(--headless) ;;
esac

leisaac() {
  set +u
  # shellcheck disable=SC1091
  source "$CONDA_DIR/etc/profile.d/conda.sh" && conda activate leisaac
  # nounset stays off: conda's activate/deactivate hooks use unset variables
  export OMNI_KIT_ACCEPT_EULA=YES
  cd "$LEISAAC_DIR"
}

cmd="${1:-help}"
shift || true
case "$cmd" in
  teleop)  # teleop [keyboard|remote]  remote = leader arm on your laptop via `ssh -R 5556:localhost:5556`
    leisaac
    mode="${1:-keyboard}"
    dev=(--teleop_device=keyboard)
    [ "$mode" = remote ] && dev=(--teleop_device=so101leader --remote_endpoint=tcp://127.0.0.1:5556)
    mkdir -p "$DATA_DIR"
    rec=(--record --dataset_file="$DATA_DIR/stack_cubes_teleop.hdf5")
    [ -f "$DATA_DIR/stack_cubes_teleop.hdf5" ] && rec+=(--resume)
    python scripts/environments/teleoperation/teleop_se3_agent.py --task="$TASK" "${dev[@]}" \
      --num_envs=1 --device=cuda --enable_cameras "${rec[@]}" "${VIEW[@]}"
    ;;
  datagen)  # datagen [N]  scripted demos until N successes; convert keeps only successful episodes
    leisaac
    mkdir -p "$DATA_DIR"
    rec=(--record --dataset_file="$HDF5" --num_demos="${1:-50}")
    [ -f "$HDF5" ] && rec+=(--resume)
    python scripts/datagen/state_machine/generate.py --task "$TASK" --num_envs 1 --device cuda \
      --enable_cameras "${rec[@]}" "${VIEW[@]}"
    ;;
  replay)  # replay [EPISODE...]
    leisaac
    sel=()
    [ $# -gt 0 ] && sel=(--select_episodes "$@")
    python scripts/datagen/state_machine/replay.py --task "$TASK" --dataset_file "$HDF5" \
      --task_type so101_state_machine --device cuda --enable_cameras --replay_mode action \
      "${sel[@]}" "${VIEW[@]}"
    ;;
  convert)  # convert [HDF5]  successful episodes -> LeRobot v2 dataset with joint-space actions
    leisaac
    python scripts/convert/isaaclab2lerobot.py --task_name="$TASK" --repo_id="$REPO_ID" \
      --hdf5_root="$DATA_DIR" --hdf5_files="$(basename "${1:-$HDF5}")"
    cp "$GR00T_DIR/examples/SO100/modality.json" "$LEROBOT_DS/meta/modality.json"
    echo "dataset: $LEROBOT_DS"
    ;;
  finetune)  # finetune [MAX_STEPS]  GR00T N1.6 on the converted dataset
    cd "$GR00T_DIR"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run python gr00t/experiment/launch_finetune.py \
      --base_model_path nvidia/GR00T-N1.6-3B \
      --dataset_path "$LEROBOT_DS" \
      --modality_config_path examples/SO100/so100_config.py \
      --embodiment_tag NEW_EMBODIMENT \
      --num_gpus 1 \
      --output_dir "$CKPT_DIR" \
      --save_steps 1000 --save_total_limit 3 \
      --max_steps "${1:-5000}" \
      --global_batch_size 32 \
      --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
      --dataloader_num_workers 4
    ;;
  serve)  # serve [CHECKPOINT_DIR]  GR00T policy server on :5555
    cd "$GR00T_DIR"
    ckpt="${1:-$(ls -d "$CKPT_DIR"/checkpoint-* | sort -t- -k2 -n | tail -1)}"
    uv run python gr00t/eval/run_gr00t_server.py --model-path "$ckpt" --embodiment-tag NEW_EMBODIMENT
    ;;
  eval)  # eval [ROUNDS]  closed loop in Isaac Sim against `serve`
    leisaac
    python scripts/evaluation/policy_inference.py --task="$TASK" --eval_rounds="${1:-10}" \
      --policy_type=gr00tn1.6 --policy_host=localhost --policy_port=5555 \
      --policy_action_horizon=16 --policy_language_instruction="$PROMPT" \
      --device=cuda --enable_cameras "${VIEW[@]}"
    ;;
  dental)  # dental [route|teeth] [extra args]  jaw_sim brain driving the SO-101 in Isaac Sim (e.g. --video out/x.mp4 --seconds 30)
    leisaac
    export LEISAAC_ASSETS_ROOT="$LEISAAC_DIR/assets"  # we run from our repo, not from the LeIsaac checkout
    cd "$REPO"
    mode="${1:-route}"
    shift || true
    python -m isaac.dental.run_isaac --mode "$mode" --device cuda "$@" "${VIEW[@]}"
    ;;
  *)
    sed -n 's/^  [a-z]*)  # /  /p' "$0"
    echo "env: LIVESTREAM=1|window|0, PUBLIC_IP, REPO_ID, WORK"
    ;;
esac
