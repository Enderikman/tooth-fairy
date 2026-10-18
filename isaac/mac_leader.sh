#!/usr/bin/env bash
# Laptop side of remote teleoperation: reads the SO-101 *leader* arm over USB and publishes its joint
# states on tcp://0.0.0.0:5556 (ZMQ). The VM reaches it through an SSH reverse tunnel:
#   ssh -R 5556:localhost:5556 <user>@<vm-ip>      then on the VM: bash isaac/run.sh teleop remote
# Usage: bash isaac/mac_leader.sh [/dev/tty.usbmodemXXXX] [--recalibrate]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$REPO/isaac/.vendor"
LEISAAC_REF=24d3bcd3f1e4585740fc79921782c41617237812  # keep in sync with isaac/setup_vm.sh

if [ ! -d "$VENDOR/leisaac/.git" ]; then
  git clone -q https://github.com/LightwheelAI/leisaac.git "$VENDOR/leisaac"
fi
git -C "$VENDOR/leisaac" fetch -q origin "$LEISAAC_REF"
git -C "$VENDOR/leisaac" checkout -q "$LEISAAC_REF"
if [ ! -x "$VENDOR/venv/bin/python" ]; then
  uv venv -q --python 3.11 "$VENDOR/venv"
  VIRTUAL_ENV="$VENDOR/venv" uv pip install -q "$VENDOR/leisaac/source/leisaac[remote]"
fi

ports=(/dev/tty.usbmodem*)
port="${1:-}"
[ -n "$port" ] || [ ! -e "${ports[0]}" ] || port="${ports[0]}"
[ -n "$port" ] || { echo "No /dev/tty.usbmodem* found: plug in the leader arm or pass the port"; exit 1; }
shift || true
echo "Leader arm on $port -> publishing tcp://0.0.0.0:5556"
exec "$VENDOR/venv/bin/python" "$VENDOR/leisaac/scripts/environments/teleoperation/so101_joint_state_server.py" \
  --port "$port" --id leader_arm --rate 50 "$@"
