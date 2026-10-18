#!/usr/bin/env bash
# Bootstrap a Nebius GPU VM (Ubuntu 22.04/24.04 with NVIDIA driver, e.g. image family ubuntu24.04-cuda12)
# for: Isaac Sim 5.1 + Isaac Lab 2.3 + LeIsaac (SO-101) + our StackCubes task + GR00T N1.6.
#
# Usage (on the VM, from the copied repo):  bash isaac/setup_vm.sh [all|isaac|gr00t]
# Idempotent: re-running skips finished steps. Logs to ~/hackathon/setup.log.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${WORK:-$HOME/hackathon}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
LEISAAC_DIR="$WORK/leisaac"
GR00T_DIR="$WORK/Isaac-GR00T"
# Pinned versions: our task code and patch are written against these commits.
LEISAAC_REF=24d3bcd3f1e4585740fc79921782c41617237812  # 2026-04-15, IsaacSim 5.1 / IsaacLab 2.3.0
GR00T_REF=e8e625f4f21898c506a1d8f7d20a289c97a52acf    # GR00T N1.6, the version LeIsaac's gr00tn1.6 client targets

mkdir -p "$WORK"
exec > >(tee -a "$WORK/setup.log") 2>&1
step() { printf '\n=== %s ===\n' "$*"; }

preflight() {
  step "GPU check"
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi missing: use a GPU image with the NVIDIA driver"; exit 1; }
  local gpu; gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "GPU: $gpu"
  case "$gpu" in
    *H100*|*H200*|*B200*|*B300*|*A100*)
      echo "WARNING: $gpu has no RT cores. Isaac Sim rendering/cameras will not work."
      echo "         Fine for GR00T fine-tuning ('gr00t' target); use L40S or RTX PRO 6000 for Isaac Sim." ;;
  esac
  step "System packages"
  sudo apt-get update -y
  sudo apt-get install -y git git-lfs cmake build-essential ffmpeg curl unzip \
    libglu1-mesa libxt6 libxi6 libxrender1 libxrandr2 libxkbcommon0 libsm6 libegl1
  git lfs install
}

conda_env() {
  if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    step "Miniconda"
    curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
  fi
  set +u  # conda's activate scripts reference unset variables; restored after activation
  # shellcheck disable=SC1091
  source "$CONDA_DIR/etc/profile.d/conda.sh"
  "$CONDA_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
  "$CONDA_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
}

setup_isaac() {
  conda_env
  if ! conda env list | grep -q '^leisaac '; then
    step "conda env leisaac (Python 3.11)"
    conda create -y -n leisaac python=3.11
  fi
  conda activate leisaac
  # nounset stays off: conda's activate/deactivate hooks (run again by `conda install`) use unset variables
  export OMNI_KIT_ACCEPT_EULA=YES

  if ! python -c "import torch" 2>/dev/null; then
    step "CUDA toolkit 12.8 + PyTorch 2.7"
    conda install -y -c "nvidia/label/cuda-12.8.1" cuda-toolkit
    pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
  fi
  if ! python -c "import isaacsim" 2>/dev/null; then
    step "Isaac Sim 5.1 (large download)"
    pip install --upgrade pip
    pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
  fi

  if [ ! -d "$LEISAAC_DIR/.git" ]; then
    step "LeIsaac @ $LEISAAC_REF"
    git clone https://github.com/LightwheelAI/leisaac.git "$LEISAAC_DIR"
  fi
  cd "$LEISAAC_DIR"
  if [ "$(git rev-parse HEAD)" != "$LEISAAC_REF" ]; then
    git fetch origin "$LEISAAC_REF" && git checkout -q "$LEISAAC_REF"
  fi
  git submodule update --init --recursive

  if ! python -c "import isaaclab" 2>/dev/null; then
    step "Isaac Lab (LeIsaac submodule)"
    # flatdict (an Isaac Lab dependency) has a setup.py that imports pkg_resources, which current
    # setuptools no longer ships: build it once against an older setuptools, without isolation.
    pip install "setuptools<81"
    pip install --no-build-isolation flatdict==4.0.1
    (cd dependencies/IsaacLab && ./isaaclab.sh --install)
  fi

  step "LeIsaac + LeRobot 0.3.3 (dataset v2 conversion for GR00T N1.6)"
  # evdev (a LeRobot dependency) must be compiled with the system gcc: the conda CUDA toolkit's
  # compiler ships older kernel headers than the /usr/include/linux/input.h evdev reads.
  CC=/usr/bin/gcc pip install evdev
  pip install -e "source/leisaac[gr00t,remote]"
  pip install lerobot==0.3.3
  pip install numpy==1.26.0  # LeIsaac docs: lerobot pulls a numpy that breaks Isaac Lab

  step "Assets: SO-101 follower USD + table_with_cube scene"
  mkdir -p assets/robots assets/scenes
  [ -f assets/robots/so101_follower.usd ] || curl -fL -o assets/robots/so101_follower.usd \
    https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd
  if [ ! -f assets/scenes/table_with_cube/scene.usd ]; then
    curl -fL -o /tmp/table_with_cube.zip https://github.com/LightwheelAI/leisaac/releases/download/v0.1.2/table_with_cube.zip
    unzip -oq /tmp/table_with_cube.zip -d assets/scenes
  fi

  step "StackCubes task + scripted demo generator"
  ln -sfn "$REPO/isaac/leisaac_ext/stack_cubes" source/leisaac/leisaac/tasks/stack_cubes
  ln -sfn "$REPO/isaac/leisaac_ext/stack_cubes_state_machine.py" \
    source/leisaac/leisaac/datagen/state_machine/stack_cubes.py
  if git apply --check "$REPO/isaac/leisaac_ext/register_stack_cubes.patch" 2>/dev/null; then
    git apply "$REPO/isaac/leisaac_ext/register_stack_cubes.patch"
  fi
  grep -q StackCubesStateMachine scripts/datagen/state_machine/generate.py

  step "Import check"
  python - <<'EOF'
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym
import leisaac.tasks  # noqa: F401
from leisaac.datagen.state_machine.stack_cubes import StackCubesStateMachine  # noqa: F401
assert "LeIsaac-SO101-StackCubes-v0" in gym.registry, "StackCubes not registered"
print("OK: LeIsaac-SO101-StackCubes-v0 registered")
app.close()
EOF
}

setup_gr00t() {
  if ! command -v uv >/dev/null; then
    step "uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  export PATH="$HOME/.local/bin:$PATH"
  if [ ! -d "$GR00T_DIR/.git" ]; then
    step "Isaac-GR00T @ $GR00T_REF"
    git clone https://github.com/NVIDIA/Isaac-GR00T.git "$GR00T_DIR"
  fi
  cd "$GR00T_DIR"
  if [ "$(git rev-parse HEAD)" != "$GR00T_REF" ]; then
    git fetch origin "$GR00T_REF" && git checkout -q "$GR00T_REF"
  fi
  git submodule update --init --recursive
  step "GR00T env (Python 3.10, flash-attn; slow first build)"
  uv sync --python 3.10
  uv pip install -e .
  uv run python -c "import gr00t; print('OK: gr00t import')"
}

target="${1:-all}"
preflight
case "$target" in
  isaac) setup_isaac ;;
  gr00t) setup_gr00t ;;
  all) setup_isaac; setup_gr00t ;;
  *) echo "usage: $0 [all|isaac|gr00t]"; exit 2 ;;
esac
step "Done. Next: bash isaac/run.sh help"
