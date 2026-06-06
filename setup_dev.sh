#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WORKSPACE_DIR="$SCRIPT_DIR/thirdparty"
ENV_NAME="${ENV_NAME:-rl-motion}"
SENTINEL_FILE=".env_setup_finished_${ENV_NAME}_local"

mkdir -p "$WORKSPACE_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH. Activate your local conda first."
  exit 1
fi

eval "$(conda shell.bash hook)"

if [[ ! -f "$SCRIPT_DIR/$SENTINEL_FILE" ]]; then
  if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.8
  fi

  conda activate "$ENV_NAME"

  # Fix newer Ubuntu libstdc++ mismatch inside the active env.
  conda install -c conda-forge -y libstdcxx-ng

  # Isaac Gym source is stored under repo-local thirdparty/.
  if [[ ! -d "$WORKSPACE_DIR/isaacgym" ]]; then
    wget https://developer.nvidia.com/isaac-gym-preview-4 -O "$WORKSPACE_DIR/IsaacGym_Preview_4_Package.tar.gz"
    tar -xzf "$WORKSPACE_DIR/IsaacGym_Preview_4_Package.tar.gz" -C "$WORKSPACE_DIR"
  fi

  # Install Isaac Gym into the currently active env even if the source dir already exists.
  if ! python -c "import isaacgym" >/dev/null 2>&1; then
    cd "$WORKSPACE_DIR/isaacgym/python"
    pip install -e .
  fi

  cd "$SCRIPT_DIR"
  pip install -e rsl_rl
  pip install -e legged_gym
  pip install -e pose

  pip install "numpy==1.23.0" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill gdown hydra-core imageio[ffmpeg] mujoco mujoco-python-viewer isaacgym-stubs pytorch-kinematics rich termcolor
  pip install scipy
  pip install "redis[hiredis]"

  if ! command -v redis-server >/dev/null 2>&1; then
    sudo apt install -y redis-server
  fi

  pip install pyttsx3
  pip install trimesh
  touch "$SCRIPT_DIR/$SENTINEL_FILE"
else
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
fi

echo "[READY] Environment '$ENV_NAME' is active."
echo "[INFO] If needed later, activate it with: conda activate $ENV_NAME"