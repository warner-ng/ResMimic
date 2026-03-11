# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Create overall workspace
WORKSPACE_DIR=$SCRIPT_DIR/thirdparty
CONDA_ROOT=$WORKSPACE_DIR/miniconda3
ENV_ROOT=$CONDA_ROOT/envs/rl-motion
SENTINEL_FILE=.env_setup_finished

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  # Install miniconda
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o $CONDA_ROOT/miniconda.sh
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n rl-motion python=3.8
  fi

  source $CONDA_ROOT/bin/activate rl-motion

  # Install libstdcxx-ng to fix the error: `version `GLIBCXX_3.4.32' not found` on Ubuntu 24.04 
  conda install -c conda-forge -y libstdcxx-ng

  # Install Isaac Gym
  if [[ ! -d $WORKSPACE_DIR/isaacgym ]]; then
    wget https://developer.nvidia.com/isaac-gym-preview-4 -O $WORKSPACE_DIR/IsaacGym_Preview_4_Package.tar.gz
    tar -xzf $WORKSPACE_DIR/IsaacGym_Preview_4_Package.tar.gz -C $WORKSPACE_DIR
    cd $WORKSPACE_DIR/isaacgym/python
    $ENV_ROOT/bin/pip install -e .
  fi

  cd $SCRIPT_DIR
  pip install -e rsl_rl
  pip install -e legged_gym
  pip install -e pose

  pip install "numpy==1.23.0" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill gdown hydra-core imageio[ffmpeg] mujoco mujoco-python-viewer isaacgym-stubs pytorch-kinematics rich termcolor 
  pip install scipy
  pip install "redis[hiredis]"
  if ! which redis-server; then
    sudo apt install -y redis-server
  fi

  pip install pyttsx3 # for voice control
  pip install trimesh
  touch $SENTINEL_FILE
fi
