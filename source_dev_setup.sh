SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CONDA_ROOT="${SCRIPT_DIR}/thirdparty/miniconda3"

if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate rl-motion
else
    source "${CONDA_ROOT}/bin/activate" rl-motion
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${CONDA_ROOT}/envs/rl-motion/lib"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"